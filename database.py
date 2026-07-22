"""
database.py
Citizen-report backend. This version adds:
  - Real display names instead of random per-session IDs (stored alongside
    every report and vote, so "who reported/verified this" is answerable)
  - Basic anti-abuse rules: a daily report limit per person, and a location
    sanity check (reports must fall within the Delhi region)
  - Everything else (duplicate detection, auto-verify at 3 upvotes) works
    the same as before.
"""

import sqlite3
import math
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "cleartrace.db"
IMAGES_DIR = Path(__file__).parent / "reports_images"
IMAGES_DIR.mkdir(exist_ok=True)


def resolve_image_path(image_path):
    """Resolve portable filenames and legacy absolute Windows image paths."""
    if not image_path:
        return None

    # Extract only the filename, including from old Windows paths saved by a
    # teammate. New reports store just this filename in SQLite.
    filename = Path(str(image_path).replace("\\", "/")).name
    local_path = IMAGES_DIR / filename

    if local_path.is_file():
        return str(local_path)

    return None


VERIFY_UPVOTES_NEEDED = 3
DUPLICATE_RADIUS_METERS = 200
DUPLICATE_WINDOW_HOURS = 24
MAX_REPORTS_PER_DAY = 10  # basic anti-spam limit per person

SOURCE_CATEGORIES = ["construction", "traffic", "waste_burning", "dust", "industry", "other"]

# Rough bounding box around Delhi NCR — reports outside this are rejected
DELHI_BOUNDS = {"lat_min": 28.30, "lat_max": 28.95, "lon_min": 76.75, "lon_max": 77.45}


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _add_column_if_missing(conn, table_name, column_name, column_type):
    """Apply a small SQLite migration for databases made by older app versions."""
    existing_columns = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in existing_columns:
        conn.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
        )


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS citizen_reports (
        report_id TEXT PRIMARY KEY,
        image_path TEXT,
        lat REAL NOT NULL,
        lon REAL NOT NULL,
        description TEXT,
        category_guess TEXT,
        user_id TEXT,
        reporter_name TEXT,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending'
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS report_votes (
        vote_id TEXT PRIMARY KEY,
        report_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        voter_name TEXT,
        vote_type TEXT NOT NULL DEFAULT 'upvote',
        created_at TEXT NOT NULL,
        UNIQUE(report_id, user_id),
        FOREIGN KEY(report_id) REFERENCES citizen_reports(report_id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS verified_reports (
        report_id TEXT PRIMARY KEY,
        source_type TEXT,
        confidence_score REAL,
        lat REAL,
        lon REAL,
        verified_at TEXT,
        FOREIGN KEY(report_id) REFERENCES citizen_reports(report_id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS health_risk_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        location TEXT,
        forecast_aqi REAL,
        duration_hours REAL,
        user_category TEXT,
        hazard_level TEXT,
        recommendations TEXT,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT,
        location TEXT,
        timestamp TEXT,
        sources_used TEXT
    )
    """)

    # CREATE TABLE IF NOT EXISTS does not add new columns to an existing local
    # database. These migrations keep old teammate databases compatible.
    _add_column_if_missing(conn, "citizen_reports", "reporter_name", "TEXT")
    _add_column_if_missing(conn, "report_votes", "voter_name", "TEXT")

    conn.commit()
    conn.close()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def is_within_delhi(lat, lon):
    b = DELHI_BOUNDS
    return b["lat_min"] <= lat <= b["lat_max"] and b["lon_min"] <= lon <= b["lon_max"]


def reports_today_count(user_id):
    conn = get_conn()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    count = conn.execute(
        "SELECT COUNT(*) c FROM citizen_reports WHERE user_id=? AND created_at >= ?", (user_id, cutoff)
    ).fetchone()["c"]
    conn.close()
    return count


def create_report(
    lat,
    lon,
    description,
    category_guess,
    user_id,
    image_bytes=None,
    image_ext="jpg",
    reporter_name="Anonymous",
):
    if not is_within_delhi(lat, lon):
        return {"status": "rejected", "reason": "Location is outside the Delhi NCR region."}

    if reports_today_count(user_id) >= MAX_REPORTS_PER_DAY:
        return {"status": "rejected", "reason": f"Daily report limit reached ({MAX_REPORTS_PER_DAY}/day)."}

    report_id = str(uuid.uuid4())[:8]
    image_path = None
    if image_bytes is not None:
        safe_extension = str(image_ext).lower().lstrip(".")
        if safe_extension not in {"jpg", "jpeg", "png"}:
            safe_extension = "jpg"

        filename = f"{report_id}.{safe_extension}"
        local_path = IMAGES_DIR / filename

        with open(local_path, "wb") as f:
            f.write(image_bytes)

        # Store only the filename so the row works on every teammate's machine.
        image_path = filename

    conn = get_conn()
    conn.execute(
        """INSERT INTO citizen_reports
           (report_id, image_path, lat, lon, description, category_guess, user_id, reporter_name, created_at, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
        (report_id, image_path, lat, lon, description, category_guess, user_id, reporter_name, _now()),
    )
    conn.commit()
    conn.close()
    return {"status": "created", "report_id": report_id}


def get_nearby_reports(lat, lon, radius_m=3000, include_verified=True):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM citizen_reports ORDER BY created_at DESC").fetchall()
    conn.close()
    out = []
    for r in rows:
        d = _haversine_m(lat, lon, r["lat"], r["lon"])
        if d <= radius_m:
            if not include_verified and r["status"] == "verified":
                continue
            row = dict(r)
            row["distance_m"] = round(d, 1)
            out.append(row)
    return sorted(out, key=lambda x: x["distance_m"])


def vote_report(report_id, user_id, vote_type="upvote", voter_name="Anonymous"):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO report_votes (vote_id, report_id, user_id, voter_name, vote_type, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4())[:8], report_id, user_id, voter_name, vote_type, _now()),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return {"status": "already_voted"}

    upvotes = cur.execute(
        "SELECT COUNT(*) c FROM report_votes WHERE report_id=? AND vote_type='upvote'", (report_id,)
    ).fetchone()["c"]

    report = cur.execute("SELECT * FROM citizen_reports WHERE report_id=?", (report_id,)).fetchone()
    confidence = min(1.0, upvotes / VERIFY_UPVOTES_NEEDED)

    result = {"status": "voted", "upvotes": upvotes, "confidence_score": round(confidence, 2)}

    if upvotes >= VERIFY_UPVOTES_NEEDED and report["status"] != "verified":
        is_dup = _is_duplicate(cur, report)
        if not is_dup:
            cur.execute("UPDATE citizen_reports SET status='verified' WHERE report_id=?", (report_id,))
            cur.execute(
                """INSERT OR REPLACE INTO verified_reports
                   (report_id, source_type, confidence_score, lat, lon, verified_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (report_id, report["category_guess"], confidence, report["lat"], report["lon"], _now()),
            )
            result["status"] = "verified"
        else:
            cur.execute("UPDATE citizen_reports SET status='duplicate' WHERE report_id=?", (report_id,))
            result["status"] = "duplicate_rejected"

    conn.commit()
    conn.close()
    return result


def get_voters(report_id):
    """Who verified/upvoted a given report — answers 'who verified this?'"""
    conn = get_conn()
    rows = conn.execute(
        "SELECT voter_name, created_at FROM report_votes WHERE report_id=? ORDER BY created_at", (report_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _is_duplicate(cur, report):
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=DUPLICATE_WINDOW_HOURS)).isoformat()
    others = cur.execute(
        "SELECT * FROM verified_reports WHERE verified_at >= ?", (cutoff,)
    ).fetchall()
    for o in others:
        if o["report_id"] == report["report_id"]:
            continue
        if _haversine_m(report["lat"], report["lon"], o["lat"], o["lon"]) <= DUPLICATE_RADIUS_METERS:
            return True
    return False


def get_verified_reports():
    conn = get_conn()
    rows = conn.execute("""
        SELECT vr.*, cr.description, cr.image_path, cr.reporter_name
        FROM verified_reports vr
        JOIN citizen_reports cr ON cr.report_id = vr.report_id
        ORDER BY vr.verified_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_citizen_features(lat, lon, radius_km=1.0):
    verified = get_verified_reports()
    nearby = [v for v in verified if _haversine_m(lat, lon, v["lat"], v["lon"]) <= radius_km * 1000]
    category_counts = {}
    for v in nearby:
        cat = v["source_type"] or "other"
        category_counts[cat] = category_counts.get(cat, 0) + 1
    return {
        "reported_source_nearby": len(nearby) > 0,
        "citizen_source_score": round(sum(v["confidence_score"] for v in nearby), 2),
        "report_density_1km": len(nearby),
        "source_category_count": category_counts,
    }


def log_health_risk(location, forecast_aqi, duration_hours, user_category, hazard_level, recommendations):
    conn = get_conn()
    conn.execute(
        """INSERT INTO health_risk_logs
           (location, forecast_aqi, duration_hours, user_category, hazard_level, recommendations, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (location, forecast_aqi, duration_hours, user_category, hazard_level, recommendations, _now()),
    )
    conn.commit()
    conn.close()


def log_chat(question, answer, location, sources_used):
    conn = get_conn()
    conn.execute(
        "INSERT INTO chat_logs (question, answer, location, timestamp, sources_used) VALUES (?, ?, ?, ?, ?)",
        (question, answer, location, _now(), sources_used),
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("Initialized cleartrace.db at", DB_PATH)
