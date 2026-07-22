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
import base64
import hashlib
import hmac
import secrets
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
SESSION_LIFETIME_DAYS = 7
PASSWORD_HASH_ITERATIONS = 310_000

SOURCE_CATEGORIES = ["construction", "traffic", "waste_burning", "dust", "industry", "other"]

# Rough bounding box around Delhi NCR — reports outside this are rejected
DELHI_BOUNDS = {"lat_min": 28.30, "lat_max": 28.95, "lon_min": 76.75, "lon_max": 77.45}


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
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
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        email TEXT NOT NULL UNIQUE COLLATE NOCASE,
        display_name TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        password_salt TEXT NOT NULL,
        created_at TEXT NOT NULL,
        last_login_at TEXT,
        is_active INTEGER NOT NULL DEFAULT 1
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_sessions (
        session_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        token_hash TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        revoked_at TEXT,
        FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
    )
    """)

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

    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_token_hash "
        "ON user_sessions(token_hash)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_user_id "
        "ON user_sessions(user_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_reports_user_id "
        "ON citizen_reports(user_id)"
    )

    conn.commit()
    conn.close()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash_password(password, salt=None):
    """Hash a password with PBKDF2 using only the Python standard library."""
    salt_bytes = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_bytes,
        PASSWORD_HASH_ITERATIONS,
    )
    return (
        base64.b64encode(digest).decode("ascii"),
        base64.b64encode(salt_bytes).decode("ascii"),
    )


def _public_user(row):
    if row is None:
        return None
    return {
        "user_id": row["user_id"],
        "email": row["email"],
        "display_name": row["display_name"],
        "created_at": row["created_at"],
    }


def create_user(email, display_name, password):
    """Create a local ClearTrace account and return its public profile."""
    email = str(email).strip().lower()
    display_name = " ".join(str(display_name).strip().split())
    password_hash, password_salt = _hash_password(password)
    user_id = str(uuid.uuid4())

    conn = get_conn()
    try:
        conn.execute(
            """INSERT INTO users
               (user_id, email, display_name, password_hash, password_salt, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, email, display_name, password_hash, password_salt, _now()),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
        return _public_user(row)
    except sqlite3.IntegrityError as error:
        raise ValueError("An account with this email already exists.") from error
    finally:
        conn.close()


def authenticate_user(email, password):
    """Return a public user profile when the credentials are valid."""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE email=? COLLATE NOCASE AND is_active=1",
        (str(email).strip().lower(),),
    ).fetchone()

    if row is None:
        conn.close()
        return None

    try:
        salt = base64.b64decode(row["password_salt"])
        candidate_hash, _ = _hash_password(password, salt=salt)
    except (TypeError, ValueError):
        conn.close()
        return None

    if not hmac.compare_digest(candidate_hash, row["password_hash"]):
        conn.close()
        return None

    conn.execute(
        "UPDATE users SET last_login_at=? WHERE user_id=?",
        (_now(), row["user_id"]),
    )
    conn.commit()
    profile = _public_user(row)
    conn.close()
    return profile


def create_user_session(user_id, lifetime_days=SESSION_LIFETIME_DAYS):
    """Create a revocable bearer session; only its SHA-256 hash is stored."""
    token = "ct_" + secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    created_at = datetime.now(timezone.utc)
    expires_at = created_at + timedelta(days=max(1, int(lifetime_days)))

    conn = get_conn()
    conn.execute(
        """INSERT INTO user_sessions
           (session_id, user_id, token_hash, created_at, expires_at)
           VALUES (?, ?, ?, ?, ?)""",
        (
            str(uuid.uuid4()),
            user_id,
            token_hash,
            created_at.isoformat(),
            expires_at.isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return token


def get_user_for_session(token):
    """Resolve an active bearer token to its public user profile."""
    if not token:
        return None
    token_hash = hashlib.sha256(str(token).encode("utf-8")).hexdigest()
    now = _now()
    conn = get_conn()
    row = conn.execute(
        """SELECT users.*
           FROM user_sessions
           JOIN users ON users.user_id = user_sessions.user_id
           WHERE user_sessions.token_hash=?
             AND user_sessions.revoked_at IS NULL
             AND user_sessions.expires_at>?
             AND users.is_active=1""",
        (token_hash, now),
    ).fetchone()
    profile = _public_user(row)
    conn.close()
    return profile


def revoke_user_session(token):
    if not token:
        return False
    token_hash = hashlib.sha256(str(token).encode("utf-8")).hexdigest()
    conn = get_conn()
    cursor = conn.execute(
        "UPDATE user_sessions SET revoked_at=? "
        "WHERE token_hash=? AND revoked_at IS NULL",
        (_now(), token_hash),
    )
    conn.commit()
    revoked = cursor.rowcount > 0
    conn.close()
    return revoked


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


def get_report(report_id):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM citizen_reports WHERE report_id=?", (report_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_reports(user_id):
    conn = get_conn()
    rows = conn.execute(
        """SELECT citizen_reports.*,
                  COUNT(report_votes.vote_id) AS upvotes
           FROM citizen_reports
           LEFT JOIN report_votes
             ON report_votes.report_id = citizen_reports.report_id
            AND report_votes.vote_type = 'upvote'
           WHERE citizen_reports.user_id=?
           GROUP BY citizen_reports.report_id
           ORDER BY citizen_reports.created_at DESC""",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_report_upvotes(report_id):
    conn = get_conn()
    count = conn.execute(
        """SELECT COUNT(*) AS count FROM report_votes
           WHERE report_id=? AND vote_type='upvote'""",
        (report_id,),
    ).fetchone()["count"]
    conn.close()
    return int(count)


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
