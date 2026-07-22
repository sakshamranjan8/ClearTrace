"""ClearTrace Module 3 — Utility Functions.

Shared helpers used by attribution.py, chatbot.py, and main.py:
  - haversine_distance   → distance between two lat/lon points
  - find_nearest_station → locate closest monitoring station
  - calculate_risk       → health risk scoring (ported from health_engine_07.py)
  - get_aqi_category     → AQI number → CPCB category string
  - fetch_teammate_api   → HTTP caller with timeout + mock fallback
  - truncate_context     → keep text within a token budget (Issue #6)
  - get_current_ist_time → current timestamp in IST
  - extract_time_from_query → parse natural-language time references
  - get_forecast_at_time    → look up forecast for a specific clock hour
  - format_time_for_display → "21:00 (9 PM)" style formatting
"""

import math
import re
from datetime import datetime, timezone, timedelta

import httpx

from rag.config import settings


# ---------------------------------------------------------------------------
# IST timezone offset (UTC+5:30)
# ---------------------------------------------------------------------------
IST = timezone(timedelta(hours=5, minutes=30))


def get_current_ist_time() -> datetime:
    """Return the current time in Indian Standard Time (UTC+5:30)."""
    return datetime.now(IST)


# ---------------------------------------------------------------------------
# Geographic helpers
# ---------------------------------------------------------------------------

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance (km) between two lat/lon points.

    Uses the Haversine formula.  Good enough for distances under 500 km,
    which covers all of Delhi easily.

    Args:
        lat1, lon1: First point (degrees).
        lat2, lon2: Second point (degrees).

    Returns:
        Distance in kilometres.
    """
    R = 6371.0  # Earth's radius in km

    # Convert degrees to radians
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def find_nearest_station(lat: float, lon: float, stations_df):
    """Find the monitoring station closest to the given coordinates.

    Args:
        lat, lon: User's location (degrees).
        stations_df: DataFrame with columns [location_id, station_name,
                      station_latitude, station_longitude].

    Returns:
        Tuple of (station_row_as_dict, distance_km).
        Returns (None, None) if stations_df is empty.
    """
    if stations_df is None or stations_df.empty:
        print("[WARN] find_nearest_station: stations_df is empty")
        return None, None

    best_row = None
    best_dist = float("inf")

    # Iterate over unique stations (not over every source-link row)
    for _, row in stations_df.iterrows():
        dist = haversine_distance(
            lat, lon, row["station_latitude"], row["station_longitude"]
        )
        if dist < best_dist:
            best_dist = dist
            best_row = row

    return best_row.to_dict() if best_row is not None else None, best_dist


# ---------------------------------------------------------------------------
# AQI classification (matches CPCB National AQI standard)
# ---------------------------------------------------------------------------

AQI_CATEGORIES = [
    (0, 50, "good"),
    (51, 100, "satisfactory"),
    (101, 200, "moderate"),
    (201, 300, "poor"),
    (301, 400, "very_poor"),
    (401, 500, "severe"),
]


def get_aqi_category(aqi: float) -> str:
    """Map an AQI value to its CPCB category string.

    Args:
        aqi: Numeric AQI value.

    Returns:
        One of: "good", "satisfactory", "moderate", "poor", "very_poor", "severe".
        Returns "severe" for AQI > 500.
        Returns "unknown" for NaN or negative values.
    """
    if aqi is None or math.isnan(aqi) or aqi < 0:
        return "unknown"

    for low, high, category in AQI_CATEGORIES:
        if low <= aqi <= high:
            return category

    return "severe"  # AQI > 500


def get_aqi_category_label(aqi: float) -> str:
    """Friendly label version: 'Good', 'Very Poor', etc."""
    return get_aqi_category(aqi).replace("_", " ").title()


# ---------------------------------------------------------------------------
# Health risk calculator (ported from notebooks/health_engine_07.py)
# ---------------------------------------------------------------------------
# Issue #7: This is called SEPARATELY from the LLM. The chatbot uses the
# returned hazard_level and recommendations directly — the LLM only writes
# the answer text.

VULNERABILITY_MULTIPLIERS = {
    "adult": 1.0,
    "child": 1.8,
    "elderly": 2.2,
    "asthma": 2.8,
    "outdoor_worker": 1.4,
    "pregnant_woman": 2.2,
}


def calculate_risk(aqi: float, duration_hours: float, user_category: str) -> dict:
    """Calculate health risk score for a given AQI, exposure duration, and user type.

    This is a deterministic calculation — no LLM involved.
    Ported exactly from your notebooks/health_engine_07.py.

    Args:
        aqi: Current or forecast AQI value.
        duration_hours: Expected outdoor exposure time.
        user_category: One of the VULNERABILITY_MULTIPLIERS keys.

    Returns:
        Dict with keys: score, level, color, recommendation.
    """
    # AQI weight mapping (same as your original)
    if aqi <= 50:
        weight = 10
    elif aqi <= 100:
        weight = 20
    elif aqi <= 200:
        weight = 40
    elif aqi <= 300:
        weight = 70
    else:
        weight = 100

    # Vulnerability multiplier — default to adult if category unknown
    multiplier = VULNERABILITY_MULTIPLIERS.get(user_category, 1.0)

    # Calculate risk score
    score = weight * duration_hours * multiplier

    # Determine level, colour, recommendation
    if score <= 30:
        level = "Low"
        color = "#00FF00"
        recommendation = (
            "Outdoor activity is generally acceptable. Keep checking forecast."
        )
    elif score <= 60:
        level = "Moderate"
        color = "#FFFF00"
        recommendation = (
            "Limit prolonged outdoor exposure. "
            "Sensitive users should be careful."
        )
    elif score <= 90:
        level = "High"
        color = "#FFA500"
        recommendation = (
            "Wear N95/KN95 mask outdoors. Reduce travel time. "
            "Use indoor purifier."
        )
    else:
        level = "Severe"
        color = "#FF0000"
        recommendation = (
            "Avoid outdoor activity. Shift plans indoors. "
            "Alerts for children, elderly, respiratory sensitivity."
        )

    return {
        "score": round(score, 2),
        "level": level,
        "color": color,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Teammate API caller (with timeout + mock fallback)
# ---------------------------------------------------------------------------

# Mock responses — used when MOCK_MODE=true or when real APIs fail

def _build_mock_forecast() -> dict:
    """Build a fresh mock forecast with current timestamps.

    Separated into a function so timestamps are always current,
    not stale from module-load time.
    """
    now = get_current_ist_time()
    base_ts = now.replace(minute=0, second=0, microsecond=0)

    return {
        "status": "success",
        "request": {"latitude": 28.6139, "longitude": 77.209},
        "generated_at": now.isoformat(),
        "forecast_window": {
            "start": (base_ts + timedelta(hours=1)).isoformat(),
            "end": (base_ts + timedelta(hours=25)).isoformat(),
            "total_hours": 24,
        },
        "nearest_stations": [
            {
                "station_name": "Mandir Marg, New Delhi - DPCC",
                "distance_km": 2.622,
                "blend_weight": 0.3831,
            },
            {
                "station_name": "Major Dhyan Chand National Stadium, Delhi - DPCC",
                "distance_km": 2.82,
                "blend_weight": 0.3312,
            },
            {
                "station_name": "Lodhi Road, New Delhi - IMD",
                "distance_km": 3.036,
                "blend_weight": 0.2857,
            },
        ],
        "forecast": [
            {"timestamp": (base_ts + timedelta(hours=h)).isoformat(),
             "horizon_hours": h,
             "predicted_aqi": aqi,
             "category": _aqi_to_category_label(aqi)}
            for h, aqi in [
                (1, 80.36), (2, 83.99), (3, 88.12), (4, 92.45),
                (5, 97.00), (6, 103.50), (7, 110.20), (8, 115.80),
                (9, 118.30), (10, 120.00), (11, 117.50), (12, 112.00),
                (13, 105.40), (14, 98.70), (15, 93.20), (16, 89.50),
                (17, 86.00), (18, 90.30), (19, 98.10), (20, 108.50),
                (21, 115.60), (22, 110.40), (23, 102.30), (24, 95.00),
            ]
        ],
    }


def _aqi_to_category_label(aqi: float) -> str:
    """Quick AQI → label for mock data generation."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    return "Severe"


# Lazy-built so timestamps are fresh on each server session
MOCK_FORECAST = None


def get_mock_forecast() -> dict:
    """Return the mock forecast, building it on first call."""
    global MOCK_FORECAST
    if MOCK_FORECAST is None:
        MOCK_FORECAST = _build_mock_forecast()
    return MOCK_FORECAST


# ---------------------------------------------------------------------------
# Forecast data extraction helpers (new Module 2 format)
# ---------------------------------------------------------------------------
# These use .get() everywhere so malformed / partial responses never crash.

def extract_current_aqi(forecast_data: dict) -> float:
    """Extract the predicted AQI for horizon_hours=1 (nearest hour).

    Falls back to the first available forecast entry, then to 200.0
    (cautious default) if nothing is found.

    Args:
        forecast_data: Raw dict from Module 2 API or mock.

    Returns:
        AQI value as float.
    """
    forecast_list = forecast_data.get("forecast")
    if not forecast_list or not isinstance(forecast_list, list):
        print("[FORECAST] No forecast array — returning default AQI 200")
        return 200.0

    # Look for horizon_hours == 1 first
    for entry in forecast_list:
        if isinstance(entry, dict) and entry.get("horizon_hours") == 1:
            aqi = entry.get("predicted_aqi")
            if aqi is not None:
                return float(aqi)

    # Fallback: return the first entry's AQI
    first = forecast_list[0] if forecast_list else {}
    aqi = first.get("predicted_aqi") if isinstance(first, dict) else None
    if aqi is not None:
        print(f"[FORECAST] No horizon_hours=1 found, using first entry AQI: {aqi}")
        return float(aqi)

    print("[FORECAST] Could not extract AQI — returning default 200")
    return 200.0


def get_forecast_category(forecast_data: dict) -> str:
    """Extract the CPCB category string for horizon_hours=1.

    Args:
        forecast_data: Raw dict from Module 2 API or mock.

    Returns:
        Category string like "Satisfactory", "Moderate", etc.
        Returns "Unknown" if not found.
    """
    forecast_list = forecast_data.get("forecast")
    if not forecast_list or not isinstance(forecast_list, list):
        return "Unknown"

    for entry in forecast_list:
        if isinstance(entry, dict) and entry.get("horizon_hours") == 1:
            return entry.get("category", "Unknown")

    return "Unknown"


def get_nearest_station_name(forecast_data: dict) -> str:
    """Extract the primary (nearest) station name from the forecast response.

    Args:
        forecast_data: Raw dict from Module 2 API or mock.

    Returns:
        Station name string, or "Unknown Station" if not found.
    """
    stations = forecast_data.get("nearest_stations")
    if stations and isinstance(stations, list) and len(stations) > 0:
        first = stations[0]
        if isinstance(first, dict):
            return first.get("station_name", "Unknown Station")
    return "Unknown Station"


def format_forecast_for_prompt(forecast_data: dict) -> str:
    """Format the forecast array into a readable string for the super prompt.

    Displays up to 8 key horizons with predicted_aqi and category.
    Also includes the blended station names for context.

    Args:
        forecast_data: Raw dict from Module 2 API or mock.

    Returns:
        Multi-line formatted string, or a "no data" note.
    """
    forecast_list = forecast_data.get("forecast")
    if not forecast_list or not isinstance(forecast_list, list):
        return "No forecast data available."

    # Station context
    station_name = get_nearest_station_name(forecast_data)
    stations = forecast_data.get("nearest_stations", [])
    station_names = []
    for s in stations[:3]:
        if isinstance(s, dict):
            name = s.get("station_name", "?")
            dist = s.get("distance_km", "?")
            station_names.append(f"{name} ({dist}km)")

    lines = []
    lines.append(f"Primary station: {station_name}")
    if station_names:
        lines.append(f"Blended from: {', '.join(station_names)}")

    # Select key horizons to keep the prompt compact: 1,2,3,6,9,12,18,24
    KEY_HORIZONS = {1, 2, 3, 6, 9, 12, 18, 24}

    for entry in forecast_list:
        if not isinstance(entry, dict):
            continue
        h = entry.get("horizon_hours")
        if h is None or h not in KEY_HORIZONS:
            continue
        aqi = entry.get("predicted_aqi", "N/A")
        cat = entry.get("category", "")
        aqi_str = f"{aqi:.0f}" if isinstance(aqi, (int, float)) else str(aqi)
        lines.append(f"  +{h}h: AQI {aqi_str} ({cat})")

    return "\n".join(lines)

MOCK_REPORTS = {
    "reports": [
        {
            "id": "mock-001",
            "type": "smoke",
            "description": "Visible smoke from nearby construction site",
            "verified": True,
            "timestamp": get_current_ist_time().isoformat(),
        },
        {
            "id": "mock-002",
            "type": "dust",
            "description": "Heavy dust on road near metro construction",
            "verified": True,
            "timestamp": get_current_ist_time().isoformat(),
        },
        {
            "id": "mock-003",
            "type": "waste_burning",
            "description": "Open waste burning spotted in residential area",
            "verified": False,
            "timestamp": get_current_ist_time().isoformat(),
        },
    ],
    "total_count": 3,
    "verified_count": 2,
}


async def fetch_teammate_api(
    url: str,
    params: dict = None,
    timeout: float = None,
    mock_fallback: dict = None,
) -> dict:
    """Call a teammate's REST API with timeout and mock fallback.

    This function handles three scenarios:
      1. MOCK_MODE=true → return mock data immediately (no network call).
      2. Real API call succeeds → return parsed JSON.
      3. Real API call fails → return mock_fallback (graceful degradation).

    Args:
        url: The teammate's endpoint URL.
        params: Query parameters dict.
        timeout: Request timeout in seconds (default from settings).
        mock_fallback: Data to return if the real call fails.

    Returns:
        Parsed JSON response as a dict.
    """
    if timeout is None:
        timeout = settings.TEAMMATE_API_TIMEOUT

    if mock_fallback is None:
        mock_fallback = {}

    # --- Scenario 1: Mock mode ---
    if settings.MOCK_MODE:
        print(f"[MOCK] Skipping real API call to {url}")
        return mock_fallback

    # --- Scenario 2 & 3: Real API call with fallback ---
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            print(f"[API] {url} → {response.status_code}")
            return response.json()

    except httpx.TimeoutException:
        print(f"[WARN] Timeout calling {url} after {timeout}s — using fallback")
        return mock_fallback

    except httpx.HTTPStatusError as e:
        print(f"[WARN] HTTP {e.response.status_code} from {url} — using fallback")
        return mock_fallback

    except Exception as e:
        print(f"[WARN] Error calling {url}: {e} — using fallback")
        return mock_fallback


# ---------------------------------------------------------------------------
# Token / text truncation (Issue #6)
# ---------------------------------------------------------------------------

def truncate_context(text: str, max_tokens: int = 4000) -> str:
    """Truncate text to approximately fit within a token budget.

    Groq's gemma2-9b-it has a 6000 tokens/min limit on the free tier.
    We keep the super-prompt under max_tokens to leave room for the response.

    Approximation: 1 token ≈ 4 characters (conservative for English text).
    This avoids importing a tokenizer just for truncation.

    Args:
        text: The input text to truncate.
        max_tokens: Maximum number of tokens allowed.

    Returns:
        Truncated text, with a note appended if truncation happened.
    """
    max_chars = max_tokens * 4  # ~4 chars per token (conservative estimate)

    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]

    # Try to cut at a sentence boundary
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:  # Don't cut too far back
        truncated = truncated[: last_period + 1]

    print(f"[TRUNCATE] Cut context from {len(text)} to {len(truncated)} chars")
    return truncated + "\n[... context truncated to fit token limit ...]"


# ---------------------------------------------------------------------------
# Time extraction from natural language queries
# ---------------------------------------------------------------------------

# Maps relative time-of-day words to representative clock hours (24h format)
_TIME_OF_DAY_MAP = {
    "morning": 9,
    "afternoon": 14,
    "evening": 18,
    "night": 21,
}


def extract_time_from_query(query: str) -> dict:
    """Extract a time reference from a user's natural-language query.

    Supports patterns like:
      - "at 9 PM", "at 9pm", "at 9:00 PM", "at 21:00"
      - "9 PM", "9pm"
      - "tomorrow morning", "tomorrow at 10 AM"
      - "tonight", "this evening"
      - "in 3 hours", "3 hours from now"
      - "at noon", "at midnight"
      - "at 6 in the evening"

    Returns:
        {
            "hour": 21,              # 24-hour format (0-23)
            "hour_12": 9,            # 12-hour format (1-12)
            "ampm": "PM",            # "AM" or "PM"
            "time_str": "9 PM",      # Human-readable time string
            "relative": "today",     # "today" or "tomorrow"
            "found": True,           # Whether a time was found
        }
        If no time is found, returns {"found": False}.
    """
    q = query.strip()
    q_lower = q.lower()
    now = get_current_ist_time()
    current_hour = now.hour

    # --- Pattern 1: "at noon" / "at midnight" ---
    if re.search(r"\bnoon\b", q_lower):
        return _build_time_result(12, "today", current_hour)
    if re.search(r"\bmidnight\b", q_lower):
        # Midnight = 0:00 of the *next* day from the user's perspective
        return _build_time_result(0, "tomorrow", current_hour)

    # --- Pattern 2: "in N hours" / "N hours from now" / "N hours later" ---
    m = re.search(r"(\d+)\s*hours?\s*(?:from\s*now|later)", q_lower)
    if not m:
        m = re.search(r"in\s*(\d+)\s*hours?", q_lower)
    if m:
        delta = int(m.group(1))
        target = (current_hour + delta) % 24
        relative = "today" if (current_hour + delta) < 24 else "tomorrow"
        return _build_time_result(target, relative, current_hour)

    # --- Pattern 3: "tomorrow morning/afternoon/evening/night" ---
    m = re.search(
        r"tomorrow\s*(morning|afternoon|evening|night)?", q_lower
    )
    if m:
        period = m.group(1)
        hour = _TIME_OF_DAY_MAP.get(period, 9)  # default "tomorrow" → 9 AM
        return _build_time_result(hour, "tomorrow", current_hour)

    # --- Pattern 4: "tonight" / "this morning/evening/..." ---
    m = re.search(r"tonight", q_lower)
    if m:
        # "tonight" → 21:00 today if it hasn't passed, else treat as later tonight
        return _build_time_result(21, "today", current_hour)

    m = re.search(r"this\s*(morning|afternoon|evening|night)", q_lower)
    if m:
        period = m.group(1)
        hour = _TIME_OF_DAY_MAP.get(period, current_hour)
        return _build_time_result(hour, "today", current_hour)

    # --- Pattern 5: "at 6 in the evening/morning" ---
    m = re.search(
        r"at\s*(\d{1,2})\s*in\s*the\s*(morning|afternoon|evening|night)",
        q_lower,
    )
    if m:
        raw_hour = int(m.group(1))
        period = m.group(2)
        hour_24 = _resolve_hour_with_period(raw_hour, period)
        return _build_time_result(hour_24, "today", current_hour)

    # --- Pattern 6: "at 9 PM" / "at 9:30 PM" / "at 21:00" / "at 9pm" ---
    m = re.search(
        r"at\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)?", q_lower
    )
    if m:
        raw_hour = int(m.group(1))
        ampm = (m.group(3) or "").upper()
        hour_24 = _resolve_hour_with_ampm(raw_hour, ampm, current_hour)
        return _build_time_result(hour_24, "today", current_hour)

    # --- Pattern 7: standalone "9 PM" / "9pm" / "10:30 AM" (no "at") ---
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)\b", q_lower)
    if m:
        raw_hour = int(m.group(1))
        ampm = m.group(3).upper()
        hour_24 = _resolve_hour_with_ampm(raw_hour, ampm, current_hour)
        return _build_time_result(hour_24, "today", current_hour)

    # --- No time found ---
    print(f"[TIME] No time reference found in: '{query}'")
    return {"found": False}


def _resolve_hour_with_ampm(raw_hour: int, ampm: str, current_hour: int) -> int:
    """Convert a raw hour + AM/PM indicator to 24-hour format.

    If no AM/PM is given (e.g., "at 6"), infer from current time:
    if the hour has already passed today, assume tomorrow's context.
    For ambiguous cases (no AM/PM), prefer the next upcoming occurrence.
    """
    if raw_hour > 23:
        raw_hour = raw_hour % 24

    if ampm == "PM" and raw_hour != 12:
        return raw_hour + 12
    elif ampm == "AM" and raw_hour == 12:
        return 0
    elif ampm:
        return raw_hour
    else:
        # No AM/PM given — if hour is 1-12, pick the upcoming occurrence
        if 1 <= raw_hour <= 12:
            # Try both AM and PM, pick whichever is next
            am_hour = raw_hour if raw_hour != 12 else 0
            pm_hour = raw_hour + 12 if raw_hour != 12 else 12
            if am_hour > current_hour:
                return am_hour
            elif pm_hour > current_hour:
                return pm_hour
            else:
                return am_hour  # Both passed → tomorrow AM
        return raw_hour  # Already looks like 24h format (13-23)


def _resolve_hour_with_period(raw_hour: int, period: str) -> int:
    """Convert '6 in the evening' → 18."""
    if period in ("afternoon", "evening", "night"):
        return raw_hour + 12 if raw_hour < 12 else raw_hour
    return raw_hour  # morning


def _build_time_result(hour_24: int, relative: str, current_hour: int) -> dict:
    """Build the standard time-extraction result dict."""
    hour_24 = hour_24 % 24

    # If the requested hour has already passed today, treat as tomorrow
    if relative == "today" and hour_24 <= current_hour:
        relative = "tomorrow"

    # Convert to 12-hour for display
    if hour_24 == 0:
        hour_12, ampm = 12, "AM"
    elif hour_24 < 12:
        hour_12, ampm = hour_24, "AM"
    elif hour_24 == 12:
        hour_12, ampm = 12, "PM"
    else:
        hour_12, ampm = hour_24 - 12, "PM"

    time_str = f"{hour_12} {ampm}"
    result = {
        "hour": hour_24,
        "hour_12": hour_12,
        "ampm": ampm,
        "time_str": time_str,
        "relative": relative,
        "found": True,
    }
    print(f"[TIME] Extracted: {result}")
    return result


# ---------------------------------------------------------------------------
# Forecast lookup for a specific time
# ---------------------------------------------------------------------------

def get_forecast_at_time(forecast_data: dict, target_hour: int,
                         relative: str = "today") -> dict | None:
    """Find the forecast entry for a specific target clock-hour.

    Converts the target clock-hour into the correct horizon_hours offset,
    then finds the matching (or closest) forecast entry.

    Args:
        forecast_data: The Module 2 forecast response dict.
        target_hour: Target time in 24-hour format (0-23).
        relative: "today" or "tomorrow" — shifts the horizon accordingly.

    Returns:
        Dict with keys {horizon_hours, predicted_aqi, category, timestamp}
        or None if forecast data is unavailable.
    """
    forecast_list = forecast_data.get("forecast")
    if not forecast_list or not isinstance(forecast_list, list):
        print("[TIME] No forecast array available for time lookup")
        return None

    now = get_current_ist_time()
    current_hour = now.hour

    # Calculate how many hours from now until the target time
    if relative == "tomorrow":
        # Hours remaining today + target hour tomorrow
        hours_until = (24 - current_hour) + target_hour
    else:
        hours_until = target_hour - current_hour
        if hours_until <= 0:
            # Already passed today — wrap to tomorrow
            hours_until += 24

    print(
        f"[TIME] Looking up forecast: target={target_hour}:00 "
        f"({relative}), hours_until={hours_until}"
    )

    # Find exact horizon match
    for entry in forecast_list:
        if not isinstance(entry, dict):
            continue
        if entry.get("horizon_hours") == hours_until:
            print(
                f"[TIME] Exact match: horizon={hours_until}h, "
                f"AQI={entry.get('predicted_aqi')}"
            )
            return entry

    # No exact match — find the closest horizon
    best_entry = None
    best_diff = float("inf")
    for entry in forecast_list:
        if not isinstance(entry, dict):
            continue
        h = entry.get("horizon_hours")
        if h is None:
            continue
        diff = abs(h - hours_until)
        if diff < best_diff:
            best_diff = diff
            best_entry = entry

    if best_entry:
        print(
            f"[TIME] Closest match: horizon={best_entry.get('horizon_hours')}h "
            f"(wanted {hours_until}h, diff={best_diff}h), "
            f"AQI={best_entry.get('predicted_aqi')}"
        )
    return best_entry


def format_time_for_display(hour_24: int) -> str:
    """Format a 24-hour value as a friendly string like '9:00 PM (21:00)'.

    Args:
        hour_24: Hour in 24-hour format (0-23).

    Returns:
        String like "9 PM (21:00)" or "12 PM (12:00)".
    """
    hour_24 = hour_24 % 24
    if hour_24 == 0:
        return "12 AM (00:00)"
    elif hour_24 < 12:
        return f"{hour_24} AM ({hour_24:02d}:00)"
    elif hour_24 == 12:
        return "12 PM (12:00)"
    else:
        return f"{hour_24 - 12} PM ({hour_24:02d}:00)"
