"""Frontend adapter for the ClearTrace FastAPI forecast service.

The public functions keep the shape used by the Streamlit app and the
existing chatbot while replacing the frozen CSV/model prediction path with
the live ``GET /forecast`` endpoint.
"""

from functools import lru_cache
import os
from pathlib import Path

import pandas as pd
import requests


HERE = Path(__file__).parent
API_BASE_URL = os.getenv("CLEARTRACE_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_TIMEOUT_SECONDS = float(os.getenv("CLEARTRACE_API_TIMEOUT_SECONDS", "15"))

if (HERE / "ml_assets" / "data" / "features_core_v2.csv").exists():
    FEATURES_CSV = HERE / "ml_assets" / "data" / "features_core_v2.csv"
else:
    FEATURES_CSV = HERE / "data" / "features" / "features_core_v2.csv"


@lru_cache(maxsize=1)
def list_stations():
    """Return the station selector metadata without loading model assets."""
    df = pd.read_csv(
        FEATURES_CSV,
        usecols=["station_name", "latitude", "longitude"],
    )
    stations = df.drop_duplicates()
    return sorted(
        stations.itertuples(index=False, name=None),
        key=lambda row: row[0],
    )


def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def _station_coordinates(location):
    for station_name, latitude, longitude in list_stations():
        if station_name == location:
            return float(latitude), float(longitude)
    return None, None


def _error_forecast(location, latitude, longitude, code, message):
    return {
        "location": location,
        "lat": latitude,
        "lon": longitude,
        "generated_at": None,
        "hourly": [],
        "nearest_stations": [],
        "available_forecast_hours": 0,
        "error": code,
        "error_message": message,
    }


def generate_forecast(
    location=None,
    hours=24,
    citizen_boost=0,
    *,
    latitude=None,
    longitude=None,
):
    """Request a coordinate forecast and adapt it for existing consumers.

    ``location`` remains supported for the chatbot and station dropdown.
    New frontend code can pass ``latitude`` and ``longitude`` explicitly.
    ``citizen_boost`` is retained only for signature compatibility; live model
    predictions are never modified in the frontend.
    """
    del citizen_boost

    hours = max(1, min(int(hours), 24))
    if latitude is None or longitude is None:
        latitude, longitude = _station_coordinates(location)

    if latitude is None or longitude is None:
        return _error_forecast(
            location,
            latitude,
            longitude,
            "missing_coordinates",
            "Latitude and longitude are required for a forecast.",
        )

    latitude = float(latitude)
    longitude = float(longitude)

    try:
        response = requests.get(
            f"{API_BASE_URL}/forecast",
            params={"latitude": latitude, "longitude": longitude},
            timeout=API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout:
        return _error_forecast(
            location,
            latitude,
            longitude,
            "forecast_api_timeout",
            "The forecast service did not respond in time.",
        )
    except requests.RequestException as error:
        detail = None
        if error.response is not None:
            try:
                detail = error.response.json().get("detail")
            except (ValueError, AttributeError):
                detail = None
        return _error_forecast(
            location,
            latitude,
            longitude,
            "forecast_api_unavailable",
            detail or "The forecast service is unavailable.",
        )
    except (TypeError, ValueError) as error:
        return _error_forecast(
            location,
            latitude,
            longitude,
            "invalid_forecast_response",
            f"The forecast service returned invalid JSON: {error}",
        )

    raw_points = payload.get("forecast")
    if not isinstance(raw_points, list):
        return _error_forecast(
            location,
            latitude,
            longitude,
            "invalid_forecast_response",
            "The forecast response does not contain a forecast array.",
        )

    points = []
    try:
        for future_index, point in enumerate(raw_points[:hours], start=1):
            predicted_aqi = float(point["predicted_aqi"])
            points.append(
                {
                    "timestamp": point["timestamp"],
                    "hour_offset": future_index,
                    "model_horizon_hours": point.get("horizon_hours"),
                    "aqi": round(predicted_aqi, 2),
                    "category": point.get("category") or aqi_category(predicted_aqi),
                }
            )
    except (KeyError, TypeError, ValueError) as error:
        return _error_forecast(
            location,
            latitude,
            longitude,
            "invalid_forecast_response",
            f"A forecast point failed validation: {error}",
        )

    return {
        "location": location or "Current location",
        "lat": latitude,
        "lon": longitude,
        "generated_at": payload.get("generated_at"),
        "forecast_origin": payload.get("forecast_origin"),
        "cache_updated_at": payload.get("cache_updated_at"),
        "is_stale": bool(payload.get("is_stale", False)),
        "requested_forecast_hours": payload.get("requested_forecast_hours", 24),
        "available_forecast_hours": len(points),
        "nearest_stations": payload.get("nearest_stations", []),
        "hourly": points,
        "error": None,
    }


def forecast_summary(forecast, window_hours=24):
    pts = forecast.get("hourly", [])[:window_hours]
    if not pts:
        return {
            "avg_aqi": None,
            "peak_aqi": None,
            "peak_time": None,
            "category": "Unknown",
        }

    avg = sum(point["aqi"] for point in pts) / len(pts)
    peak = max(pts, key=lambda point: point["aqi"])
    return {
        "avg_aqi": round(avg, 1),
        "peak_aqi": peak["aqi"],
        "peak_time": peak["timestamp"],
        "category": aqi_category(avg),
    }
