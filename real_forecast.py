"""
real_forecast.py
Replaces the synthetic forecast.py with the actual trained models delivered by
the forecasting teammate: 24 CatBoost models (models/horizon_01h.joblib ...
horizon_24h.joblib), one per forecast hour, trained on features_core_v2.csv.
"""

import joblib
import pandas as pd
from pathlib import Path
from functools import lru_cache

HERE = Path(__file__).parent

# Looks for the delivered files in your sibling ClearTrace/ folder
if (HERE / "ml_assets" / "models").exists():
    MODELS_DIR = HERE / "ml_assets" / "models"
    FEATURES_CSV = HERE / "ml_assets" / "data" / "features_core_v2.csv"
else:
    MODELS_DIR = HERE / "models"
    FEATURES_CSV = HERE / "data" / "features" / "features_core_v2.csv"

MODEL_FEATURE_COLUMNS = None


@lru_cache(maxsize=1)
def _load_models():
    """Load all 24 horizon models once and cache them for the app's lifetime."""
    models = {}
    for h in range(1, 25):
        path = MODELS_DIR / f"horizon_{h:02d}h.joblib"
        models[h] = joblib.load(path)
    return models


@lru_cache(maxsize=1)
def _load_feature_table():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp_hour"])
    return df


@lru_cache(maxsize=1)
def list_stations():
    """Returns [(station_name, lat, lon), ...] for the 38 real Delhi stations."""
    df = _load_feature_table()
    stations = df[["station_name", "latitude", "longitude"]].drop_duplicates()
    return sorted(stations.itertuples(index=False, name=None), key=lambda r: r[0])


def _latest_valid_row(station_name):
    df = _load_feature_table()
    station_rows = df[(df["station_name"] == station_name) & (df["aqi_calculation_valid"] == True)]  # noqa: E712
    if station_rows.empty:
        return None
    return station_rows.sort_values("timestamp_hour").iloc[-1]


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


def generate_forecast(location, hours=24, citizen_boost=0):
    """Same function name and same return shape as the old forecast.py,
    so app.py barely needs to change."""
    hours = min(hours, 24)
    models = _load_models()
    row = _latest_valid_row(location)

    if row is None:
        return {
            "location": location, "lat": None, "lon": None,
            "generated_at": None, "hourly": [], "error": "no_valid_recent_data",
        }

    feature_cols = models[1].feature_names_
    x = row[feature_cols].to_frame().T
    x["station_name"] = str(row["station_name"])

    base_time = row["timestamp_hour"]
    points = []
    for h in range(1, hours + 1):
        pred = float(models[h].predict(x)[0])
        pred = max(pred + citizen_boost, 0)
        ts = base_time + pd.Timedelta(hours=h)
        points.append({
            "timestamp": ts.isoformat(),
            "hour_offset": h,
            "aqi": round(pred, 1),
            "pm25": round(pred * 0.6, 1),
            "category": aqi_category(pred),
        })

    return {
        "location": location,
        "lat": float(row["latitude"]),
        "lon": float(row["longitude"]),
        "generated_at": base_time.isoformat(),
        "as_of_note": f"Based on last valid reading at {base_time} (frozen dataset, not live yet)",
        "current_aqi": float(row["current_aqi"]),
        "hourly": points,
    }


def forecast_summary(forecast, window_hours=24):
    pts = forecast["hourly"][:window_hours]
    if not pts:
        return {"avg_aqi": None, "peak_aqi": None, "peak_time": None, "category": "Unknown"}
    avg = sum(p["aqi"] for p in pts) / len(pts)
    peak = max(pts, key=lambda p: p["aqi"])
    return {
        "avg_aqi": round(avg, 1),
        "peak_aqi": peak["aqi"],
        "peak_time": peak["timestamp"],
        "category": aqi_category(avg),
    }