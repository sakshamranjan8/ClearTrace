import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("OPENAQ_API_KEY")
if not API_KEY:
    raise ValueError("OPENAQ_API_KEY not found.")

HEADERS = {"X-API-Key": API_KEY}

# Delhi bounding box: approximate, enough for exploration
# min_lon, min_lat, max_lon, max_lat
DELHI_BBOX = "76.80,28.40,77.35,28.90"

url = "https://api.openaq.org/v3/locations"

params = {
    "bbox": DELHI_BBOX,
    "limit": 1000,
    "page": 1,
}

response = requests.get(url, headers=HEADERS, params=params, timeout=30)
response.raise_for_status()

data = response.json()
locations = data.get("results", [])

rows = []

for loc in locations:
    coordinates = loc.get("coordinates") or {}
    sensors = loc.get("sensors") or []

    for sensor in sensors:
        parameter = sensor.get("parameter") or {}

        rows.append({
            "location_id": loc.get("id"),
            "station_name": loc.get("name"),
            "locality": loc.get("locality"),
            "latitude": coordinates.get("latitude"),
            "longitude": coordinates.get("longitude"),
            "timezone": loc.get("timezone"),
            "provider_name": (loc.get("provider") or {}).get("name"),
            "is_monitor": loc.get("isMonitor"),
            "is_mobile": loc.get("isMobile"),
            "sensor_id": sensor.get("id"),
            "sensor_name": sensor.get("name"),
            "parameter_id": parameter.get("id"),
            "parameter_name": parameter.get("name"),
            "parameter_unit": parameter.get("units"),
            "parameter_display_name": parameter.get("displayName"),
            "datetime_first_local": (loc.get("datetimeFirst") or {}).get("local"),
            "datetime_last_local": (loc.get("datetimeLast") or {}).get("local"),
        })

availability_df = pd.DataFrame(rows)

output_dir = PROJECT_ROOT / "data" / "raw"
output_path = output_dir / "openaq_delhi_pollutant_availability.csv"
availability_df.to_csv(output_path, index=False)

print(f"Locations found: {len(locations)}")
print(f"Sensor rows saved: {len(availability_df)}")
print(f"Saved to: {output_path}")

print("\nPollutant availability count:")
print(
    availability_df
    .groupby("parameter_name")["location_id"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nPreview:")
print(availability_df.head(20))