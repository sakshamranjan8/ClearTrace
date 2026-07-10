import os
import json
from pathlib import Path

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    raise ValueError("OPENAQ_API_KEY not found.")

HEADERS = {
    "X-API-Key": API_KEY
}

LOCATION_ID = 8118
SENSOR_ID = 23534


def fetch_json(url, params=None):
    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=30
    )
    response.raise_for_status()
    return response.json()


location_url = f"https://api.openaq.org/v3/locations/{LOCATION_ID}"
measurement_url = f"https://api.openaq.org/v3/sensors/{SENSOR_ID}/measurements"

location_data = fetch_json(location_url)
measurement_data = fetch_json(measurement_url, params={"limit": 1})


output_dir = PROJECT_ROOT / "data" / "raw" / "field_inventory"
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "location_8118_sample.json", "w", encoding="utf-8") as f:
    json.dump(location_data, f, indent=2)

with open(output_dir / "sensor_23534_measurement_sample.json", "w", encoding="utf-8") as f:
    json.dump(measurement_data, f, indent=2)

print("Saved OpenAQ sample JSON files.")
print("Location sample:", output_dir / "location_8118_sample.json")
print("Measurement sample:", output_dir / "sensor_23534_measurement_sample.json")