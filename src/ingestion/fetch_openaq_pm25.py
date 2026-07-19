import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# 1. Locate project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 2. Load API key from .env
load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("OPENAQ_API_KEY")

if not API_KEY:
    raise ValueError("OPENAQ_API_KEY not found. Check your .env file.")


# 3. Known OpenAQ metadata from our earlier test
LOCATION_ID = 8118
SENSOR_ID = 23534
STATION_NAME = "New Delhi"


# 4. API endpoint
url = f"https://api.openaq.org/v3/sensors/{SENSOR_ID}/measurements"

headers = {
    "X-API-Key": API_KEY
}

params = {
    "datetime_from": "2026-06-28T00:00:00+05:30",
    "datetime_to": "2026-07-05T23:59:59+05:30",
    "limit": 1000,
    "page": 1
}


# 5. Make request
response = requests.get(url, headers=headers, params=params, timeout=30)
response.raise_for_status()

data = response.json()
results = data.get("results", [])


# 6. Convert nested JSON into table rows
rows = []

for item in results:
    row = {
        "location_id": LOCATION_ID,
        "sensor_id": SENSOR_ID,
        "station_name": STATION_NAME,
        "parameter": item["parameter"]["name"],
        "unit": item["parameter"]["units"],
        "value": item["value"],
        "timestamp_utc": item["period"]["datetimeFrom"]["utc"],
        "timestamp_local": item["period"]["datetimeFrom"]["local"],
        "interval": item["period"]["interval"],
        "has_flags": item["flagInfo"]["hasFlags"],
        "percent_complete": item["coverage"]["percentComplete"],
        "percent_coverage": item["coverage"]["percentCoverage"],
    }

    rows.append(row)


# 7. Create DataFrame
df = pd.DataFrame(rows)


# 8. Save to data/raw
output_dir = PROJECT_ROOT / "data" / "raw"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "openaq_pm25_newdelhi_raw.csv"
df.to_csv(output_path, index=False)


print(f"Fetched {len(df)} rows.")
print(f"Saved raw data to: {output_path}")
print(df.head())
