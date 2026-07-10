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

LOCATION_ID = 8118

url = f"https://api.openaq.org/v3/locations/{LOCATION_ID}"
headers = {"X-API-Key": API_KEY}

response = requests.get(url, headers=headers, timeout=30)
response.raise_for_status()

data = response.json()
location = data["results"][0]

rows = []

for sensor in location["sensors"]:
    row = {
        "location_id": location["id"],
        "sensor_id": sensor["id"],
        "station_name": location["name"],
        "latitude": location["coordinates"]["latitude"],
        "longitude": location["coordinates"]["longitude"],
        "timezone": location["timezone"],
        "parameter": sensor["parameter"]["name"],
        "unit": sensor["parameter"]["units"],
    }
    rows.append(row)

df = pd.DataFrame(rows)

output_path = PROJECT_ROOT / "data" / "raw" / "openaq_station_metadata.csv"
df.to_csv(output_path, index=False)

print(f"Saved station metadata to: {output_path}")
print(df)