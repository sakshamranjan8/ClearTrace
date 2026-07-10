import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("OPENAQ_API_KEY")
if not API_KEY:
    raise ValueError("OPENAQ_API_KEY not found in .env file.")

HEADERS = {"X-API-Key": API_KEY}

CONTROL_TABLE_PATH = PROJECT_ROOT / "data" / "processed" / "final_delhi_sensor_control_table.csv"

# Sample mode first. After testing, change this to None.
MAX_SENSORS = None

DAYS_BACK = 90
LIMIT = 1000
REQUEST_SLEEP_SECONDS = 0.15

BASE_URL = "https://api.openaq.org/v3/sensors"


def get_time_window():
    """
    Creates the 90-day time window.

    We use Asia/Kolkata because Delhi station timestamps are local to India.
    OpenAQ accepts ISO datetime strings with timezone offsets.
    """
    timezone = ZoneInfo("Asia/Kolkata")

    datetime_to = datetime.now(timezone).replace(minute=0, second=0, microsecond=0)
    datetime_from = datetime_to - timedelta(days=DAYS_BACK)

    return datetime_from.isoformat(), datetime_to.isoformat()


def fetch_sensor_hourly_measurements(sensor_row, datetime_from, datetime_to):
    """
    Fetch hourly measurements for one sensor_id.

    One sensor_id represents:
    one station + one pollutant

    Example:
    Anand Vihar + PM2.5 sensor
    """
    sensor_id = int(sensor_row["sensor_id"])

    url = f"{BASE_URL}/{sensor_id}/hours"

    all_rows = []
    page = 1

    while True:
        params = {
            "datetime_from": datetime_from,
            "datetime_to": datetime_to,
            "limit": LIMIT,
            "page": page,
        }

        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=40,
        )

        # Handle rate limit or temporary server issues gently
        if response.status_code in [429, 500, 502, 503, 504]:
            print(f"Temporary issue for sensor {sensor_id}, page {page}. Retrying...")
            time.sleep(5)
            response = requests.get(
                url,
                headers=HEADERS,
                params=params,
                timeout=40,
            )

        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        if not results:
            break

        for item in results:
            parameter = item.get("parameter") or {}
            flag_info = item.get("flagInfo") or {}
            coverage = item.get("coverage") or {}
            period = item.get("period") or {}
            datetime_from_obj = period.get("datetimeFrom") or {}
            datetime_to_obj = period.get("datetimeTo") or {}

            all_rows.append(
                {
                    "location_id": sensor_row["location_id"],
                    "station_name": sensor_row["station_name"],
                    "latitude": sensor_row["latitude"],
                    "longitude": sensor_row["longitude"],
                    "provider_name": sensor_row["provider_name"],
                    "sensor_id": sensor_id,
                    "parameter_name": sensor_row["parameter_name"],
                    "parameter_unit": sensor_row["parameter_unit"],
                    "timestamp_utc": datetime_from_obj.get("utc"),
                    "timestamp_local": datetime_from_obj.get("local"),
                    "timestamp_to_utc": datetime_to_obj.get("utc"),
                    "timestamp_to_local": datetime_to_obj.get("local"),
                    "value": item.get("value"),
                    "has_flags": flag_info.get("hasFlags"),
                    "period_label": period.get("label"),
                    "period_interval": period.get("interval"),
                    "percent_complete": coverage.get("percentComplete"),
                    "percent_coverage": coverage.get("percentCoverage"),
                }
            )

        if len(results) < LIMIT:
            break

        page += 1
        time.sleep(REQUEST_SLEEP_SECONDS)

    return all_rows


def main():
    control_df = pd.read_csv(CONTROL_TABLE_PATH)

    if MAX_SENSORS is not None:
        control_df = control_df.head(MAX_SENSORS).copy()
        output_path = PROJECT_ROOT / "data" / "raw" / "openaq_delhi_pollutants_90d_long_sample.csv"
    else:
        output_path = PROJECT_ROOT / "data" / "raw" / "openaq_delhi_pollutants_90d_long.csv"

    datetime_from, datetime_to = get_time_window()

    print(f"Fetching from: {datetime_from}")
    print(f"Fetching to:   {datetime_to}")
    print(f"Sensors to fetch: {len(control_df)}")

    all_measurements = []
    failures = []

    for index, sensor_row in control_df.iterrows():
        sensor_id = sensor_row["sensor_id"]
        station_name = sensor_row["station_name"]
        pollutant = sensor_row["parameter_name"]

        print(
            f"[{index + 1}/{len(control_df)}] "
            f"Fetching sensor_id={sensor_id}, station={station_name}, pollutant={pollutant}"
        )

        try:
            sensor_measurements = fetch_sensor_hourly_measurements(
                sensor_row=sensor_row,
                datetime_from=datetime_from,
                datetime_to=datetime_to,
            )

            print(f"  Rows fetched: {len(sensor_measurements)}")
            all_measurements.extend(sensor_measurements)

        except Exception as error:
            print(f"  FAILED: {error}")
            failures.append(
                {
                    "sensor_id": sensor_id,
                    "station_name": station_name,
                    "parameter_name": pollutant,
                    "error": str(error),
                }
            )

        time.sleep(REQUEST_SLEEP_SECONDS)

    measurements_df = pd.DataFrame(all_measurements)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    measurements_df.to_csv(output_path, index=False)

    failures_path = PROJECT_ROOT / "data" / "raw" / "openaq_90d_fetch_failures.csv"
    pd.DataFrame(failures).to_csv(failures_path, index=False)

    print(f"\nSaved measurements to: {output_path}")
    print(f"Saved failures to: {failures_path}")

    print(f"\nTotal measurement rows: {len(measurements_df)}")
    print(f"Failed sensors: {len(failures)}")

    if not measurements_df.empty:
        print("\nRows by pollutant:")
        print(
            measurements_df
            .groupby("parameter_name")
            .size()
            .sort_values(ascending=False)
        )

        print("\nTimestamp range:")
        print("min:", measurements_df["timestamp_local"].min())
        print("max:", measurements_df["timestamp_local"].max())

        print("\nPreview:")
        print(measurements_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()