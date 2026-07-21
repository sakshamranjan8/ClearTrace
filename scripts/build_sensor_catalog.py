import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATION_CATALOG_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "station_catalog_v1.csv"
)

SENSOR_CATALOG_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "sensor_catalog_v1.csv"
)

OPENAQ_URL = "https://api.openaq.org/v3"

POLLUTANTS = {
    "pm25",
    "pm10",
    "no2",
    "co",
    "so2",
    "o3",
}


load_dotenv(PROJECT_ROOT / ".env")

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

if not OPENAQ_API_KEY:
    raise ValueError("OPENAQ_API_KEY is missing.")


def get_utc_datetime(sensor, field):
    datetime_data = sensor.get(field)

    if datetime_data is None:
        return None

    return datetime_data.get("utc")


def fetch_sensors(location_id):
    response = requests.get(
        f"{OPENAQ_URL}/locations/{location_id}/sensors",
        headers={
            "X-API-Key": OPENAQ_API_KEY,
        },
        params={
            "limit": 100,
        },
        timeout=30,
    )

    response.raise_for_status()

    return response.json()["results"]


def build_sensor_catalog():
    stations = pd.read_csv(STATION_CATALOG_FILE)

    sensor_rows = []

    for number, station in enumerate(
        stations.itertuples(index=False),
        start=1,
    ):
        print(
            f"[{number}/{len(stations)}] "
            f"{station.station_name}"
        )

        sensors = fetch_sensors(station.location_id)

        for sensor in sensors:
            parameter = sensor.get("parameter", {})
            pollutant = parameter.get("name")

            if pollutant not in POLLUTANTS:
                continue

            sensor_rows.append(
                {
                    "location_id": station.location_id,
                    "station_name": station.station_name,
                    "sensor_id": sensor["id"],
                    "pollutant": pollutant,
                    "unit": parameter.get("units"),
                    "datetime_first_utc": get_utc_datetime(
                        sensor,
                        "datetimeFirst",
                    ),
                    "datetime_last_utc": get_utc_datetime(
                        sensor,
                        "datetimeLast",
                    ),
                }
            )

        time.sleep(0.1)

    sensor_catalog = (
        pd.DataFrame(sensor_rows)
        .sort_values(
            ["location_id", "pollutant", "sensor_id"]
        )
        .reset_index(drop=True)
    )

    sensor_catalog.to_csv(
        SENSOR_CATALOG_FILE,
        index=False,
    )

    return sensor_catalog


if __name__ == "__main__":
    catalog = build_sensor_catalog()

    print("\nSensor catalogue created.")
    print("Sensor rows:", len(catalog))

    print("\nStation coverage by pollutant:")

    print(
        catalog
        .groupby("pollutant")["location_id"]
        .nunique()
        .sort_index()
    )

    duplicate_rows = catalog.duplicated(
        ["location_id", "pollutant"],
        keep=False,
    ).sum()

    print(
        "\nRows in station-pollutant groups "
        "with multiple sensors:",
        duplicate_rows,
    )

    print("\nSaved:", SENSOR_CATALOG_FILE)