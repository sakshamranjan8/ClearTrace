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

OPENAQ_BASE_URL = "https://api.openaq.org/v3"

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
    raise ValueError(
        "OPENAQ_API_KEY was not found in the .env file."
    )


def fetch_location_sensors(location_id):
    """Fetch all sensors belonging to one OpenAQ location."""

    url = (
        f"{OPENAQ_BASE_URL}"
        f"/locations/{location_id}/sensors"
    )

    response = requests.get(
        url,
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


def extract_utc_datetime(sensor, field_name):
    """Safely extract a UTC datetime from an OpenAQ sensor."""

    datetime_object = sensor.get(field_name)

    if not datetime_object:
        return None

    return datetime_object.get("utc")


def build_sensor_catalog(station_catalog):
    """Build the pollutant-sensor mapping for all 38 stations."""

    sensor_rows = []
    failed_stations = []

    for station in station_catalog.itertuples(index=False):
        print(
            f"Fetching sensors for "
            f"{station.station_name}..."
        )

        try:
            sensors = fetch_location_sensors(
                station.location_id
            )
        except requests.RequestException as error:
            failed_stations.append(
                {
                    "location_id": station.location_id,
                    "station_name": station.station_name,
                    "error": str(error),
                }
            )
            continue

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
                    "datetime_first_utc": extract_utc_datetime(
                        sensor,
                        "datetimeFirst",
                    ),
                    "datetime_last_utc": extract_utc_datetime(
                        sensor,
                        "datetimeLast",
                    ),
                }
            )

        # Avoid sending all 38 requests simultaneously.
        time.sleep(0.1)

    if failed_stations:
        print("\nFailed stations:")

        for failure in failed_stations:
            print(failure)

        raise RuntimeError(
            "Sensor catalogue was not saved because "
            "some station requests failed."
        )

    sensor_catalog = (
        pd.DataFrame(sensor_rows)
        .sort_values(
            [
                "station_name",
                "pollutant",
                "sensor_id",
            ]
        )
        .reset_index(drop=True)
    )

    sensor_catalog.to_csv(
        SENSOR_CATALOG_FILE,
        index=False,
    )

    return sensor_catalog


if __name__ == "__main__":
    stations = pd.read_csv(STATION_CATALOG_FILE)

    sensor_catalog = build_sensor_catalog(stations)

    print("\nSensor catalogue created.")
    print("Stations requested:", len(stations))
    print("Sensors retained:", len(sensor_catalog))

    print("\nStations available per pollutant:")

    print(
        sensor_catalog
        .groupby("pollutant")["location_id"]
        .nunique()
        .sort_index()
    )

    duplicate_pairs = sensor_catalog.duplicated(
        ["location_id", "pollutant"],
        keep=False,
    ).sum()

    print(
        "\nRows belonging to station-pollutant "
        "pairs with multiple sensors:",
        duplicate_pairs,
    )

    print("\nSaved to:", SENSOR_CATALOG_FILE)