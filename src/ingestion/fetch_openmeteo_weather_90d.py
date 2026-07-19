import time
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STATION_REVIEW_PATH = PROJECT_ROOT / "data" / "processed" / "final_delhi_station_review.csv"
POLLUTANT_WIDE_PATH = PROJECT_ROOT / "data" / "processed" / "delhi_pollutants_90d_wide.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "openmeteo_delhi_weather_90d.csv"
FAILURES_PATH = PROJECT_ROOT / "data" / "raw" / "openmeteo_weather_fetch_failures.csv"

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]

REQUEST_SLEEP_SECONDS = 0.2


def get_pollutant_date_range():
    """
    Use the pollutant wide table to derive the exact date range.

    This keeps weather aligned with the pollutant dataset instead of hardcoding dates.
    Open-Meteo needs start_date and end_date, not full timestamps.
    """
    pollutants_df = pd.read_csv(POLLUTANT_WIDE_PATH, usecols=["timestamp_hour"])

    pollutants_df["timestamp_hour"] = pd.to_datetime(
        pollutants_df["timestamp_hour"],
        errors="coerce"
    )

    min_timestamp = pollutants_df["timestamp_hour"].min()
    max_timestamp = pollutants_df["timestamp_hour"].max()

    start_date = min_timestamp.date().isoformat()
    end_date = max_timestamp.date().isoformat()

    return start_date, end_date, min_timestamp, max_timestamp


def fetch_weather_for_station(station_row, start_date, end_date):
    """
    Fetch hourly weather for one AQI station coordinate.

    One station gets one weather time series.
    """
    params = {
        "latitude": station_row["latitude"],
        "longitude": station_row["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "Asia/Kolkata",
    }

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=40,
    )

    if response.status_code in [429, 500, 502, 503, 504]:
        print("Temporary API issue. Retrying after 5 seconds...")
        time.sleep(5)

        response = requests.get(
            BASE_URL,
            params=params,
            timeout=40,
        )

    response.raise_for_status()

    data = response.json()
    hourly = data.get("hourly", {})

    if "time" not in hourly:
        return pd.DataFrame()

    weather_df = pd.DataFrame(hourly)

    weather_df["location_id"] = station_row["location_id"]
    weather_df["station_name"] = station_row["station_name"]
    weather_df["latitude"] = station_row["latitude"]
    weather_df["longitude"] = station_row["longitude"]

    weather_df = weather_df.rename(columns={"time": "timestamp_hour"})

    weather_df["timestamp_hour"] = pd.to_datetime(
        weather_df["timestamp_hour"],
        errors="coerce"
    )

    final_columns = [
        "location_id",
        "station_name",
        "latitude",
        "longitude",
        "timestamp_hour",
    ] + HOURLY_VARIABLES

    return weather_df[final_columns]


def main():
    stations_df = pd.read_csv(STATION_REVIEW_PATH)

    start_date, end_date, min_timestamp, max_timestamp = get_pollutant_date_range()

    print(f"Pollutant timestamp range:")
    print(f"min: {min_timestamp}")
    print(f"max: {max_timestamp}")

    print(f"\nWeather fetch date range:")
    print(f"start_date: {start_date}")
    print(f"end_date:   {end_date}")

    print(f"\nStations to fetch weather for: {len(stations_df)}")

    all_weather = []
    failures = []

    for index, station_row in stations_df.iterrows():
        location_id = station_row["location_id"]
        station_name = station_row["station_name"]

        print(
            f"[{index + 1}/{len(stations_df)}] "
            f"Fetching weather for location_id={location_id}, station={station_name}"
        )

        try:
            station_weather_df = fetch_weather_for_station(
                station_row=station_row,
                start_date=start_date,
                end_date=end_date,
            )

            print(f"  Rows fetched: {len(station_weather_df)}")
            all_weather.append(station_weather_df)

        except Exception as error:
            print(f"  FAILED: {error}")

            failures.append(
                {
                    "location_id": location_id,
                    "station_name": station_name,
                    "error": str(error),
                }
            )

        time.sleep(REQUEST_SLEEP_SECONDS)

    if all_weather:
        weather_df = pd.concat(all_weather, ignore_index=True)
    else:
        weather_df = pd.DataFrame()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    weather_df.to_csv(OUTPUT_PATH, index=False)
    pd.DataFrame(failures).to_csv(FAILURES_PATH, index=False)

    print(f"\nSaved weather data to: {OUTPUT_PATH}")
    print(f"Saved failures to: {FAILURES_PATH}")

    print(f"\nTotal weather rows: {len(weather_df)}")
    print(f"Failed stations: {len(failures)}")

    if not weather_df.empty:
        print(f"\nWeather stations: {weather_df['location_id'].nunique()}")

        print("\nTimestamp range:")
        print("min:", weather_df["timestamp_hour"].min())
        print("max:", weather_df["timestamp_hour"].max())

        print("\nMissing values:")
        print(weather_df[HOURLY_VARIABLES].isna().sum())

        print("\nPreview:")
        print(weather_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()