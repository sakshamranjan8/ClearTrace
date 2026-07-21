import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLLUTANT_HISTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "station_pollutant_history_48h.csv"
)

WEATHER_CACHE_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "station_weather_history_48h.csv"
)

OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]

BATCH_SIZE = 5
MAX_ATTEMPTS = 5


def request_weather_batch(session, params):
    """Request one weather batch with retry and backoff."""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(
                OPEN_METEO_URL,
                params=params,
                timeout=(20, 60),
            )

            response.raise_for_status()

            return response.json()

        except requests.RequestException as error:
            if attempt == MAX_ATTEMPTS:
                raise

            wait_seconds = 2 ** attempt

            print(
                f"Weather request failed: {error}"
            )

            print(
                f"Retrying in {wait_seconds} seconds "
                f"({attempt}/{MAX_ATTEMPTS})..."
            )

            time.sleep(wait_seconds)


def fetch_station_weather():
    pollutant_history = pd.read_csv(
        POLLUTANT_HISTORY_FILE
    )

    pollutant_history["timestamp_hour"] = pd.to_datetime(
        pollutant_history["timestamp_hour"]
    )

    stations = (
        pollutant_history[
            [
                "location_id",
                "station_name",
                "latitude",
                "longitude",
            ]
        ]
        .drop_duplicates("location_id")
        .sort_values("location_id")
        .reset_index(drop=True)
    )

    start_hour = (
        pollutant_history["timestamp_hour"]
        .min()
        .strftime("%Y-%m-%dT%H:%M")
    )

    end_hour = (
        pollutant_history["timestamp_hour"]
        .max()
        .strftime("%Y-%m-%dT%H:%M")
    )

    weather_responses = []

    with requests.Session() as session:
        for start_index in range(
            0,
            len(stations),
            BATCH_SIZE,
        ):
            station_batch = stations.iloc[
                start_index:start_index + BATCH_SIZE
            ]

            batch_responses = request_weather_batch(
                session=session,
                params={
                    "latitude": ",".join(
                        station_batch[
                            "latitude"
                        ].astype(str)
                    ),
                    "longitude": ",".join(
                        station_batch[
                            "longitude"
                        ].astype(str)
                    ),
                    "hourly": ",".join(
                        WEATHER_VARIABLES
                    ),
                    "start_hour": start_hour,
                    "end_hour": end_hour,
                    "timezone": "Asia/Kolkata",
                    "wind_speed_unit": "kmh",
                },
            )

            if isinstance(batch_responses, dict):
                batch_responses = [batch_responses]

            if len(batch_responses) != len(station_batch):
                raise RuntimeError(
                    "Open-Meteo batch response count "
                    "does not match the station batch."
                )

            weather_responses.extend(batch_responses)

            print(
                "Weather fetched:",
                min(
                    start_index + BATCH_SIZE,
                    len(stations),
                ),
                "/",
                len(stations),
            )

            time.sleep(1)

    if len(weather_responses) != len(stations):
        raise RuntimeError(
            "Total Open-Meteo response count does not "
            "match the station count."
        )

    weather_frames = []

    for station, weather_response in zip(
        stations.itertuples(index=False),
        weather_responses,
    ):
        hourly = weather_response["hourly"]

        station_weather = pd.DataFrame(hourly)

        station_weather["timestamp_hour"] = (
            pd.to_datetime(station_weather["time"])
            .dt.tz_localize("Asia/Kolkata")
        )

        station_weather["location_id"] = (
            station.location_id
        )

        station_weather["station_name"] = (
            station.station_name
        )

        weather_frames.append(station_weather)

    weather = pd.concat(
        weather_frames,
        ignore_index=True,
    )

    # Match the weather transformations used for training.
    weather["wind_speed_10m_ms"] = (
        weather["wind_speed_10m"] / 3.6
    )

    wind_angle = np.deg2rad(
        weather["wind_direction_10m"]
    )

    weather["wind_u_ms"] = (
        -weather["wind_speed_10m_ms"]
        * np.sin(wind_angle)
    )

    weather["wind_v_ms"] = (
        -weather["wind_speed_10m_ms"]
        * np.cos(wind_angle)
    )

    weather["is_raining"] = (
        weather["precipitation"] > 0
    ).astype(int)

    output_columns = [
        "location_id",
        "station_name",
        "timestamp_hour",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "surface_pressure",
        "is_raining",
        "wind_speed_10m_ms",
        "wind_u_ms",
        "wind_v_ms",
    ]

    weather = (
        weather[output_columns]
        .sort_values(
            ["location_id", "timestamp_hour"]
        )
        .reset_index(drop=True)
    )

    weather.to_csv(
        WEATHER_CACHE_FILE,
        index=False,
    )

    return weather


if __name__ == "__main__":
    weather = fetch_station_weather()

    print("\nWeather rows:", len(weather))

    print(
        "Duplicate station-hours:",
        weather.duplicated(
            ["location_id", "timestamp_hour"]
        ).sum(),
    )

    print(
        "Missing weather values:",
        weather.isna().sum().sum(),
    )

    print("\nSaved:", WEATHER_CACHE_FILE)