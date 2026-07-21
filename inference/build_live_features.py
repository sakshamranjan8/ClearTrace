from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

POLLUTANT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "station_pollutant_history_48h.csv"
)

WEATHER_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "station_weather_history_48h.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "live_feature_base_48h.csv"
)

WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "is_raining",
    "wind_speed_10m_ms",
    "wind_u_ms",
    "wind_v_ms",
]


def build_live_feature_base():
    pollutants = pd.read_csv(POLLUTANT_FILE)
    weather = pd.read_csv(WEATHER_FILE)

    pollutants["timestamp_hour"] = pd.to_datetime(
        pollutants["timestamp_hour"]
    )

    weather["timestamp_hour"] = pd.to_datetime(
        weather["timestamp_hour"]
    )

    live_features = pollutants.merge(
        weather[
            [
                "location_id",
                "timestamp_hour",
            ]
            + WEATHER_COLUMNS
        ],
        on=["location_id", "timestamp_hour"],
        how="left",
        validate="one_to_one",
    )

    # Calendar features: identical to training.
    live_features["hour"] = (
        live_features["timestamp_hour"].dt.hour
    )

    live_features["day_of_week"] = (
        live_features["timestamp_hour"].dt.dayofweek
    )

    live_features["is_weekend"] = (
        live_features["day_of_week"] >= 5
    ).astype(int)

    live_features["month"] = (
        live_features["timestamp_hour"].dt.month
    )

    live_features["hour_sin"] = np.sin(
        2 * np.pi * live_features["hour"] / 24
    )

    live_features["hour_cos"] = np.cos(
        2 * np.pi * live_features["hour"] / 24
    )

    live_features["day_sin"] = np.sin(
        2
        * np.pi
        * live_features["day_of_week"]
        / 7
    )

    live_features["day_cos"] = np.cos(
        2
        * np.pi
        * live_features["day_of_week"]
        / 7
    )

    live_features = (
        live_features
        .sort_values(["location_id", "timestamp_hour"])
        .reset_index(drop=True)
    )

    live_features.to_csv(OUTPUT_FILE, index=False)

    return live_features


if __name__ == "__main__":
    live_features = build_live_feature_base()

    print("Rows:", len(live_features))

    print(
        "Stations:",
        live_features["location_id"].nunique(),
    )

    print(
        "Hours:",
        live_features["timestamp_hour"].nunique(),
    )

    print(
        "Duplicate station-hours:",
        live_features.duplicated(
            ["location_id", "timestamp_hour"]
        ).sum(),
    )

    print(
        "Missing weather values:",
        live_features[WEATHER_COLUMNS]
        .isna()
        .sum()
        .sum(),
    )

    print("\nSaved:", OUTPUT_FILE)