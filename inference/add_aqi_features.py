from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "live_feature_base_48h.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "live_features_with_aqi_48h.csv"
)

POLLUTANTS = [
    "pm25",
    "pm10",
    "no2",
    "co",
    "so2",
    "o3",
]

BREAKPOINTS = {
    "pm25": [
        (0, 30, 0, 50),
        (30, 60, 51, 100),
        (60, 90, 101, 200),
        (90, 120, 201, 300),
        (120, 250, 301, 400),
        (250, 500, 401, 500),
    ],
    "pm10": [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 250, 101, 200),
        (250, 350, 201, 300),
        (350, 430, 301, 400),
        (430, 600, 401, 500),
    ],
    "no2": [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 180, 101, 200),
        (180, 280, 201, 300),
        (280, 400, 301, 400),
        (400, 800, 401, 500),
    ],
    "so2": [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 380, 101, 200),
        (380, 800, 201, 300),
        (800, 1600, 301, 400),
        (1600, 2000, 401, 500),
    ],
    "co": [
        (0, 1, 0, 50),
        (1, 2, 51, 100),
        (2, 10, 101, 200),
        (10, 17, 201, 300),
        (17, 34, 301, 400),
        (34, 50, 401, 500),
    ],
    "o3": [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 168, 101, 200),
        (168, 208, 201, 300),
        (208, 748, 301, 400),
        (748, 1000, 401, 500),
    ],
}


def calculate_subindex(value, pollutant):
    if pd.isna(value):
        return np.nan

    for (
        concentration_low,
        concentration_high,
        index_low,
        index_high,
    ) in BREAKPOINTS[pollutant]:
        if value <= concentration_high:
            return (
                (index_high - index_low)
                / (
                    concentration_high
                    - concentration_low
                )
                * (value - concentration_low)
                + index_low
            )

    return 500.0


def add_aqi_features():
    features = pd.read_csv(INPUT_FILE)

    features["timestamp_hour"] = pd.to_datetime(
        features["timestamp_hour"]
    )

    features = (
        features
        .sort_values(["location_id", "timestamp_hour"])
        .reset_index(drop=True)
    )

    averaging_rules = {
        "pm25": (24, 18),
        "pm10": (24, 18),
        "no2": (24, 18),
        "so2": (24, 18),
        "co": (8, 6),
        "o3": (8, 6),
    }

    subindex_columns = []

    for pollutant, (
        window_hours,
        minimum_observations,
    ) in averaging_rules.items():
        observed_column = (
            f"{pollutant}_observed"
        )

        average_column = (
            f"{pollutant}_{window_hours}h_avg"
        )

        subindex_column = (
            f"{pollutant}_subindex"
        )

        features[average_column] = (
            features
            .groupby("location_id")[observed_column]
            .transform(
                lambda values: values.rolling(
                    window=window_hours,
                    min_periods=minimum_observations,
                ).mean()
            )
        )

        features[subindex_column] = features[
            average_column
        ].apply(
            lambda value: calculate_subindex(
                value,
                pollutant,
            )
        )

        subindex_columns.append(subindex_column)

    features["available_subindex_count"] = (
        features[subindex_columns]
        .notna()
        .sum(axis=1)
    )

    features["has_particulate_subindex"] = (
        features[
            [
                "pm25_subindex",
                "pm10_subindex",
            ]
        ]
        .notna()
        .any(axis=1)
    )

    features["aqi_calculation_valid"] = (
        (
            features["available_subindex_count"]
            >= 3
        )
        & features["has_particulate_subindex"]
    )

    maximum_subindex = (
        features[subindex_columns]
        .max(axis=1)
    )

    features["current_aqi"] = (
        maximum_subindex
        .round()
        .where(
            features["aqi_calculation_valid"]
        )
    )

    for lag_hours in [1, 6, 12, 24]:
        features[f"aqi_lag_{lag_hours}h"] = (
            features
            .groupby("location_id")["current_aqi"]
            .shift(lag_hours)
        )

    features.to_csv(OUTPUT_FILE, index=False)

    return features


if __name__ == "__main__":
    features = add_aqi_features()

    latest_timestamp = features[
        "timestamp_hour"
    ].max()

    latest_rows = features[
        features["timestamp_hour"]
        == latest_timestamp
    ]

    print("Latest timestamp:", latest_timestamp)

    print(
        "Valid current AQI stations:",
        latest_rows["current_aqi"].notna().sum(),
        "/",
        len(latest_rows),
    )

    print("\nLatest AQI lag availability:")

    print(
        latest_rows[
            [
                "aqi_lag_1h",
                "aqi_lag_6h",
                "aqi_lag_12h",
                "aqi_lag_24h",
            ]
        ].notna().sum()
    )

    print("\nSaved:", OUTPUT_FILE)