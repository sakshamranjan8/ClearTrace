from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "live_features_with_aqi_48h.csv"
)

SENSOR_CATALOG_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "sensor_catalog_v1.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "live_features_complete_48h.csv"
)

POLLUTANTS = [
    "pm25",
    "pm10",
    "no2",
    "co",
    "so2",
    "o3",
]

LAG_CONFIG = {
    "pm25": [1, 6, 12, 24],
    "pm10": [1, 6, 12, 24],
    "no2": [1, 6, 24],
    "co": [1, 6, 24],
    "o3": [1, 6, 24],
    "so2": [1, 6],
}

ROLLING_CONFIG = {
    "pm25": [6, 12, 24],
    "pm10": [6, 12, 24],
    "no2": [6, 12],
    "co": [6, 12],
    "o3": [6, 12],
    "so2": [6],
}

SELECTED_RADII = {
    "pm25": 15.0,
    "pm10": 12.5,
    "no2": 15.0,
    "co": 12.5,
    "so2": 15.0,
    "o3": 15.0,
}

MAX_NEIGHBOURS = 3
IDW_POWER = 2


def haversine_km(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    latitude_difference = lat2 - lat1
    longitude_difference = lon2 - lon1

    a = (
        sin(latitude_difference / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(longitude_difference / 2) ** 2
    )

    return (
        earth_radius_km
        * 2
        * atan2(sqrt(a), sqrt(1 - a))
    )


def add_availability_features(features, sensor_catalog):
    historical_capability = (
        sensor_catalog.assign(has_sensor=True)
        .pivot_table(
            index="location_id",
            columns="pollutant",
            values="has_sensor",
            aggfunc="any",
            fill_value=False,
        )
    )

    for pollutant in POLLUTANTS:
        has_column = f"has_{pollutant}"

        previously_seen = (
            features["location_id"]
            .map(
                historical_capability.get(
                    pollutant,
                    pd.Series(dtype=bool),
                )
            )
            .fillna(False)
            .astype(bool)
        )

        seen_in_live_window = (
            features
            .groupby("location_id")[has_column]
            .cummax()
            .astype(bool)
        )

        seen_so_far = (
            previously_seen | seen_in_live_window
        )

        features[
            f"station_seen_{pollutant}_so_far"
        ] = seen_so_far

        features[
            f"{pollutant}_temporary_missing_causal"
        ] = (
            seen_so_far
            & ~features[has_column].astype(bool)
        )

        features[
            f"{pollutant}_not_yet_observed_causal"
        ] = ~seen_so_far

    return features


def add_observed_lags(features):
    for pollutant, lag_hours in LAG_CONFIG.items():
        observed_column = f"{pollutant}_observed"

        for hour in lag_hours:
            features[
                f"{pollutant}_observed_lag_{hour}h"
            ] = (
                features
                .groupby("location_id")[observed_column]
                .shift(hour)
            )

    return features


def add_observed_rolling_features(features):
    for pollutant, windows in ROLLING_CONFIG.items():
        observed_column = f"{pollutant}_observed"

        for window in windows:
            count_column = (
                f"{pollutant}_observed_count_{window}h"
            )

            mean_column = (
                f"{pollutant}_observed_rolling_mean_"
                f"{window}h"
            )

            rolling_count = (
                features
                .groupby("location_id")[observed_column]
                .transform(
                    lambda values: values.rolling(
                        window,
                        min_periods=1,
                    ).count()
                )
            )

            rolling_mean = (
                features
                .groupby("location_id")[observed_column]
                .transform(
                    lambda values: values.rolling(
                        window,
                        min_periods=1,
                    ).mean()
                )
            )

            minimum_required = max(
                2,
                window // 2,
            )

            features[count_column] = (
                rolling_count.astype("int8")
            )

            features[mean_column] = (
                rolling_mean.where(
                    rolling_count >= minimum_required
                )
            )

    return features


def build_station_distances(features):
    stations = (
        features[
            [
                "location_id",
                "station_name",
                "latitude",
                "longitude",
            ]
        ]
        .drop_duplicates("location_id")
        .reset_index(drop=True)
    )

    distance_rows = []

    for target in stations.itertuples(index=False):
        for donor in stations.itertuples(index=False):
            if target.location_id == donor.location_id:
                continue

            distance = haversine_km(
                target.latitude,
                target.longitude,
                donor.latitude,
                donor.longitude,
            )

            distance_rows.append(
                {
                    "target_location_id":
                        target.location_id,
                    "donor_location_id":
                        donor.location_id,
                    "distance_km": distance,
                }
            )

    return pd.DataFrame(distance_rows)


def add_spatial_features(features, station_distances):
    row_index = pd.MultiIndex.from_arrays(
        [
            features["location_id"],
            features["timestamp_hour"],
        ],
        names=[
            "target_location_id",
            "timestamp_hour",
        ],
    )

    for pollutant in POLLUTANTS:
        observed_column = (
            f"{pollutant}_observed"
        )

        eligible_distances = station_distances[
            station_distances["distance_km"]
            <= SELECTED_RADII[pollutant]
        ]

        donor_observations = (
            features[
                [
                    "location_id",
                    "timestamp_hour",
                    observed_column,
                ]
            ]
            .dropna(subset=[observed_column])
            .rename(
                columns={
                    "location_id":
                        "donor_location_id",
                    observed_column:
                        "donor_value",
                }
            )
        )

        candidates = eligible_distances.merge(
            donor_observations,
            on="donor_location_id",
            how="inner",
        )

        candidates = (
            candidates
            .sort_values(
                [
                    "target_location_id",
                    "timestamp_hour",
                    "distance_km",
                ]
            )
            .groupby(
                [
                    "target_location_id",
                    "timestamp_hour",
                ],
                sort=False,
            )
            .head(MAX_NEIGHBOURS)
            .copy()
        )

        candidates["weight"] = (
            1
            / candidates["distance_km"].pow(
                IDW_POWER
            )
        )

        candidates["weighted_value"] = (
            candidates["donor_value"]
            * candidates["weight"]
        )

        estimates = (
            candidates
            .groupby(
                [
                    "target_location_id",
                    "timestamp_hour",
                ],
                as_index=False,
            )
            .agg(
                weighted_sum=(
                    "weighted_value",
                    "sum",
                ),
                weight_sum=("weight", "sum"),
                neighbour_count=(
                    "donor_location_id",
                    "size",
                ),
                nearest_km=(
                    "distance_km",
                    "min",
                ),
            )
        )

        estimates["neighbour_estimate"] = (
            estimates["weighted_sum"]
            / estimates["weight_sum"]
        )

        lookup = estimates.set_index(
            [
                "target_location_id",
                "timestamp_hour",
            ]
        )

        prefix = f"{pollutant}_neighbor"

        features[f"{prefix}_idw_v2"] = (
            lookup["neighbour_estimate"]
            .reindex(row_index)
            .to_numpy()
        )

        features[f"{prefix}_count_v2"] = (
            lookup["neighbour_count"]
            .reindex(row_index)
            .fillna(0)
            .astype("int8")
            .to_numpy()
        )

        features[f"{prefix}_nearest_km_v2"] = (
            lookup["nearest_km"]
            .reindex(row_index)
            .to_numpy()
        )

    return features


def build_complete_features():
    features = pd.read_csv(INPUT_FILE)
    sensor_catalog = pd.read_csv(
        SENSOR_CATALOG_FILE
    )

    features["timestamp_hour"] = pd.to_datetime(
        features["timestamp_hour"]
    )

    features = (
        features
        .sort_values(["location_id", "timestamp_hour"])
        .reset_index(drop=True)
    )

    features = add_availability_features(
        features,
        sensor_catalog,
    )

    features = add_observed_lags(features)

    features = add_observed_rolling_features(
        features
    )

    station_distances = build_station_distances(
        features
    )

    features = add_spatial_features(
        features,
        station_distances,
    )

    features.to_csv(OUTPUT_FILE, index=False)

    return features


if __name__ == "__main__":
    features = build_complete_features()

    latest_timestamp = features[
        "timestamp_hour"
    ].max()

    latest_rows = features[
        features["timestamp_hour"]
        == latest_timestamp
    ]

    spatial_columns = [
        column
        for column in features.columns
        if "_neighbor_" in column
    ]

    lag_columns = [
        column
        for column in features.columns
        if "_observed_lag_" in column
    ]

    rolling_mean_columns = [
        column
        for column in features.columns
        if "_observed_rolling_mean_" in column
    ]

    rolling_count_columns = [
        column
        for column in features.columns
        if "_observed_count_" in column
    ]

    print("Latest rows:", len(latest_rows))
    print("Observed lag columns:", len(lag_columns))
    print(
        "Rolling mean columns:",
        len(rolling_mean_columns),
    )
    print(
        "Rolling count columns:",
        len(rolling_count_columns),
    )
    print("Spatial columns:", len(spatial_columns))

    print(
        "Infinite numeric values:",
        np.isinf(
            latest_rows.select_dtypes(
                include=np.number
            )
        ).sum().sum(),
    )

    print("\nSaved:", OUTPUT_FILE)