from pathlib import Path

from math import atan2, cos, radians, sin, sqrt
import pandas as pd


# station_selector.py is inside ClearTrace/inference/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "features_core_v2.csv"
)

STATION_CATALOG_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "station_catalog_v1.csv"
)


def build_station_catalog() -> pd.DataFrame:
    """
    Extract one permanent record for each monitoring station.

    We use the feature file only once to build this small catalogue.
    The live prediction pipeline should not repeatedly load the
    approximately 79 MB feature file.
    """

    station_catalog = (
        pd.read_csv(
            FEATURE_FILE,
            usecols=[
                "location_id",
                "station_name",
                "provider_name",
                "latitude",
                "longitude",
            ],
        )
        .drop_duplicates(subset=["location_id"])
        .sort_values("station_name")
        .reset_index(drop=True)
    )

    station_catalog.to_csv(STATION_CATALOG_FILE, index=False)

    return station_catalog

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in kilometres."""

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


def find_three_nearest_stations(
    user_latitude,
    user_longitude,
    station_catalog,
):
    stations_with_distance = station_catalog.copy()

    stations_with_distance["distance_km"] = [
        haversine_km(
            user_latitude,
            user_longitude,
            station.latitude,
            station.longitude,
        )
        for station in stations_with_distance.itertuples()
    ]

    return (
        stations_with_distance
        .sort_values("distance_km")
        .head(3)
        .reset_index(drop=True)
    )

def add_blend_weights(
    nearest_stations,
    idw_power=2,
):
    weighted_stations = nearest_stations.copy()

    exact_station = (
        weighted_stations["distance_km"] <= 0.001
    )

    if exact_station.any():
        weighted_stations["blend_weight"] = 0.0

        weighted_stations.loc[
            exact_station,
            "blend_weight",
        ] = 1.0 / exact_station.sum()

        return weighted_stations

    raw_weights = (
        1
        / weighted_stations[
            "distance_km"
        ].pow(idw_power)
    )

    weighted_stations["blend_weight"] = (
        raw_weights / raw_weights.sum()
    )

    return weighted_stations


if __name__ == "__main__":
    if STATION_CATALOG_FILE.exists():
        stations = pd.read_csv(STATION_CATALOG_FILE)
    else:
        stations = build_station_catalog()

    # Temporary test location: central Delhi
    user_latitude = 28.6139
    user_longitude = 77.2090

    nearest_three = find_three_nearest_stations(
        user_latitude,
        user_longitude,
        stations,
    )

    nearest_three = add_blend_weights(
        nearest_three
    )

    print(
        nearest_three[
            [
                "location_id",
                "station_name",
                "distance_km",
                "blend_weight",
            ]
        ].to_string(index=False)
    )

    print(
        "\nTotal weight:",
        nearest_three["blend_weight"].sum(),
    )