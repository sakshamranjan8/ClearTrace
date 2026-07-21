import argparse
from pathlib import Path

import pandas as pd

from inference.station_selector import (
    add_blend_weights,
    find_three_nearest_stations,
)






PROJECT_ROOT = Path(__file__).resolve().parents[1]

STATION_CATALOG_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "station_catalog_v1.csv"
)

STATION_FORECAST_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "station_forecasts_24h.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "user_forecast_example.csv"
)


def predict_user_aqi(
    user_latitude,
    user_longitude,
):
    stations = pd.read_csv(
        STATION_CATALOG_FILE
    )

    station_forecasts = pd.read_csv(
        STATION_FORECAST_FILE
    )

    station_forecasts["forecast_origin"] = (
        pd.to_datetime(
            station_forecasts["forecast_origin"]
        )
    )

    station_forecasts["forecast_timestamp"] = (
        pd.to_datetime(
            station_forecasts["forecast_timestamp"]
        )
    )

    nearest_three = find_three_nearest_stations(
        user_latitude=user_latitude,
        user_longitude=user_longitude,
        station_catalog=stations,
    )

    nearest_three = add_blend_weights(
        nearest_three
    )

    selected_forecasts = station_forecasts.merge(
        nearest_three[
            [
                "location_id",
                "distance_km",
                "blend_weight",
            ]
        ],
        on="location_id",
        how="inner",
        validate="many_to_one",
    )

    expected_rows = 3 * 24

    if len(selected_forecasts) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} selected "
            f"station forecasts, found "
            f"{len(selected_forecasts)}."
        )

    selected_forecasts["weighted_aqi"] = (
        selected_forecasts["predicted_aqi"]
        * selected_forecasts["blend_weight"]
    )

    user_forecast = (
        selected_forecasts
        .groupby(
            [
                "forecast_origin",
                "forecast_timestamp",
                "horizon_hours",
            ],
            as_index=False,
        )
        .agg(
            predicted_aqi=(
                "weighted_aqi",
                "sum",
            )
        )
        .sort_values("horizon_hours")
        .reset_index(drop=True)
    )

    user_forecast["predicted_aqi"] = (
        user_forecast["predicted_aqi"]
        .round(2)
    )

    return nearest_three, user_forecast


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Predict a user's next 24 hours of AQI."
        )
    )

    parser.add_argument(
        "--latitude",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--longitude",
        type=float,
        required=True,
    )

    arguments = parser.parse_args()

    nearest_stations, forecast = predict_user_aqi(
        user_latitude=arguments.latitude,
        user_longitude=arguments.longitude,
    )

    forecast.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nNearest stations:")

    print(
        nearest_stations[
            [
                "station_name",
                "distance_km",
                "blend_weight",
            ]
        ].to_string(index=False)
    )

    print("\nUser AQI forecast:")

    print(
        forecast[
            [
                "forecast_timestamp",
                "horizon_hours",
                "predicted_aqi",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:", OUTPUT_FILE)