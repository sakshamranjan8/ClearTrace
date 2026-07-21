from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "latest_model_input.csv"
)

FEATURE_LIST_FILE = (
    PROJECT_ROOT
    / "reports"
    / "feature_list.txt"
)

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "station_forecasts_24h.csv"
)


def load_feature_list():
    with open(
        FEATURE_LIST_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


def predict_all_stations():
    model_input = pd.read_csv(
        MODEL_INPUT_FILE
    )

    model_input["timestamp_hour"] = pd.to_datetime(
        model_input["timestamp_hour"]
    )

    feature_columns = load_feature_list()

    if len(feature_columns) != 118:
        raise ValueError(
            f"Expected 118 model features, "
            f"found {len(feature_columns)}."
        )

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in model_input.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    forecast_rows = []

    for horizon in range(1, 25):
        model_path = (
            MODEL_DIRECTORY
            / f"horizon_{horizon:02d}h.joblib"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing model: {model_path}"
            )

        model = joblib.load(model_path)

        model_feature_names = list(
            model.feature_names_
        )

        if set(model_feature_names) != set(feature_columns):
            missing_from_model = sorted(
                set(feature_columns)
                - set(model_feature_names)
            )

            missing_from_feature_list = sorted(
                set(model_feature_names)
                - set(feature_columns)
            )

            raise ValueError(
                f"Horizon {horizon} has a true "
                f"feature-contract mismatch. "
                f"Missing from model: "
                f"{missing_from_model}. "
                f"Missing from feature list: "
                f"{missing_from_feature_list}."
            )

        # The saved CatBoost model determines the exact order.
        prediction_input = model_input[
            model_feature_names
        ].copy()

        predictions = model.predict(
            prediction_input
        )

        # CPCB AQI is bounded between 0 and 500.
        predictions = np.clip(
            predictions,
            0,
            500,
        )

        horizon_rows = model_input[
            [
                "location_id",
                "station_name",
                "latitude",
                "longitude",
                "timestamp_hour",
            ]
        ].copy()

        horizon_rows.rename(
            columns={
                "timestamp_hour":
                    "forecast_origin"
            },
            inplace=True,
        )

        horizon_rows["horizon_hours"] = horizon

        horizon_rows["forecast_timestamp"] = (
            horizon_rows["forecast_origin"]
            + pd.Timedelta(hours=horizon)
        )

        horizon_rows["predicted_aqi"] = (
            predictions
        )

        forecast_rows.append(horizon_rows)

        print(
            f"Predicted horizon "
            f"{horizon:02d}/24"
        )

    forecasts = pd.concat(
        forecast_rows,
        ignore_index=True,
    )

    forecasts["predicted_aqi"] = (
        forecasts["predicted_aqi"].round(2)
    )

    forecasts = (
        forecasts
        .sort_values(
            ["location_id", "horizon_hours"]
        )
        .reset_index(drop=True)
    )

    forecasts.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return forecasts


if __name__ == "__main__":
    forecasts = predict_all_stations()

    print("\nForecast rows:", len(forecasts))

    print(
        "Stations:",
        forecasts["location_id"].nunique(),
    )

    print(
        "Horizons:",
        forecasts["horizon_hours"].nunique(),
    )

    print(
        "Missing predictions:",
        forecasts["predicted_aqi"]
        .isna()
        .sum(),
    )

    print(
        "Prediction range:",
        round(
            forecasts["predicted_aqi"].min(),
            2,
        ),
        "to",
        round(
            forecasts["predicted_aqi"].max(),
            2,
        ),
    )

    print("\nSaved:", OUTPUT_FILE)