from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COMPLETE_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "live_features_complete_48h.csv"
)

FEATURE_LIST_FILE = (
    PROJECT_ROOT
    / "reports"
    / "feature_list.txt"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "latest_model_input.csv"
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


def build_latest_model_input():
    features = pd.read_csv(COMPLETE_FEATURE_FILE)

    features["timestamp_hour"] = pd.to_datetime(
        features["timestamp_hour"]
    )

    model_features = load_feature_list()

    missing_features = [
        column
        for column in model_features
        if column not in features.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    latest_timestamp = features[
        "timestamp_hour"
    ].max()

    latest_rows = (
        features[
            features["timestamp_hour"]
            == latest_timestamp
        ]
        .sort_values("location_id")
        .reset_index(drop=True)
    )

    # location_id and timestamp are metadata.
    # They are not passed into CatBoost.
    output = latest_rows[
        [
            "location_id",
            "timestamp_hour",
        ]
        + model_features
    ].copy()

    numeric_model_input = output[
        model_features
    ].select_dtypes(include=np.number)

    infinite_count = np.isinf(
        numeric_model_input
    ).sum().sum()

    if infinite_count:
        raise ValueError(
            f"Model input contains "
            f"{infinite_count} infinite values."
        )

    output.to_csv(OUTPUT_FILE, index=False)

    return output, model_features


if __name__ == "__main__":
    model_input, model_features = (
        build_latest_model_input()
    )

    print("Model-input rows:", len(model_input))
    print("Model features:", len(model_features))

    print(
        "Duplicate stations:",
        model_input["location_id"]
        .duplicated()
        .sum(),
    )

    print(
        "Infinite numeric values:",
        np.isinf(
            model_input[
                model_features
            ].select_dtypes(include=np.number)
        ).sum().sum(),
    )

    print(
        "Missing numeric values:",
        model_input[
            model_features
        ].select_dtypes(include=np.number)
        .isna()
        .sum()
        .sum(),
    )

    print("\nSaved:", OUTPUT_FILE)