from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CACHE_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "openaq_hourly_48h.csv"
)

STATION_CATALOG_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "station_catalog_v1.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "station_pollutant_history_48h.csv"
)

POLLUTANTS = [
    "pm25",
    "pm10",
    "no2",
    "co",
    "so2",
    "o3",
]

# Same conversions used by the training pipeline.
PPB_TO_UG_M3 = {
    "no2": 1.88,
    "so2": 2.62,
}


def prepare_pollutant_history():
    observations = pd.read_csv(CACHE_FILE)
    stations = pd.read_csv(STATION_CATALOG_FILE)

    observations["timestamp_utc"] = pd.to_datetime(
        observations["timestamp_utc"],
        utc=True,
    )

    observations["value"] = pd.to_numeric(
        observations["value"],
        errors="coerce",
    )

    # Only direct, unflagged, non-negative observations.
    observations = observations[
        (~observations["has_flags"].astype(bool))
        & observations["value"].notna()
        & (observations["value"] >= 0)
    ].copy()

    observations["observed_value"] = observations["value"]

    for pollutant, conversion_factor in PPB_TO_UG_M3.items():
        pollutant_mask = (
            observations["pollutant"] == pollutant
        )

        observations.loc[
            pollutant_mask,
            "observed_value",
        ] *= conversion_factor

    # CO remains unchanged because the source values are treated
    # as mg/m³-scale values, matching the training decision.

    observations["timestamp_hour"] = (
        observations["timestamp_utc"]
        .dt.tz_convert("Asia/Kolkata")
        .dt.floor("h")
    )

    latest_by_pollutant = (
        observations
        .groupby("pollutant")["timestamp_hour"]
        .max()
    )

    # Use the newest hour available for all six pollutants.
    data_origin = latest_by_pollutant.reindex(
        POLLUTANTS
    ).min()

    history_hours = pd.date_range(
        end=data_origin,
        periods=48,
        freq="h",
    )

    observations = observations[
        observations["timestamp_hour"].isin(
            history_hours
        )
    ]

    pollutant_table = observations.pivot_table(
        index=["location_id", "timestamp_hour"],
        columns="pollutant",
        values="observed_value",
        aggfunc="mean",
    ).reset_index()

    pollutant_table.columns.name = None

    station_hours = (
        stations.assign(_join_key=1)
        .merge(
            pd.DataFrame(
                {
                    "timestamp_hour": history_hours,
                    "_join_key": 1,
                }
            ),
            on="_join_key",
        )
        .drop(columns="_join_key")
    )

    history = station_hours.merge(
        pollutant_table,
        on=["location_id", "timestamp_hour"],
        how="left",
    )

    for pollutant in POLLUTANTS:
        history.rename(
            columns={
                pollutant: f"{pollutant}_observed"
            },
            inplace=True,
        )

        history[f"has_{pollutant}"] = history[
            f"{pollutant}_observed"
        ].notna()

    history = (
        history
        .sort_values(["location_id", "timestamp_hour"])
        .reset_index(drop=True)
    )

    history.to_csv(OUTPUT_FILE, index=False)

    return history, data_origin


if __name__ == "__main__":
    history, data_origin = prepare_pollutant_history()

    print("Data origin:", data_origin)
    print("Rows:", len(history))
    print(
        "Duplicate station-hours:",
        history.duplicated(
            ["location_id", "timestamp_hour"]
        ).sum(),
    )

    print("\nObserved values per pollutant:")

    print(
        history[
            [
                f"{pollutant}_observed"
                for pollutant in POLLUTANTS
            ]
        ].notna().sum()
    )

    print("\nSaved:", OUTPUT_FILE)