from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

POLLUTANTS_PATH = PROJECT_ROOT / "data" / "processed" / "delhi_pollutants_90d_wide.csv"
WEATHER_PATH = PROJECT_ROOT / "data" / "raw" / "openmeteo_delhi_weather_90d.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "delhi_environment_90d_merged.csv"
QUALITY_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "environment_merge_quality_report.csv"

POLLUTANT_COLUMNS = ["pm25", "pm10", "no2", "co", "so2", "o3"]

WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]


def main():
    pollutants_df = pd.read_csv(POLLUTANTS_PATH)
    weather_df = pd.read_csv(WEATHER_PATH)

    print(f"Pollutant rows: {len(pollutants_df)}")
    print(f"Weather rows: {len(weather_df)}")

    print(f"Pollutant stations: {pollutants_df['location_id'].nunique()}")
    print(f"Weather stations: {weather_df['location_id'].nunique()}")

    # Normalize ID types
    pollutants_df["location_id"] = pollutants_df["location_id"].astype(str)
    weather_df["location_id"] = weather_df["location_id"].astype(str)

    # Normalize timestamps
    pollutants_df["timestamp_hour"] = pd.to_datetime(
        pollutants_df["timestamp_hour"],
        errors="coerce"
    )

    weather_df["timestamp_hour"] = pd.to_datetime(
        weather_df["timestamp_hour"],
        errors="coerce"
    )

    # Check duplicate weather keys before merge
    weather_duplicate_count = weather_df.duplicated(
        subset=["location_id", "timestamp_hour"],
        keep=False
    ).sum()

    print(f"\nDuplicate weather location-hour rows: {weather_duplicate_count}")

    if weather_duplicate_count > 0:
        weather_df = weather_df.drop_duplicates(
            subset=["location_id", "timestamp_hour"],
            keep="first"
        ).copy()

        print("Dropped duplicate weather rows.")

    # Keep only columns needed from weather table
    weather_keep_columns = ["location_id", "timestamp_hour"] + WEATHER_COLUMNS

    weather_df = weather_df[weather_keep_columns].copy()

    # Main merge
    merged_df = pollutants_df.merge(
        weather_df,
        on=["location_id", "timestamp_hour"],
        how="left",
        indicator=True,
    )

    # Merge quality
    merge_counts = merged_df["_merge"].value_counts(dropna=False)

    print("\nMerge result:")
    print(merge_counts)

    missing_weather_rows = (merged_df["_merge"] == "left_only").sum()

    print(f"\nRows without weather match: {missing_weather_rows}")

    # Remove merge indicator after checking
    merged_df = merged_df.drop(columns=["_merge"])

    # Sort cleanly
    merged_df = merged_df.sort_values(
        by=["location_id", "timestamp_hour"]
    ).reset_index(drop=True)

    # Quality report
    quality_rows = []

    for column in POLLUTANT_COLUMNS + WEATHER_COLUMNS:
        quality_rows.append(
            {
                "column_name": column,
                "missing_count": merged_df[column].isna().sum(),
                "missing_percent": round(merged_df[column].isna().mean() * 100, 2),
            }
        )

    quality_df = pd.DataFrame(quality_rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    merged_df.to_csv(OUTPUT_PATH, index=False)
    quality_df.to_csv(QUALITY_REPORT_PATH, index=False)

    print(f"\nSaved merged environment table to: {OUTPUT_PATH}")
    print(f"Saved quality report to: {QUALITY_REPORT_PATH}")

    print(f"\nMerged rows: {len(merged_df)}")
    print(f"Merged stations: {merged_df['location_id'].nunique()}")

    print("\nTimestamp range:")
    print("min:", merged_df["timestamp_hour"].min())
    print("max:", merged_df["timestamp_hour"].max())

    print("\nMissing percentage report:")
    print(quality_df.to_string(index=False))

    print("\nPreview:")
    print(merged_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()