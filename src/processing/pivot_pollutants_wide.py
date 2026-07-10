from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

input_path = PROJECT_ROOT / "data" / "raw" / "openaq_delhi_pollutants_90d_long.csv"
output_path = PROJECT_ROOT / "data" / "processed" / "delhi_pollutants_90d_wide.csv"
quality_report_path = PROJECT_ROOT / "data" / "processed" / "pollutant_pivot_quality_report.csv"

SELECTED_POLLUTANTS = ["pm25", "pm10", "no2", "co", "so2", "o3"]

MIN_PERCENT_COVERAGE = 75


def normalize_flag_column(series):
    """
    Makes the has_flags column safe even if pandas reads it as bool or string.
    """
    return series.astype(str).str.lower().map(
        {
            "true": True,
            "false": False,
            "nan": False,
        }
    ).fillna(False)


def main():
    df = pd.read_csv(input_path)

    print(f"Raw long rows: {len(df)}")
    print(f"Unique stations: {df['location_id'].nunique()}")
    print(f"Unique sensors: {df['sensor_id'].nunique()}")

    # Keep only our selected AQI pollutants
    df = df[df["parameter_name"].isin(SELECTED_POLLUTANTS)].copy()

    # Convert timestamp to local station-hour.
    # We parse as UTC first because the timestamp contains +05:30 offset,
    # then convert to Asia/Kolkata and remove timezone for easy merging later.
    df["timestamp_local"] = pd.to_datetime(
        df["timestamp_local"],
        utc=True,
        errors="coerce"
    ).dt.tz_convert("Asia/Kolkata")

    df["timestamp_hour"] = df["timestamp_local"].dt.floor("h").dt.tz_localize(None)

    # Make numeric columns safe
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["percent_coverage"] = pd.to_numeric(df["percent_coverage"], errors="coerce")

    # Normalize flag column
    df["has_flags"] = normalize_flag_column(df["has_flags"])

    # Quality rules for values used in modeling/AQI
    df["is_null_value"] = df["value"].isna()
    df["is_flagged"] = df["has_flags"]
    df["is_low_coverage"] = df["percent_coverage"] < MIN_PERCENT_COVERAGE

    df["is_valid_measurement"] = (
        ~df["is_null_value"]
        & ~df["is_flagged"]
        & ~df["is_low_coverage"]
        & df["timestamp_hour"].notna()
    )

    # Keep raw value, but create cleaned value for pivoting
    df["value_clean"] = df["value"].where(df["is_valid_measurement"])

    # Quality report before pivot
    quality_summary = (
        df.groupby("parameter_name")
        .agg(
            raw_rows=("value", "size"),
            valid_rows=("is_valid_measurement", "sum"),
            null_values=("is_null_value", "sum"),
            flagged_rows=("is_flagged", "sum"),
            low_coverage_rows=("is_low_coverage", "sum"),
            unique_stations=("location_id", "nunique"),
        )
        .reset_index()
    )

    quality_summary["invalid_rows"] = (
        quality_summary["raw_rows"] - quality_summary["valid_rows"]
    )

    quality_summary["valid_percent"] = (
        quality_summary["valid_rows"] / quality_summary["raw_rows"] * 100
    ).round(2)

    # Pivot pollutants into columns
    wide_df = df.pivot_table(
        index=[
            "location_id",
            "station_name",
            "latitude",
            "longitude",
            "provider_name",
            "timestamp_hour",
        ],
        columns="parameter_name",
        values="value_clean",
        aggfunc="mean",
    ).reset_index()

    # Remove pivot column index name
    wide_df.columns.name = None

    # Ensure all selected pollutant columns exist even if one is missing
    for pollutant in SELECTED_POLLUTANTS:
        if pollutant not in wide_df.columns:
            wide_df[pollutant] = pd.NA

    # Add missingness flags for later model/reasoning use
    for pollutant in SELECTED_POLLUTANTS:
        wide_df[f"has_{pollutant}"] = wide_df[pollutant].notna()

    # Sort for readability
    wide_df = wide_df.sort_values(
        by=["location_id", "timestamp_hour"]
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    wide_df.to_csv(output_path, index=False)
    quality_summary.to_csv(quality_report_path, index=False)

    print(f"\nSaved wide pollutant table to: {output_path}")
    print(f"Saved quality report to: {quality_report_path}")

    print(f"\nWide rows: {len(wide_df)}")
    print(f"Wide stations: {wide_df['location_id'].nunique()}")
    print(f"Timestamp range:")
    print("min:", wide_df["timestamp_hour"].min())
    print("max:", wide_df["timestamp_hour"].max())

    print("\nQuality summary:")
    print(quality_summary.to_string(index=False))

    print("\nMissing percentage in wide pollutant columns:")
    missing_percent = (
        wide_df[SELECTED_POLLUTANTS]
        .isna()
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )
    print(missing_percent)

    print("\nPreview:")
    print(wide_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()