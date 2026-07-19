from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

selected_sensors_path = PROJECT_ROOT / "data" / "processed" / "selected_pollutant_sensors.csv"
final_station_review_path = PROJECT_ROOT / "data" / "processed" / "final_delhi_station_review.csv"

output_path = PROJECT_ROOT / "data" / "processed" / "final_delhi_sensor_control_table.csv"
mismatch_output_path = PROJECT_ROOT / "data" / "processed" / "sensor_control_mismatches.csv"

SELECTED_POLLUTANTS = ["pm25", "pm10", "no2", "co", "so2", "o3"]


def provider_rank(provider_name):
    provider = str(provider_name).lower()

    if provider == "cpcb":
        return 0
    if provider == "nan":
        return 1
    if provider == "caaqm":
        return 2

    return 3


sensors_df = pd.read_csv(selected_sensors_path)
review_df = pd.read_csv(final_station_review_path)

# Normalize IDs for safe matching
sensors_df["location_id"] = sensors_df["location_id"].astype(str)
review_df["location_id"] = review_df["location_id"].astype(str)

accepted_location_ids = set(review_df["location_id"])

control_df = sensors_df[
    sensors_df["location_id"].isin(accepted_location_ids)
    & sensors_df["parameter_name"].isin(SELECTED_POLLUTANTS)
].copy()

control_df["datetime_last_local"] = pd.to_datetime(
    control_df["datetime_last_local"],
    errors="coerce"
)

control_df["provider_rank"] = control_df["provider_name"].apply(provider_rank)

print(f"Accepted stations from review file: {len(accepted_location_ids)}")
print(f"Sensor rows before deduplication: {len(control_df)}")

duplicates_before = control_df.duplicated(
    subset=["location_id", "parameter_name"],
    keep=False
).sum()

print(f"Duplicate station-pollutant rows before deduplication: {duplicates_before}")

# Keep one sensor per station-pollutant.
# Preference:
# 1. latest active sensor
# 2. preferred provider
# 3. highest sensor_id as deterministic tie-breaker
control_df = control_df.sort_values(
    by=[
        "location_id",
        "parameter_name",
        "datetime_last_local",
        "provider_rank",
        "sensor_id",
    ],
    ascending=[True, True, False, True, False],
)

control_df = control_df.drop_duplicates(
    subset=["location_id", "parameter_name"],
    keep="first"
).copy()

control_df = control_df[
    [
        "location_id",
        "station_name",
        "latitude",
        "longitude",
        "timezone",
        "provider_name",
        "sensor_id",
        "parameter_name",
        "parameter_unit",
        "datetime_first_local",
        "datetime_last_local",
    ]
].copy()

control_df = control_df.sort_values(
    by=["station_name", "parameter_name"]
).reset_index(drop=True)

# Validate coverage against review file
actual_coverage = (
    control_df.groupby("location_id")["parameter_name"]
    .nunique()
    .reset_index(name="actual_sensor_count")
)

expected_coverage = review_df[
    ["location_id", "station_name", "pollutant_count", "pollutants_available"]
].copy()

validation_df = expected_coverage.merge(
    actual_coverage,
    on="location_id",
    how="left"
)

validation_df["actual_sensor_count"] = validation_df["actual_sensor_count"].fillna(0).astype(int)

mismatches = validation_df[
    validation_df["pollutant_count"] != validation_df["actual_sensor_count"]
].copy()

output_path.parent.mkdir(parents=True, exist_ok=True)

control_df.to_csv(output_path, index=False)
mismatches.to_csv(mismatch_output_path, index=False)

print(f"\nSaved final sensor control table to: {output_path}")
print(f"Saved mismatch check to: {mismatch_output_path}")

print(f"\nFinal stations in control table: {control_df['location_id'].nunique()}")
print(f"Final sensor-control rows: {len(control_df)}")

print("\nPollutant availability in final control table:")
print(
    control_df
    .groupby("parameter_name")["location_id"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nStation pollutant count distribution:")
print(
    control_df
    .groupby("location_id")["parameter_name"]
    .nunique()
    .value_counts()
    .sort_index(ascending=False)
)

print(f"\nCoverage mismatches: {len(mismatches)}")
if len(mismatches) > 0:
    print(mismatches.to_string(index=False))