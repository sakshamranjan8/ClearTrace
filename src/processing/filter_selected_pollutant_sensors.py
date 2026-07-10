from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

input_path = PROJECT_ROOT / "data" / "raw" / "openaq_delhi_pollutant_availability.csv"
output_path = PROJECT_ROOT / "data" / "processed" / "selected_pollutant_sensors.csv"

selected_pollutants = ["pm25", "pm10", "no2", "co", "so2", "o3"]

df = pd.read_csv(input_path)

selected_df = df[df["parameter_name"].isin(selected_pollutants)].copy()

# Keep only useful columns for ingestion
selected_df = selected_df[
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
]

# Avoid accidental duplicate sensor rows
selected_df = selected_df.drop_duplicates(subset=["sensor_id"])

# Sort for readability
selected_df = selected_df.sort_values(
    by=["location_id", "parameter_name"]
).reset_index(drop=True)

output_path.parent.mkdir(parents=True, exist_ok=True)
selected_df.to_csv(output_path, index=False)

print(f"Saved to: {output_path}")
print(f"Total selected sensor rows: {len(selected_df)}")
print(f"Unique stations: {selected_df['location_id'].nunique()}")

print("\nPollutant availability in selected data:")
print(
    selected_df
    .groupby("parameter_name")["location_id"]
    .nunique()
    .sort_values(ascending=False)
)

station_matrix = selected_df.pivot_table(
    index=["location_id", "station_name"],
    columns="parameter_name",
    values="sensor_id",
    aggfunc="first"
)

stations_with_all_6 = station_matrix.dropna().shape[0]

print(f"\nStations with all 6 pollutants: {stations_with_all_6}")
print("\nMissing sensor count per pollutant:")
print(station_matrix.isna().sum())