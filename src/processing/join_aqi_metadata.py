from pathlib import Path

import pandas as pd


# 1. Find project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 2. Input file paths
aqi_path = PROJECT_ROOT / "data" / "raw" / "openaq_pm25_newdelhi_raw.csv"
metadata_path = PROJECT_ROOT / "data" / "raw" / "openaq_station_metadata.csv"

# 3. Output file path
output_dir = PROJECT_ROOT / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "aqi_with_station_metadata.csv"


# 4. Read both CSV files
aqi_df = pd.read_csv(aqi_path)
metadata_df = pd.read_csv(metadata_path)


# 5. Join AQI readings with station metadata
merged_df = aqi_df.merge(
    metadata_df,
    on=["location_id", "sensor_id"],
    how="left",
    suffixes=("_aqi", "_station")
)


# 6. Save merged file
merged_df.to_csv(output_path, index=False)


# 7. Quick verification
print(f"AQI rows: {len(aqi_df)}")
print(f"Metadata rows: {len(metadata_df)}")
print(f"Merged rows: {len(merged_df)}")
print(f"Saved to: {output_path}")

print("\nColumns:")
print(merged_df.columns.tolist())

print("\nPreview:")
print(merged_df.head())