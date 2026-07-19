from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

input_path = PROJECT_ROOT / "data" / "processed" / "selected_pollutant_sensors.csv"
output_path = PROJECT_ROOT / "data" / "processed" / "selected_stations_review.csv"

df = pd.read_csv(input_path)

# One row per station for manual review
station_df = (
    df.groupby(["location_id", "station_name"], as_index=False)
    .agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        timezone=("timezone", "first"),
        provider_name=("provider_name", "first"),
        datetime_last_local=("datetime_last_local", "first"),
        pollutant_count=("parameter_name", "nunique"),
        pollutants_available=("parameter_name", lambda x: ", ".join(sorted(x.unique()))),
    )
)

# Sort best-looking stations first
station_df = station_df.sort_values(
    by=["pollutant_count", "station_name"],
    ascending=[False, True]
).reset_index(drop=True)

station_df.to_csv(output_path, index=False)

print(f"Saved station review file to: {output_path}")
print(f"Total unique stations: {len(station_df)}")

print("\nPollutant count distribution:")
print(station_df["pollutant_count"].value_counts().sort_index(ascending=False))

print("\nTop 30 stations by pollutant coverage:")
print(
    station_df[
        [
            "location_id",
            "station_name",
            "latitude",
            "longitude",
            "provider_name",
            "pollutant_count",
            "pollutants_available",
            "datetime_last_local",
        ]
    ].head(30).to_string(index=False)
)