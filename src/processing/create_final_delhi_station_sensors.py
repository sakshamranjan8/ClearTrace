from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

input_path = PROJECT_ROOT / "data" / "processed" / "selected_pollutant_sensors.csv"

output_path = PROJECT_ROOT / "data" / "processed" / "final_delhi_station_sensors.csv"
review_output_path = PROJECT_ROOT / "data" / "processed" / "final_delhi_station_review.csv"

SELECTED_POLLUTANTS = ["pm25", "pm10", "no2", "co", "so2", "o3"]

EXCLUDE_KEYWORDS = [
    "ghaziabad",
    "noida",
    "greater noida",
    "gurugram",
    "gurgaon",
    "faridabad",
    "bahadurgarh",
    "sonipat",
    "baghpat",
]


def clean_station_base_name(station_name: str) -> str:
    name = str(station_name).lower()

    # Remove provider suffix after " - "
    name = name.split(" - ")[0]

    # Normalize Delhi naming
    name = name.replace("new delhi", "")
    name = name.replace("delhi", "")

    # Remove punctuation and extra spaces
    name = re.sub(r"[^a-z0-9]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


def provider_rank(provider_name):
    provider = str(provider_name).lower()

    if provider == "cpcb":
        return 0
    if provider == "nan":
        return 1
    if provider == "caaqm":
        return 2

    return 3


df = pd.read_csv(input_path)

# Keep only selected AQI pollutants
df = df[df["parameter_name"].isin(SELECTED_POLLUTANTS)].copy()

# Parse timestamps
df["datetime_last_local"] = pd.to_datetime(
    df["datetime_last_local"],
    errors="coerce"
)

# Keep only Delhi/New Delhi stations by name
df["station_name_lower"] = df["station_name"].str.lower()

df = df[
    df["station_name_lower"].str.contains("delhi", na=False)
].copy()

# Remove obvious NCR stations captured by broad bounding box
for keyword in EXCLUDE_KEYWORDS:
    df = df[
        ~df["station_name_lower"].str.contains(keyword, na=False)
    ].copy()

# Add station base name for deduplication
df["station_base_name"] = df["station_name"].apply(clean_station_base_name)

# Add provider ranking
df["provider_rank"] = df["provider_name"].apply(provider_rank)

# Dynamic recency filter
# Use the freshest timestamp in the current OpenAQ pull, then keep stations active within 30 days of it.
latest_timestamp = df["datetime_last_local"].max()
recent_cutoff = latest_timestamp - pd.Timedelta(days=30)

df = df[df["datetime_last_local"] >= recent_cutoff].copy()

# Build station-level coverage table
station_coverage = (
    df.groupby(
        [
            "location_id",
            "station_base_name",
            "station_name",
            "latitude",
            "longitude",
            "timezone",
            "provider_name",
            "provider_rank",
        ],
        as_index=False,
    )
    .agg(
        datetime_last_local=("datetime_last_local", "max"),
        pollutant_count=("parameter_name", "nunique"),
        pollutants_available=("parameter_name", lambda x: sorted(x.unique())),
    )
)

station_coverage["has_pm25"] = station_coverage["pollutants_available"].apply(
    lambda pollutants: "pm25" in pollutants
)

station_coverage["has_pm10"] = station_coverage["pollutants_available"].apply(
    lambda pollutants: "pm10" in pollutants
)

station_coverage["has_particulate"] = (
    station_coverage["has_pm25"] | station_coverage["has_pm10"]
)

# New professional rule:
# Keep stations with decent pollutant coverage, but do not require all 6.
station_coverage = station_coverage[
    (station_coverage["pollutant_count"] >= 3)
    & (station_coverage["has_particulate"])
].copy()

# Deduplicate physical stations.
# If the same physical station appears multiple times, keep:
# 1. more pollutant coverage
# 2. latest active record
# 3. preferred provider
station_coverage = station_coverage.sort_values(
    by=[
        "station_base_name",
        "pollutant_count",
        "datetime_last_local",
        "provider_rank",
    ],
    ascending=[True, False, False, True],
)

best_locations = (
    station_coverage
    .drop_duplicates(subset=["station_base_name"], keep="first")
    ["location_id"]
)

final_df = df[df["location_id"].isin(best_locations)].copy()

# Keep useful columns only
final_df = final_df[
    [
        "location_id",
        "station_base_name",
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

final_df = final_df.sort_values(
    by=["station_base_name", "parameter_name"]
).reset_index(drop=True)

# Final review table
review_df = (
    final_df.groupby(
        [
            "location_id",
            "station_base_name",
            "station_name",
            "latitude",
            "longitude",
            "provider_name",
        ],
        as_index=False,
    )
    .agg(
        datetime_last_local=("datetime_last_local", "max"),
        pollutant_count=("parameter_name", "nunique"),
        pollutants_available=("parameter_name", lambda x: ", ".join(sorted(x.unique()))),
    )
)

review_df = review_df.sort_values(
    by=["station_base_name"]
).reset_index(drop=True)

output_path.parent.mkdir(parents=True, exist_ok=True)

final_df.to_csv(output_path, index=False)
review_df.to_csv(review_output_path, index=False)

print(f"Latest timestamp in selected data: {latest_timestamp}")
print(f"Recent cutoff used: {recent_cutoff}")

print(f"\nSaved final sensor table to: {output_path}")
print(f"Saved station review table to: {review_output_path}")

print(f"\nFinal unique stations: {final_df['location_id'].nunique()}")
print(f"Final sensor rows: {len(final_df)}")

print("\nPollutant availability in final stations:")
print(
    final_df
    .groupby("parameter_name")["location_id"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nPollutant count distribution:")
print(
    review_df["pollutant_count"]
    .value_counts()
    .sort_index(ascending=False)
)

print("\nFinal station review:")
print(
    review_df[
        [
            "location_id",
            "station_base_name",
            "station_name",
            "provider_name",
            "pollutant_count",
            "pollutants_available",
            "datetime_last_local",
        ]
    ].to_string(index=False)
)