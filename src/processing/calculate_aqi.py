from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_PATH = PROJECT_ROOT / "data" / "processed" / "delhi_environment_90d_merged.csv"
WEATHER_PATH = PROJECT_ROOT / "data" / "raw" / "openmeteo_delhi_weather_90d.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "delhi_aqi_90d.csv"
QUALITY_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "aqi_calculation_quality_report.csv"

NO2_PPB_TO_UGM3 = 46.0055 / 24.45
SO2_PPB_TO_UGM3 = 64.066 / 24.45

POLLUTANT_COLUMNS = ["pm25", "pm10", "no2", "co", "so2", "o3"]

WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]

BREAKPOINTS = {
    "pm25": [
        (0, 30, 0, 50),
        (30, 60, 51, 100),
        (60, 90, 101, 200),
        (90, 120, 201, 300),
        (120, 250, 301, 400),
        (250, 500, 401, 500),
    ],
    "pm10": [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 250, 101, 200),
        (250, 350, 201, 300),
        (350, 430, 301, 400),
        (430, 600, 401, 500),
    ],
    "no2": [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 180, 101, 200),
        (180, 280, 201, 300),
        (280, 400, 301, 400),
        (400, 1000, 401, 500),
    ],
    "so2": [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 380, 101, 200),
        (380, 800, 201, 300),
        (800, 1600, 301, 400),
        (1600, 2000, 401, 500),
    ],
    "co": [
        (0, 1, 0, 50),
        (1, 2, 51, 100),
        (2, 10, 101, 200),
        (10, 17, 201, 300),
        (17, 34, 301, 400),
        (34, 50, 401, 500),
    ],
    "o3": [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 168, 101, 200),
        (168, 208, 201, 300),
        (208, 748, 301, 400),
        (748, 1000, 401, 500),
    ],
}


def calculate_subindex(concentration, pollutant):
    if pd.isna(concentration) or concentration < 0:
        return np.nan

    for c_low, c_high, i_low, i_high in BREAKPOINTS[pollutant]:
        if c_low <= concentration <= c_high:
            value = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            return round(value, 2)

    if concentration > BREAKPOINTS[pollutant][-1][1]:
        return 500.0

    return np.nan


def clean_negative_values(df, columns):
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df.loc[df[column] < 0, column] = np.nan

    return df


def apply_sanity_caps(df):
    caps = {
        "pm25": 1000,
        "pm10": 2000,
        "no2_ugm3": 1000,
        "so2_ugm3": 2000,
        "co_mgm3": 50,
        "o3": 1000,
    }

    for column, cap in caps.items():
        df.loc[df[column] > cap, column] = np.nan

    return df


def build_hourly_base(env_df, weather_df):
    env_df["location_id"] = env_df["location_id"].astype(str)
    weather_df["location_id"] = weather_df["location_id"].astype(str)

    env_df["timestamp_hour"] = pd.to_datetime(env_df["timestamp_hour"], errors="coerce")
    weather_df["timestamp_hour"] = pd.to_datetime(weather_df["timestamp_hour"], errors="coerce")

    min_timestamp = env_df["timestamp_hour"].min()
    max_timestamp = env_df["timestamp_hour"].max()

    weather_df = weather_df[
        (weather_df["timestamp_hour"] >= min_timestamp)
        & (weather_df["timestamp_hour"] <= max_timestamp)
    ].copy()

    station_meta = (
        env_df.groupby("location_id", as_index=False)
        .agg(
            station_name=("station_name", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            provider_name=("provider_name", "first"),
        )
    )

    pollutant_df = env_df[
        ["location_id", "timestamp_hour"] + POLLUTANT_COLUMNS
    ].copy()

    pollutant_df = pollutant_df.drop_duplicates(
        subset=["location_id", "timestamp_hour"],
        keep="first"
    )

    weather_df = weather_df[
        ["location_id", "timestamp_hour"] + WEATHER_COLUMNS
    ].copy()

    base_df = weather_df.merge(
        station_meta,
        on="location_id",
        how="left"
    )

    base_df = base_df.merge(
        pollutant_df,
        on=["location_id", "timestamp_hour"],
        how="left"
    )

    ordered_columns = [
        "location_id",
        "station_name",
        "latitude",
        "longitude",
        "provider_name",
        "timestamp_hour",
    ] + POLLUTANT_COLUMNS + WEATHER_COLUMNS

    return base_df[ordered_columns].copy()


def add_unit_conversions(df):
    df["no2_ugm3"] = df["no2"] * NO2_PPB_TO_UGM3
    df["so2_ugm3"] = df["so2"] * SO2_PPB_TO_UGM3
    df["co_mgm3"] = df["co"]

    return df


def add_availability_flags(df):
    for pollutant in POLLUTANT_COLUMNS:
        df[f"has_{pollutant}"] = df[pollutant].notna()

    return df


def add_rolling_averages(df):
    df = df.sort_values(["location_id", "timestamp_hour"]).copy()

    rolling_specs = {
        "pm25_24h_avg": ("pm25", 24, 18),
        "pm10_24h_avg": ("pm10", 24, 18),
        "no2_24h_avg": ("no2_ugm3", 24, 18),
        "so2_24h_avg": ("so2_ugm3", 24, 18),
        "co_8h_avg": ("co_mgm3", 8, 6),
        "o3_8h_avg": ("o3", 8, 6),
    }

    for output_column, spec in rolling_specs.items():
        input_column, window, min_periods = spec
        df[output_column] = (
            df.groupby("location_id")[input_column]
            .transform(lambda s: s.rolling(window=window, min_periods=min_periods).mean())
        )

    return df


def add_subindices(df):
    subindex_inputs = {
        "pm25_subindex": ("pm25_24h_avg", "pm25"),
        "pm10_subindex": ("pm10_24h_avg", "pm10"),
        "no2_subindex": ("no2_24h_avg", "no2"),
        "so2_subindex": ("so2_24h_avg", "so2"),
        "co_subindex": ("co_8h_avg", "co"),
        "o3_subindex": ("o3_8h_avg", "o3"),
    }

    for output_column, spec in subindex_inputs.items():
        input_column, pollutant = spec
        df[output_column] = df[input_column].apply(
            lambda value: calculate_subindex(value, pollutant)
        )

    return df


def add_final_aqi(df):
    subindex_columns = [
        "pm25_subindex",
        "pm10_subindex",
        "no2_subindex",
        "so2_subindex",
        "co_subindex",
        "o3_subindex",
    ]

    df["available_subindex_count"] = df[subindex_columns].notna().sum(axis=1)

    df["has_particulate_subindex"] = (
        df["pm25_subindex"].notna() | df["pm10_subindex"].notna()
    )

    df["aqi_calculation_valid"] = (
        (df["available_subindex_count"] >= 3)
        & (df["has_particulate_subindex"])
    )

    df["current_aqi"] = df[subindex_columns].max(axis=1)

    dominant_map = {
        "pm25_subindex": "pm25",
        "pm10_subindex": "pm10",
        "no2_subindex": "no2",
        "so2_subindex": "so2",
        "co_subindex": "co",
        "o3_subindex": "o3",
    }

    df["dominant_pollutant"] = df[subindex_columns].apply(
        lambda row: row.idxmax() if row.notna().any() else np.nan,
        axis=1
    )

    df["dominant_pollutant"] = df["dominant_pollutant"].map(dominant_map)

    df.loc[~df["aqi_calculation_valid"], "current_aqi"] = np.nan
    df.loc[~df["aqi_calculation_valid"], "dominant_pollutant"] = np.nan

    df["current_aqi"] = df["current_aqi"].round(0)

    return df


def create_quality_report(df):
    columns = [
        "pm25",
        "pm10",
        "no2",
        "no2_ugm3",
        "co",
        "co_mgm3",
        "so2",
        "so2_ugm3",
        "o3",
        "pm25_24h_avg",
        "pm10_24h_avg",
        "no2_24h_avg",
        "so2_24h_avg",
        "co_8h_avg",
        "o3_8h_avg",
        "current_aqi",
    ]

    rows = []

    for column in columns:
        rows.append(
            {
                "metric": f"{column}_missing_percent",
                "value": round(df[column].isna().mean() * 100, 2),
            }
        )

    rows.extend(
        [
            {"metric": "total_rows", "value": len(df)},
            {"metric": "valid_aqi_rows", "value": int(df["aqi_calculation_valid"].sum())},
            {"metric": "valid_aqi_percent", "value": round(df["aqi_calculation_valid"].mean() * 100, 2)},
            {"metric": "unique_stations", "value": int(df["location_id"].nunique())},
        ]
    )

    return pd.DataFrame(rows)


def main():
    env_df = pd.read_csv(ENV_PATH)
    weather_df = pd.read_csv(WEATHER_PATH)

    df = build_hourly_base(env_df, weather_df)

    print(f"Hourly base rows: {len(df)}")
    print(f"Stations: {df['location_id'].nunique()}")

    df = clean_negative_values(df, POLLUTANT_COLUMNS)
    df = add_unit_conversions(df)
    df = apply_sanity_caps(df)
    df = add_availability_flags(df)
    df = add_rolling_averages(df)
    df = add_subindices(df)
    df = add_final_aqi(df)

    df = df.sort_values(["location_id", "timestamp_hour"]).reset_index(drop=True)

    quality_report = create_quality_report(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)
    quality_report.to_csv(QUALITY_REPORT_PATH, index=False)

    print(f"\nSaved AQI table to: {OUTPUT_PATH}")
    print(f"Saved AQI quality report to: {QUALITY_REPORT_PATH}")

    print(f"\nOutput rows: {len(df)}")
    print(f"Output stations: {df['location_id'].nunique()}")

    print("\nAQI validity:")
    print(df["aqi_calculation_valid"].value_counts(dropna=False))

    print("\nDominant pollutant counts:")
    print(df["dominant_pollutant"].value_counts(dropna=False))

    print("\nCurrent AQI summary:")
    print(df["current_aqi"].describe())

    print("\nQuality report:")
    print(quality_report.to_string(index=False))

    preview_columns = [
        "location_id",
        "station_name",
        "timestamp_hour",
        "pm25_24h_avg",
        "pm10_24h_avg",
        "no2_24h_avg",
        "so2_24h_avg",
        "co_8h_avg",
        "o3_8h_avg",
        "pm25_subindex",
        "pm10_subindex",
        "no2_subindex",
        "so2_subindex",
        "co_subindex",
        "o3_subindex",
        "current_aqi",
        "dominant_pollutant",
        "aqi_calculation_valid",
    ]

    print("\nPreview:")
    print(df[preview_columns].head(20).to_string(index=False))


if __name__ == "__main__":
    main()