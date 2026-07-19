from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_LONG_PATH = PROJECT_ROOT / "data" / "raw" / "openaq_delhi_pollutants_90d_long.csv"
MERGED_ENV_PATH = PROJECT_ROOT / "data" / "processed" / "delhi_environment_90d_merged.csv"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "pollutant_unit_audit.csv"

POLLUTANTS = ["pm25", "pm10", "no2", "co", "so2", "o3"]


def main():
    raw_df = pd.read_csv(RAW_LONG_PATH)
    merged_df = pd.read_csv(MERGED_ENV_PATH)

    raw_df["value"] = pd.to_numeric(raw_df["value"], errors="coerce")

    audit_rows = []

    for pollutant in POLLUTANTS:
        raw_pollutant_df = raw_df[raw_df["parameter_name"] == pollutant].copy()

        units = (
            raw_pollutant_df["parameter_unit"]
            .dropna()
            .astype(str)
            .value_counts()
            .to_dict()
        )

        merged_values = pd.to_numeric(
            merged_df[pollutant],
            errors="coerce"
        )

        audit_rows.append(
            {
                "pollutant": pollutant,
                "raw_unit_counts": units,
                "raw_rows": len(raw_pollutant_df),
                "merged_non_null_rows": merged_values.notna().sum(),
                "min": merged_values.min(),
                "p25": merged_values.quantile(0.25),
                "median": merged_values.median(),
                "mean": merged_values.mean(),
                "p75": merged_values.quantile(0.75),
                "p95": merged_values.quantile(0.95),
                "p99": merged_values.quantile(0.99),
                "max": merged_values.max(),
            }
        )

    audit_df = pd.DataFrame(audit_rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved pollutant unit audit to: {OUTPUT_PATH}")

    print("\nPollutant unit/value audit:")
    print(audit_df.to_string(index=False))

    print("\nCO-specific warning check:")
    co_row = audit_df[audit_df["pollutant"] == "co"].iloc[0]

    print(f"CO units in raw file: {co_row['raw_unit_counts']}")
    print(f"CO median value: {co_row['median']}")
    print(f"CO max value: {co_row['max']}")

    if "ppb" in str(co_row["raw_unit_counts"]).lower() and co_row["median"] < 10:
        print(
            "\nWARNING: CO is labelled as ppb, but the values look too small for ppb "
            "and more like mg/m³ or ppm-scale values. Do not calculate AQI until this is resolved."
        )
    else:
        print("\nCO values do not trigger the simple ppb mismatch warning.")


if __name__ == "__main__":
    main()