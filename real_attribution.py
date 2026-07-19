"""
real_attribution.py
Uses the teammate's real station_source_links_v1.csv (real mapped
construction/traffic/industrial/waste/power sites near each station)
instead of guessed percentages.
"""

from pathlib import Path
from functools import lru_cache
import pandas as pd

import database

HERE = Path(__file__).parent
if (HERE / "ml_assets" / "data" / "station_source_links_v1.csv").exists():
    LINKS_CSV = HERE / "ml_assets" / "data" / "station_source_links_v1.csv"
else:
    LINKS_CSV = HERE / "data" / "context" / "station_source_links_v1.csv"


@lru_cache(maxsize=1)
def _load_links():
    df = pd.read_csv(LINKS_CSV)
    return df


def get_attribution(location, lat, lon, radius_km=3.0):
    links = _load_links()
    station_links = links[(links["station_name"] == location) & (links["centroid_distance_km"] <= radius_km)]

    if station_links.empty:
        breakdown = [{"source": "unknown", "share": 1.0}]
        top_source = "unknown"
        base_evidence = [f"No mapped sources found within {radius_km}km of this station."]
    else:
        weighted = station_links.groupby("source_category").apply(
            lambda g: (1.0 / g["centroid_distance_km"].clip(lower=0.1)).sum()
        )
        shares = (weighted / weighted.sum()).round(3)
        ranked = shares.sort_values(ascending=False)
        breakdown = [{"source": s, "share": float(p)} for s, p in ranked.items()]
        top_source = ranked.index[0]
        base_evidence = [
            f"{len(station_links)} mapped source(s) within {radius_km}km "
            f"(OpenStreetMap-derived construction/traffic/industrial/waste/power sites)."
        ]

    features = database.get_citizen_features(lat, lon)
    cat_counts = features["source_category_count"]
    if cat_counts:
        total_reports = sum(cat_counts.values())
        blended = {}
        base_map = {b["source"]: b["share"] for b in breakdown}
        all_cats = set(base_map) | set(cat_counts)
        for cat in all_cats:
            base_share = base_map.get(cat, 0.05)
            report_share = cat_counts.get(cat, 0) / total_reports
            blended[cat] = 0.7 * base_share + 0.3 * report_share
        total = sum(blended.values())
        breakdown = sorted(
            [{"source": k, "share": round(v / total, 3)} for k, v in blended.items()],
            key=lambda d: d["share"], reverse=True,
        )
        top_source = breakdown[0]["source"]
        base_evidence.append(
            f"{features['report_density_1km']} verified citizen report(s) within 1km blended in "
            f"(citizen_source_score={features['citizen_source_score']})"
        )
    else:
        base_evidence.append("No verified citizen reports nearby yet to blend in.")

    return {
        "location": location,
        "breakdown": breakdown,
        "top_source": top_source,
        "evidence": base_evidence,
        "citizen_features": features,
    }