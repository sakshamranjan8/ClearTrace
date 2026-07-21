"""Location-centred pollution-source indicators for ClearTrace.

This is contextual evidence, not emissions source apportionment.  Sources are
measured from the exact user coordinates using the geometry in
``source_inventory_v1.csv``.  The module intentionally returns qualitative
signal strength, evidence counts and distances instead of contribution shares.
"""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import re

import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE_INVENTORY_CANDIDATES = [
    HERE / "data" / "context" / "source_inventory_v1.csv",
    HERE / "data" / "source_inventory_v1.csv",
    HERE / "source_inventory_v1.csv",
]

CATEGORY_META = {
    "traffic": {"label": "Road traffic", "icon": "🚗"},
    "industrial": {"label": "Industrial activity", "icon": "🏭"},
    "waste": {"label": "Waste sites", "icon": "🗑️"},
    "power": {"label": "Power generation", "icon": "⚡"},
    "construction": {"label": "Construction context", "icon": "🏗️"},
    "wastewater": {"label": "Wastewater facilities", "icon": "💧"},
}

CONFIDENCE_NAMES = {
    "A": "High-confidence official record",
    "B": "Official or strongly verified record",
    "C": "Mapped contextual record",
    "D": "Unverified mapped context",
}


def _inventory_path() -> Path:
    for path in SOURCE_INVENTORY_CANDIDATES:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in SOURCE_INVENTORY_CANDIDATES)
    raise FileNotFoundError(
        "source_inventory_v1.csv was not found. Expected one of:\n" + searched
    )


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


@lru_cache(maxsize=1)
def load_source_inventory() -> pd.DataFrame:
    df = pd.read_csv(_inventory_path(), low_memory=False)
    required = {
        "source_id",
        "source_category",
        "source_name",
        "geometry_type",
        "latitude",
        "longitude",
        "geometry_wkt",
        "default_include",
        "confidence_tier",
        "officially_listed",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Source inventory is missing columns: " + ", ".join(sorted(missing))
        )
    df = df.copy()
    df["default_include"] = df["default_include"].map(_as_bool)
    df["officially_listed"] = df["officially_listed"].map(_as_bool)
    return df


@lru_cache(maxsize=8192)
def _wkt_coordinates(wkt: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(wkt, str) or not wkt:
        return ()
    pairs = re.findall(
        r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)",
        wkt,
    )
    # WKT order is longitude, latitude.
    return tuple((float(lon), float(lat)) for lon, lat in pairs)


def _xy_km(lon: float, lat: float, origin_lon: float, origin_lat: float) -> tuple[float, float]:
    x = (lon - origin_lon) * 111.32 * math.cos(math.radians(origin_lat))
    y = (lat - origin_lat) * 110.57
    return x, y


def _point_to_segment_km(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _distance_to_geometry_km(row, latitude: float, longitude: float) -> float:
    coordinates = _wkt_coordinates(row.geometry_wkt)
    if not coordinates:
        if pd.isna(row.latitude) or pd.isna(row.longitude):
            return float("inf")
        x, y = _xy_km(float(row.longitude), float(row.latitude), longitude, latitude)
        return math.hypot(x, y)

    xy = [_xy_km(lon, lat, longitude, latitude) for lon, lat in coordinates]
    if len(xy) == 1:
        return math.hypot(*xy[0])
    return min(
        _point_to_segment_km(0.0, 0.0, *xy[index], *xy[index + 1])
        for index in range(len(xy) - 1)
    )


def _strength_for_category(category: str, group: pd.DataFrame, radius_km: float) -> tuple[str, int]:
    nearest = float(group["distance_km"].min())
    count = len(group)
    official = int(group["officially_listed"].sum())

    if category == "traffic":
        road_km = float(group["road_length_km"].fillna(0).sum())
        if nearest <= 0.25 or road_km >= 8:
            return "High", 3
        if nearest <= 0.75 or road_km >= 3:
            return "Medium", 2
        return "Low", 1

    if category == "construction":
        return "Unverified", 0

    if official and nearest <= min(3.0, radius_km):
        return "High", 3
    if count >= 3 or nearest <= min(2.0, radius_km):
        return "Medium", 2
    return "Low", 1


def _evidence_text(category: str, group: pd.DataFrame, radius_km: float) -> list[str]:
    nearest_row = group.sort_values("distance_km").iloc[0]
    nearest = float(nearest_row["distance_km"])
    evidence = [
        f"Nearest mapped feature: {nearest_row['source_name']} ({nearest:.2f} km)",
        f"{len(group)} eligible mapped feature(s) within {radius_km:g} km",
    ]
    if category == "traffic":
        road_km = float(group["road_length_km"].fillna(0).sum())
        evidence[1] = f"{road_km:.1f} km of mapped major-road segments within {radius_km:g} km"
    official = int(group["officially_listed"].sum())
    if official:
        evidence.append(f"{official} official or officially listed record(s)")
    return evidence


@lru_cache(maxsize=256)
def _cached_indicators(latitude: float, longitude: float, radius_km: float) -> dict:
    df = load_source_inventory()
    rows = []
    for row in df.itertuples(index=False):
        distance = _distance_to_geometry_km(row, latitude, longitude)
        if distance <= radius_km:
            item = row._asdict()
            item["distance_km"] = distance
            rows.append(item)

    nearby = pd.DataFrame(rows)
    if nearby.empty:
        return {
            "indicators": [],
            "context_only": [],
            "radius_km": radius_km,
            "total_features": 0,
            "eligible_features": 0,
            "confidence": "Low",
        }

    # Only records explicitly approved by the inventory are scored.  Tier-D
    # construction polygons remain visible as context but never become a claim.
    eligible = nearby[nearby["default_include"]].copy()
    context_only = nearby[~nearby["default_include"]].copy()

    indicators = []
    for category, group in eligible.groupby("source_category"):
        meta = CATEGORY_META.get(
            category,
            {"label": category.replace("_", " ").title(), "icon": "📍"},
        )
        strength, sort_score = _strength_for_category(category, group, radius_km)
        confidence_tier = min(group["confidence_tier"].dropna().astype(str), default="D")
        indicators.append(
            {
                "category": category,
                "label": meta["label"],
                "icon": meta["icon"],
                "strength": strength,
                "sort_score": sort_score,
                "feature_count": int(len(group)),
                "nearest_distance_km": round(float(group["distance_km"].min()), 2),
                "confidence_tier": confidence_tier,
                "confidence_label": CONFIDENCE_NAMES.get(confidence_tier, "Context record"),
                "evidence": _evidence_text(category, group, radius_km),
            }
        )

    indicators.sort(
        key=lambda item: (
            -item["sort_score"],
            item["nearest_distance_km"],
            item["label"],
        )
    )

    context_summary = []
    for category, group in context_only.groupby("source_category"):
        meta = CATEGORY_META.get(category, {"label": category.title(), "icon": "📍"})
        context_summary.append(
            {
                "label": meta["label"],
                "icon": meta["icon"],
                "count": int(len(group)),
                "nearest_distance_km": round(float(group["distance_km"].min()), 2),
                "note": "Mapped presence only; current activity is not verified.",
            }
        )

    has_official = bool(eligible["officially_listed"].any())
    confidence = "Medium" if indicators else "Low"
    if has_official and len(indicators) >= 2:
        confidence = "Medium-high"

    return {
        "indicators": indicators,
        "context_only": context_summary,
        "radius_km": radius_km,
        "total_features": int(len(nearby)),
        "eligible_features": int(len(eligible)),
        "confidence": confidence,
    }


def get_source_indicators(latitude: float, longitude: float, radius_km: float = 5.0) -> dict:
    """Return contextual source indicators centred on exact coordinates."""
    latitude = round(float(latitude), 5)
    longitude = round(float(longitude), 5)
    radius_km = round(float(radius_km), 1)
    result = dict(_cached_indicators(latitude, longitude, radius_km))
    result.update(
        {
            "latitude": latitude,
            "longitude": longitude,
            "method": "Exact-coordinate geometry proximity",
            "wind_adjusted": False,
            "disclaimer": (
                "These are nearby source indicators, not estimated pollution-contribution "
                "percentages. Wind, live activity and pollutant chemistry are not yet included."
            ),
        }
    )
    return result


def get_attribution(station_name: str, user_lat: float, user_lon: float) -> dict:
    """Backward-compatible shape for older chatbot consumers."""
    result = get_source_indicators(user_lat, user_lon, radius_km=5.0)
    indicators = result["indicators"]
    breakdown = [
        {
            "source": item["label"],
            "share": 0.0,
            "strength": item["strength"],
        }
        for item in indicators
    ]
    return {
        "top_source": indicators[0]["label"] if indicators else "No strong nearby indicator",
        "breakdown": breakdown,
        "evidence": [result["disclaimer"]],
        "reference_station": station_name,
        "source_indicators": indicators,
    }
