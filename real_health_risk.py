"""Transparent AQI exposure guidance for the ClearTrace frontend.

This module deliberately does not calculate a medical-risk probability or an
arbitrary numeric hazard score.  It selects the hours a user actually plans to
be outdoors, summarizes that forecast window, and maps the result to CPCB AQI
health messaging with a more cautious action for susceptible groups.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd


SENSITIVITY_GROUPS = [
    "General population",
    "Child or teenager",
    "Older adult (65+)",
    "Pregnant",
    "Asthma or COPD",
    "Heart condition",
]

ACTIVITY_LEVELS = [
    "Light activity (walking / commuting)",
    "Moderate activity (brisk walk / cycling)",
    "Strenuous activity (running / outdoor sport)",
]

SENSITIVE_GROUPS = set(SENSITIVITY_GROUPS[1:])


AQI_BANDS = [
    {
        "max": 50,
        "category": "Good",
        "color": "#22c55e",
        "impact": "Minimal expected health impact for most people.",
    },
    {
        "max": 100,
        "category": "Satisfactory",
        "color": "#84cc16",
        "impact": "Minor breathing discomfort may occur in sensitive people.",
    },
    {
        "max": 200,
        "category": "Moderate",
        "color": "#f59e0b",
        "impact": (
            "Breathing discomfort may occur in people with lung, asthma or "
            "heart conditions, children and older adults."
        ),
    },
    {
        "max": 300,
        "category": "Poor",
        "color": "#f97316",
        "impact": (
            "Breathing discomfort may affect most people with prolonged "
            "outdoor exposure."
        ),
    },
    {
        "max": 400,
        "category": "Very Poor",
        "color": "#ef4444",
        "impact": "Prolonged exposure may cause respiratory illness.",
    },
    {
        "max": float("inf"),
        "category": "Severe",
        "color": "#7f1d1d",
        "impact": (
            "Air quality can affect healthy people and seriously affect people "
            "with existing health conditions."
        ),
    },
]


ACTION_LABELS = [
    ("Low concern", "Outdoor plans can continue", "#16a34a"),
    ("Be aware", "Keep your usual plan and monitor symptoms", "#65a30d"),
    ("Use caution", "Reduce prolonged or strenuous outdoor activity", "#d97706"),
    ("Limit exposure", "Shorten outdoor time and choose a less intense activity", "#ea580c"),
    ("Avoid strenuous exposure", "Move strenuous or prolonged activity indoors", "#dc2626"),
    ("Avoid outdoor exposure", "Postpone non-essential outdoor activity", "#7f1d1d"),
]


def aqi_band(aqi: float) -> dict:
    """Return the official Indian AQI band metadata for an AQI value."""
    value = float(aqi)
    for band in AQI_BANDS:
        if value <= band["max"]:
            return band.copy()
    return AQI_BANDS[-1].copy()


def _action_index(
    peak_aqi: float,
    sensitivity_group: str,
    activity_level: str,
    duration_hours: int = 1,
) -> int:
    base = next(
        index for index, band in enumerate(AQI_BANDS) if peak_aqi <= band["max"]
    )

    # Susceptibility and breathing rate affect the action, not the reported AQI
    # category.  Capping prevents the UI from inventing a new medical category.
    if sensitivity_group in SENSITIVE_GROUPS and base >= 1:
        base += 1
    if activity_level.startswith("Strenuous") and base >= 2:
        base += 1
    if duration_hours >= 4 and base >= 2:
        base += 1
    return min(base, len(ACTION_LABELS) - 1)


def _recommendations(
    peak_aqi: float,
    sensitivity_group: str,
    activity_level: str,
    duration_hours: int,
    alternative: dict | None,
) -> list[str]:
    action_index = _action_index(
        peak_aqi,
        sensitivity_group,
        activity_level,
        duration_hours,
    )
    recommendations = []

    if action_index <= 1:
        recommendations.append("Keep the planned activity, but check the forecast again before leaving.")
    elif action_index == 2:
        recommendations.append("Reduce duration or intensity if you develop coughing, wheezing or unusual breathlessness.")
    elif action_index == 3:
        recommendations.append("Prefer a shorter, lighter activity and take breaks away from busy roads.")
    else:
        recommendations.append("Move strenuous or prolonged activity indoors, or postpone it if practical.")

    if sensitivity_group in SENSITIVE_GROUPS:
        recommendations.append(
            "Because you selected a susceptible group, follow your clinician's existing action plan and carry prescribed medication."
        )

    if activity_level.startswith("Strenuous") and peak_aqi > 100:
        recommendations.append(
            "Strenuous exercise increases breathing rate; choose a lighter activity during this window."
        )

    if alternative and alternative["mean_aqi"] + 5 < alternative["selected_mean_aqi"]:
        start = pd.to_datetime(alternative["start"]).strftime("%d %b, %I:%M %p")
        recommendations.append(
            f"A cleaner {alternative['duration_hours']}h window begins around {start} "
            f"(forecast mean AQI {alternative['mean_aqi']:.0f})."
        )

    recommendations.append(
        "Seek medical help for severe breathlessness, chest pain, confusion, or symptoms that do not settle."
    )
    return recommendations


def _best_window(hourly: list[dict], duration_hours: int, selected_mean: float) -> dict | None:
    if not hourly or duration_hours > len(hourly):
        return None

    best = None
    for start_index in range(0, len(hourly) - duration_hours + 1):
        window = hourly[start_index : start_index + duration_hours]
        mean_aqi = sum(float(point["aqi"]) for point in window) / len(window)
        candidate = {
            "start": window[0]["timestamp"],
            "end": window[-1]["timestamp"],
            "duration_hours": duration_hours,
            "mean_aqi": mean_aqi,
            "selected_mean_aqi": selected_mean,
        }
        if best is None or candidate["mean_aqi"] < best["mean_aqi"]:
            best = candidate
    return best


def build_exposure_advisory(
    hourly: list[dict],
    start_index: int,
    duration_hours: int,
    sensitivity_group: str,
    activity_level: str,
) -> dict:
    """Build an advisory from the exact selected forecast hours."""
    if not hourly:
        raise ValueError("At least one hourly forecast point is required.")

    start_index = max(0, min(int(start_index), len(hourly) - 1))
    duration_hours = max(1, int(duration_hours))
    end_index = min(len(hourly), start_index + duration_hours)
    window = hourly[start_index:end_index]

    values = [float(point["aqi"]) for point in window]
    mean_aqi = sum(values) / len(values)
    peak_point = max(window, key=lambda point: float(point["aqi"]))
    peak_aqi = float(peak_point["aqi"])
    band = aqi_band(peak_aqi)
    action_index = _action_index(
        peak_aqi,
        sensitivity_group,
        activity_level,
        len(window),
    )
    action_level, headline, action_color = ACTION_LABELS[action_index]
    alternative = _best_window(hourly, len(window), mean_aqi)

    start_time = pd.to_datetime(window[0]["timestamp"])
    # Forecast points denote hourly starts; add one hour for a human-readable end.
    end_time = pd.to_datetime(window[-1]["timestamp"]) + timedelta(hours=1)

    return {
        "window": window,
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "hours_used": len(window),
        "mean_aqi": round(mean_aqi),
        "peak_aqi": round(peak_aqi),
        "peak_time": peak_point["timestamp"],
        "category": band["category"],
        "category_color": band["color"],
        "health_message": band["impact"],
        "action_level": action_level,
        "headline": headline,
        "action_color": action_color,
        "recommendations": _recommendations(
            peak_aqi,
            sensitivity_group,
            activity_level,
            len(window),
            alternative,
        ),
        "sensitivity_group": sensitivity_group,
        "activity_level": activity_level,
        "method_note": (
            "Guidance uses the maximum AQI in the selected forecast window and "
            "CPCB category messaging. It is not a diagnosis or medical-risk probability."
        ),
    }


def compute_health_risk(aqi: float, duration: float, category: str) -> dict:
    """Backward-compatible wrapper for older chatbot/admin consumers.

    No arbitrary numeric score is returned. New UI code should use
    :func:`build_exposure_advisory` with actual hourly forecast points.
    """
    band = aqi_band(aqi)
    group = "General population" if category in {"Normal adult", "General population"} else category
    if group not in SENSITIVITY_GROUPS:
        group = "General population"
    action_index = _action_index(
        float(aqi),
        group,
        ACTIVITY_LEVELS[0],
        max(1, round(float(duration))),
    )
    level, headline, color = ACTION_LABELS[action_index]
    return {
        "score": None,
        "hazard_level": level,
        "headline": headline,
        "color_hex": color,
        "recommendations": [band["impact"]],
        "duration_hours": duration,
        "category": band["category"],
    }


# Alias retained for old selectboxes that import this name.
VULNERABILITY_MULTIPLIER = {group: 1.0 for group in SENSITIVITY_GROUPS}
