"""
real_health_risk.py
Wraps the teammate's real health_engine_07.py so it matches what app.py
already expects to call.
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
if (HERE / "ml_assets" / "health_engine_07.py").exists():
    sys.path.insert(0, str(HERE / "ml_assets"))
else:
    sys.path.insert(0, str(HERE / "notebooks"))
from health_engine_07 import calculate_risk  # type: ignore

# UI label -> the key health_engine_07.py expects
UI_TO_ENGINE_CATEGORY = {
    "Normal adult": "adult",
    "Child": "child",
    "Elderly": "elderly",
    "Asthma / respiratory sensitivity": "asthma",
    "Outdoor worker": "outdoor_worker",
    "Pregnant": "pregnant_woman",
}

VULNERABILITY_MULTIPLIER = {k: None for k in UI_TO_ENGINE_CATEGORY}


def compute_health_risk(aqi, duration_hours, user_category):
    engine_category = UI_TO_ENGINE_CATEGORY.get(user_category, "adult")
    result = calculate_risk(aqi, duration_hours, engine_category)
    return {
        "score": result["score"],
        "hazard_level": result["level"],
        "color_hex": result["color"],
        "headline": result["recommendation"],
        "recommendations": [result["recommendation"]],
    }