"""ClearTrace Module 3 — Pydantic Data Models.

Defines the request/response schemas that match the API contracts
your teammates expect.  Pydantic validates incoming JSON automatically
and generates nice error messages when the frontend sends bad data.

Why Pydantic?
  - FastAPI uses it natively for request/response validation
  - Auto-generates OpenAPI/Swagger docs from these models
  - Gives you free type checking and serialisation
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ===========================================================================
# Attribution Endpoint — GET /attribution/latest
# ===========================================================================

class AttributionResponse(BaseModel):
    """Response for GET /attribution/latest?lat=...&lon=...

    Example:
        {
            "sources": {"traffic": 35, "industry": 25, ...},
            "evidence": {"traffic": "NH-24 highway within 1km, 12 road segments nearby"},
            "confidence_score": 0.78,
            "timestamp": "2026-07-18T10:30:00Z"
        }
    """

    # Percentage contribution per source category (should roughly sum to 100)
    sources: Dict[str, float] = Field(
        ...,
        description="Pollution source contributions as percentages",
        examples=[{"traffic": 35, "industry": 25, "construction": 20, "waste_burning": 15, "other": 5}],
    )

    # Human-readable evidence strings with SPECIFIC source names (Issue #5)
    evidence: Dict[str, str] = Field(
        ...,
        description="Evidence text per category, including specific source names",
    )

    confidence_score: float = Field(
    ...,
    ge=0.0,
    le=100.0,  
    description="Overall confidence in the attribution (0-100%)",
)

    timestamp: datetime


# ===========================================================================
# Chat Endpoint — POST /chat/query
# ===========================================================================

class ChatRequest(BaseModel):
    """Request body for POST /chat/query.

    Example:
        {
            "question": "Is it safe to jog tomorrow?",
            "lat": 28.6139,
            "lon": 77.2090,
            "user_category": "adult"
        }
    """

    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The user's question about air quality or health",
    )

    lat: float = Field(
        ...,
        ge=20.0,
        le=35.0,
        description="Latitude (must be within India roughly)",
    )

    lon: float = Field(
        ...,
        ge=65.0,
        le=100.0,
        description="Longitude (must be within India roughly)",
    )

    # Issue #4: Default to "adult" so the endpoint never breaks if omitted
    user_category: str = Field(
        default="adult",
        description="User vulnerability category",
        examples=["adult", "child", "elderly", "asthma", "outdoor_worker", "pregnant_woman"],
    )


class ChatResponse(BaseModel):
    """Response for POST /chat/query.

    Example:
        {
            "answer": "Based on AQI 280 (Severe), not safe to jog...",
            "sources_used": ["CPCB Guidelines 2024", "Forecast Data"],
            "hazard_level": "severe",
            "recommendations": ["Avoid outdoor activity", "Use N95 mask"],
            "timestamp": "2026-07-18T10:30:00Z"
        }
    """

    answer: str = Field(
        ...,
        description="LLM-generated answer text",
    )

    sources_used: List[str] = Field(
        ...,
        description="List of data sources used to build the answer",
    )

    # Issue #7: This is set programmatically from health engine, NOT by the LLM
    hazard_level: str = Field(
        ...,
        description="Risk level: good / satisfactory / moderate / poor / very_poor / severe",
    )

    recommendations: List[str] = Field(
        ...,
        description="Actionable recommendations for the user",
    )

    timestamp: datetime


# ===========================================================================
# Internal Models — for teammate API responses
# ===========================================================================

class ForecastItem(BaseModel):
    """A single hourly forecast entry from Module 2.

    Example:
        {"timestamp": "...", "horizon_hours": 1, "predicted_aqi": 80.36, "category": "Satisfactory"}
    """

    timestamp: Optional[datetime] = None
    horizon_hours: Optional[int] = None
    predicted_aqi: Optional[float] = None
    category: Optional[str] = None


class ForecastWindow(BaseModel):
    """The time window covered by the forecast."""

    start: Optional[datetime] = None
    end: Optional[datetime] = None
    total_hours: Optional[int] = None


class NearestStation(BaseModel):
    """A station used by Module 2 for blended prediction."""

    station_name: Optional[str] = None
    distance_km: Optional[float] = None
    blend_weight: Optional[float] = None


class ForecastData(BaseModel):
    """Expected shape of Module 2 forecast response (actual format).

    All fields are Optional so partial responses or schema changes
    never crash our chatbot.

    Example:
        {
            "status": "success",
            "nearest_stations": [{"station_name": "...", "distance_km": 2.6, ...}],
            "forecast": [{"horizon_hours": 1, "predicted_aqi": 80.36, ...}, ...]
        }
    """

    status: Optional[str] = None
    request: Optional[Dict] = None                       # {"latitude": ..., "longitude": ...}
    generated_at: Optional[datetime] = None
    forecast_window: Optional[ForecastWindow] = None
    nearest_stations: Optional[List[NearestStation]] = None
    forecast: Optional[List[ForecastItem]] = None         # Array of hourly forecasts


class ReportsData(BaseModel):
    """Expected shape of Module 4 crowd reports response."""

    reports: Optional[List[Dict]] = None  # List of report dicts
    total_count: Optional[int] = 0
    verified_count: Optional[int] = 0


class HealthRisk(BaseModel):
    """Output of the health risk calculator (ported from health_engine_07.py).

    This is computed deterministically — the LLM never decides hazard levels.
    """

    score: float
    level: str       # "Low", "Moderate", "High", "Severe"
    color: str       # Hex colour for frontend
    recommendation: str
