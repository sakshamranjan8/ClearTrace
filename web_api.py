"""HTTP API for the optional React frontend.

The original Streamlit app keeps calling the existing Python modules directly.
This service exposes those same modules to the React client and adds local,
revocable authentication for citizen reports.  It requires no external auth
provider or API key.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

import chatbot as chatbot_mod
import database
import real_attribution as attribution_mod
import real_forecast as forecast_mod
import real_health_risk as health_mod


REPORT_CATEGORIES = set(database.SOURCE_CATEGORIES)
MAX_IMAGE_BYTES = 5 * 1024 * 1024
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
bearer_scheme = HTTPBearer(auto_error=False)


app = FastAPI(
    title="ClearTrace Web API",
    description="Backend-for-frontend for the optional React experience.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

database.init_db()


class RegisterRequest(BaseModel):
    display_name: str = Field(min_length=2, max_length=80)
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    location: str = Field(default="Current location", max_length=120)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    user_category: str = Field(default="adult", max_length=80)


class ExposureRequest(BaseModel):
    location: str = Field(default="Current location", max_length=120)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    start_index: int = Field(default=0, ge=0, le=23)
    duration_hours: int = Field(default=1, ge=1, le=8)
    sensitivity_group: str = Field(default="General population", max_length=80)
    activity_level: str = Field(
        default="Light activity (walking / commuting)", max_length=120
    )


class VoteRequest(BaseModel):
    vote_type: str = Field(default="upvote", pattern="^(upvote)$")


def _validate_email(email: str) -> str:
    value = email.strip().lower()
    if not EMAIL_PATTERN.fullmatch(value):
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    return value


def _token_from_credentials(
    credentials: HTTPAuthorizationCredentials | None,
) -> str | None:
    if credentials is None or credentials.scheme.lower() != "bearer":
        return None
    return credentials.credentials


def require_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    token = _token_from_credentials(credentials)
    user = database.get_user_for_session(token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in to use citizen reports.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def _auth_payload(user: dict, token: str) -> dict:
    return {
        "token": token,
        "token_type": "bearer",
        "expires_in_days": database.SESSION_LIFETIME_DAYS,
        "user": user,
    }


def _serialize_report(report: dict) -> dict:
    result = dict(report)
    result.pop("image_path", None)
    result["has_image"] = bool(report.get("image_path"))
    result["image_url"] = (
        f"/api/reports/{report['report_id']}/image"
        if report.get("image_path")
        else None
    )
    if "upvotes" not in result:
        result["upvotes"] = database.get_report_upvotes(report["report_id"])
    return result


@app.get("/")
def root():
    return {
        "service": "ClearTrace Web API",
        "status": "running",
        "documentation": "/docs",
        "authentication": "optional except for citizen-report routes",
    }


@app.get("/api/health")
def health():
    return {"status": "healthy", "database": str(database.DB_PATH.name)}


@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest):
    email = _validate_email(payload.email)
    display_name = " ".join(payload.display_name.strip().split())
    if len(display_name) < 2:
        raise HTTPException(status_code=422, detail="Enter your full name.")
    if not any(char.isalpha() for char in payload.password) or not any(
        char.isdigit() for char in payload.password
    ):
        raise HTTPException(
            status_code=422,
            detail="Password must contain at least one letter and one number.",
        )

    try:
        user = database.create_user(email, display_name, payload.password)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

    token = database.create_user_session(user["user_id"])
    return _auth_payload(user, token)


@app.post("/api/auth/login")
def login(payload: LoginRequest):
    email = _validate_email(payload.email)
    user = database.authenticate_user(email, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    token = database.create_user_session(user["user_id"])
    return _auth_payload(user, token)


@app.get("/api/auth/me")
def me(user=Depends(require_user)):
    return {"user": user}


@app.post("/api/auth/logout")
def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    token = _token_from_credentials(credentials)
    if token:
        database.revoke_user_session(token)
    return {"status": "signed_out"}


@app.get("/api/forecast")
def forecast(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    location: str = Query(default="Current location", max_length=120),
):
    result = forecast_mod.generate_forecast(
        location,
        latitude=latitude,
        longitude=longitude,
    )
    if result.get("error"):
        raise HTTPException(
            status_code=503,
            detail=result.get("error_message") or "Forecast is unavailable.",
        )
    return {"forecast": result, "summary": forecast_mod.forecast_summary(result)}


@app.get("/api/source-indicators")
def source_indicators(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(default=5.0, ge=0.5, le=20),
):
    try:
        return attribution_mod.get_source_indicators(
            latitude, longitude, radius_km=radius_km
        )
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/api/exposure-advisory")
def exposure_advisory(payload: ExposureRequest):
    forecast_result = forecast_mod.generate_forecast(
        payload.location,
        latitude=payload.latitude,
        longitude=payload.longitude,
    )
    if forecast_result.get("error") or not forecast_result.get("hourly"):
        raise HTTPException(
            status_code=503,
            detail=forecast_result.get("error_message") or "Forecast is unavailable.",
        )
    try:
        return health_mod.build_exposure_advisory(
            forecast_result["hourly"],
            payload.start_index,
            payload.duration_hours,
            payload.sensitivity_group,
            payload.activity_level,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/chat")
def chat(payload: ChatRequest):
    try:
        return chatbot_mod.answer_query(
            payload.question,
            payload.location,
            lat=payload.latitude,
            lon=payload.longitude,
            user_category=payload.user_category,
        )
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="The AQI assistant is temporarily unavailable.",
        ) from error


@app.get("/api/reports/nearby")
def nearby_reports(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    radius_m: int = Query(default=5000, ge=100, le=20000),
    user=Depends(require_user),
):
    del user
    reports = database.get_nearby_reports(
        latitude, longitude, radius_m=radius_m, include_verified=True
    )
    return {"reports": [_serialize_report(report) for report in reports[:50]]}


@app.get("/api/reports/mine")
def my_reports(user=Depends(require_user)):
    reports = database.get_user_reports(user["user_id"])
    return {"reports": [_serialize_report(report) for report in reports]}


@app.post("/api/reports", status_code=status.HTTP_201_CREATED)
async def submit_report(
    latitude: float = Form(...),
    longitude: float = Form(...),
    description: str = Form(..., min_length=8, max_length=1000),
    category: str = Form(...),
    image: UploadFile | None = File(default=None),
    user=Depends(require_user),
):
    if category not in REPORT_CATEGORIES:
        raise HTTPException(status_code=422, detail="Choose a valid source category.")

    image_bytes = None
    image_ext = "jpg"
    if image is not None:
        if image.content_type not in {"image/jpeg", "image/png"}:
            raise HTTPException(
                status_code=422, detail="Upload a JPEG or PNG image."
            )
        image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image must be 5 MB or smaller.")
        image_ext = Path(image.filename or "report.jpg").suffix.lstrip(".") or "jpg"

    result = database.create_report(
        latitude,
        longitude,
        description.strip(),
        category,
        user["user_id"],
        image_bytes=image_bytes,
        image_ext=image_ext,
        reporter_name=user["display_name"],
    )
    if result.get("status") == "rejected":
        raise HTTPException(status_code=400, detail=result.get("reason"))
    return result


@app.post("/api/reports/{report_id}/vote")
def vote(report_id: str, payload: VoteRequest, user=Depends(require_user)):
    report = database.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    if report.get("user_id") == user["user_id"]:
        raise HTTPException(status_code=400, detail="You cannot verify your own report.")

    result = database.vote_report(
        report_id,
        user["user_id"],
        vote_type=payload.vote_type,
        voter_name=user["display_name"],
    )
    if result.get("status") == "already_voted":
        raise HTTPException(status_code=409, detail="You already verified this report.")
    return result


@app.get("/api/reports/{report_id}/image")
def report_image(report_id: str, user=Depends(require_user)):
    del user
    report = database.get_report(report_id)
    if report is None or not report.get("image_path"):
        raise HTTPException(status_code=404, detail="Report image not found.")
    resolved = database.resolve_image_path(report["image_path"])
    if resolved is None:
        raise HTTPException(status_code=404, detail="Report image is unavailable.")
    return FileResponse(resolved)

