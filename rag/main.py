"""ClearTrace Module 3 — FastAPI Server.

The HTTP layer that exposes two endpoints:
  GET  /attribution/latest  → source attribution for a location
  POST /chat/query          → RAG-powered chatbot answer
  GET  /health              → health check for monitoring

Run locally:
    python -m uvicorn app.main:app --reload --port 8000

Then visit:
    http://localhost:8000/docs  — interactive Swagger UI
    http://localhost:8000/health — health check

ISSUE #9:  Rate limiting is DISABLED for the hackathon demo.
ISSUE #11: Request logging middleware tracks method, path, and response time.
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.models import ChatRequest, ChatResponse, AttributionResponse
from app import chatbot
from app import vector_store


# ===========================================================================
# Lifespan (startup / shutdown events)
# ===========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on server startup and shutdown.

    Startup:
      - Build or load the FAISS vector store index.
      - NOTE: CSV data is NOT loaded here (lazy loading — Issue #2).

    Shutdown:
      - Cleanup if needed (currently nothing).
    """
    print("\n" + "=" * 60)
    print("  ClearTrace Module 3 — Starting up")
    print("=" * 60)

    # Build/load FAISS index (this downloads the embedding model on first run)
    print("\n[STARTUP] Building FAISS index...")
    index_ready = vector_store.build_or_load_index()
    if index_ready:
        print("[STARTUP] FAISS index ready ✓")
    else:
        print("[STARTUP] FAISS index NOT ready — chatbot will use fallbacks")

    print("\n[STARTUP] Attribution CSV will load on first request (lazy loading)")
    print(f"[STARTUP] Mock mode: {settings.MOCK_MODE}")
    print(f"[STARTUP] Groq model: {settings.GROQ_MODEL}")
    print("\n[STARTUP] Ready! Visit http://localhost:8000/docs for Swagger UI")
    print("=" * 60 + "\n")

    yield  # Server is running

    # Shutdown
    print("\n[SHUTDOWN] ClearTrace Module 3 shutting down")


# ===========================================================================
# FastAPI app
# ===========================================================================

app = FastAPI(
    title="ClearTrace Module 3",
    description=(
        "Attribution Engine + RAG AI Chatbot for Delhi Air Quality.\n\n"
        "**Endpoints:**\n"
        "- `GET /attribution/latest` — pollution source attribution\n"
        "- `POST /chat/query` — AI-powered Q&A about air quality and health\n"
        "- `GET /health` — service health check"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ===========================================================================
# CORS middleware (allows frontend to call our API)
# ===========================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for hackathon demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Request logging middleware (Issue #11)
# ===========================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with method, path, and response time.

    This is invaluable for debugging during the hackathon.
    Example output:
        [REQUEST] GET /health → 200 (12ms)
        [REQUEST] POST /chat/query → 200 (2340ms)
    """
    start_time = time.time()

    response = await call_next(request)

    duration_ms = (time.time() - start_time) * 1000
    print(
        f"[REQUEST] {request.method} {request.url.path} "
        f"→ {response.status_code} ({duration_ms:.0f}ms)"
    )

    return response


# ===========================================================================
# Issue #9: Rate limiting is DISABLED for the hackathon demo.
#
# Why? Hugging Face Spaces free tier restarts frequently, resetting any
# in-memory counters. A proper solution would use Redis or a database,
# which is overkill for a hackathon. If you need rate limiting later,
# consider using the `slowapi` library:
#   pip install slowapi
#   from slowapi import Limiter
#   limiter = Limiter(key_func=get_remote_address)
# ===========================================================================


# ===========================================================================
# Endpoints
# ===========================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint.

    Used by Hugging Face Spaces to know the app is alive.
    Also useful for teammates to verify your service is running.
    """
    vs_status = vector_store.get_status()

    return {
        "status": "healthy",
        "module": "ClearTrace Module 3 — Attribution + Chatbot",
        "mock_mode": settings.MOCK_MODE,
        "groq_model": settings.GROQ_MODEL,
        "vector_store": vs_status,
    }


@app.get("/attribution/latest", response_model=AttributionResponse)
async def get_attribution(
    lat: float = Query(
        ...,
        description="Latitude of the location",
        example=28.6139,
        ge=20.0,
        le=35.0,
    ),
    lon: float = Query(
        ...,
        description="Longitude of the location",
        example=77.2090,
        ge=65.0,
        le=100.0,
    ),
):
    """Get pollution source attribution for a location.

    Identifies which sources (traffic, industry, construction, waste, etc.)
    are likely contributing to air pollution at the given coordinates.

    Uses the pre-built station-source linkage data from OpenStreetMap.
    """
    try:
        # Import here to trigger lazy loading on first call
        from app.attribution import get_attribution as compute_attribution

        result = compute_attribution(lat, lon)
        return result

    except Exception as e:
        print(f"[ERROR] Attribution failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Attribution computation failed: {str(e)}",
        )


@app.post("/chat/query", response_model=ChatResponse)
async def chat_query(request: ChatRequest):
    """Ask the AI chatbot a question about air quality or health.

    The chatbot combines:
    - CPCB guidelines and health recommendations (via RAG)
    - Pollution source attribution for your location
    - AQI forecast data (from Module 2)
    - Community reports (from Module 4)

    All of this context is sent to the LLM, which generates a
    personalised, evidence-based answer.
    """
    try:
        result = await chatbot.ask(
            question=request.question,
            lat=request.lat,
            lon=request.lon,
            user_category=request.user_category,
        )
        return result

    except Exception as e:
        print(f"[ERROR] Chat query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat query failed: {str(e)}",
        )


# ===========================================================================
# Error handlers
# ===========================================================================

@app.exception_handler(422)
async def validation_error_handler(request: Request, exc):
    """Return a beginner-friendly error when request validation fails."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid request",
            "detail": str(exc),
            "hint": (
                "For /attribution/latest: use ?lat=28.6139&lon=77.2090\n"
                "For /chat/query: send JSON with 'question', 'lat', 'lon'"
            ),
        },
    )


# ===========================================================================
# Run with: python -m uvicorn app.main:app --reload --port 8000
# Or:       python app/main.py
# ===========================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
