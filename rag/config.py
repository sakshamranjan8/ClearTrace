"""ClearTrace Module 3 — Configuration.

Loads environment variables from .env file and provides a Settings object
that every other module imports.  Validates required keys at import time
so the app fails fast with a clear message instead of crashing mid-request.

Usage:
    from rag.config import settings
    print(settings.GROQ_API_KEY)
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
# PROJECT_ROOT is the ClearTrace/ directory (two levels above app/config.py).
# This mirrors the pattern used in your existing src/ scripts.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load .env from project root (same directory as requirements.txt)
_dotenv_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_dotenv_path)


# ---------------------------------------------------------------------------
# Settings dataclass (plain class — avoids extra dependency on pydantic-settings)
# ---------------------------------------------------------------------------
class Settings:
    """Centralised settings loaded once from environment variables.

    Every path is resolved relative to PROJECT_ROOT so the code works
    regardless of where you run `uvicorn` from.
    """

    def __init__(self):
        # --- Groq LLM ---
        self.GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
        self.GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        # --- Teammate APIs ---
        self.MOCK_MODE: bool = os.getenv("MOCK_MODE", "true").lower() == "true"
        self.MODULE2_FORECAST_URL: str = os.getenv(
            "MODULE2_FORECAST_URL", "http://localhost:8000/forecast"
        )
        self.MODULE4_REPORTS_URL: str = os.getenv(
            "MODULE4_REPORTS_URL", "http://localhost:8002/reports"
        )

        # --- Embedding model (runs locally, no API key needed) ---
        self.EMBEDDING_MODEL: str = os.getenv(
            "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )

        # --- Data paths (resolved from PROJECT_ROOT) ---
        self.STATION_SOURCE_LINKS_PATH: Path = (
            PROJECT_ROOT / "data" / "context" / "station_source_links_v1.csv"
        )
        self.SOURCE_INVENTORY_PATH: Path = (
            PROJECT_ROOT / "data" / "context" / "source_inventory_v1.csv"
        )

        # --- Knowledge base for RAG (Issue #3: explicit path resolution) ---
        self.KNOWLEDGE_DIR: Path = PROJECT_ROOT / "rag" / "knowledge"
        self.FAISS_INDEX_DIR: Path = PROJECT_ROOT / "data" / "faiss_index"

        # --- Logging ---
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

        # --- Request / prompt limits ---
        # Issue #6: keep super-prompt under this token budget
        self.MAX_PROMPT_TOKENS: int = 4000

        # Timeout (seconds) for teammate API calls
        self.TEAMMATE_API_TIMEOUT: int = 10


# ---------------------------------------------------------------------------
# Validation (Issue #10)
# ---------------------------------------------------------------------------
def validate_config(s: Settings) -> None:
    """Check that critical settings are present.

    Called at import time so the app fails immediately with a clear
    message instead of crashing when the first chat request arrives.
    """
    if not s.GROQ_API_KEY or s.GROQ_API_KEY == "your_groq_api_key_here":
        raise ValueError(
            "\n"
            "╔══════════════════════════════════════════════════════════╗\n"
            "║  GROQ_API_KEY is missing or still set to the placeholder.║\n"
            "║                                                          ║\n"
            "║  1. Copy .env.example → .env                             ║\n"
            "║  2. Paste your Groq API key into .env                    ║\n"
            "║  3. Get a free key at https://console.groq.com/keys      ║\n"
            "╚══════════════════════════════════════════════════════════╝"
        )

    if not s.STATION_SOURCE_LINKS_PATH.exists():
        print(
            f"[WARNING] station_source_links_v1.csv not found at "
            f"{s.STATION_SOURCE_LINKS_PATH}.  Attribution will use fallback data."
        )

    if not s.KNOWLEDGE_DIR.exists():
        print(
            f"[WARNING] Knowledge directory not found at {s.KNOWLEDGE_DIR}.  "
            f"RAG search will return empty results."
        )

    print(f"[CONFIG] Project root   : {PROJECT_ROOT}")
    print(f"[CONFIG] Mock mode      : {s.MOCK_MODE}")
    print(f"[CONFIG] Groq model     : {s.GROQ_MODEL}")
    print(f"[CONFIG] Embedding model: {s.EMBEDDING_MODEL}")
    print(f"[CONFIG] Knowledge dir  : {s.KNOWLEDGE_DIR}")
    print(f"[CONFIG] FAISS index dir: {s.FAISS_INDEX_DIR}")


# ---------------------------------------------------------------------------
# Module-level singleton — every file does `from rag.config import settings`
# ---------------------------------------------------------------------------
settings = Settings()
validate_config(settings)
