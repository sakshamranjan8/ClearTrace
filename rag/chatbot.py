"""ClearTrace Module 3 — RAG Chatbot.

The main intelligence layer. Combines:
  1. FAISS-retrieved knowledge chunks (CPCB guidelines, health recommendations)
  2. Source attribution data (from attribution.py)
  3. Forecast data (from Module 2 teammate API, or mock)
  4. Crowd reports (from Module 4 teammate API, or mock)
  5. Health risk calculation (deterministic, from utils.py)

Then sends everything to Groq's gemma2-9b-it via a structured "super prompt"
and returns a formatted ChatResponse.

ISSUE #6: Super prompt is truncated to stay within Groq's 6000 tokens/min limit.
ISSUE #7: hazard_level and recommendations come from calculate_risk(), NOT the LLM.
           The LLM only generates the answer text.
"""

from groq import Groq

from rag.config import settings
from rag.utils import (
    calculate_risk,
    extract_current_aqi,
    extract_time_from_query,
    fetch_teammate_api,
    format_forecast_for_prompt,
    format_time_for_display,
    get_aqi_category,
    get_aqi_category_label,
    get_current_ist_time,
    get_forecast_at_time,
    get_forecast_category,
    get_mock_forecast,
    get_nearest_station_name,
    truncate_context,
    MOCK_REPORTS,
)
from rag import attribution
from rag import vector_store


# ===========================================================================
# Groq client (lazy initialisation)
# ===========================================================================
_groq_client = None


def _get_groq_client() -> Groq:
    """Get or create the Groq API client."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.GROQ_API_KEY)
        print(f"[CHATBOT] Groq client initialised (model: {settings.GROQ_MODEL})")
    return _groq_client


# ===========================================================================
# Main entry point
# ===========================================================================

async def ask(
    question: str,
    lat: float,
    lon: float,
    user_category: str = "adult",
) -> dict:
    """Answer a user's air quality / health question using RAG + LLM.

    This is the full pipeline:
      1. Retrieve context (RAG + attribution + forecast + reports)
      2. Build the super prompt
      3. Call Groq LLM
      4. Compute hazard_level deterministically (Issue #7)
      5. Return structured ChatResponse

    Args:
        question: The user's natural-language question.
        lat: User's latitude.
        lon: User's longitude.
        user_category: Vulnerability category (default: "adult").

    Returns:
        Dict matching the ChatResponse schema.
    """
    timestamp = get_current_ist_time()
    sources_used = []

    print(f"\n{'='*60}")
    print(f"[CHATBOT] New query: '{question}'")
    print(f"[CHATBOT] Location: ({lat}, {lon}), Category: {user_category}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Step 1: Gather context from all sources (parallel-ish)
    # ------------------------------------------------------------------

    # 1a. RAG — search knowledge base
    rag_chunks = _get_rag_context(question)
    if rag_chunks:
        sources_used.append("CPCB Guidelines 2024")
        sources_used.append("ClearTrace Knowledge Base")

    # 1b. Attribution — get source breakdown
    attribution_data = _get_attribution_context(lat, lon)
    if attribution_data:
        sources_used.append("Source Attribution Data")

    # 1c. Forecast — call Module 2 API (or mock)
    forecast_data = await _get_forecast_context(lat, lon)
    if forecast_data:
        sources_used.append("Forecast Data")

    # 1d. Crowd reports — call Module 4 API (or mock)
    reports_data = await _get_reports_context(lat, lon)
    if reports_data:
        verified = reports_data.get("verified_count", 0)
        sources_used.append(f"{verified} Verified Reports")

    # ------------------------------------------------------------------
    # Step 2a: Extract time reference from the question (if any)
    # ------------------------------------------------------------------
    time_info = extract_time_from_query(question)
    time_forecast = None

    if time_info.get("found") and forecast_data:
        time_forecast = get_forecast_at_time(
            forecast_data,
            target_hour=time_info["hour"],
            relative=time_info.get("relative", "today"),
        )
        if time_forecast:
            print(
                f"[CHATBOT] Time-specific forecast: "
                f"{time_info['time_str']} → AQI {time_forecast.get('predicted_aqi')}"
            )

    # ------------------------------------------------------------------
    # Step 2b: Compute health risk DETERMINISTICALLY (Issue #7)
    # ------------------------------------------------------------------
    # If the user asked about a specific time, use THAT time's AQI for risk;
    # otherwise use the nearest-hour (horizon_hours=1) AQI.
    if time_forecast and time_forecast.get("predicted_aqi") is not None:
        current_aqi = float(time_forecast["predicted_aqi"])
        print(f"[CHATBOT] Using time-specific AQI for risk: {current_aqi}")
    elif forecast_data:
        current_aqi = extract_current_aqi(forecast_data)
    else:
        current_aqi = 200.0
        print("[CHATBOT] No forecast data — using cautious default AQI (200)")

    # Assume 1 hour outdoor exposure for risk calculation
    health_risk = calculate_risk(
        aqi=current_aqi,
        duration_hours=1.0,
        user_category=user_category,
    )

    aqi_category = get_aqi_category(current_aqi)
    aqi_label = get_aqi_category_label(current_aqi)

    print(f"[CHATBOT] AQI: {current_aqi} → {aqi_label}, Risk: {health_risk['level']}")

    # ------------------------------------------------------------------
    # Step 3: Build the super prompt
    # ------------------------------------------------------------------
    super_prompt = _build_super_prompt(
        question=question,
        user_category=user_category,
        lat=lat,
        lon=lon,
        current_aqi=current_aqi,
        aqi_label=aqi_label,
        health_risk=health_risk,
        rag_chunks=rag_chunks,
        attribution_data=attribution_data,
        forecast_data=forecast_data,
        reports_data=reports_data,
        time_info=time_info,
        time_forecast=time_forecast,
    )

    # ------------------------------------------------------------------
    # Step 4: Call Groq LLM
    # ------------------------------------------------------------------
    llm_answer = _call_groq(super_prompt, question)

    # ------------------------------------------------------------------
    # Step 5: Build response (Issue #7 — health risk OVERRIDES LLM)
    # ------------------------------------------------------------------
    # The LLM generates the answer text.
    # hazard_level and recommendations are computed deterministically.

    recommendations = _build_recommendations(
        aqi_category=aqi_category,
        user_category=user_category,
        health_risk=health_risk,
    )

    return {
        "answer": llm_answer,
        "sources_used": sources_used,
        "hazard_level": aqi_category,  # From get_aqi_category(), not LLM
        "recommendations": recommendations,
        "timestamp": timestamp.isoformat(),
    }


# ===========================================================================
# Context gathering helpers
# ===========================================================================

def _get_rag_context(question: str) -> list[str]:
    """Search the FAISS knowledge base for relevant chunks."""
    try:
        chunks = vector_store.search(question, top_k=5)
        print(f"[CHATBOT] RAG returned {len(chunks)} chunks")
        return chunks
    except Exception as e:
        print(f"[CHATBOT] RAG search failed: {e}")
        return []


def _get_attribution_context(lat: float, lon: float) -> dict:
    """Get source attribution for the user's location."""
    try:
        result = attribution.get_attribution(lat, lon)
        print(f"[CHATBOT] Attribution: {result.get('sources', {})}")
        return result
    except Exception as e:
        print(f"[CHATBOT] Attribution failed: {e}")
        return {}


async def _get_forecast_context(lat: float, lon: float) -> dict:
    """Fetch AQI forecast from Module 2 (or mock).

    Uses the new nested format with forecast array and nearest_stations.
    """
    mock = get_mock_forecast()
    try:
        result = await fetch_teammate_api(
            url=settings.MODULE2_FORECAST_URL,
            params={"lat": lat, "lon": lon},
            mock_fallback=mock,
        )
        # Log extracted AQI from the new format
        aqi = extract_current_aqi(result)
        station = get_nearest_station_name(result)
        print(f"[CHATBOT] Forecast AQI: {aqi} (station: {station})")
        return result
    except Exception as e:
        print(f"[CHATBOT] Forecast fetch failed: {e}")
        return mock  # Graceful degradation


async def _get_reports_context(lat: float, lon: float) -> dict:
    """Fetch crowd reports from Module 4 (or mock)."""
    try:
        result = await fetch_teammate_api(
            url=settings.MODULE4_REPORTS_URL,
            params={"lat": lat, "lon": lon},
            mock_fallback=MOCK_REPORTS,
        )
        print(f"[CHATBOT] Reports: {result.get('total_count', 0)} total")
        return result
    except Exception as e:
        print(f"[CHATBOT] Reports fetch failed: {e}")
        return MOCK_REPORTS  # Graceful degradation


# ===========================================================================
# Super prompt builder
# ===========================================================================

def _build_super_prompt(
    question: str,
    user_category: str,
    lat: float,
    lon: float,
    current_aqi: float,
    aqi_label: str,
    health_risk: dict,
    rag_chunks: list,
    attribution_data: dict,
    forecast_data: dict,
    reports_data: dict,
    time_info: dict = None,
    time_forecast: dict = None,
) -> str:
    """Build the structured prompt that goes to the LLM.

    The prompt is organized into sections so the LLM can easily find
    relevant information.  Each section is optional — if data is missing,
    that section is replaced with a note.

    When the user asked about a specific time, a TIME-SPECIFIC FORECAST
    section is added so the LLM can give a precise answer.

    Issue #6: The total prompt is truncated to MAX_PROMPT_TOKENS.
    """

    sections = []

    # --- System instruction ---
    sections.append(
        "You are ClearTrace AI, a Delhi air quality expert assistant. "
        "You provide accurate, helpful, and empathetic answers about air quality, "
        "health risks, and protective measures. "
        "Base your answers ONLY on the context provided below. "
        "Cite specific data (AQI numbers, source names, guidelines) when possible. "
        "Keep your response concise (3-5 sentences for simple questions, "
        "up to 8 sentences for complex ones). "
        "Do NOT make up data. If information is missing, say so honestly."
    )

    # --- Current conditions ---
    sections.append(
        f"\n== CURRENT CONDITIONS ==\n"
        f"AQI: {current_aqi} ({aqi_label})\n"
        f"Health Risk Level: {health_risk['level']} (score: {health_risk['score']})\n"
        f"User Category: {user_category}\n"
        f"Location: ({lat}, {lon})"
    )

    # --- Forecast (from Module 2, new nested format) ---
    if forecast_data and forecast_data.get("forecast"):
        forecast_text = format_forecast_for_prompt(forecast_data)
        forecast_cat = get_forecast_category(forecast_data)

        sections.append(
            f"\n== FORECAST (next 24h) ==\n"
            f"Current forecast category: {forecast_cat}\n"
            f"{forecast_text}"
        )
    else:
        sections.append("\n== FORECAST ==\nNo forecast data available.")

    # --- Time-specific forecast (new: time-aware queries) ---
    if time_info and time_info.get("found") and time_forecast:
        display_time = format_time_for_display(time_info["hour"])
        aqi_val = time_forecast.get("predicted_aqi", "N/A")
        category = time_forecast.get("category", "Unknown")
        relative = time_info.get("relative", "today")
        horizon = time_forecast.get("horizon_hours", "?")

        aqi_str = f"{aqi_val:.1f}" if isinstance(aqi_val, (int, float)) else str(aqi_val)

        sections.append(
            f"\n== TIME-SPECIFIC FORECAST ==\n"
            f"The user asked about AQI at a specific time: {time_info['time_str']} ({relative})\n"
            f"Forecast at that time:\n"
            f"  Time: {display_time}\n"
            f"  AQI: {aqi_str}\n"
            f"  Category: {category}\n"
            f"  Horizon: +{horizon}h from now\n"
            f"\n"
            f"IMPORTANT: Include this specific AQI value ({aqi_str}) and time "
            f"({display_time}) in your answer. Be precise."
        )
    elif time_info and time_info.get("found"):
        # User asked about a time but no forecast data available
        sections.append(
            f"\n== TIME-SPECIFIC FORECAST ==\n"
            f"The user asked about {time_info['time_str']} but no forecast data "
            f"is available for that time. Let the user know."
        )

    # --- Attribution (from our engine) ---
    if attribution_data and attribution_data.get("sources"):
        attr_lines = []
        for cat, pct in sorted(
            attribution_data["sources"].items(), key=lambda x: -x[1]
        ):
            evidence = attribution_data.get("evidence", {}).get(cat, "")
            attr_lines.append(f"  {cat}: {pct}% — {evidence}")
        attr_text = "\n".join(attr_lines)

        confidence = attribution_data.get("confidence_score", "N/A")
        sections.append(
            f"\n== POLLUTION SOURCES ==\n"
            f"Confidence: {confidence}\n"
            f"{attr_text}"
        )
    else:
        sections.append("\n== POLLUTION SOURCES ==\nNo attribution data available.")

    # --- Crowd reports (from Module 4) ---
    if reports_data and reports_data.get("reports"):
        report_lines = []
        for r in reports_data["reports"][:3]:  # Limit to 3 reports
            status = "✓ Verified" if r.get("verified") else "○ Unverified"
            report_lines.append(f"  [{status}] {r.get('description', 'No description')}")
        reports_text = "\n".join(report_lines)

        sections.append(
            f"\n== COMMUNITY REPORTS ==\n"
            f"Total: {reports_data.get('total_count', 0)}, "
            f"Verified: {reports_data.get('verified_count', 0)}\n"
            f"{reports_text}"
        )

    # --- RAG knowledge chunks ---
    if rag_chunks:
        # Join chunks and truncate to keep the prompt within budget
        knowledge_text = "\n\n".join(rag_chunks)
        sections.append(
            f"\n== KNOWLEDGE BASE (retrieved documents) ==\n"
            f"{knowledge_text}"
        )

    # --- Combine and truncate (Issue #6) ---
    full_prompt = "\n".join(sections)
    full_prompt = truncate_context(full_prompt, max_tokens=settings.MAX_PROMPT_TOKENS)

    print(f"[CHATBOT] Super prompt: {len(full_prompt)} chars")
    return full_prompt


# ===========================================================================
# Groq LLM caller
# ===========================================================================

def _call_groq(system_prompt: str, user_question: str) -> str:
    """Call Groq's LLM API and return the answer text.

    Falls back to a template-based response if Groq fails.

    Args:
        system_prompt: The super prompt with all context.
        user_question: The user's original question.

    Returns:
        The LLM's answer as a string.
    """
    try:
        client = _get_groq_client()

        print(f"[CHATBOT] Calling Groq ({settings.GROQ_MODEL})...")
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question},
            ],
            model=settings.GROQ_MODEL,
            temperature=0.3,   # Low temp → more factual, less creative
            max_tokens=1024,   # Keep response concise
            top_p=0.9,
        )

        answer = chat_completion.choices[0].message.content
        usage = chat_completion.usage
        print(
            f"[CHATBOT] Groq response: {len(answer)} chars, "
            f"tokens used: prompt={usage.prompt_tokens}, "
            f"completion={usage.completion_tokens}"
        )

        return answer

    except Exception as e:
        print(f"[CHATBOT] Groq API failed: {e}")
        return _get_fallback_answer(user_question)


def _get_fallback_answer(question: str) -> str:
    """Generate a template-based answer when Groq is unavailable.

    This ensures the chatbot ALWAYS returns something useful,
    even if the LLM is down or rate-limited.
    """
    # Try to get RAG chunks for a knowledge-based fallback
    chunks = vector_store.search(question, top_k=3)

    if chunks:
        context = " ".join(chunks[:2])
        return (
            f"Based on CPCB guidelines: {context[:500]}. "
            f"Note: This is an automated response — our AI assistant is "
            f"temporarily unavailable. Please check CPCB's AQI dashboard "
            f"for real-time data."
        )

    return (
        "I'm currently unable to process your question due to a temporary "
        "service issue. Please check the CPCB AQI dashboard at "
        "https://app.cpcbccr.com for real-time air quality data, or try "
        "again in a few minutes."
    )


# ===========================================================================
# Recommendations builder (Issue #7 — deterministic, not LLM-generated)
# ===========================================================================

def _build_recommendations(
    aqi_category: str,
    user_category: str,
    health_risk: dict,
) -> list[str]:
    """Build actionable recommendations based on AQI and user category.

    These are returned directly in the API response — the LLM does NOT
    generate these. This ensures consistent, evidence-based advice.

    Args:
        aqi_category: CPCB category string (good/satisfactory/moderate/...).
        user_category: User vulnerability type.
        health_risk: Output from calculate_risk().

    Returns:
        List of recommendation strings.
    """
    recs = []

    # Base recommendation from health engine
    recs.append(health_risk["recommendation"])

    # Category-specific recommendations
    CATEGORY_RECS = {
        "good": [
            "Air quality is good — enjoy outdoor activities",
            "Good day for exercise and outdoor sports",
        ],
        "satisfactory": [
            "Air quality is acceptable for most people",
            "Sensitive individuals should monitor symptoms",
        ],
        "moderate": [
            "Reduce prolonged outdoor exertion",
            "Keep windows closed during peak traffic hours",
            "Consider wearing a mask if outdoors for extended periods",
        ],
        "poor": [
            "Avoid prolonged outdoor activity",
            "Use N95/KN95 mask if going outdoors",
            "Use air purifier indoors",
            "Keep windows and doors closed",
        ],
        "very_poor": [
            "Avoid all outdoor physical activity",
            "Use N95/KN95 mask for any outdoor exposure",
            "Run air purifier continuously indoors",
            "Consider working from home if possible",
        ],
        "severe": [
            "Avoid outdoor activity completely",
            "Use N95/KN95 mask if outdoors is unavoidable",
            "Seek medical attention if breathing difficulty occurs",
            "Keep all windows sealed; use air purifier",
            "Limit travel — use AC on recirculate mode in vehicles",
        ],
    }

    aqi_recs = CATEGORY_RECS.get(aqi_category, CATEGORY_RECS["moderate"])
    recs.extend(aqi_recs)

    # Vulnerability-specific additions
    VULN_RECS = {
        "child": ["Keep children indoors during school breaks", "Avoid outdoor PE classes"],
        "elderly": ["Monitor blood pressure and oxygen levels", "Avoid morning walks when AQI is high"],
        "asthma": ["Keep rescue inhaler accessible", "Consider preventive medication if AQI > 200"],
        "pregnant_woman": ["Avoid areas with heavy traffic", "Consult doctor if experiencing discomfort"],
        "outdoor_worker": ["Take breaks in sheltered/indoor areas every hour", "Stay hydrated"],
    }

    if user_category in VULN_RECS:
        recs.extend(VULN_RECS[user_category])

    # De-duplicate while preserving order
    seen = set()
    unique_recs = []
    for r in recs:
        if r not in seen:
            seen.add(r)
            unique_recs.append(r)

    return unique_recs
