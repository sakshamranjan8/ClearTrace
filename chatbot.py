"""
chatbot.py
Streamlit chatbot backend — tries the Module 3 RAG API first, then falls
back to the original Claude / template pipeline.

Priority order:
  1. Module 3 RAG API  (POST /chat/query on the RAG FastAPI server)
     → Groq LLM + FAISS knowledge base + forecast + attribution + reports
  2. Claude LLM        (if ANTHROPIC_API_KEY is set)
     → Grounded on forecast + attribution + verified reports + advisory docs
  3. Template fallback  (always available, no API key needed)

The Streamlit frontend calls ``answer_query(question, location, lat, lon)``.
If lat/lon are provided, the RAG API gets exact coordinates.
If only location is provided, coordinates are resolved from the forecast.
"""

import os

import requests

import database
import real_forecast as forecast_mod
import real_health_risk as health_mod
import real_attribution as attribution_mod
from chatbot_data.advisory_corpus import keyword_search

# ---------------------------------------------------------------------------
# Module 3 RAG API configuration
# ---------------------------------------------------------------------------
RAG_API_URL = os.getenv("CLEARTRACE_RAG_URL", "http://127.0.0.1:8001")
RAG_API_TIMEOUT = float(os.getenv("CLEARTRACE_RAG_TIMEOUT", "30"))

SYSTEM_PROMPT = (
    "You are the ClearTrace AQI assistant for Delhi. Answer ONLY using the "
    "CONTEXT provided (forecast, attribution, verified citizen reports, advisory docs). "
    "Be practical and concise. Do not make medical claims or promise cures. "
    "If symptoms are mentioned, suggest consulting a doctor. If the context does not "
    "cover something, say so plainly instead of guessing."
)


# ---------------------------------------------------------------------------
# Module 3 RAG API caller (new)
# ---------------------------------------------------------------------------

def _rag_answer(question, lat, lon, user_category="adult"):
    """Try to get an answer from the Module 3 RAG API.

    Returns the full response dict on success, or None if the server
    is unreachable or returns an error.
    """
    if lat is None or lon is None:
        return None

    try:
        response = requests.post(
            f"{RAG_API_URL}/chat/query",
            json={
                "question": question,
                "lat": lat,
                "lon": lon,
                "user_category": user_category,
            },
            timeout=RAG_API_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        # Map the RAG API response to the shape the frontend expects
        sources_str = ", ".join(data.get("sources_used", []))
        return {
            "answer": data.get("answer", ""),
            "sources_used": sources_str + ",rag_api",
            "hazard_level": data.get("hazard_level", ""),
            "recommendations": data.get("recommendations", []),
        }
    except requests.ConnectionError:
        # RAG server not running — silent fallback
        return None
    except requests.Timeout:
        print("[CHATBOT] RAG API timed out — falling back")
        return None
    except Exception as exc:
        print(f"[CHATBOT] RAG API error: {exc} — falling back")
        return None


# ---------------------------------------------------------------------------
# Original context builder (unchanged)
# ---------------------------------------------------------------------------

def _build_context(location):
    fc = forecast_mod.generate_forecast(location)
    summary = forecast_mod.forecast_summary(fc)
    loc_meta = fc["lat"], fc["lon"]
    attr = attribution_mod.get_attribution(location, *loc_meta)
    nearby_reports = database.get_nearby_reports(*loc_meta, radius_m=2000)
    verified_nearby = [r for r in nearby_reports if r["status"] == "verified"]
    docs = keyword_search(location)

    # Build attribution breakdown string safely
    breakdown = ", ".join(
        f"{b['source']}={int(b['share'] * 100)}%"
        for b in attr["breakdown"]
    )

    context_lines = [
        (
            f"Forecast for {location}: avg next-24h AQI "
            f"{summary['avg_aqi']} ({summary['category']}), "
            f"peak {summary['peak_aqi']} at {summary['peak_time']}."
        ),
        (
            f"Top attributed source: {attr['top_source']} "
            f"(breakdown: {breakdown})."
        ),
    ]

    if verified_nearby:
        cats = ", ".join(
            sorted(set(r["category_guess"] or "unspecified" for r in verified_nearby))
        )
        context_lines.append(
            f"{len(verified_nearby)} verified citizen report(s) nearby, "
            f"categories: {cats}."
        )
    else:
        context_lines.append("No verified citizen reports nearby yet.")

    for d in docs:
        context_lines.append(f"[{d['title']}] {d['text']}")

    return {
        "summary": summary,
        "attribution": attr,
        "verified_nearby": verified_nearby,
        "docs": docs,
        "context_text": "\n".join(context_lines),
        "lat": loc_meta[0],
        "lon": loc_meta[1],
    }


def _template_answer(question, ctx):
    s = ctx["summary"]
    a = ctx["attribution"]
    q_lower = question.lower()

    lines = []
    if "safe" in q_lower and ("jog" in q_lower or "run" in q_lower or "outdoor" in q_lower):
        risk = health_mod.compute_health_risk(s["avg_aqi"], 1, "Normal adult")
        lines.append(
            f"Average forecast AQI is {s['avg_aqi']} ({s['category']}), giving a {risk['hazard_level']} "
            f"hazard level for a ~1 hour outdoor session."
        )
        lines.append(risk['headline'])
    elif "why" in q_lower or "source" in q_lower or "polluted" in q_lower:
        top = a["breakdown"][0]
        lines.append(
            f"The largest attributed source right now is {top['source']} (~{int(top['share']*100)}% share)."
        )
        lines.append(a["evidence"][0])
    elif "school" in q_lower:
        lines.append(
            f"At {s['category']} AQI ({s['avg_aqi']} avg), schools should move assemblies and PE indoors "
            "and keep windows closed until conditions improve."
        )
    elif "verified" in q_lower and "report" in q_lower:
        if ctx["verified_nearby"]:
            lines.append(f"{len(ctx['verified_nearby'])} verified report(s) nearby right now.")
            for r in ctx["verified_nearby"][:3]:
                lines.append(f"- {r['category_guess'] or 'unspecified source'}: {r['description']}")
        else:
            lines.append("No verified citizen reports near this location yet.")
    else:
        lines.append(f"Forecast AQI is {s['avg_aqi']} ({s['category']}); top source is {a['top_source']}.")

    lines.append("This is general guidance, not medical advice — please consult a doctor for symptoms.")
    return " ".join(lines)


def _claude_answer(question, ctx):
    try:
        import anthropic
    except ImportError:
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"CONTEXT:\n{ctx['context_text']}\n\nQUESTION: {question}",
            }],
        )
        return "".join(b.text for b in msg.content if hasattr(b, "text"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main entry point — used by Streamlit frontend
# ---------------------------------------------------------------------------

def answer_query(question, location, lat=None, lon=None, user_category="adult"):
    """Answer a user question using the best available backend.

    Priority:
      1. Module 3 RAG API (if running)
      2. Claude LLM (if ANTHROPIC_API_KEY is set)
      3. Template fallback

    Args:
        question:       The user's natural-language question.
        location:       Station name string (for context/fallback).
        lat, lon:       Optional coordinates for Module 3 API.
        user_category:  Vulnerability category (default: "adult").

    Returns:
        Dict with 'answer' and 'sources_used' keys.
    """
    # --- Step 0: If lat/lon are missing, resolve from forecast ---
    if lat is None or lon is None:
        try:
            fc = forecast_mod.generate_forecast(location)
            lat = fc.get("lat")
            lon = fc.get("lon")
        except Exception:
            pass  # Continue without coordinates

    # --- Step 1: Try Module 3 RAG API ---
    rag_result = _rag_answer(question, lat, lon, user_category)
    if rag_result is not None:
        database.log_chat(question, rag_result["answer"], location, rag_result["sources_used"])
        return rag_result

    # --- Step 2 & 3: Fall back to Claude / template ---
    ctx = _build_context(location)
    answer = _claude_answer(question, ctx)
    used_llm = answer is not None
    if answer is None:
        answer = _template_answer(question, ctx)

    sources_used = "forecast,attribution,verified_reports,advisory_docs" + (",claude_llm" if used_llm else "")
    database.log_chat(question, answer, location, sources_used)
    return {"answer": answer, "sources_used": sources_used, "context": ctx}
