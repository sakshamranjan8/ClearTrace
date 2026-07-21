"""
chatbot.py
Stands in for Member 3's POST /chat/query endpoint (Section 7C).

Always grounds on four sources, per the plan:
  1. current/forecast AQI      -> forecast.py
  2. attribution output        -> attribution.py
  3. verified citizen reports  -> database.get_verified_reports / get_citizen_features
  4. advisory/regulation docs  -> data/advisory_corpus.py

If ANTHROPIC_API_KEY is set in the environment, the grounded context is
passed to a real Claude call for a fluent answer. Otherwise it falls back to
a template-based answer built from the same grounded context, so the demo
works with zero external dependencies / no API key required.
"""

import os
import database
import real_forecast as forecast_mod
import real_health_risk as health_mod
import real_attribution as attribution_mod
from chatbot_data.advisory_corpus import keyword_search

SYSTEM_PROMPT = (
    "You are the ClearTrace AQI assistant for Delhi. Answer ONLY using the "
    "CONTEXT provided (forecast, attribution, verified citizen reports, advisory docs). "
    "Be practical and concise. Do not make medical claims or promise cures. "
    "If symptoms are mentioned, suggest consulting a doctor. If the context does not "
    "cover something, say so plainly instead of guessing."
)


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


def answer_query(question, location):
    ctx = _build_context(location)
    answer = _claude_answer(question, ctx)
    used_llm = answer is not None
    if answer is None:
        answer = _template_answer(question, ctx)

    sources_used = "forecast,attribution,verified_reports,advisory_docs" + (",claude_llm" if used_llm else "")
    database.log_chat(question, answer, location, sources_used)
    return {"answer": answer, "sources_used": sources_used, "context": ctx}
