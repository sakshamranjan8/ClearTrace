import uuid
import streamlit as st


def get_demo_user_id():
    """Simple per-browser-session demo user id (no real auth for the hackathon MVP)."""
    if "demo_user_id" not in st.session_state:
        st.session_state.demo_user_id = "demo-" + str(uuid.uuid4())[:6]
    return st.session_state.demo_user_id


def status_badge(status):
    colors = {"pending": "🟡 Pending", "verified": "🟢 Verified", "duplicate": "🔴 Duplicate (rejected)"}
    return colors.get(status, status)


def hazard_emoji(level):
    return {"Low": "🟢", "Moderate": "🟡", "High": "🟠", "Severe": "🔴"}.get(level, "⚪")
