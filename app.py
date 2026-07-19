import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

import database
import real_forecast as forecast_mod
import real_health_risk as health_mod
import real_attribution as attribution_mod
import chatbot as chatbot_mod
from utils import get_demo_user_id, status_badge, hazard_emoji

st.set_page_config(page_title="ClearTrace - AQI Intelligence Platform", page_icon="🌫️", layout="wide")
database.init_db()

STATIONS = forecast_mod.list_stations()  # returns [(name, lat, lon), ...] for all 38 real stations
LOCATIONS = [name for name, lat, lon in STATIONS]
STATION_META = {name: {"lat": lat, "lon": lon} for name, lat, lon in STATIONS}


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🌫️ ClearTrace")
st.sidebar.caption("Closed-loop AQI intelligence for Delhi")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Citizen Dashboard", "📸 Report a Source", "✅ Verify Reports", "🗺️ Admin Dashboard", "💬 AQI Chatbot"],
)
location = st.sidebar.selectbox("Location", LOCATIONS, key="global_location")
user_id = get_demo_user_id()
display_name = st.sidebar.text_input("Your name (required)", value=st.session_state.get("display_name", ""))
st.session_state["display_name"] = display_name.strip()
name_missing = not display_name.strip()
if name_missing:
    st.sidebar.warning("Enter your name to report or vote.")

loc_meta = STATION_META[location]


# ---------------------------------------------------------------------------
# 1. Citizen Dashboard — forecast + personalized health advisory
# ---------------------------------------------------------------------------
if page == "🏠 Citizen Dashboard":
    st.title("Citizen Dashboard")
    st.caption(f"📍 {location}, Delhi")

    fc = forecast_mod.generate_forecast(location)
    summary = forecast_mod.forecast_summary(fc)

    col1, col2, col3 = st.columns(3)
    col1.metric("Next 24h avg AQI", summary["avg_aqi"], summary["category"])
    col2.metric("Peak AQI (24h window)", summary["peak_aqi"])
    col3.metric("Peak time", summary["peak_time"][11:16])

    df = pd.DataFrame(fc["hourly"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["hour_offset"], y=df["aqi"], mode="lines", name="AQI", line=dict(color="#d97706")))
    fig.update_layout(
        title="24-hour AQI forecast",
        xaxis_title="Hours from now",
        yaxis_title="AQI",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, width='stretch')

    st.divider()
    st.subheader("Personalized health-risk guidance")
    hc1, hc2 = st.columns(2)
    with hc1:
        duration = st.slider("Time you plan to spend there (hours)", 0.5, 12.0, 2.0, 0.5)
        category = st.selectbox("You are a...", list(health_mod.VULNERABILITY_MULTIPLIER.keys()))
    with hc2:
        risk = health_mod.compute_health_risk(summary["avg_aqi"], duration, category)
        st.markdown(
            f"""
            <div style="border-radius:12px;padding:18px;background:{risk['color_hex']}22;
                        border:1px solid {risk['color_hex']};">
                <h3 style="margin:0;color:{risk['color_hex']};">
                    {hazard_emoji(risk['hazard_level'])} {risk['hazard_level']} hazard — score {risk['score']}
                </h3>
                <p style="margin-top:6px;">{risk['headline']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for rec in risk["recommendations"]:
            st.markdown(f"- {rec}")
        database.log_health_risk(location, summary["avg_aqi"], duration, category, risk["hazard_level"], "; ".join(risk["recommendations"]))

    st.divider()
    st.subheader("Likely pollution sources here")
    attr = attribution_mod.get_attribution(location, loc_meta["lat"], loc_meta["lon"])
    bdf = pd.DataFrame(attr["breakdown"])
    fig2 = go.Figure(go.Bar(x=bdf["share"] * 100, y=bdf["source"], orientation="h", marker_color="#0ea5e9"))
    fig2.update_layout(height=280, xaxis_title="Share (%)", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, width='stretch')
    for e in attr["evidence"]:
        st.caption(f"ℹ️ {e}")


# ---------------------------------------------------------------------------
# 2. Report a Source — citizen upload
# ---------------------------------------------------------------------------
elif page == "📸 Report a Source":
    st.title("Report a pollution source")
    st.caption("Photo + location + description. Other users will verify it before it enters the AI pipeline.")

    with st.form("report_form", clear_on_submit=True):
        img = st.file_uploader("Photo of the source", type=["jpg", "jpeg", "png"])
        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input("Latitude", value=loc_meta["lat"], format="%.5f")
        with c2:
            lon = st.number_input("Longitude", value=loc_meta["lon"], format="%.5f")
        category_guess = st.selectbox("Source category", database.SOURCE_CATEGORIES)
        description = st.text_area("Description", placeholder="e.g. Open waste burning behind the market, visible black smoke")
        submitted = st.form_submit_button("Submit report")

        if submitted:
            if name_missing:
                st.error("Please enter your name in the sidebar before submitting a report.")
            else:
                image_bytes = img.read() if img is not None else None
                ext = img.name.split(".")[-1] if img is not None else "jpg"
                result = database.create_report(lat, lon, description, category_guess, user_id, image_bytes, ext,
                                                  reporter_name=display_name)
                if result["status"] == "rejected":
                    st.error(result["reason"])
                else:
                    st.success(f"Report submitted by {display_name}! ID `{result['report_id']}`. "
                               f"It needs {database.VERIFY_UPVOTES_NEEDED} upvotes to be verified.")

    st.divider()
    st.subheader("Your recent reports nearby")
    nearby = database.get_nearby_reports(loc_meta["lat"], loc_meta["lon"], radius_m=5000)
    if not nearby:
        st.info("No reports near this location yet — be the first!")
    for r in nearby[:10]:
        with st.container(border=True):
            cols = st.columns([1, 3])
            if r["image_path"]:
                cols[0].image(r["image_path"], width='stretch')
            else:
                cols[0].caption("No photo")
            cols[1].markdown(f"**{r['category_guess']}** — {status_badge(r['status'])}")
            cols[1].write(r["description"])
            cols[1].caption(f"{r['distance_m']} m away · {r['created_at'][:16]}")


# ---------------------------------------------------------------------------
# 3. Verify Reports — upvote/verification queue
# ---------------------------------------------------------------------------
elif page == "✅ Verify Reports":
    st.title("Verification queue")
    if "vote_message" in st.session_state:
        kind, msg = st.session_state.pop("vote_message")
        getattr(st, kind)(msg)
    st.caption(f"A report becomes verified at {database.VERIFY_UPVOTES_NEEDED} upvotes (with duplicate check).")

    pending = [r for r in database.get_nearby_reports(loc_meta["lat"], loc_meta["lon"], radius_m=8000) if r["status"] == "pending"]
    if not pending:
        st.info("No pending reports near this location right now.")
    for r in pending:
        with st.container(border=True):
            cols = st.columns([1, 3, 1])
            if r["image_path"]:
                cols[0].image(r["image_path"], width='stretch')
            cols[1].markdown(f"**{r['category_guess']}**")
            cols[1].write(r["description"])
            cols[1].caption(f"{r['distance_m']} m away · reported {r['created_at'][:16]}")
            if cols[2].button("👍 Verify / Upvote", key=f"vote-{r['report_id']}", disabled=name_missing):
                result = database.vote_report(r["report_id"], user_id, voter_name=display_name)
                if result["status"] == "already_voted":
                    st.session_state["vote_message"] = ("warning", "You already voted on this report.")
                elif result["status"] == "verified":
                    st.session_state["vote_message"] = ("success", "Threshold reached — report is now VERIFIED and feeds the AI pipeline!")
                elif result["status"] == "duplicate_rejected":
                    st.session_state["vote_message"] = ("warning", "Rejected as a likely duplicate of an existing verified report.")
                else:
                    st.session_state["vote_message"] = ("info", f"Upvoted ({result['upvotes']}/{database.VERIFY_UPVOTES_NEEDED}, confidence {result['confidence_score']})")
                st.rerun()

    st.divider()
    st.subheader("Recently verified")
    verified = database.get_verified_reports()
    if not verified:
        st.caption("Nothing verified yet.")
    for v in verified[:5]:
        st.markdown(f"🟢 **{v['source_type']}** — {v['description']} (confidence {v['confidence_score']})")


# ---------------------------------------------------------------------------
# 4. Admin Dashboard — map with forecast, attribution, verified reports
# ---------------------------------------------------------------------------
elif page == "🗺️ Admin Dashboard":
    st.title("Admin Dashboard")
    st.caption("Forecast + attribution + verified citizen signals, all in one map.")

    m = folium.Map(location=[28.6139, 77.2090], zoom_start=11, tiles="CartoDB positron")

    for name, lat, lon in STATIONS:
        fc = forecast_mod.generate_forecast(name)
        if fc.get("error") or not fc["hourly"]:
            continue
        summary = forecast_mod.forecast_summary(fc)
        color = {"Good": "green", "Satisfactory": "green", "Moderate": "orange",
                 "Poor": "orange", "Very Poor": "red", "Severe": "darkred"}.get(summary["category"], "gray")
        folium.CircleMarker(
            location=[lat, lon],
            radius=10,
            popup=f"{name}: AQI {summary['avg_aqi']} ({summary['category']})",
            tooltip=name,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
        ).add_to(m)

    for v in database.get_verified_reports():
        folium.Marker(
            location=[v["lat"], v["lon"]],
            popup=f"Verified: {v['source_type']} (confidence {v['confidence_score']})",
            icon=folium.Icon(color="blue", icon="camera"),
        ).add_to(m)

    st_folium(m, width=None, height=500)

    st.divider()
    st.subheader(f"Detail: {location}")
    fc = forecast_mod.generate_forecast(location)
    summary = forecast_mod.forecast_summary(fc)
    attr = attribution_mod.get_attribution(location, loc_meta["lat"], loc_meta["lon"])
    risk = health_mod.compute_health_risk(summary["avg_aqi"], 2, "Normal adult")

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg AQI (24h)", summary["avg_aqi"], summary["category"])
    c2.metric("Top attributed source", attr["top_source"])
    c3.metric("2h exposure hazard (adult)", f"{hazard_emoji(risk['hazard_level'])} {risk['hazard_level']}")

    st.markdown("**Recommended action:** " + risk["headline"])
    for e in attr["evidence"]:
        st.caption(f"ℹ️ {e}")


# ---------------------------------------------------------------------------
# 5. AQI Chatbot
# ---------------------------------------------------------------------------
elif page == "💬 AQI Chatbot":
    st.title("AQI Chatbot")
    st.caption("Grounded on forecast + attribution + verified citizen reports + advisory docs.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(text)

    sample_qs = [
        f"Is it safe to jog in {location} tomorrow morning?",
        f"Why is AQI high near {location}?",
        "What should a school do if AQI is severe?",
        "Which reports near me are verified?",
    ]
    st.caption("Try: " + " · ".join(f"`{q}`" for q in sample_qs))

    q = st.chat_input("Ask about AQI, sources, or health guidance...")
    if q:
        st.session_state.chat_history.append(("user", q))
        with st.chat_message("user"):
            st.write(q)
        resp = chatbot_mod.answer_query(q, location)
        st.session_state.chat_history.append(("assistant", resp["answer"]))
        with st.chat_message("assistant"):
            st.write(resp["answer"])
            st.caption(f"Sources used: {resp['sources_used']}")
