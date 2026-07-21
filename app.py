from html import escape

import pandas as pd
import plotly.graph_objects as go
import folium
import streamlit as st
from streamlit_folium import st_folium

import database
import real_attribution as attribution_mod
import real_forecast as forecast_mod
import real_health_risk as health_mod
from utils import get_demo_user_id, status_badge

try:
    from streamlit_geolocation import streamlit_geolocation
except ImportError:
    streamlit_geolocation = None


st.set_page_config(
    page_title="ClearTrace - AQI Intelligence Platform",
    page_icon="🌫️",
    layout="wide",
)
database.init_db()

st.markdown(
    """
    <style>
      .stApp { background: #f7f9fc; color: #14213d; }
      [data-testid="stSidebar"] { background: #eef3f8; border-right: 1px solid #dbe5ef; }
      [data-testid="stMetric"] {
        background: white; border: 1px solid #e2e8f0; border-radius: 16px;
        padding: 16px 18px; box-shadow: 0 6px 24px rgba(15, 23, 42, .04);
      }
      [data-testid="stMetricValue"] { color: #0f172a; }
      .ct-eyebrow {
        color: #0f766e; font-size: .78rem; font-weight: 800; letter-spacing: .12em;
        text-transform: uppercase; margin-bottom: .35rem;
      }
      .ct-hero {
        background: linear-gradient(120deg, #073b4c 0%, #0f766e 60%, #14b8a6 100%);
        color: white; padding: 24px 28px; border-radius: 22px; margin-bottom: 20px;
        box-shadow: 0 14px 36px rgba(15, 118, 110, .18);
      }
      .ct-hero h1 { color: white; margin: 0 0 6px; font-size: 2rem; }
      .ct-hero p { color: #d8fffa; margin: 0; }
      .ct-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 18px;
        padding: 18px 20px; height: 100%; box-shadow: 0 6px 24px rgba(15, 23, 42, .045);
      }
      .ct-card-title { font-size: 1.04rem; font-weight: 750; color: #0f172a; margin: 0 0 8px; }
      .ct-muted { color: #64748b; font-size: .9rem; }
      .ct-pill {
        display: inline-block; padding: 4px 10px; border-radius: 999px;
        font-size: .78rem; font-weight: 750; margin-right: 6px;
      }
      .ct-advisory {
        background: white; border-radius: 18px; padding: 22px;
        border: 1px solid #e2e8f0; border-left: 6px solid var(--accent);
        box-shadow: 0 8px 26px rgba(15, 23, 42, .06);
      }
      .ct-advisory h3 { margin: 5px 0 8px; color: #0f172a; }
      .ct-big { font-size: 1.55rem; font-weight: 820; color: #0f172a; }
      .ct-source {
        background: white; border: 1px solid #e2e8f0; border-radius: 16px;
        padding: 16px 18px; min-height: 172px; margin-bottom: 10px;
      }
      .ct-source-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
      .ct-source-name { font-weight: 780; color: #0f172a; font-size: 1rem; }
      .ct-signal { font-size: .76rem; font-weight: 800; padding: 4px 9px; border-radius: 999px; }
      .ct-evidence { color: #475569; font-size: .86rem; margin-top: 8px; line-height: 1.45; }
      .ct-note {
        background: #eff6ff; border: 1px solid #bfdbfe; color: #1e3a8a;
        border-radius: 12px; padding: 12px 14px; font-size: .88rem;
      }
      div[data-testid="stPlotlyChart"] { background: white; border: 1px solid #e2e8f0; border-radius: 18px; padding: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60, show_spinner=False)
def get_live_forecast(location_name, latitude, longitude):
    return forecast_mod.generate_forecast(
        location_name,
        latitude=latitude,
        longitude=longitude,
    )


def forecast_or_stop(location_name, latitude, longitude):
    with st.spinner("Loading the live AQI forecast..."):
        forecast = get_live_forecast(location_name, latitude, longitude)

    if forecast.get("error") or not forecast.get("hourly"):
        st.error(
            forecast.get("error_message")
            or "No forecast is currently available for this location."
        )
        st.caption(
            "Make sure FastAPI is running and CLEARTRACE_API_URL points to it."
        )
        st.stop()

    if forecast.get("is_stale"):
        st.warning("The forecast cache is stale. Run the hourly update pipeline.")

    return forecast


def format_forecast_time(timestamp):
    if not timestamp:
        return "Unavailable"
    return pd.to_datetime(timestamp).strftime("%d %b, %I:%M %p")


def render_advisory_card(advisory):
    start = format_forecast_time(advisory["start"])
    end = format_forecast_time(advisory["end"])
    st.markdown(
        f"""
        <div class="ct-advisory" style="--accent:{advisory['action_color']};">
          <div class="ct-eyebrow">YOUR SELECTED WINDOW · {start} — {end}</div>
          <span class="ct-pill" style="background:{advisory['category_color']}20;color:{advisory['category_color']};">
            Peak category · {escape(advisory['category'])}
          </span>
          <span class="ct-pill" style="background:{advisory['action_color']}18;color:{advisory['action_color']};">
            {escape(advisory['action_level'])}
          </span>
          <h3>{escape(advisory['headline'])}</h3>
          <p style="margin:.2rem 0 1rem;color:#475569;">{escape(advisory['health_message'])}</p>
          <div style="display:flex;gap:28px;flex-wrap:wrap;">
            <div><span class="ct-muted">Window mean</span><br><span class="ct-big">{advisory['mean_aqi']} AQI</span></div>
            <div><span class="ct-muted">Window peak</span><br><span class="ct-big">{advisory['peak_aqi']} AQI</span></div>
            <div><span class="ct-muted">Forecast hours used</span><br><span class="ct-big">{advisory['hours_used']}</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_source_card(indicator):
    colors = {
        "High": ("#fee2e2", "#b91c1c"),
        "Medium": ("#fef3c7", "#b45309"),
        "Low": ("#dcfce7", "#15803d"),
        "Unverified": ("#e2e8f0", "#475569"),
    }
    background, foreground = colors.get(
        indicator["strength"], ("#e2e8f0", "#475569")
    )
    evidence = "".join(
        f"<div>• {escape(item)}</div>" for item in indicator["evidence"]
    )
    st.markdown(
        f"""
        <div class="ct-source">
          <div class="ct-source-head">
            <div class="ct-source-name">{indicator['icon']} {escape(indicator['label'])}</div>
            <span class="ct-signal" style="background:{background};color:{foreground};">
              {escape(indicator['strength'])} signal
            </span>
          </div>
          <div class="ct-evidence">{evidence}</div>
          <div class="ct-muted" style="margin-top:10px;">Data: {escape(indicator['confidence_label'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


try:
    STATIONS = forecast_mod.list_stations()
except (FileNotFoundError, KeyError, ValueError) as error:
    st.error(f"Could not load the monitoring-station list: {error}")
    st.stop()

LOCATIONS = [name for name, _lat, _lon in STATIONS]
STATION_META = {
    name: {"lat": float(lat), "lon": float(lon)}
    for name, lat, lon in STATIONS
}


# ---------------------------------------------------------------------------
# Sidebar navigation and location
# ---------------------------------------------------------------------------
st.sidebar.title("🌫️ ClearTrace")
st.sidebar.caption("Closed-loop AQI intelligence for Delhi")
page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Citizen Dashboard",
        "📸 Report a Source",
        "✅ Verify Reports",
        "🗺️ Admin Dashboard",
        "💬 AQI Chatbot",
    ],
)

location_mode = st.sidebar.radio(
    "Forecast location",
    [
        "Monitoring station",
        "Use my current location",
        "Enter coordinates manually",
    ],
    key="location_mode",
)

# Keep a valid station name available as a fallback for modules that still use
# station-linked data (notably source attribution and the existing chatbot).
location = st.session_state.get("global_location", LOCATIONS[0])
if location not in STATION_META:
    location = LOCATIONS[0]

if location_mode == "Monitoring station":
    location = st.sidebar.selectbox(
        "Monitoring station",
        LOCATIONS,
        index=LOCATIONS.index(location),
        key="global_location",
    )
    loc_meta = STATION_META[location]
    effective_lat = loc_meta["lat"]
    effective_lon = loc_meta["lon"]
    location_label = location

elif location_mode == "Use my current location":
    if streamlit_geolocation is None:
        st.sidebar.error(
            "Location detection is unavailable. Install streamlit-geolocation."
        )
    else:
        with st.sidebar:
            location_data = streamlit_geolocation()

        if isinstance(location_data, dict):
            detected_lat = location_data.get("latitude")
            detected_lon = location_data.get("longitude")
            if detected_lat is not None and detected_lon is not None:
                st.session_state["user_latitude"] = float(detected_lat)
                st.session_state["user_longitude"] = float(detected_lon)
                if location_data.get("accuracy") is not None:
                    st.session_state["location_accuracy"] = float(
                        location_data["accuracy"]
                    )

    has_device_coordinates = (
        "user_latitude" in st.session_state
        and "user_longitude" in st.session_state
    )

    if has_device_coordinates:
        effective_lat = st.session_state["user_latitude"]
        effective_lon = st.session_state["user_longitude"]
        location_label = "Your current location"
        st.sidebar.success(
            f"Location detected: {effective_lat:.5f}, {effective_lon:.5f}"
        )
        if "location_accuracy" in st.session_state:
            st.sidebar.caption(
                f"Estimated accuracy: "
                f"±{st.session_state['location_accuracy']:.0f} m"
            )
    else:
        # This fallback keeps the app usable before the browser returns its
        # asynchronous geolocation result or when permission is denied.
        loc_meta = STATION_META[location]
        effective_lat = loc_meta["lat"]
        effective_lon = loc_meta["lon"]
        location_label = location
        st.sidebar.info(
            "Click the location button and allow browser permission. "
            "A monitoring station is used until coordinates arrive."
        )

else:
    manual_lat = st.sidebar.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=float(st.session_state.get("manual_latitude", 28.6139)),
        step=0.0001,
        format="%.6f",
        key="manual_latitude",
    )
    manual_lon = st.sidebar.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=float(st.session_state.get("manual_longitude", 77.2090)),
        step=0.0001,
        format="%.6f",
        key="manual_longitude",
    )
    effective_lat = float(manual_lat)
    effective_lon = float(manual_lon)
    location_label = "Manually entered coordinates"
    st.sidebar.caption(
        f"Using {effective_lat:.6f}, {effective_lon:.6f}"
    )

user_id = get_demo_user_id()
display_name = st.sidebar.text_input(
    "Your name (required)",
    value=st.session_state.get("display_name", ""),
)
st.session_state["display_name"] = display_name.strip()
name_missing = not display_name.strip()
if name_missing:
    st.sidebar.warning("Enter your name to report or vote.")


# ---------------------------------------------------------------------------
# 1. Citizen Dashboard — live forecast + personalized health advisory
# ---------------------------------------------------------------------------
if page == "🏠 Citizen Dashboard":
    st.markdown(
        f"""
        <div class="ct-hero">
          <div style="font-size:.78rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#99f6e4;">
            DELHI AIR INTELLIGENCE
          </div>
          <h1>Your next 24 hours, made clearer.</h1>
          <p>📍 {escape(location_label)} · {effective_lat:.5f}, {effective_lon:.5f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fc = forecast_or_stop(location_label, effective_lat, effective_lon)
    summary = forecast_mod.forecast_summary(fc)
    hours_available = len(fc["hourly"])

    col1, col2, col3 = st.columns(3)
    col1.metric(
        f"{hours_available}h average",
        f"{summary['avg_aqi']:.0f} AQI",
        summary["category"],
    )
    col2.metric(
        "Forecast peak",
        f"{summary['peak_aqi']:.0f} AQI",
        format_forecast_time(summary["peak_time"]),
    )
    col3.metric("Peak time", format_forecast_time(summary["peak_time"]))

    requested_hours = int(fc.get("requested_forecast_hours") or 24)
    if hours_available < requested_hours:
        st.warning(
            f"The API currently has {hours_available} future hourly points, "
            f"not {requested_hours}. The graph shows only genuine future forecasts."
        )

    df = pd.DataFrame(fc["hourly"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    point_colors = [
        health_mod.aqi_band(value)["color"] for value in df["aqi"]
    ]
    fig = go.Figure()
    for lower, upper, color in [
        (0, 50, "#22c55e"),
        (50, 100, "#84cc16"),
        (100, 200, "#f59e0b"),
        (200, 300, "#f97316"),
        (300, 400, "#ef4444"),
        (400, max(500, float(df["aqi"].max()) + 20), "#7f1d1d"),
    ]:
        fig.add_hrect(y0=lower, y1=upper, fillcolor=color, opacity=0.045, line_width=0)
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["aqi"],
            mode="lines+markers",
            name="AQI",
            line={"color": "#0f766e", "width": 3, "shape": "spline"},
            marker={"color": point_colors, "size": 9, "line": {"color": "white", "width": 1.5}},
            fill="tozeroy",
            fillcolor="rgba(20, 184, 166, 0.07)",
            hovertemplate="%{x|%d %b, %I:%M %p}<br><b>%{y:.0f} AQI</b><extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": f"Next {hours_available}-hour AQI forecast", "font": {"size": 18}},
        xaxis_title=None,
        yaxis_title="AQI",
        height=390,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        showlegend=False,
        margin={"l": 34, "r": 18, "t": 54, "b": 24},
        xaxis={"showgrid": False},
        yaxis={"gridcolor": "#e2e8f0", "rangemode": "tozero"},
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How this location estimate was produced"):
        stations = fc.get("nearest_stations") or []
        if stations:
            st.dataframe(pd.DataFrame(stations), use_container_width=True)
            nearest_distance = stations[0].get("distance_km")
            if nearest_distance is not None:
                st.caption(
                    f"The nearest contributing monitor is {float(nearest_distance):.1f} km away. "
                    "This is a blended neighbourhood estimate, not a sensor reading at your exact point."
                )
        else:
            st.caption("The API did not return station-blend metadata.")

    st.divider()
    st.markdown('<div class="ct-eyebrow">PERSONAL GUIDANCE</div>', unsafe_allow_html=True)
    st.subheader("Plan your outdoor exposure")
    st.caption(
        "Choose when you will be outside. Guidance is calculated from those exact forecast hours—not the full-day average."
    )

    control_col, result_col = st.columns([0.36, 0.64], gap="large")
    with control_col:
        start_index = st.selectbox(
            "Planned start time",
            options=list(range(hours_available)),
            format_func=lambda index: format_forecast_time(fc["hourly"][index]["timestamp"]),
        )
        max_duration = max(1, min(12, hours_available - start_index))
        duration = st.slider(
            "Time outdoors",
            min_value=1,
            max_value=max_duration,
            value=min(2, max_duration),
            step=1,
            format="%d hour(s)",
        )
        sensitivity_group = st.selectbox(
            "Who is this guidance for?",
            health_mod.SENSITIVITY_GROUPS,
        )
        activity_level = st.selectbox(
            "Planned activity",
            health_mod.ACTIVITY_LEVELS,
        )

        advisory = health_mod.build_exposure_advisory(
            fc["hourly"],
            start_index,
            duration,
            sensitivity_group,
            activity_level,
        )

    with result_col:
        render_advisory_card(advisory)
        st.markdown("#### What to do")
        for recommendation in advisory["recommendations"]:
            st.markdown(f"- {recommendation}")
        st.caption(advisory["method_note"])

        if st.button("Save this plan", type="primary"):
            database.log_health_risk(
                location_label,
                advisory["mean_aqi"],
                advisory["hours_used"],
                sensitivity_group,
                advisory["action_level"],
                "; ".join(advisory["recommendations"]),
            )
            st.success("Outdoor plan saved.")

    st.divider()
    st.markdown('<div class="ct-eyebrow">LOCATION CONTEXT</div>', unsafe_allow_html=True)
    source_title_col, radius_col = st.columns([0.72, 0.28])
    with source_title_col:
        st.subheader("Nearby pollution-source indicators")
        st.caption(
            "Mapped features around your exact coordinates. Strength describes nearby evidence—not pollution contribution."
        )
    with radius_col:
        source_radius = st.select_slider(
            "Search radius",
            options=[2.0, 3.0, 5.0, 8.0, 10.0],
            value=5.0,
            format_func=lambda value: f"{value:g} km",
        )

    try:
        source_context = attribution_mod.get_source_indicators(
            effective_lat,
            effective_lon,
            radius_km=source_radius,
        )
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Source inventory could not be loaded: {error}")
        source_context = None

    if source_context:
        s1, s2, s3 = st.columns(3)
        s1.metric("Evidence confidence", source_context["confidence"])
        s2.metric("Eligible mapped features", source_context.get("eligible_features", 0))
        s3.metric("Wind adjustment", "Not available")

        indicators = source_context["indicators"]
        if not indicators:
            st.info(
                f"No eligible mapped source indicators were found within {source_radius:g} km. "
                "This does not mean the area has no pollution sources."
            )
        else:
            source_columns = st.columns(2, gap="medium")
            for index, indicator in enumerate(indicators):
                with source_columns[index % 2]:
                    render_source_card(indicator)

        context_only = source_context.get("context_only") or []
        if context_only:
            with st.expander("Unverified mapped context (excluded from indicator ranking)"):
                for item in context_only:
                    st.write(
                        f"{item['icon']} **{item['label']}** — {item['count']} mapped feature(s); "
                        f"nearest {item['nearest_distance_km']:.2f} km. {item['note']}"
                    )

        st.markdown(
            f'<div class="ct-note"><b>How to read this:</b> {escape(source_context["disclaimer"])}</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Method: {source_context['method']} · Centre: {effective_lat:.5f}, {effective_lon:.5f} · "
            f"Radius: {source_radius:g} km"
        )


# ---------------------------------------------------------------------------
# 2. Report a Source — citizen upload
# ---------------------------------------------------------------------------
elif page == "📸 Report a Source":
    st.title("Report a pollution source")
    st.caption(
        "Photo + location + description. Other users will verify it before "
        "it enters the AI pipeline."
    )

    with st.form("report_form", clear_on_submit=True):
        img = st.file_uploader("Photo of the source", type=["jpg", "jpeg", "png"])
        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input("Latitude", value=effective_lat, format="%.5f")
        with c2:
            lon = st.number_input("Longitude", value=effective_lon, format="%.5f")
        category_guess = st.selectbox(
            "Source category",
            database.SOURCE_CATEGORIES,
        )
        description = st.text_area(
            "Description",
            placeholder=(
                "e.g. Open waste burning behind the market, visible black smoke"
            ),
        )
        submitted = st.form_submit_button("Submit report")

        if submitted:
            if name_missing:
                st.error(
                    "Please enter your name in the sidebar before submitting a report."
                )
            else:
                image_bytes = img.read() if img is not None else None
                extension = img.name.rsplit(".", 1)[-1] if img is not None else "jpg"
                result = database.create_report(
                    lat,
                    lon,
                    description,
                    category_guess,
                    user_id,
                    image_bytes,
                    extension,
                    reporter_name=display_name,
                )
                if result["status"] == "rejected":
                    st.error(result["reason"])
                else:
                    st.success(
                        f"Report submitted by {display_name}! ID "
                        f"`{result['report_id']}`. It needs "
                        f"{database.VERIFY_UPVOTES_NEEDED} upvotes to be verified."
                    )

    st.divider()
    st.subheader("Your recent reports nearby")
    nearby = database.get_nearby_reports(
        effective_lat,
        effective_lon,
        radius_m=5000,
    )
    if not nearby:
        st.info("No reports near this location yet — be the first!")
    for report in nearby[:10]:
        with st.container(border=True):
            cols = st.columns([1, 3])
            if report["image_path"]:
                cols[0].image(report["image_path"], use_container_width=True)
            else:
                cols[0].caption("No photo")
            cols[1].markdown(
                f"**{report['category_guess']}** — {status_badge(report['status'])}"
            )
            cols[1].write(report["description"])
            cols[1].caption(
                f"{report['distance_m']} m away · {report['created_at'][:16]}"
            )


# ---------------------------------------------------------------------------
# 3. Verify Reports — upvote/verification queue
# ---------------------------------------------------------------------------
elif page == "✅ Verify Reports":
    st.title("Verification queue")
    if "vote_message" in st.session_state:
        kind, message = st.session_state.pop("vote_message")
        getattr(st, kind)(message)
    st.caption(
        f"A report becomes verified at {database.VERIFY_UPVOTES_NEEDED} "
        "upvotes (with duplicate check)."
    )

    pending = [
        report
        for report in database.get_nearby_reports(
            effective_lat,
            effective_lon,
            radius_m=8000,
        )
        if report["status"] == "pending"
    ]
    if not pending:
        st.info("No pending reports near this location right now.")
    for report in pending:
        with st.container(border=True):
            cols = st.columns([1, 3, 1])
            if report["image_path"]:
                cols[0].image(report["image_path"], use_container_width=True)
            cols[1].markdown(f"**{report['category_guess']}**")
            cols[1].write(report["description"])
            cols[1].caption(
                f"{report['distance_m']} m away · "
                f"reported {report['created_at'][:16]}"
            )
            if cols[2].button(
                "👍 Verify / Upvote",
                key=f"vote-{report['report_id']}",
                disabled=name_missing,
            ):
                result = database.vote_report(
                    report["report_id"],
                    user_id,
                    voter_name=display_name,
                )
                if result["status"] == "already_voted":
                    st.session_state["vote_message"] = (
                        "warning",
                        "You already voted on this report.",
                    )
                elif result["status"] == "verified":
                    st.session_state["vote_message"] = (
                        "success",
                        "Threshold reached — report is now VERIFIED!",
                    )
                elif result["status"] == "duplicate_rejected":
                    st.session_state["vote_message"] = (
                        "warning",
                        "Rejected as a likely duplicate of a verified report.",
                    )
                else:
                    st.session_state["vote_message"] = (
                        "info",
                        f"Upvoted ({result['upvotes']}/"
                        f"{database.VERIFY_UPVOTES_NEEDED}, confidence "
                        f"{result['confidence_score']})",
                    )
                st.rerun()

    st.divider()
    st.subheader("Recently verified")
    verified = database.get_verified_reports()
    if not verified:
        st.caption("Nothing verified yet.")
    for report in verified[:5]:
        st.markdown(
            f"🟢 **{report['source_type']}** — {report['description']} "
            f"(confidence {report['confidence_score']})"
        )


# ---------------------------------------------------------------------------
# 4. Admin Dashboard — map with forecast and verified reports
# ---------------------------------------------------------------------------
elif page == "🗺️ Admin Dashboard":
    st.title("Admin Dashboard")
    st.caption("Forecast + attribution + verified citizen signals in one map.")

    map_view = folium.Map(
        location=[28.6139, 77.2090],
        zoom_start=11,
        tiles="CartoDB positron",
    )

    failed_station_count = 0
    for station_name, station_lat, station_lon in STATIONS:
        station_fc = get_live_forecast(station_name, station_lat, station_lon)
        if station_fc.get("error") or not station_fc.get("hourly"):
            failed_station_count += 1
            continue
        station_summary = forecast_mod.forecast_summary(station_fc)
        color = {
            "Good": "green",
            "Satisfactory": "green",
            "Moderate": "orange",
            "Poor": "orange",
            "Very Poor": "red",
            "Severe": "darkred",
        }.get(station_summary["category"], "gray")
        folium.CircleMarker(
            location=[station_lat, station_lon],
            radius=10,
            popup=(
                f"{station_name}: AQI {station_summary['avg_aqi']} "
                f"({station_summary['category']})"
            ),
            tooltip=station_name,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
        ).add_to(map_view)

    for report in database.get_verified_reports():
        folium.Marker(
            location=[report["lat"], report["lon"]],
            popup=(
                f"Verified: {report['source_type']} "
                f"(confidence {report['confidence_score']})"
            ),
            icon=folium.Icon(color="blue", icon="camera"),
        ).add_to(map_view)

    st_folium(map_view, width=None, height=500)
    if failed_station_count:
        st.warning(
            f"Forecasts were unavailable for {failed_station_count} station(s)."
        )

    st.divider()
    st.subheader(f"Detail: {location_label}")
    fc = forecast_or_stop(location_label, effective_lat, effective_lon)
    summary = forecast_mod.forecast_summary(fc)
    source_context = attribution_mod.get_source_indicators(
        effective_lat,
        effective_lon,
        radius_km=5.0,
    )
    advisory = health_mod.build_exposure_advisory(
        fc["hourly"],
        0,
        2,
        "General population",
        health_mod.ACTIVITY_LEVELS[0],
    )
    top_indicator = (
        source_context["indicators"][0]["label"]
        if source_context["indicators"]
        else "No strong nearby indicator"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        f"Avg AQI ({len(fc['hourly'])}h)",
        summary["avg_aqi"],
        summary["category"],
    )
    c2.metric("Strongest nearby indicator", top_indicator)
    c3.metric(
        "Next 2h outdoor action",
        advisory["action_level"],
    )

    st.markdown("**Recommended action:** " + advisory["headline"])
    st.caption(source_context["disclaimer"])


# ---------------------------------------------------------------------------
# 5. AQI Chatbot — isolated so chatbot errors do not break other pages
# ---------------------------------------------------------------------------
elif page == "💬 AQI Chatbot":
    st.title("AQI Chatbot")
    st.caption(
        "Grounded on forecast + attribution + verified citizen reports + "
        "advisory docs."
    )

    try:
        import chatbot as chatbot_mod
    except Exception as error:
        st.error("The chatbot is temporarily unavailable while it is being updated.")
        st.code(f"{type(error).__name__}: {error}")
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(text)

    sample_questions = [
        f"Is it safe to jog in {location} tomorrow morning?",
        f"Why is AQI high near {location}?",
        "What should a school do if AQI is severe?",
        "Which reports near me are verified?",
    ]
    st.caption(
        "Try: " + " · ".join(f"`{question}`" for question in sample_questions)
    )

    question = st.chat_input("Ask about AQI, sources, or health guidance...")
    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.write(question)
        try:
            response = chatbot_mod.answer_query(question, location)
        except Exception as error:
            st.error("The chatbot could not answer this request.")
            st.code(f"{type(error).__name__}: {error}")
        else:
            st.session_state.chat_history.append(
                ("assistant", response["answer"])
            )
            with st.chat_message("assistant"):
                st.write(response["answer"])
                st.caption(f"Sources used: {response['sources_used']}")
