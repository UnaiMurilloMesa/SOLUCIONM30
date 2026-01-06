import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.data_loader import load_csv_data, load_metadata
from src.preprocessor import DataPreprocessor
from src.config import DATA_PATH_RAW, DATA_PATH_PROCESSED, M30_EAST_SENSORS
from src.physics import TrafficPhysics
from src.optimizer import TrafficOptimizer
from src.kpi_analyzer import HourlyKPIAnalyzer

st.set_page_config(page_title="M-30 Digital Twin", layout="wide")

# --- CUSTOM CSS ---
st.markdown(
    """
<style>
    .road-container {
        position: relative;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        border: 2px solid #555;
        transition: background-color 0.5s ease;
    }
    .lane-marking {
        position: absolute;
        width: 100%;
        height: 4px;
        background: repeating-linear-gradient(90deg, #fff 0px, #fff 40px, transparent 40px, transparent 80px);
        top: 50%;
    }
    .car-overlay {
        font-size: 2.5em;
        font-weight: bold;
        color: white;
        z-index: 10;
        text-shadow: 2px 2px 4px #000;
        background-color: rgba(0,0,0,0.3);
        padding: 5px 15px;
        border-radius: 5px;
    }
    
    /* Metrics Layout */
    .metrics-container {
        display: flex;
        justify-content: space-around;
        align-items: center;
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #444;
        margin-top: 5px;
    }
    .metric-item {
        text-align: center;
    }
    .metric-label {
        font-size: 0.9em;
        color: #aaa;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.4em;
        font-weight: bold;
        color: white;
    }
    
    /* Speed Sign */
    .speed-sign {
        width: 60px;
        height: 60px;
        background-color: white;
        border: 6px solid #cc0000;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        font-weight: bold;
        color: black;
        box-shadow: 0 2px 5px rgba(0,0,0,0.5);
    }
    
    /* Clock */
    .digital-clock {
        text-align: center; 
        background-color: #000; 
        color: #0f0; 
        font-family: 'Courier New', Courier, monospace; 
        padding: 10px; 
        border-radius: 10px; 
        border: 2px solid #555;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(0, 255, 0, 0.5);
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_all_data(selected_date):
    # 1. Traffic Data
    month_str = selected_date.strftime("%m")
    year_str = selected_date.strftime("%Y")
    folder_name = f"{month_str}-{year_str}"
    file_name = f"{month_str}-{year_str}.csv"

    file_path = DATA_PATH_RAW / "trafico" / folder_name / file_name

    # 2. Metadata (Map) - Static for now
    meta_path = DATA_PATH_RAW / "meta" / "pmed_ubicacion_10_2018.csv"

    # 3. Real Limits
    limits_path = DATA_PATH_PROCESSED / "realvlimit" / "sensor_limits.csv"

    with st.spinner(f"Loading data from {file_name}..."):
        if not file_path.exists():
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        df = load_csv_data(file_path)
        meta_df = load_metadata(meta_path)

        limits_df = pd.DataFrame()
        if limits_path.exists():
            limits_df = pd.read_csv(limits_path)

    # FILTER: Only keep M-30 sensors
    if "tipo_elem" in meta_df.columns:
        meta_df = meta_df[meta_df["tipo_elem"] == "M30"]

    m30_ids = meta_df["id"].unique()

    preprocessor = DataPreprocessor(sensor_ids=m30_ids)
    df_clean = preprocessor.clean_data(df)
    df_features = preprocessor.create_features(df_clean)

    return df_features, meta_df, limits_df


def get_road_color(speed):
    # Map speed (0-90) to Red-Green
    # 0 km/h -> Red (255, 0, 0)
    # 90 km/h -> Green (0, 255, 0)
    # 45 km/h -> Yellow (255, 255, 0)

    speed = max(0, min(90, speed))
    factor = speed / 90.0

    # Simple Linear Interpolation
    r = int(255 * (1 - factor))
    g = int(255 * factor)
    b = 0

    # Boost colors for "Neon" look
    # If speed is low, make it very red.
    if speed < 30:
        r = 255
        g = 0
    elif speed > 70:
        r = 0
        g = 255

    return f"rgb({r}, {g}, {b})"


# --- SESSION STATE ---
if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = False
if "selected_sensor" not in st.session_state:
    st.session_state.selected_sensor = None
if "current_frame_idx" not in st.session_state:
    st.session_state.current_frame_idx = 0

# ... (middle content preserved via broad match or careful editing?)
# Actually simpler to just target the specific blocks. Use separate chunks.

# 2. Slider Update
# 3. Loop Replacement


# --- LOAD DATA ---
# 1. Day Selection (Before loading data to potentially filter file loading in future)
# Moved Day Selection to Sidebar start, but we need data for map first.
# Let's keep logic: Load default -> Show Map -> User selects -> Filter.

st.sidebar.header("🕹️ Simulation Controls")
selected_date = st.sidebar.date_input(
    "📅 Date to Analyze", value=pd.to_datetime("2019-01-01")
)

# Reset if Date changes
if "last_date" not in st.session_state or st.session_state.last_date != selected_date:
    st.session_state.simulation_running = False
    st.session_state.current_frame_idx = 0
    st.session_state.last_date = selected_date

df_raw, df_meta, df_limits = load_all_data(selected_date)

if df_raw.empty:
    st.error("Data not found.")
    st.stop()

# --- MAP VISUALIZATION & SELECTION ---
st.markdown("### 🗺️ Select Sensor from Map")

# Filter metadata to only include sensors present in raw data
valid_sensors = df_raw["id"].unique()
map_data = df_meta[df_meta["id"].isin(valid_sensors)].copy()

# Define Color and Size based on selection
map_data["color"] = "blue"
map_data["size"] = 10
if st.session_state.selected_sensor:
    mask = map_data["id"] == st.session_state.selected_sensor
    map_data.loc[mask, "color"] = "red"
    map_data.loc[mask, "size"] = 20

# Add hover info
if not map_data.empty:
    fig_map = px.scatter_mapbox(
        map_data,
        lat="latitud",
        lon="longitud",
        hover_name="nombre",
        hover_data=["id", "distrito"],
        zoom=11,
        height=350,
        color="color",
        size="size",
        color_discrete_map={"blue": "#3498db", "red": "#e74c3c"},
        size_max=20,
    )
    fig_map.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=False,
    )
    # fig_map.update_traces(marker=dict(size=12, color='blue')) # Removed in favor of dynamic size/color

    # Enable Selection (Streamlit 1.37+)
    selection = st.plotly_chart(fig_map, on_select="rerun", use_container_width=True)

    # Handle Selection Event
    if selection and selection["selection"]["points"]:
        point_idx = selection["selection"]["points"][0]["point_index"]
        # The chart might resort data, but px usually keeps order if not aggregating?
        # Safest to use custom data, but let's assume index matches for now
        # or better: rely on the filtering we did.
        # Actually, clicking a point returns the index inthe dataframe passed to plotly.
        # Since we pass map_data, point_index 0 refers to map_data.iloc[0].
        sensor_id_selected = map_data.iloc[point_idx]["id"]

        # Only update if changed to avoid loop
        if st.session_state.selected_sensor != sensor_id_selected:
            st.session_state.selected_sensor = sensor_id_selected
            st.rerun()

# Fallback if no map selection or first run
available_sensors = sorted(valid_sensors)
default_idx = 0
if st.session_state.selected_sensor in available_sensors:
    default_idx = available_sensors.index(st.session_state.selected_sensor)

# --- CONTROLS SIDEBAR ---
# 2. Sensor Selection
selected_sensor = st.sidebar.selectbox(
    "📍 Sensor Location", options=available_sensors, index=default_idx
)

# Sync Sidebar -> Session State (if changed manually)
if selected_sensor != st.session_state.selected_sensor:
    st.session_state.selected_sensor = selected_sensor
    st.session_state.simulation_running = False
    st.session_state.current_frame_idx = 0

# Process Data for Sensor
sensor_data = df_raw[df_raw["id"] == selected_sensor].copy()

# Look up Real Limit
real_limit_val = 90  # Default
if not df_limits.empty:
    limit_row = df_limits[df_limits["id"] == selected_sensor]
    if not limit_row.empty:
        real_limit_val = int(limit_row.iloc[0]["inferred_limit"])

k_crit = TrafficPhysics.calculate_critical_density(sensor_data)
q_max = TrafficPhysics.calculate_max_capacity(sensor_data)
optimizer = TrafficOptimizer(
    critical_density_override=k_crit,
    max_capacity_override=q_max,
    base_speed_limit=real_limit_val,
)
df_opt = optimizer.optimize_traffic(sensor_data)

# Use optimized intensity if available, else real
intensity_col = "intensidad_opt" if "intensidad_opt" in df_opt.columns else "intensidad"
df_opt["simulated_density"] = df_opt.apply(
    lambda row: row[intensity_col] / row["simulated_speed"]
    if row["simulated_speed"] > 0
    else 0,
    axis=1,
)

st.sidebar.markdown("---")

# 3. Speed Control
speed_factor = st.sidebar.slider(
    "⏩ Simulation Speed",
    min_value=1,
    max_value=20,
    value=5,
    help="Increase to make time fly.",
)
# Base sleep time for real-time feel (2 min duration for 24h)
# 1440 mins / (120s * 10fps) = 1.2 real mins per frame.
# We will just step through the dataframe.
# Simulation Speed simply reduces sleep time.

# 4. Filter to Date
# Filter df_opt to selected date (ignoring year/month if we are reusing sample data for demo)
# For demo purposes, we usually just take the first 24h of the sample.
unique_dates = df_opt["fecha"].dt.date.unique()
if selected_date not in unique_dates:
    st.warning(
        f"Data for {selected_date} not in sample. Using first available date: {unique_dates[0]}"
    )
    target_date = unique_dates[0]
else:
    target_date = selected_date

daily_data = (
    df_opt[df_opt["fecha"].dt.date == target_date]
    .sort_values("fecha")
    .reset_index(drop=True)
)

# Resample for smooth animation (1 minute intervals)
# Resample for smooth animation (1 minute intervals)
# We need to handle discrete variables (limits) differently from continuous (speed, density)
df_indexed = daily_data.set_index("fecha")

# 1. Continuous Variables: Linear Interpolation
continuous_cols = [
    "intensidad",
    "ocupacion",
    "carga",
    "vmed",
    "density",
    "intensidad_opt",
    "velocidad_opt",
    "simulated_speed",
    "simulated_density",
]
# Ensure they exist
continuous_cols = [c for c in continuous_cols if c in df_indexed.columns]
df_continuous = df_indexed[continuous_cols].resample("1T").interpolate(method="linear")

# 2. Discrete Variables: Forward Fill (Step function)
# Limits should jump, not fade.
discrete_cols = ["limite_dinamico", "optimal_speed_limit"]
discrete_cols = [c for c in discrete_cols if c in df_indexed.columns]
df_discrete = df_indexed[discrete_cols].resample("1T").ffill()

# Combine
daily_data_resampled = pd.concat([df_continuous, df_discrete], axis=1).reset_index()

# --- KPI ANALYZER INITIALIZATION ---
# Initialize KPI Analyzer for metrics calculation
reality_data = daily_data_resampled[["fecha", "vmed", "intensidad", "density"]].copy()
twin_data = daily_data_resampled[
    ["fecha", "simulated_speed", "intensidad_opt", "simulated_density"]
].copy()
kpi_analyzer = HourlyKPIAnalyzer(reality_data, twin_data)

# Session state for KPI tracking
if "last_kpi_update_hour" not in st.session_state:
    st.session_state.last_kpi_update_hour = -1

if "hourly_kpi_data" not in st.session_state:
    st.session_state.hourly_kpi_data = []

# --- MAIN INTERFACE ---

# Layout: Clock | Slider
# Layout: Clock | Controls
col_clock, col_controls = st.columns([1, 2])

# Logic for Hour Jumping
current_idx = st.session_state.current_frame_idx
# Assuming 1T resampling, 1 index = 1 minute. 60 indices = 1 hour.
current_hour = current_idx // 60
current_minute = current_idx % 60

# Buttons layout
c1, c2, c3 = col_controls.columns([1, 2, 1])

if c1.button("⏪ -1 Hr", use_container_width=True):
    # Logic:
    # If minute > 5 -> Go to start of Current Hour (XX:00)
    # If minute <= 5 -> Go to start of Previous Hour ((XX-1):00)
    if current_minute > 5:
        new_idx = current_hour * 60
    else:
        new_idx = (current_hour - 1) * 60

    st.session_state.current_frame_idx = max(0, int(new_idx))
    st.rerun()

if c3.button("⏩ +1 Hr", use_container_width=True):
    # Logic: Go to start of Next Hour
    new_idx = (current_hour + 1) * 60
    st.session_state.current_frame_idx = min(
        len(daily_data_resampled) - 1, int(new_idx)
    )
    st.rerun()

current_frame = st.session_state.current_frame_idx

with col_clock:
    clock_ph = st.empty()

# Extract Current Row based on Slider (Initial)
row = daily_data_resampled.iloc[current_frame]
curr_time_str = row["fecha"].strftime("%H:%M")

clock_ph.markdown(
    f"""
<div class="digital-clock">
    <div style="font-size: 1.2em; color: #888;">{target_date}</div>
    <div style="font-size: 3em; font-weight: bold;">{curr_time_str}</div>
</div>
""",
    unsafe_allow_html=True,
)

with c2:
    if st.session_state.simulation_running:
        if st.button("⏸️ PAUSE", use_container_width=True):
            st.session_state.simulation_running = False
            st.rerun()
    else:
        if st.button("▶️ START", use_container_width=True):
            st.session_state.simulation_running = True
            st.session_state.current_frame_idx = 0  # Reset to beginning
            st.session_state.last_kpi_update_hour = -1  # Reset KPI tracking
            st.session_state.simulation_completed = False  # Reset completion flag
            st.rerun()

# --- LAYOUT VISUALS ---
col1, col2 = st.columns(2)

# Placeholders for dynamic content
with col1:
    st.markdown("### 🔴 REALITY")
    real_road_ph = st.empty()
    real_metrics_ph = st.empty()

with col2:
    st.markdown("### 🟢 DIGITAL TWIN")
    opt_road_ph = st.empty()
    opt_metrics_ph = st.empty()

# --- KPI METRICS SECTION ---
st.markdown("---")
st.markdown("## 📊 KPI Metrics - Hourly Analysis")

col_kpi1, col_kpi2 = st.columns(2)

with col_kpi1:
    st.markdown("#### 🚀 Speed Improvement")
    kpi_speed_ph = st.empty()

with col_kpi2:
    st.markdown("#### 📉 Density Reduction")
    kpi_density_ph = st.empty()


def render_frame(data_row):
    # Reality
    r_speed = data_row["vmed"]
    r_dens = data_row["density"]
    r_color = get_road_color(r_speed)

    real_road_ph.markdown(
        f"""
    <div class="road-container" style="background-color: {r_color};">
        <div class="lane-marking"></div>
        <div class="car-overlay">{r_speed:.0f} km/h</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    real_metrics_ph.markdown(
        f"""
    <div class="metrics-container">
        <div class="metric-item">
            <div class="metric-label">Density</div>
            <div class="metric-value">{r_dens:.0f}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Limit</div>
             <div class="speed-sign">{real_limit_val}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Optimized
    o_speed = data_row["simulated_speed"]
    o_dens = data_row["simulated_density"]
    o_limit = int(data_row["optimal_speed_limit"])
    o_color = get_road_color(o_speed)

    # Highlight if limit is lower than base (VSL Active)
    is_active = o_limit < real_limit_val
    sign_style = (
        "border-color: #ffcc00; box-shadow: 0 0 15px #ffcc00;"
        if is_active
        else "border-color: #cc0000;"
    )

    opt_road_ph.markdown(
        f"""
    <div class="road-container" style="background-color: {o_color};">
        <div class="lane-marking"></div>
        <div class="car-overlay">{o_speed:.0f} km/h</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    opt_metrics_ph.markdown(
        f"""
    <div class="metrics-container">
        <div class="metric-item">
            <div class="metric-label">Density</div>
             <div class="metric-value">{o_dens:.0f}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Limit</div>
             <div class="speed-sign" style="{sign_style}">{o_limit}</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_kpi_metrics(current_hour):
    """
    Render KPI metrics with line charts showing Reality vs Digital Twin.
    Updates speed, flow, and density improvement metrics with visual charts.
    """

    # Check if simulation is completed - show daily summary instead of hourly
    if st.session_state.get("simulation_completed", False):
        # Calculate overall daily metrics
        hourly_metrics = kpi_analyzer.calculate_hourly_metrics()

        if len(hourly_metrics) > 0:
            # Calculate daily averages
            daily_speed_reality = hourly_metrics["vmed"].mean()
            daily_speed_twin = hourly_metrics["simulated_speed"].mean()
            daily_speed_imp = (
                ((daily_speed_twin - daily_speed_reality) / daily_speed_reality * 100)
                if daily_speed_reality > 0
                else 0
            )

            daily_flow_reality = hourly_metrics["intensidad"].mean()
            daily_flow_twin = hourly_metrics["intensidad_opt"].mean()
            daily_flow_imp = (
                ((daily_flow_twin - daily_flow_reality) / daily_flow_reality * 100)
                if daily_flow_reality > 0
                else 0
            )

            daily_density_reality = hourly_metrics["density"].mean()
            daily_density_twin = hourly_metrics["simulated_density"].mean()
            daily_density_red = (
                (
                    (daily_density_reality - daily_density_twin)
                    / daily_density_reality
                    * 100
                )
                if daily_density_reality > 0
                else 0
            )

            # Determine colors
            speed_color = (
                "#ff0000"
                if daily_speed_imp < 0
                else "#00bfff"
                if abs(daily_speed_imp) < 0.1
                else "#00ff00"
            )
            flow_color = (
                "#ff0000"
                if daily_flow_imp < 0
                else "#00bfff"
                if abs(daily_flow_imp) < 0.1
                else "#00ff00"
            )
            density_color = (
                "#ff0000"
                if daily_density_red < 0
                else "#00bfff"
                if abs(daily_density_red) < 0.1
                else "#00ff00"
            )

            # Create summary charts with daily average lines
            # Speed Chart
            fig_speed = go.Figure()
            fig_speed.add_trace(
                go.Scatter(
                    x=hourly_metrics["hour"],
                    y=hourly_metrics["vmed"],
                    name="Reality",
                    line=dict(color="#e74c3c", width=2),
                    mode="lines+markers",
                    marker=dict(size=6),
                )
            )
            fig_speed.add_trace(
                go.Scatter(
                    x=hourly_metrics["hour"],
                    y=hourly_metrics["simulated_speed"],
                    name="Digital Twin",
                    line=dict(color="#2ecc71", width=2),
                    mode="lines+markers",
                    marker=dict(size=6),
                )
            )
            fig_speed.update_layout(
                title=dict(
                    text=f"<b>DAILY Speed Improvement: {daily_speed_imp:+.1f}%</b>",
                    font=dict(size=20, color=speed_color),
                    x=0.5,
                    xanchor="center",
                ),
                xaxis=dict(
                    title=dict(
                        text="Hour", font=dict(size=14, color="white"), standoff=30
                    ),
                    gridcolor="#34495e",
                    showgrid=True,
                    color="white",
                    tickfont=dict(size=12),
                ),
                yaxis=dict(
                    title=dict(text="km/h", font=dict(size=14, color="white")),
                    gridcolor="#34495e",
                    showgrid=True,
                    color="white",
                    tickfont=dict(size=12),
                ),
                plot_bgcolor="#2c3e50",
                paper_bgcolor="#2c3e50",
                font=dict(color="white", size=12),
                height=300,
                margin=dict(l=40, r=20, t=70, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.2,
                    xanchor="right",
                    x=0.98,
                    font=dict(size=11, color="white"),
                    bgcolor="rgba(44, 62, 80, 0.8)",
                    bordercolor="rgba(0, 0, 0, 0)",
                    borderwidth=0,
                ),
                hovermode="x unified",
            )
            kpi_speed_ph.plotly_chart(
                fig_speed, use_container_width=True, key="speed_daily_summary"
            )

            # Density Chart
            fig_density = go.Figure()
            fig_density.add_trace(
                go.Scatter(
                    x=hourly_metrics["hour"],
                    y=hourly_metrics["density"],
                    name="Reality",
                    line=dict(color="#e74c3c", width=2),
                    mode="lines+markers",
                    marker=dict(size=6),
                )
            )
            fig_density.add_trace(
                go.Scatter(
                    x=hourly_metrics["hour"],
                    y=hourly_metrics["simulated_density"],
                    name="Digital Twin",
                    line=dict(color="#2ecc71", width=2),
                    mode="lines+markers",
                    marker=dict(size=6),
                )
            )
            fig_density.update_layout(
                title=dict(
                    text=f"<b>DAILY Density Reduction: {daily_density_red:+.1f}%</b>",
                    font=dict(size=20, color=density_color),
                    x=0.5,
                    xanchor="center",
                ),
                xaxis=dict(
                    title=dict(
                        text="Hour", font=dict(size=14, color="white"), standoff=30
                    ),
                    gridcolor="#34495e",
                    showgrid=True,
                    color="white",
                    tickfont=dict(size=12),
                ),
                yaxis=dict(
                    title=dict(text="veh/km", font=dict(size=14, color="white")),
                    gridcolor="#34495e",
                    showgrid=True,
                    color="white",
                    tickfont=dict(size=12),
                ),
                plot_bgcolor="#2c3e50",
                paper_bgcolor="#2c3e50",
                font=dict(color="white", size=12),
                height=300,
                margin=dict(l=40, r=20, t=70, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.2,
                    xanchor="right",
                    x=0.98,
                    font=dict(size=11, color="white"),
                    bgcolor="rgba(44, 62, 80, 0.8)",
                    bordercolor="rgba(0, 0, 0, 0)",
                    borderwidth=0,
                ),
                hovermode="x unified",
            )
            kpi_density_ph.plotly_chart(
                fig_density, use_container_width=True, key="density_daily_summary"
            )

        return  # Exit early, don't show hourly metrics

    if current_hour == 0:
        # No data yet for hour 0
        for ph in [kpi_speed_ph, kpi_density_ph]:
            ph.markdown(
                """
            <div style="text-align: center; padding: 20px; background-color: #2c3e50; border-radius: 10px;">
                <div style="font-size: 2em; color: #95a5a6;">━</div>
                <div style="font-size: 0.9em; color: #7f8c8d; margin-top: 10px;">Waiting for data...</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        return

    # Get cumulative hourly metrics up to current hour
    hourly_metrics = kpi_analyzer.calculate_hourly_metrics()
    hourly_metrics_subset = hourly_metrics[hourly_metrics["hour"] <= current_hour]

    if len(hourly_metrics_subset) == 0:
        return

    # Get current hour metrics
    speed_metrics = kpi_analyzer.get_last_hour_improvement(current_hour)
    density_metrics = kpi_analyzer.get_density_metrics(current_hour)

    # --- SPEED CHART ---
    speed_imp = speed_metrics["speed_improvement"]

    fig_speed = go.Figure()

    # Reality line
    fig_speed.add_trace(
        go.Scatter(
            x=hourly_metrics_subset["hour"],
            y=hourly_metrics_subset["vmed"],
            name="Reality",
            line=dict(color="#e74c3c", width=2),
            mode="lines+markers",
            marker=dict(size=6),
        )
    )

    # Digital Twin line
    fig_speed.add_trace(
        go.Scatter(
            x=hourly_metrics_subset["hour"],
            y=hourly_metrics_subset["simulated_speed"],
            name="Digital Twin",
            line=dict(color="#2ecc71", width=2),
            mode="lines+markers",
            marker=dict(size=6),
        )
    )

    fig_speed.update_layout(
        title=dict(
            text=f"<b>Speed Improvement: {speed_imp:+.1f}%</b>",
            font=dict(
                size=18,
                color="#ff0000"
                if speed_imp < 0
                else "#00bfff"
                if speed_imp == 0
                else "#00ff00",
            ),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title=dict(text="Hour", font=dict(size=14, color="white"), standoff=30),
            gridcolor="#34495e",
            showgrid=True,
            color="white",
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title=dict(text="km/h", font=dict(size=14, color="white")),
            gridcolor="#34495e",
            showgrid=True,
            color="white",
            tickfont=dict(size=12),
        ),
        plot_bgcolor="#2c3e50",
        paper_bgcolor="#2c3e50",
        font=dict(color="white", size=12),
        height=300,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="right",
            x=0.98,
            font=dict(size=11, color="white"),
            bgcolor="rgba(44, 62, 80, 0.8)",
            bordercolor="rgba(0, 0, 0, 0)",
            borderwidth=0,
        ),
        hovermode="x unified",
    )

    kpi_speed_ph.plotly_chart(
        fig_speed, use_container_width=True, key=f"speed_chart_{current_hour}"
    )

    # --- DENSITY CHART ---
    density_red = density_metrics["density_reduction"]

    fig_density = go.Figure()

    # Reality line
    fig_density.add_trace(
        go.Scatter(
            x=hourly_metrics_subset["hour"],
            y=hourly_metrics_subset["density"],
            name="Reality",
            line=dict(color="#e74c3c", width=2),
            mode="lines+markers",
            marker=dict(size=6),
        )
    )

    # Digital Twin line
    fig_density.add_trace(
        go.Scatter(
            x=hourly_metrics_subset["hour"],
            y=hourly_metrics_subset["simulated_density"],
            name="Digital Twin",
            line=dict(color="#2ecc71", width=2),
            mode="lines+markers",
            marker=dict(size=6),
        )
    )

    fig_density.update_layout(
        title=dict(
            text=f"<b>Density Reduction: {density_red:+.1f}%</b>",
            font=dict(
                size=18,
                color="#ff0000"
                if density_red < 0
                else "#00bfff"
                if density_red == 0
                else "#00ff00",
            ),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title=dict(text="Hour", font=dict(size=14, color="white"), standoff=30),
            gridcolor="#34495e",
            showgrid=True,
            color="white",
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title=dict(text="veh/km", font=dict(size=14, color="white")),
            gridcolor="#34495e",
            showgrid=True,
            color="white",
            tickfont=dict(size=12),
        ),
        plot_bgcolor="#2c3e50",
        paper_bgcolor="#2c3e50",
        font=dict(color="white", size=12),
        height=300,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="right",
            x=0.98,
            font=dict(size=11, color="white"),
            bgcolor="rgba(44, 62, 80, 0.8)",
            bordercolor="rgba(0, 0, 0, 0)",
            borderwidth=0,
        ),
        hovermode="x unified",
    )

    kpi_density_ph.plotly_chart(
        fig_density, use_container_width=True, key=f"density_chart_{current_hour}"
    )


# Initial Render (Static)
render_frame(row)

# Initial KPI Render
current_hour = current_frame // 60
render_kpi_metrics(current_hour)


# --- ANIMATION LOGIC ---
# --- ANIMATION LOGIC ---
if st.session_state.simulation_running:
    # Use a placeholder for the Stop button to allow breaking? date updates?
    # Actually, in Streamlit, clicking a button interrupts the script.
    # So we just run the loop. If user clicks PAUSE, script restarts, running is checked...
    # We need to ensure we don't reset index on restart unless intended.

    start_idx = st.session_state.current_frame_idx

    # Create a container for the loop to reuse
    # We already have placeholders: clock_ph, real_road_ph, opt_road_ph...

    for i in range(start_idx, len(daily_data_resampled)):
        # Check if we assume it's still running?
        # We can't detect button press inside loop easily without special components.
        # But clicking "Pause" defines a new run.

        # Update Session State Index (so we resume from here)
        st.session_state.current_frame_idx = i

        row = daily_data_resampled.iloc[i]

        # Update Clock
        curr_time_str = row["fecha"].strftime("%H:%M")
        clock_ph.markdown(
            f"""
        <div class="digital-clock">
            <div style="font-size: 1.2em; color: #888;">{target_date}</div>
            <div style="font-size: 3em; font-weight: bold;">{curr_time_str}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Render Visuals
        render_frame(row)

        # Update KPIs every hour
        current_hour = i // 60
        if current_hour > st.session_state.last_kpi_update_hour:
            render_kpi_metrics(current_hour)
            st.session_state.last_kpi_update_hour = current_hour

        # Sleep
        base_sleep = 0.08
        actual_sleep = base_sleep / speed_factor
        time.sleep(actual_sleep)

    # If loop finishes normally, just stop running but keep the data
    st.session_state.simulation_running = False
    st.session_state.simulation_completed = (
        True  # Mark as completed for summary display
    )
    # Don't reset current_frame_idx to keep charts visible
    st.rerun()
