import os
import io
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pyarrow.parquet as pq

# Import Worker & API Client
from worker_client import (
    get_cloudflare_config,
    get_api_url,
    check_worker_health,
    fetch_dataset_bytes,
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="NLNG Predictive Maintenance IDSS",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 700 !important;
        color: #00D4FF !important;
    }
    div[data-testid="metric-container"] {
        background-color: #1E222D;
        border: 1px solid #2A2E3D;
        border-radius: 8px;
        padding: 14px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.25);
    }
    [data-testid="stVerticalBlock"] > div[data-testid="stBlock"] {
        border-color: #2A2E3D !important;
    }
</style>
""", unsafe_allow_html=True)

SHAP_PATH = os.path.join(os.path.dirname(__file__), "shap_feature_importance.csv")
WORKER_URL, _ = get_cloudflare_config()
API_URL = get_api_url()

FEATURE_COLUMNS = [
    "commission_year", "asset_age_years", "hours_since_maint", "cumulative_op_hours",
    "bearing_temperature", "rpm", "vibration", "overall_vibration", "motor_current",
    "oil_pressure", "oil_particles_ppm", "bearing_index", "discharge_pressure",
    "feed_gas_pressure", "lng_output_tph", "ambient_temperature", "load_factor",
    "wear_level", "lubrication_health_index", "production_efficiency", "quality_factor"
]

DASHBOARD_COLUMNS = [
    "timestamp", "train", "equipment_id", "equipment_name", "equipment_type",
    "operating_state", "criticality"
] + FEATURE_COLUMNS

# =============================================================================
# DATA LOADING & CACHING
# =============================================================================

@st.cache_data(show_spinner="Streaming dataset...")
def load_data():
    try:
        dataset_bytes = fetch_dataset_bytes()
        buffer = io.BytesIO(dataset_bytes)
        parquet_file = pq.ParquetFile(buffer)
        available_cols = [c for c in DASHBOARD_COLUMNS if c in parquet_file.schema.names]
        table = parquet_file.read(columns=available_cols)
        df = table.to_pandas()
        
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')
    except Exception as e:
        local_path = "data/NLNG_cleaned_leakage_controlled.parquet"
        if os.path.exists(local_path):
            df = pd.read_parquet(local_path)
        else:
            raise e

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

@st.cache_data
def load_shap_data():
    if not os.path.exists(SHAP_PATH):
        return pd.DataFrame()
    return pd.read_csv(SHAP_PATH)

try:
    df = load_data()
except Exception as exc:
    st.error(f"Unable to load resources: {exc}")
    st.stop()

shap_df = load_shap_data()

# =============================================================================
# CACHED SAFE PREDICTION FETCHING
# =============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_prediction_cached(asset_id, train_id, feature_dict):
    """Cached prediction call to prevent freezing/crashing on equipment selection."""
    payload = {
        "equipment_id": str(asset_id),
        "train": str(train_id)
    }
    
    for k, v in feature_dict.items():
        if pd.isna(v) or v is None:
            payload[k] = None
        else:
            payload[k] = float(v)

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🎛️ IDSS Control Center")

trains = sorted(df["train"].dropna().unique())
selected_train = st.sidebar.selectbox("LNG Train", trains)

# Filter by train first
train_mask = df["train"] == selected_train
train_df = df[train_mask]

assets = sorted(train_df["equipment_id"].dropna().unique())
selected_asset = st.sidebar.selectbox("Equipment", assets)

# Slice asset history
asset_history = train_df[train_df["equipment_id"] == selected_asset].sort_values("timestamp")

if asset_history.empty:
    st.warning("No data found for selected asset.")
    st.stop()

asset_current = asset_history.iloc[-1]

# Build feature dictionary safely
feature_dict = {}
for col in FEATURE_COLUMNS:
    if col in asset_current:
        val = asset_current[col]
        feature_dict[col] = float(val) if pd.notna(val) else None

# Get prediction safely from cache
prediction = fetch_prediction_cached(selected_asset, selected_train, feature_dict)

# =============================================================================
# DASHBOARD UI
# =============================================================================

st.title("🛡️ NLNG Predictive Maintenance IDSS")
st.caption("Pipeline 2.0 Model-Serving and Maintenance Decision-Support Interface")

# Parse KPIs
if prediction and isinstance(prediction, dict):
    failure_probability = prediction.get("failure_probability_24h", 0) or 0
    failure_percent = failure_probability * 100
    failure_risk = str(prediction.get("failure_risk", "UNKNOWN"))
    rul_days = prediction.get("rul_days", 0) or 0
else:
    failure_percent = None
    failure_risk = "UNAVAILABLE"
    rul_days = None

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric("Equipment ID", str(selected_asset))

with kpi2:
    st.metric("Facility", str(selected_train))

with kpi3:
    st.metric("24h Failure Risk", f"{failure_percent:.2f}%" if failure_percent is not None else "N/A")

with kpi4:
    st.metric("Estimated RUL", f"{rul_days:.1f} Days" if rul_days is not None else "N/A")

with kpi5:
    st.metric("Operating State", str(asset_current.get("operating_state", "N/A")))

st.markdown("<br>", unsafe_allow_html=True)

# Profile Container
with st.container(border=True):
    st.markdown("##### 🛠️ Equipment Profile")
    i1, i2, i3, i4 = st.columns(4)
    with i1:
        st.markdown(f"**Name:** {asset_current.get('equipment_name', 'N/A')}")
    with i2:
        st.markdown(f"**Type:** {asset_current.get('equipment_type', 'N/A')}")
    with i3:
        st.markdown(f"**Criticality:** `{asset_current.get('criticality', 'N/A')}`")
    with i4:
        ts = asset_current.get('timestamp')
        ts_str = ts.strftime('%Y-%m-%d %H:%M') if pd.notna(ts) else "N/A"
        st.markdown(f"**Last Sync:** {ts_str}")

st.markdown("<br>", unsafe_allow_html=True)

# Visualizations
left, right = st.columns([1.8, 1])

with left:
    with st.container(border=True):
        st.markdown("##### 📈 Equipment Condition Trends")
        recent = asset_history.tail(300).copy()
        fig = go.Figure()

        if "overall_vibration" in recent.columns:
            fig.add_trace(go.Scatter(
                x=recent["timestamp"], y=recent["overall_vibration"],
                name="Overall Vibration", mode="lines", line=dict(color="#00D4FF", width=2)
            ))

        if "oil_particles_ppm" in recent.columns:
            fig.add_trace(go.Scatter(
                x=recent["timestamp"], y=recent["oil_particles_ppm"],
                name="Oil Particles (PPM)", mode="lines", line=dict(color="#FF4B4B", width=1.5, dash="dot")
            ))

        fig.update_layout(
            paper_bgcolor="#1E222D", plot_bgcolor="#1E222D",
            font=dict(color="#E0E0E0"), xaxis=dict(showgrid=True, gridcolor="#2A2E3D"),
            yaxis=dict(showgrid=True, gridcolor="#2A2E3D"), margin=dict(l=20, r=20, t=30, b=20),
            height=360, legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
        )
        st.plotly_chart(fig, use_container_width=True)

with right:
    with st.container(border=True):
        st.markdown("##### 🎯 24-Hour Failure Risk")
        gauge_val = failure_percent if failure_percent is not None else 0
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_val,
            number={'suffix': "%", 'font': {'size': 32, 'color': '#FFFFFF'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': "#A0A0A0"},
                'bar': {'color': "#00D4FF"},
                'bgcolor': "#14171F",
                'borderwidth': 1,
                'bordercolor': "#2A2E3D",
                'steps': [
                    {'range': [0, 15], 'color': 'rgba(0, 204, 150, 0.3)'},
                    {'range': [15, 35], 'color': 'rgba(255, 170, 0, 0.3)'},
                    {'range': [35, 100], 'color': 'rgba(255, 75, 75, 0.3)'}
                ]
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#1E222D", font=dict(color="#E0E0E0"),
            height=280, margin=dict(l=20, r=20, t=20, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        if failure_risk == "LOW":
            st.success("🟢 Risk Level: LOW")
        elif failure_risk in ["MODERATE", "HIGH"]:
            st.warning(f"🟡 Risk Level: {failure_risk}")
        elif failure_risk == "CRITICAL":
            st.error("🔴 Risk Level: CRITICAL")
        else:
            st.info("ℹ️ Standby / Visualizer Mode")

st.markdown("<br>", unsafe_allow_html=True)

# SHAP & System Status
m_col, s_col = st.columns([1, 1])

with m_col:
    with st.container(border=True):
        st.markdown("##### 🤖 Active Model Suite")
        st.info("**Classification Engine:** MLP / Gradient Boosting Classifier\n\nPredicts 24-hour failure probability.")
        st.info("**Regression Engine:** Random Forest Regressor / LSTM\n\nEstimates Remaining Useful Life (RUL).")

with s_col:
    with st.container(border=True):
        st.markdown("##### 🔎 Global Model Drivers (SHAP)")
        if not shap_df.empty:
            top = shap_df.head(7).copy()
            fig_shap = go.Figure(go.Bar(
                x=top["Mean_Absolute_SHAP"], y=top["Feature"],
                orientation="h", marker=dict(color="#00D4FF")
            ))
            fig_shap.update_layout(
                paper_bgcolor="#1E222D", plot_bgcolor="#1E222D",
                font=dict(color="#E0E0E0"), xaxis=dict(showgrid=True, gridcolor="#2A2E3D"),
                margin=dict(l=20, r=20, t=20, b=20), height=250, yaxis={"categoryorder": "total ascending"}
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.caption("SHAP data unavailable.")

# Decision Support
with st.container(border=True):
    st.markdown("##### 💡 Maintenance Decision Support System (IDSS)")
    if prediction is None:
        st.info("Render API endpoint in standby or offline. Operating in visualizer mode.")
    else:
        if failure_risk == "CRITICAL":
            st.error(f"**CRITICAL RISK DETECTED** on {selected_asset}. Prioritise immediate inspection.")
        elif failure_risk in ["HIGH", "MODERATE"]:
            st.warning(f"**ELEVATED RISK DETECTED** on {selected_asset}. Schedule maintenance check.")
        else:
            st.success(f"**NORMAL OPERATING CONDITION** for {selected_asset}.")

st.divider()
st.caption("NLNG Predictive Maintenance IDSS | Streamlit → Cloudflare R2 / Render API → Pipeline 2.0")
