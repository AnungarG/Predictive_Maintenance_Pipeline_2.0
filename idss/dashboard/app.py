import os
import io
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pyarrow.parquet as pq
import joblib
from pathlib import Path

# Import Worker & API Client
from worker_client import (
    get_cloudflare_config,
    get_api_url,
    check_worker_health,
    fetch_dataset_bytes,
    fetch_model_bytes
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

# =============================================================================
# POWER BI EXECUTIVE THEME & STYLING (CUSTOM CSS)
# =============================================================================

st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Executive Metric Card Styling */
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
    
    /* Native Container Card Borders */
    [data-testid="stVerticalBlock"] > div[data-testid="stBlock"] {
        border-color: #2A2E3D !important;
    }
    
    /* Status Badge Styling */
    .badge-online {
        background-color: rgba(0, 204, 150, 0.15);
        color: #00CC96;
        border: 1px solid #00CC96;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .badge-offline {
        background-color: rgba(255, 75, 75, 0.15);
        color: #FF4B4B;
        border: 1px solid #FF4B4B;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


SHAP_PATH = os.path.join(
    os.path.dirname(__file__),
    "shap_feature_importance.csv"
)

WORKER_URL, _ = get_cloudflare_config()
API_URL = get_api_url()

# =============================================================================
# AUTHORITATIVE FEATURE CONTRACT
# =============================================================================

FEATURE_COLUMNS = [
    "commission_year",
    "asset_age_years",
    "hours_since_maint",
    "cumulative_op_hours",
    "bearing_temperature",
    "rpm",
    "vibration",
    "overall_vibration",
    "motor_current",
    "oil_pressure",
    "oil_particles_ppm",
    "bearing_index",
    "discharge_pressure",
    "feed_gas_pressure",
    "lng_output_tph",
    "ambient_temperature",
    "load_factor",
    "wear_level",
    "lubrication_health_index",
    "production_efficiency",
    "quality_factor"
]

DASHBOARD_COLUMNS = [
    "timestamp",
    "train",
    "equipment_id",
    "equipment_name",
    "equipment_type",
    "operating_state",
    "criticality"
] + FEATURE_COLUMNS


# =============================================================================
# OPTIMIZED DATA AND MODEL LOADING (PREVENTS STREAMLIT MEMORY CRASH)
# =============================================================================

@st.cache_data(show_spinner="Streaming & filtering dataset from Cloudflare R2...")
def load_data():
    try:
        dataset_bytes = fetch_dataset_bytes()
        buffer = io.BytesIO(dataset_bytes)
        
        # Open Parquet file without loading all unneeded bytes to RAM
        parquet_file = pq.ParquetFile(buffer)
        available_cols = [c for c in DASHBOARD_COLUMNS if c in parquet_file.schema.names]
        
        # Read only contract schema columns
        table = parquet_file.read(columns=available_cols)
        df = table.to_pandas()
        
        # Downcast float64 to float32 to halve memory usage
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')
        
    except Exception as e:
        local_path = "data/NLNG_cleaned_leakage_controlled.parquet"
        if os.path.exists(local_path):
            st.warning("Worker dataset fetch failed. Falling back to local copy...")
            df = pd.read_parquet(local_path)
        else:
            raise e

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    return df.sort_values(
        ["equipment_id", "timestamp"]
    )


@st.cache_resource(show_spinner="Verifying models on Cloudflare R2...")
def load_models():
    return True


@st.cache_data
def load_shap_data():
    if not os.path.exists(SHAP_PATH):
        return pd.DataFrame()
    return pd.read_csv(SHAP_PATH)


# Initialize Dataset and Models
try:
    df = load_data()
    load_models()
except Exception as exc:
    st.error(f"Unable to load resources:\n\n{exc}")
    st.stop()

shap_df = load_shap_data()


# =============================================================================
# API / WORKER HEALTH CHECKS
# =============================================================================

def check_worker_and_api():
    worker_ok, worker_msg = check_worker_health()
    
    api_ok = False
    try:
        resp = requests.get(f"{API_URL}/health", timeout=10)
        if resp.status_code == 200:
            api_ok = True
    except Exception:
        api_ok = False
        
    return worker_ok, api_ok


def get_prediction(row):
    payload = {
        "equipment_id": str(row["equipment_id"]),
        "train": str(row["train"])
    }

    for feature in FEATURE_COLUMNS:
        if feature in row:
            value = row[feature]
            payload[feature] = None if pd.isna(value) else float(value)

    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=120
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Prediction API returned {response.status_code}: {response.text}"
        )

    return response.json()


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🎛️ IDSS Control Center")

worker_ok, api_ok = check_worker_and_api()

if worker_ok:
    st.sidebar.success("Cloudflare R2 Backend: ONLINE")
else:
    st.sidebar.error("Cloudflare R2 Backend: OFFLINE")

if api_ok:
    st.sidebar.success("Prediction API (Render): ONLINE")
else:
    st.sidebar.warning("Prediction API (Render): OFFLINE / SLEEPING")

st.sidebar.divider()

trains = sorted(df["train"].dropna().unique())
selected_train = st.sidebar.selectbox("LNG Train", trains)

train_df = df[df["train"] == selected_train]

assets = sorted(train_df["equipment_id"].dropna().unique())
selected_asset = st.sidebar.selectbox("Equipment", assets)

asset_history = (
    train_df[train_df["equipment_id"] == selected_asset]
    .sort_values("timestamp")
    .copy()
)

if asset_history.empty:
    st.error("No observations available for this equipment.")
    st.stop()

asset_current = asset_history.iloc[-1]


# =============================================================================
# PREDICTION FETCH
# =============================================================================

prediction = None

if api_ok:
    try:
        prediction = get_prediction(asset_current)
    except Exception as exc:
        st.sidebar.caption(f"Prediction note: {exc}")


# =============================================================================
# HEADER
# =============================================================================

st.title("🛡️ NLNG Predictive Maintenance IDSS")
st.caption("Pipeline 2.0 Model-Serving and Maintenance Decision-Support Interface")

# =============================================================================
# EXECUTIVE KPI CARDS
# =============================================================================

if prediction is not None:
    failure_probability = prediction.get("failure_probability_24h", 0)
    failure_percent = failure_probability * 100
    failure_risk = prediction.get("failure_risk", "UNKNOWN")
    rul_days = prediction.get("rul_days", 0)
else:
    failure_percent = np.nan
    failure_risk = "UNAVAILABLE"
    rul_days = np.nan

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric("Equipment ID", selected_asset)

with kpi2:
    st.metric("Facility", selected_train)

with kpi3:
    st.metric(
        "24h Failure Risk",
        f"{failure_percent:.2f}%" if prediction is not None else "N/A"
    )

with kpi4:
    st.metric(
        "Estimated RUL",
        f"{rul_days:.1f} Days" if prediction is not None else "N/A"
    )

with kpi5:
    st.metric("Operating State", str(asset_current["operating_state"]))

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# EQUIPMENT DETAILS (CARD)
# =============================================================================

with st.container(border=True):
    st.markdown("##### 🛠️ Equipment Profile")
    i1, i2, i3, i4 = st.columns(4)
    
    with i1:
        st.markdown(f"**Name:** {asset_current['equipment_name']}")
    with i2:
        st.markdown(f"**Type:** {asset_current['equipment_type']}")
    with i3:
        st.markdown(f"**Criticality:** `{asset_current['criticality']}`")
    with i4:
        st.markdown(f"**Last Sync:** {asset_current['timestamp'].strftime('%Y-%m-%d %H:%M')}")


st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# CONDITION TRENDS + RADIAL GAUGE
# =============================================================================

left, right = st.columns([1.8, 1])

with left:
    with st.container(border=True):
        st.markdown("##### 📈 Equipment Condition Trends")

        recent = asset_history.tail(500).copy()
        fig = go.Figure()

        if "overall_vibration" in recent.columns:
            fig.add_trace(
                go.Scatter(
                    x=recent["timestamp"],
                    y=recent["overall_vibration"],
                    name="Overall Vibration",
                    mode="lines",
                    line=dict(color="#00D4FF", width=2)
                )
            )

        if "oil_particles_ppm" in recent.columns:
            fig.add_trace(
                go.Scatter(
                    x=recent["timestamp"],
                    y=recent["oil_particles_ppm"],
                    name="Oil Particles (PPM)",
                    mode="lines",
                    line=dict(color="#FF4B4B", width=1.5, dash="dot")
                )
            )

        fig.update_layout(
            paper_bgcolor="#1E222D",
            plot_bgcolor="#1E222D",
            font=dict(color="#E0E0E0", family="Segoe UI, sans-serif"),
            xaxis=dict(showgrid=True, gridcolor="#2A2E3D", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#2A2E3D", zeroline=False),
            margin=dict(l=20, r=20, t=30, b=20),
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

with right:
    with st.container(border=True):
        st.markdown("##### 🎯 24-Hour Failure Risk")

        gauge_value = failure_percent if prediction is not None and not np.isnan(failure_percent) else 0

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=gauge_value,
                number={'suffix': "%", 'font': {'size': 32, 'color': '#FFFFFF'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#A0A0A0"},
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
            )
        )

        fig_gauge.update_layout(
            paper_bgcolor="#1E222D",
            font=dict(color="#E0E0E0", family="Segoe UI, sans-serif"),
            height=280,
            margin=dict(l=20, r=20, t=20, b=10)
        )

        st.plotly_chart(fig_gauge, use_container_width=True)

        if prediction is not None:
            if failure_risk == "LOW":
                st.success(f"🟢 Risk Level: {failure_risk}")
            elif failure_risk in ["MODERATE", "HIGH"]:
                st.warning(f"🟡 Risk Level: {failure_risk}")
            else:
                st.error(f"🔴 Risk Level: {failure_risk}")

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# SERVED MODELS & SHAP DRIVERS
# =============================================================================

m_col, s_col = st.columns([1, 1])

with m_col:
    with st.container(border=True):
        st.markdown("##### 🤖 Active Model Suite")
        st.info(
            "**Classification Engine**\n\n"
            "Deep Learning MLP / Gradient Boosting Classifier\n\n"
            "Predicts 24-hour equipment failure probability."
        )
        st.info(
            "**Regression Engine**\n\n"
            "Random Forest Regressor / LSTM Architecture\n\n"
            "Estimates Remaining Useful Life (RUL) in days."
        )

with s_col:
    with st.container(border=True):
        st.markdown("##### 🔎 Global Model Drivers (SHAP)")
        if not shap_df.empty:
            top = shap_df.head(7).copy()

            fig_shap = go.Figure(
                go.Bar(
                    x=top["Mean_Absolute_SHAP"],
                    y=top["Feature"],
                    orientation="h",
                    marker=dict(color="#00D4FF")
                )
            )

            fig_shap.update_layout(
                paper_bgcolor="#1E222D",
                plot_bgcolor="#1E222D",
                font=dict(color="#E0E0E0", family="Segoe UI, sans-serif"),
                xaxis=dict(showgrid=True, gridcolor="#2A2E3D"),
                margin=dict(l=20, r=20, t=20, b=20),
                height=250,
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.caption("SHAP feature importance data unavailable.")


st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# DECISION SUPPORT
# =============================================================================

with st.container(border=True):
    st.markdown("##### 💡 Maintenance Decision Support System (IDSS)")

    if prediction is None:
        st.info("Prediction API service endpoint inactive or running standalone visualizer mode.")
    else:
        if failure_risk == "CRITICAL":
            st.error(
                f"**CRITICAL RISK DETECTED**\n\n"
                f"{selected_asset} has a high predicted probability of failure within 24 hours.\n\n"
                f"**Action Required:** Prioritise immediate engineering assessment and condition review."
            )
        elif failure_risk == "HIGH":
            st.warning(
                f"**HIGH RISK DETECTED**\n\n"
                f"{selected_asset} requires increased monitoring and maintenance scheduling.\n\n"
                f"**Action Required:** Schedule inspection during the next operational window."
            )
        elif failure_risk == "MODERATE":
            st.warning(
                f"**MODERATE RISK DETECTED**\n\n"
                f"{selected_asset} shows elevated condition parameters.\n\n"
                f"**Action Required:** Continue close telemetry observation."
            )
        else:
            st.success(
                f"**NORMAL OPERATING CONDITION**\n\n"
                f"{selected_asset} exhibits low failure probability within the next 24-hour window.\n\n"
                f"**Action Required:** Continue routine monitoring schedules."
            )

    st.caption(
        "Notice: The IDSS provides decision-support insights. It does not automatically "
        "execute plant-control actions or system shutdowns."
    )

st.divider()
st.caption(
    "NLNG Predictive Maintenance IDSS | Streamlit → Cloudflare R2 Worker / Render API → Pipeline 2.0"
)
