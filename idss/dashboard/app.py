import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import joblib
import urllib.request
from pathlib import Path
import tensorflow as tf

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="NLNG Predictive Maintenance IDSS",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

SHAP_PATH = os.path.join(
    "data",
    "04_corrected_pipeline",
    "stage_5_explainability",
    "shap_feature_importance.csv"
)

API_URL = st.secrets.get("API_URL", "http://127.0.0.1:8000")


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
# DATA AND MODEL LOADING
# =============================================================================

@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    url = "data/NLNG_cleaned_leakage_controlled.parquet"
    df = pd.read_parquet(url)
    
    # Filter to contract columns if present
    available_cols = [col for col in DASHBOARD_COLUMNS if col in df.columns]
    df = df[available_cols].copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    return df.sort_values(
        ["equipment_id", "timestamp"]
    )


@st.cache_resource
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
# API FUNCTIONS
# =============================================================================

def check_api():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=10
        )
        if response.status_code == 200:
            return True, response.json()
        return False, response.text
    except Exception as exc:
        return False, str(exc)


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
# HEADER
# =============================================================================

st.title("🛡️ NLNG Predictive Maintenance IDSS")
st.caption("Pipeline 2.0 model-serving and maintenance decision-support interface")


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🎛️ IDSS Control Center")

api_ok, api_info = check_api()

if api_ok:
    st.sidebar.success("Model API: ONLINE")
else:
    st.sidebar.error("Model API: OFFLINE")

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
# PREDICTION
# =============================================================================

prediction = None

if api_ok:
    try:
        prediction = get_prediction(asset_current)
    except Exception as exc:
        st.error(f"Prediction failed:\n\n{exc}")


# =============================================================================
# TOP KPIs
# =============================================================================

if prediction is not None:
    failure_probability = prediction["failure_probability_24h"]
    failure_percent = failure_probability * 100
    failure_risk = prediction["failure_risk"]
    rul_days = prediction["rul_days"]
else:
    failure_percent = np.nan
    failure_risk = "UNAVAILABLE"
    rul_days = np.nan

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Equipment", selected_asset)

with col2:
    st.metric("Train", selected_train)

with col3:
    st.metric(
        "24h Failure Risk",
        f"{failure_percent:.2f}%" if prediction is not None else "N/A"
    )

with col4:
    st.metric(
        "Estimated RUL",
        f"{rul_days:.2f} days" if prediction is not None else "N/A"
    )

with col5:
    st.metric("Operating State", str(asset_current["operating_state"]))

st.divider()


# =============================================================================
# ASSET INFORMATION
# =============================================================================

st.subheader("Equipment Information")

i1, i2, i3, i4 = st.columns(4)

with i1:
    st.write("**Equipment Name**")
    st.write(asset_current["equipment_name"])

with i2:
    st.write("**Equipment Type**")
    st.write(asset_current["equipment_type"])

with i3:
    st.write("**Criticality**")
    st.write(asset_current["criticality"])

with i4:
    st.write("**Latest Observation**")
    st.write(asset_current["timestamp"].strftime("%Y-%m-%d %H:%M"))

st.divider()


# =============================================================================
# CONDITION TRENDS + RISK GAUGE
# =============================================================================

left, right = st.columns([1.7, 1])

with left:
    st.subheader("📈 Equipment Condition Trends")

    recent = asset_history.tail(500).copy()
    fig = go.Figure()

    if "overall_vibration" in recent.columns:
        fig.add_trace(
            go.Scatter(
                x=recent["timestamp"],
                y=recent["overall_vibration"],
                name="Overall Vibration",
                mode="lines"
            )
        )

    if "oil_particles_ppm" in recent.columns:
        fig.add_trace(
            go.Scatter(
                x=recent["timestamp"],
                y=recent["oil_particles_ppm"],
                name="Oil Particles",
                mode="lines"
            )
        )

    fig.update_layout(
        title="Recent Equipment Condition",
        xaxis_title="Timestamp",
        yaxis_title="Value",
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("🎯 24-Hour Failure Risk")

    gauge_value = failure_percent if prediction is not None else 0

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=gauge_value,
            title={"text": "Failure Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 10], "color": "#00CC96"},
                    {"range": [10, 30], "color": "#FFAA00"},
                    {"range": [30, 100], "color": "#FF4B4B"}
                ]
            }
        )
    )

    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

    if prediction is not None:
        if failure_risk == "LOW":
            st.success(f"🟢 Risk Level: {failure_risk}")
        elif failure_risk == "MODERATE":
            st.warning(f"🟡 Risk Level: {failure_risk}")
        elif failure_risk == "HIGH":
            st.warning(f"🟠 Risk Level: {failure_risk}")
        else:
            st.error(f"🔴 Risk Level: {failure_risk}")

st.divider()


# =============================================================================
# RUL
# =============================================================================

st.subheader("⏳ Remaining Useful Life")

if prediction is not None:
    st.metric("Estimated RUL", f"{rul_days:.2f} days")
    st.info(
        "RUL is presented as a model-generated estimate for maintenance planning "
        "support and should not be interpreted as an exact failure date."
    )
else:
    st.warning("RUL prediction unavailable.")

st.divider()


# =============================================================================
# SERVED MODEL INFORMATION
# =============================================================================

st.subheader("🤖 Models Being Served")

m1, m2 = st.columns(2)

with m1:
    st.info(
        "**Classification**\n\n"
        "Deep Learning MLP / Gradient Boosting Classifier\n\n"
        "24-hour failure-risk prediction"
    )

with m2:
    st.info(
        "**Regression**\n\n"
        "Random Forest Regressor / LSTM\n\n"
        "Remaining Useful Life estimation"
    )

st.divider()


# =============================================================================
# SHAP GLOBAL DRIVERS
# =============================================================================

st.subheader("🔎 Global Model Drivers")

if not shap_df.empty:
    top = shap_df.head(10).copy()

    fig = go.Figure(
        go.Bar(
            x=top["Mean_Absolute_SHAP"],
            y=top["Feature"],
            orientation="h"
        )
    )

    fig.update_layout(
        title="Top Global Failure-Prediction Drivers",
        xaxis_title="Mean Absolute SHAP Value",
        yaxis_title="Feature",
        template="plotly_dark",
        height=450,
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "These are global model drivers from the Stage 5 SHAP analysis. They describe "
        "overall model behaviour and are not a case-specific causal explanation."
    )

st.divider()


# =============================================================================
# DECISION SUPPORT
# =============================================================================

st.subheader("💡 Predictive Maintenance Decision Support")

if prediction is None:
    st.info("Prediction unavailable. Start the FastAPI model-serving service.")
else:
    if failure_risk == "CRITICAL":
        st.error(
            f"**CRITICAL RISK**\n\n"
            f"{selected_asset} has a high predicted probability of failure within the defined 24-hour horizon.\n\n"
            f"**Suggested decision-support action:**\n"
            f"Prioritise engineering assessment and condition review."
        )
    elif failure_risk == "HIGH":
        st.warning(
            f"**HIGH RISK**\n\n"
            f"{selected_asset} requires increased monitoring and maintenance review.\n\n"
            f"**Suggested decision-support action:**\n"
            f"Prioritise condition assessment and maintenance planning."
        )
    elif failure_risk == "MODERATE":
        st.warning(
            f"**MODERATE RISK**\n\n"
            f"{selected_asset} shows elevated predicted failure risk.\n\n"
            f"**Suggested decision-support action:**\n"
            f"Review current condition indicators and continue focused monitoring."
        )
    else:
        st.success(
            f"**LOW RISK**\n\n"
            f"{selected_asset} has a low predicted probability of failure within the defined 24-hour horizon.\n\n"
            f"**Suggested decision-support action:**\n"
            f"Continue normal condition monitoring."
        )

    st.caption(
        "The IDSS provides decision-support information. It does not automatically "
        "initiate maintenance, shutdown, or plant-control actions."
    )

st.divider()

st.caption(
    "NLNG Predictive Maintenance IDSS | "
    "Streamlit → FastAPI → Pipeline 2.0 Models"
)