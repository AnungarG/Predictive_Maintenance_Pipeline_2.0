import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import gdown


# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="NLNG Predictive Maintenance IDSS",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = st.secrets.get(
    "API_URL",
    "http://127.0.0.1:8000"
)

SHAP_PATH = os.path.join(
    "data",
    "04_corrected_pipeline",
    "stage_5_explainability",
    "shap_feature_importance.csv"
)


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
# DATA LOADING
# =============================================================================

@st.cache_data(show_spinner="Loading dataset...")
def load_data():

    # -------------------------------------------------------------------------
    # DATA DIRECTORY
    # -------------------------------------------------------------------------

    data_dir = "data"

    os.makedirs(
        data_dir,
        exist_ok=True
    )

    # -------------------------------------------------------------------------
    # DATASET PATH
    # -------------------------------------------------------------------------

    file_path = os.path.join(
        data_dir,
        "NLNG_cleaned_leakage_controlled.parquet"
    )

    # -------------------------------------------------------------------------
    # GOOGLE DRIVE FILE ID
    # -------------------------------------------------------------------------

    google_drive_file_id = (
        "1glib_3N3PuvQtnvr8s8NN-MGITLLjcZ"
    )

    # -------------------------------------------------------------------------
    # DOWNLOAD DATASET IF NOT ALREADY AVAILABLE
    # -------------------------------------------------------------------------

    if not os.path.exists(file_path):

        with st.spinner(
            "Downloading NLNG dataset from Google Drive..."
        ):

            try:

                downloaded_file = gdown.download(
                    id=google_drive_file_id,
                    output=file_path,
                    quiet=False,
                    fuzzy=True
                )

            except Exception as exc:

                raise RuntimeError(
                    "Google Drive download failed.\n\n"
                    f"File ID: {google_drive_file_id}\n\n"
                    f"Error: {exc}\n\n"
                    "Please verify that the Google Drive file is "
                    "shared as 'Anyone with the link' with Viewer access."
                )

            # -----------------------------------------------------------------
            # VERIFY GDOWN RESULT
            # -----------------------------------------------------------------

            if downloaded_file is None:

                raise RuntimeError(
                    "Google Drive returned no downloadable file.\n\n"
                    f"File ID: {google_drive_file_id}\n\n"
                    "Possible causes:\n"
                    "1. The Google Drive file is not publicly accessible.\n"
                    "2. Google Drive has temporarily restricted downloads.\n"
                    "3. The file has exceeded its download quota.\n"
                    "4. The Google Drive file ID is incorrect."
                )

    # -------------------------------------------------------------------------
    # VERIFY FILE EXISTS
    # -------------------------------------------------------------------------

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            "NLNG dataset file was not found after the download attempt.\n\n"
            f"Expected location: {file_path}"
        )

    # -------------------------------------------------------------------------
    # VERIFY FILE SIZE
    # -------------------------------------------------------------------------

    file_size = os.path.getsize(file_path)

    if file_size == 0:

        try:
            os.remove(file_path)
        except Exception:
            pass

        raise RuntimeError(
            "The downloaded NLNG dataset is empty (0 bytes).\n\n"
            "Google Drive may have returned an invalid download response."
        )

    # -------------------------------------------------------------------------
    # LOAD PARQUET
    # -------------------------------------------------------------------------

    try:

        df = pd.read_parquet(
            file_path
        )

    except Exception as exc:

        # Remove corrupted/incomplete download
        try:
            os.remove(file_path)
        except Exception:
            pass

        raise RuntimeError(
            "The NLNG dataset was downloaded, but it could not be "
            "read as a valid Parquet file.\n\n"
            f"File: {file_path}\n"
            f"Size: {file_size:,} bytes\n\n"
            f"Parquet error: {exc}"
        )

    # -------------------------------------------------------------------------
    # VERIFY DATASET IS NOT EMPTY
    # -------------------------------------------------------------------------

    if df.empty:

        raise RuntimeError(
            "The NLNG dataset was loaded successfully, "
            "but it contains no records."
        )

    # -------------------------------------------------------------------------
    # KEEP ONLY DASHBOARD COLUMNS
    # -------------------------------------------------------------------------

    available_cols = [
        col
        for col in DASHBOARD_COLUMNS
        if col in df.columns
    ]

    if not available_cols:

        raise RuntimeError(
            "The downloaded dataset does not contain any of the "
            "expected dashboard columns.\n\n"
            f"Dataset columns found: {list(df.columns)}"
        )

    df = df[
        available_cols
    ].copy()

    # -------------------------------------------------------------------------
    # VERIFY REQUIRED DASHBOARD COLUMNS
    # -------------------------------------------------------------------------

    required_columns = [
        "timestamp",
        "equipment_id"
    ]

    missing_required = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_required:

        raise RuntimeError(
            "Required dashboard columns are missing:\n\n"
            + ", ".join(missing_required)
            + "\n\n"
            f"Available columns: {list(df.columns)}"
        )

    # -------------------------------------------------------------------------
    # CONVERT TIMESTAMP
    # -------------------------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # -------------------------------------------------------------------------
    # VERIFY TIMESTAMP VALUES
    # -------------------------------------------------------------------------

    valid_timestamps = (
        df["timestamp"].notna().sum()
    )

    if valid_timestamps == 0:

        raise RuntimeError(
            "The dataset contains no valid timestamp values."
        )

    # -------------------------------------------------------------------------
    # REMOVE RECORDS WITHOUT EQUIPMENT ID
    # -------------------------------------------------------------------------

    df = df[
        df["equipment_id"].notna()
    ].copy()

    if df.empty:

        raise RuntimeError(
            "No valid equipment records remain after filtering "
            "missing equipment_id values."
        )

    # -------------------------------------------------------------------------
    # SORT OBSERVATIONS
    # -------------------------------------------------------------------------

    df = df.sort_values(
        ["equipment_id", "timestamp"]
    ).reset_index(
        drop=True
    )

    # -------------------------------------------------------------------------
    # DATASET INFORMATION
    # -------------------------------------------------------------------------

    st.sidebar.caption(
        f"Dataset: {len(df):,} records"
    )

    st.sidebar.caption(
        f"Equipment: {df['equipment_id'].nunique():,}"
    )

    if "train" in df.columns:

        st.sidebar.caption(
            f"LNG Trains: {df['train'].nunique():,}"
        )

    return df


# =============================================================================
# SHAP DATA
# =============================================================================

@st.cache_data(show_spinner=False)
def load_shap_data():

    if not os.path.exists(SHAP_PATH):
        return pd.DataFrame()

    try:

        shap_data = pd.read_csv(
            SHAP_PATH
        )

        required_columns = [
            "Feature",
            "Mean_Absolute_SHAP"
        ]

        if not all(
            col in shap_data.columns
            for col in required_columns
        ):
            return pd.DataFrame()

        return shap_data.sort_values(
            "Mean_Absolute_SHAP",
            ascending=False
        )

    except Exception:

        return pd.DataFrame()


# =============================================================================
# LOAD RESOURCES
# =============================================================================

try:

    df = load_data()

    shap_df = load_shap_data()

except Exception as exc:

    st.error(
        f"Unable to load resources:\n\n{exc}"
    )

    st.stop()


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
        "equipment_id": str(
            row["equipment_id"]
        ),
        "train": str(
            row["train"]
        )
    }

    for feature in FEATURE_COLUMNS:

        if feature in row:

            value = row[feature]

            payload[feature] = (
                None
                if pd.isna(value)
                else float(value)
            )

    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=120
    )

    if response.status_code != 200:

        raise RuntimeError(
            f"Prediction API returned "
            f"{response.status_code}: "
            f"{response.text}"
        )

    return response.json()


# =============================================================================
# HEADER
# =============================================================================

st.title(
    "🛡️ NLNG Predictive Maintenance IDSS"
)

st.caption(
    "Pipeline 2.0 model-serving and maintenance "
    "decision-support interface"
)


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title(
    "🎛️ IDSS Control Center"
)

api_ok, api_info = check_api()

if api_ok:

    st.sidebar.success(
        "Model API: ONLINE"
    )

else:

    st.sidebar.error(
        "Model API: OFFLINE"
    )

st.sidebar.divider()


# =============================================================================
# TRAIN SELECTION
# =============================================================================

if "train" not in df.columns:

    st.error(
        "The dataset does not contain the required "
        "'train' column."
    )

    st.stop()


trains = sorted(
    df["train"]
    .dropna()
    .unique()
)

if not trains:

    st.error(
        "No LNG Train values are available "
        "in the dataset."
    )

    st.stop()


selected_train = st.sidebar.selectbox(
    "LNG Train",
    trains
)


train_df = df[
    df["train"] == selected_train
]


# =============================================================================
# EQUIPMENT SELECTION
# =============================================================================

assets = sorted(
    train_df["equipment_id"]
    .dropna()
    .unique()
)

if not assets:

    st.error(
        "No equipment assets are available "
        "for the selected LNG Train."
    )

    st.stop()


selected_asset = st.sidebar.selectbox(
    "Equipment",
    assets
)


asset_history = (
    train_df[
        train_df["equipment_id"] == selected_asset
    ]
    .sort_values("timestamp")
    .copy()
)


if asset_history.empty:

    st.error(
        "No observations available "
        "for this equipment."
    )

    st.stop()


asset_current = asset_history.iloc[-1]


# =============================================================================
# PREDICTION
# =============================================================================

prediction = None

if api_ok:

    try:

        prediction = get_prediction(
            asset_current
        )

    except Exception as exc:

        st.error(
            f"Prediction failed:\n\n{exc}"
        )


# =============================================================================
# TOP KPIs
# =============================================================================

if prediction is not None:

    failure_probability = prediction[
        "failure_probability_24h"
    ]

    failure_percent = (
        failure_probability * 100
    )

    failure_risk = prediction[
        "failure_risk"
    ]

    rul_days = prediction[
        "rul_days"
    ]

else:

    failure_percent = np.nan

    failure_risk = "UNAVAILABLE"

    rul_days = np.nan


col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Equipment",
        selected_asset
    )


with col2:

    st.metric(
        "Train",
        selected_train
    )


with col3:

    st.metric(
        "24h Failure Risk",
        (
            f"{failure_percent:.2f}%"
            if prediction is not None
            else "N/A"
        )
    )


with col4:

    st.metric(
        "Estimated RUL",
        (
            f"{rul_days:.2f} days"
            if prediction is not None
            else "N/A"
        )
    )


with col5:

    st.metric(
        "Operating State",
        str(
            asset_current.get(
                "operating_state",
                "Unknown"
            )
        )
    )


st.divider()


# =============================================================================
# ASSET INFORMATION
# =============================================================================

st.subheader(
    "Equipment Information"
)


i1, i2, i3, i4 = st.columns(4)


with i1:

    st.write("**Equipment Name**")

    st.write(
        asset_current.get(
            "equipment_name",
            "N/A"
        )
    )


with i2:

    st.write("**Equipment Type**")

    st.write(
        asset_current.get(
            "equipment_type",
            "N/A"
        )
    )


with i3:

    st.write("**Criticality**")

    st.write(
        asset_current.get(
            "criticality",
            "N/A"
        )
    )


with i4:

    st.write("**Latest Observation**")

    timestamp = asset_current[
        "timestamp"
    ]

    if pd.notna(timestamp):

        st.write(
            timestamp.strftime(
                "%Y-%m-%d %H:%M"
            )
        )

    else:

        st.write("N/A")


st.divider()


# =============================================================================
# CONDITION TRENDS + RISK GAUGE
# =============================================================================

left, right = st.columns(
    [1.7, 1]
)


with left:

    st.subheader(
        "📈 Equipment Condition Trends"
    )

    recent = asset_history.tail(
        500
    ).copy()

    fig = go.Figure()

    if "overall_vibration" in recent.columns:

        fig.add_trace(
            go.Scatter(
                x=recent["timestamp"],
                y=recent[
                    "overall_vibration"
                ],
                name="Overall Vibration",
                mode="lines"
            )
        )

    if "oil_particles_ppm" in recent.columns:

        fig.add_trace(
            go.Scatter(
                x=recent["timestamp"],
                y=recent[
                    "oil_particles_ppm"
                ],
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

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with right:

    st.subheader(
        "🎯 24-Hour Failure Risk"
    )

    gauge_value = (
        failure_percent
        if prediction is not None
        else 0
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=gauge_value,
            title={
                "text":
                "Failure Probability (%)"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "steps": [
                    {
                        "range": [0, 10],
                        "color": "#00CC96"
                    },
                    {
                        "range": [10, 30],
                        "color": "#FFAA00"
                    },
                    {
                        "range": [30, 100],
                        "color": "#FF4B4B"
                    }
                ]
            }
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if prediction is not None:

        if failure_risk == "LOW":

            st.success(
                f"🟢 Risk Level: "
                f"{failure_risk}"
            )

        elif failure_risk == "MODERATE":

            st.warning(
                f"🟡 Risk Level: "
                f"{failure_risk}"
            )

        elif failure_risk == "HIGH":

            st.warning(
                f"🟠 Risk Level: "
                f"{failure_risk}"
            )

        else:

            st.error(
                f"🔴 Risk Level: "
                f"{failure_risk}"
            )


st.divider()


# =============================================================================
# RUL
# =============================================================================

st.subheader(
    "⏳ Remaining Useful Life"
)


if prediction is not None:

    st.metric(
        "Estimated RUL",
        f"{rul_days:.2f} days"
    )

    st.info(
        "RUL is presented as a model-generated "
        "estimate for maintenance planning support "
        "and should not be interpreted as an exact "
        "failure date."
    )

else:

    st.warning(
        "RUL prediction unavailable."
    )


st.divider()


# =============================================================================
# SERVED MODEL INFORMATION
# =============================================================================

st.subheader(
    "🤖 Models Being Served"
)


m1, m2 = st.columns(2)


with m1:

    st.info(
        "**Classification**\n\n"
        "Deep Learning MLP\n\n"
        "24-hour failure-risk prediction"
    )


with m2:

    st.info(
        "**Regression**\n\n"
        "Gradient Boosting Regressor\n\n"
        "Remaining Useful Life estimation"
    )


st.divider()


# =============================================================================
# SHAP GLOBAL DRIVERS
# =============================================================================

st.subheader(
    "🔎 Global Model Drivers"
)


if not shap_df.empty:

    top = shap_df.head(
        10
    ).copy()

    fig = go.Figure(
        go.Bar(
            x=top[
                "Mean_Absolute_SHAP"
            ],
            y=top["Feature"],
            orientation="h"
        )
    )

    fig.update_layout(
        title=(
            "Top Global "
            "Failure-Prediction Drivers"
        ),
        xaxis_title=(
            "Mean Absolute SHAP Value"
        ),
        yaxis_title="Feature",
        template="plotly_dark",
        height=450,
        yaxis={
            "categoryorder":
            "total ascending"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "These are global model drivers from "
        "the Stage 5 SHAP analysis. They describe "
        "overall model behaviour and are not a "
        "case-specific causal explanation."
    )


st.divider()


# =============================================================================
# DECISION SUPPORT
# =============================================================================

st.subheader(
    "💡 Predictive Maintenance Decision Support"
)


if prediction is None:

    st.info(
        "Prediction unavailable. "
        "Check the FastAPI model-serving service."
    )

else:

    if failure_risk == "CRITICAL":

        st.error(
            f"**CRITICAL RISK**\n\n"
            f"{selected_asset} has a high predicted "
            f"probability of failure within the defined "
            f"24-hour horizon.\n\n"
            f"**Suggested decision-support action:**\n"
            f"Prioritise engineering assessment and "
            f"condition review."
        )

    elif failure_risk == "HIGH":

        st.warning(
            f"**HIGH RISK**\n\n"
            f"{selected_asset} requires increased "
            f"monitoring and maintenance review.\n\n"
            f"**Suggested decision-support action:**\n"
            f"Prioritise condition assessment and "
            f"maintenance planning."
        )

    elif failure_risk == "MODERATE":

        st.warning(
            f"**MODERATE RISK**\n\n"
            f"{selected_asset} shows elevated "
            f"predicted failure risk.\n\n"
            f"**Suggested decision-support action:**\n"
            f"Review current condition indicators "
            f"and continue focused monitoring."
        )

    else:

        st.success(
            f"**LOW RISK**\n\n"
            f"{selected_asset} has a low predicted "
            f"probability of failure within the defined "
            f"24-hour horizon.\n\n"
            f"**Suggested decision-support action:**\n"
            f"Continue normal condition monitoring."
        )

    st.caption(
        "The IDSS provides decision-support information. "
        "It does not automatically initiate maintenance, "
        "shutdown, or plant-control actions."
    )


st.divider()


# =============================================================================
# FOOTER
# =============================================================================

st.caption(
    "NLNG Predictive Maintenance IDSS | "
    "Streamlit → FastAPI → Pipeline 2.0 Models"
)
