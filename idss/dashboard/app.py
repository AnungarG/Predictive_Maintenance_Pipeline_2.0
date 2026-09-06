import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import gdown

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NLNG Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# ============================================================
# CONFIGURATION
# ============================================================

API_URL = st.secrets.get(
    "API_URL",
    "http://127.0.0.1:8000"
)

SHAP_PATH = (
    "data/04_corrected_pipeline/"
    "stage_5_explainability/"
    "shap_feature_importance.csv"
)

# ============================================================
# MODEL FEATURES
# ============================================================

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

# ============================================================
# DASHBOARD COLUMNS
# ============================================================

DASHBOARD_COLUMNS = [
    "timestamp",
    "train",
    "equipment_id",
    "equipment_name",
    "equipment_type",
    "operating_state",
    "criticality"
] + FEATURE_COLUMNS


# ============================================================
# LOAD NLNG DATASET
# ============================================================

@st.cache_data(show_spinner="Loading NLNG dataset...")
def load_data():

    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    file_path = os.path.join(
        data_dir,
        "NLNG_cleaned_leakage_controlled.parquet"
    )

    # --------------------------------------------------------
    # GOOGLE DRIVE FILE
    # --------------------------------------------------------

    google_drive_file_id = (
        "1glib_3N3PuvQtnvr8s8NN-MGITLLjcZ"
    )

    google_drive_url = (
        "https://drive.google.com/uc"
        f"?id={google_drive_file_id}"
    )

    # --------------------------------------------------------
    # DOWNLOAD ONLY IF FILE DOES NOT EXIST
    # --------------------------------------------------------

    if not os.path.exists(file_path):

        st.info(
            "NLNG dataset is not available locally. "
            "Downloading from Google Drive..."
        )

        try:

            downloaded_file = gdown.download(
                url=google_drive_url,
                output=file_path,
                quiet=False
            )

            # ------------------------------------------------
            # CHECK 1 — DOWNLOAD RESULT
            # ------------------------------------------------

            if downloaded_file is None:
                raise RuntimeError(
                    "Google Drive did not return the dataset."
                )

            # ------------------------------------------------
            # CHECK 2 — FILE EXISTS
            # ------------------------------------------------

            if not os.path.exists(file_path):
                raise RuntimeError(
                    "The download completed without creating "
                    "the expected Parquet file."
                )

            # ------------------------------------------------
            # CHECK 3 — FILE SIZE
            # ------------------------------------------------

            file_size = os.path.getsize(file_path)

            if file_size == 0:

                os.remove(file_path)

                raise RuntimeError(
                    "Google Drive returned an empty file."
                )

            # ------------------------------------------------
            # CHECK 4 — DETECT HTML RESPONSE
            # ------------------------------------------------

            with open(file_path, "rb") as f:
                file_header = f.read(500).lower()

            html_signatures = [
                b"<html",
                b"<!doctype",
                b"<head",
                b"google drive"
            ]

            if any(
                signature in file_header
                for signature in html_signatures
            ):

                os.remove(file_path)

                raise RuntimeError(
                    "Google Drive returned an HTML page "
                    "instead of the Parquet dataset. "
                    "The file may be unavailable for "
                    "direct download."
                )

            # ------------------------------------------------
            # CHECK 5 — VALIDATE PARQUET
            # ------------------------------------------------

            try:

                test_df = pd.read_parquet(
                    file_path
                )

                if test_df.empty:

                    os.remove(file_path)

                    raise RuntimeError(
                        "The downloaded Parquet file "
                        "contains no records."
                    )

                del test_df

            except Exception as parquet_error:

                if os.path.exists(file_path):
                    os.remove(file_path)

                raise RuntimeError(
                    "The downloaded file is not a valid "
                    "Parquet dataset."
                ) from parquet_error

            st.success(
                "NLNG dataset downloaded successfully "
                f"({file_size / (1024 * 1024):.1f} MB)."
            )

        except Exception as e:

            if os.path.exists(file_path):

                try:
                    os.remove(file_path)
                except Exception:
                    pass

            st.error(
                "Unable to download the NLNG dataset "
                "from Google Drive."
            )

            st.code(
                str(e),
                language="text"
            )

            st.info(
                "Please verify that the Google Drive file "
                "is shared as 'Anyone with the link → Viewer' "
                "and that it is available for download."
            )

            st.stop()

    # --------------------------------------------------------
    # FILE ALREADY EXISTS
    # --------------------------------------------------------

    else:

        file_size = os.path.getsize(file_path)

        if file_size == 0:

            os.remove(file_path)

            st.error(
                "The local NLNG dataset file is empty."
            )

            st.stop()

    # ========================================================
    # READ PARQUET
    # ========================================================

    try:

        df = pd.read_parquet(
            file_path
        )

    except Exception as e:

        st.error(
            "The NLNG dataset could not be read "
            "as a Parquet file."
        )

        st.code(
            str(e),
            language="text"
        )

        st.stop()

    # ========================================================
    # SELECT AVAILABLE DASHBOARD COLUMNS
    # ========================================================

    available_cols = [
        col
        for col in DASHBOARD_COLUMNS
        if col in df.columns
    ]

    missing_cols = [
        col
        for col in DASHBOARD_COLUMNS
        if col not in df.columns
    ]

    if missing_cols:

        st.warning(
            "Some expected dashboard columns are missing "
            "from the dataset:"
        )

        st.write(
            missing_cols
        )

    df = df[
        available_cols
    ].copy()

    # ========================================================
    # TIMESTAMP VALIDATION
    # ========================================================

    if "timestamp" not in df.columns:

        st.error(
            "The dataset does not contain the required "
            "'timestamp' column."
        )

        st.stop()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    invalid_timestamps = (
        df["timestamp"].isna().sum()
    )

    if invalid_timestamps > 0:

        st.warning(
            f"{invalid_timestamps:,} records contain "
            "invalid timestamps and will be removed."
        )

        df = df.dropna(
            subset=["timestamp"]
        )

    # ========================================================
    # EQUIPMENT ID VALIDATION
    # ========================================================

    if "equipment_id" not in df.columns:

        st.error(
            "The dataset does not contain the required "
            "'equipment_id' column."
        )

        st.stop()

    # ========================================================
    # SORT DATA
    # ========================================================

    df = df.sort_values(
        ["equipment_id", "timestamp"]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if df.empty:

        st.error(
            "The NLNG dataset contains no usable records."
        )

        st.stop()

    return df


# ============================================================
# LOAD SHAP FEATURE IMPORTANCE
# ============================================================

@st.cache_data
def load_shap():

    if not os.path.exists(SHAP_PATH):
        return None

    try:

        shap_df = pd.read_csv(
            SHAP_PATH
        )

        return shap_df

    except Exception:

        return None


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()

shap_df = load_shap()


# ============================================================
# HEADER
# ============================================================

st.title(
    "NLNG Predictive Maintenance Dashboard"
)

st.markdown(
    """
    **Equipment Failure Prediction and Maintenance
    Decision Support System**

    This dashboard provides equipment-level condition
    monitoring, failure-risk assessment and maintenance
    decision support for LNG plant operations.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Equipment Selection"
)

# ------------------------------------------------------------
# TRAIN SELECTION
# ------------------------------------------------------------

if "train" in df.columns:

    trains = sorted(
        df["train"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_train = st.sidebar.selectbox(
        "Select LNG Train",
        trains
    )

    train_df = df[
        df["train"].astype(str)
        == selected_train
    ].copy()

else:

    selected_train = None

    train_df = df.copy()


# ------------------------------------------------------------
# EQUIPMENT SELECTION
# ------------------------------------------------------------

equipment_list = sorted(
    train_df["equipment_id"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

if not equipment_list:

    st.error(
        "No equipment is available for the selected train."
    )

    st.stop()

selected_equipment = st.sidebar.selectbox(
    "Select Equipment",
    equipment_list
)


# ============================================================
# SELECTED EQUIPMENT DATA
# ============================================================

equipment_df = train_df[
    train_df["equipment_id"].astype(str)
    == selected_equipment
].copy()

equipment_df = equipment_df.sort_values(
    "timestamp"
)


if equipment_df.empty:

    st.warning(
        "No records found for the selected equipment."
    )

    st.stop()


latest_row = equipment_df.iloc[-1]


# ============================================================
# EQUIPMENT INFORMATION
# ============================================================

st.subheader(
    "Equipment Information"
)

info_col1, info_col2, info_col3, info_col4 = st.columns(4)

with info_col1:

    st.metric(
        "Equipment ID",
        selected_equipment
    )

with info_col2:

    if "equipment_name" in equipment_df.columns:

        value = latest_row.get(
            "equipment_name",
            "N/A"
        )

        st.metric(
            "Equipment Name",
            str(value)
        )

with info_col3:

    if "equipment_type" in equipment_df.columns:

        value = latest_row.get(
            "equipment_type",
            "N/A"
        )

        st.metric(
            "Equipment Type",
            str(value)
        )

with info_col4:

    if "criticality" in equipment_df.columns:

        value = latest_row.get(
            "criticality",
            "N/A"
        )

        st.metric(
            "Criticality",
            str(value)
        )


# ============================================================
# FASTAPI HEALTH CHECK
# ============================================================

api_status = "Unavailable"

try:

    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )

    if health_response.status_code == 200:
        api_status = "Online"

except Exception:

    api_status = "Unavailable"


# ============================================================
# PREDICTION
# ============================================================

st.subheader(
    "Failure Risk Assessment"
)

prediction = None

try:

    prediction_payload = {
        "equipment_id": selected_equipment
    }

    if selected_train is not None:
        prediction_payload["train"] = selected_train

    # --------------------------------------------------------
    # ADD MODEL FEATURES
    # --------------------------------------------------------

    for feature in FEATURE_COLUMNS:

        if feature in latest_row.index:

            value = latest_row[feature]

            if pd.isna(value):
                value = 0

            prediction_payload[feature] = float(
                value
            )

    response = requests.post(
        f"{API_URL}/predict",
        json=prediction_payload,
        timeout=15
    )

    if response.status_code == 200:

        prediction = response.json()

    else:

        st.warning(
            "Prediction API returned an error."
        )

        st.code(
            response.text,
            language="text"
        )

except Exception as e:

    st.warning(
        "Unable to connect to the prediction API."
    )

    st.code(
        str(e),
        language="text"
    )


# ============================================================
# DISPLAY PREDICTION
# ============================================================

if prediction is not None:

    prediction_col1, prediction_col2, prediction_col3 = (
        st.columns(3)
    )

    # --------------------------------------------------------
    # FAILURE PROBABILITY
    # --------------------------------------------------------

    failure_probability = prediction.get(
        "failure_probability_24h",
        prediction.get(
            "failure_probability",
            None
        )
    )

    if failure_probability is not None:

        try:

            failure_probability = float(
                failure_probability
            )

            if failure_probability <= 1:
                risk_percentage = (
                    failure_probability * 100
                )
            else:
                risk_percentage = failure_probability

            with prediction_col1:

                st.metric(
                    "Failure Risk",
                    f"{risk_percentage:.2f}%"
                )

        except Exception:

            with prediction_col1:

                st.metric(
                    "Failure Risk",
                    "N/A"
                )

    # --------------------------------------------------------
    # RUL
    # --------------------------------------------------------

    rul = prediction.get(
        "rul_days",
        prediction.get(
            "RUL_Days",
            None
        )
    )

    with prediction_col2:

        if rul is not None:

            try:

                st.metric(
                    "Estimated RUL",
                    f"{float(rul):.1f} days"
                )

            except Exception:

                st.metric(
                    "Estimated RUL",
                    str(rul)
                )

        else:

            st.metric(
                "Estimated RUL",
                "N/A"
            )

    # --------------------------------------------------------
    # API STATUS
    # --------------------------------------------------------

    with prediction_col3:

        st.metric(
            "Prediction API",
            api_status
        )


# ============================================================
# CONDITION TRENDS
# ============================================================

st.subheader(
    "Equipment Condition Trends"
)

trend_features = [
    "vibration",
    "bearing_temperature",
    "rpm",
    "motor_current",
    "oil_pressure"
]

available_trend_features = [
    feature
    for feature in trend_features
    if feature in equipment_df.columns
]

for feature in available_trend_features:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=equipment_df["timestamp"],
            y=equipment_df[feature],
            mode="lines",
            name=feature.replace(
                "_",
                " "
            ).title()
        )
    )

    fig.update_layout(
        title=feature.replace(
            "_",
            " "
        ).title(),
        xaxis_title="Timestamp",
        yaxis_title=feature.replace(
            "_",
            " "
        ).title(),
        height=350
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# FAILURE-RISK GAUGE
# ============================================================

if prediction is not None:

    failure_probability = prediction.get(
        "failure_probability_24h",
        prediction.get(
            "failure_probability",
            None
        )
    )

    if failure_probability is not None:

        try:

            probability = float(
                failure_probability
            )

            if probability <= 1:
                probability *= 100

            probability = max(
                0,
                min(
                    probability,
                    100
                )
            )

            st.subheader(
                "Failure Risk Gauge"
            )

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability,
                    title={
                        "text": "Failure Risk (%)"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        }
                    }
                )
            )

            gauge.update_layout(
                height=350
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

        except Exception:
            pass


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader(
    "Model Information"
)

model_col1, model_col2 = st.columns(2)

with model_col1:

    st.write(
        "**Prediction System:** "
        "NLNG Predictive Maintenance"
    )

    st.write(
        "**Primary Model:** Random Forest"
    )

with model_col2:

    st.write(
        "**Primary Target:** "
        "Failure_Within_7d"
    )

    st.write(
        "**Deployment API:** FastAPI"
    )


# ============================================================
# SHAP FEATURE IMPORTANCE
# ============================================================

if shap_df is not None:

    st.subheader(
        "Model Explainability"
    )

    st.write(
        "Feature importance from the model "
        "explainability analysis."
    )

    st.dataframe(
        shap_df,
        use_container_width=True
    )


# ============================================================
# DECISION SUPPORT
# ============================================================

st.subheader(
    "Maintenance Decision Support"
)

if prediction is not None:

    failure_probability = prediction.get(
        "failure_probability_24h",
        prediction.get(
            "failure_probability",
            None
        )
    )

    if failure_probability is not None:

        try:

            probability = float(
                failure_probability
            )

            if probability <= 1:
                probability *= 100

            if probability >= 70:

                st.error(
                    "HIGH RISK: Immediate maintenance "
                    "inspection is recommended."
                )

            elif probability >= 40:

                st.warning(
                    "MEDIUM RISK: Schedule a maintenance "
                    "inspection and closely monitor equipment."
                )

            else:

                st.success(
                    "LOW RISK: Continue normal monitoring "
                    "and planned maintenance."
                )

        except Exception:

            st.info(
                "Review the equipment condition indicators "
                "and maintenance history."
            )

    else:

        st.info(
            "Prediction risk is currently unavailable."
        )

else:

    st.info(
        "Connect to the prediction API to obtain "
        "maintenance recommendations."
    )


# ============================================================
# DATASET DIAGNOSTICS
# ============================================================

with st.sidebar.expander(
    "Dataset Diagnostics"
):

    st.write(
        f"Records: {len(df):,}"
    )

    st.write(
        f"Equipment: {df['equipment_id'].nunique():,}"
    )

    if "train" in df.columns:

        st.write(
            f"Trains: {df['train'].nunique():,}"
        )

    st.write(
        "Date range:"
    )

    st.write(
        f"{df['timestamp'].min()} → "
        f"{df['timestamp'].max()}"
    )

    dataset_size_mb = (
        os.path.getsize(file_path)
        / (1024 * 1024)
    )

    st.write(
        f"Dataset size: {dataset_size_mb:.1f} MB"
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NLNG Predictive Maintenance Analytics | "
    "Equipment Failure Prediction and "
    "Maintenance Decision Support"
