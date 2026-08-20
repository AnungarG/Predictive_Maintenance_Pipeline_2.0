# NLNG Predictive Maintenance — Predictive Maintenance Pipeline 2.0

An end-to-end, modular machine learning pipeline engineered to predict equipment failures and estimate Remaining Useful Life (RUL) for industrial telemetry data. 

The pipeline runs through six stages: Data Audit, Data Cleaning & Feature Engineering, ML Baselines, Deep Learning, SHAP Model Explainability, and Automated Model Comparison/Artifact Selection.

---

## 🛠 Project Architecture

```text
Predictive_Maintenance/
├── configs/
│   └── pipeline_config.json      # Central configuration for paths & params
├── notebooks/
│   └── Research.ipynb            # Exploratory research & prototype analysis
├── scripts/
│   └── run_pipeline.py           # Production execution script
├── src/
│   └── predictive_maintenance/   # Core python package
│       ├── stage1_audit.py       # Data validation & missingness audit
│       ├── stage2_cleaning.py    # Imputation & feature engineering
│       ├── stage3_ml.py          # Baseline ML models (RF, XGBoost, LogReg)
│       ├── stage4_deep_learning.py# Deep Learning (MLP & LSTM)
│       ├── stage5_explainability.py# Global SHAP feature importance
│       └── stage6_comparison.py  # Model evaluation & artifact selection
├── tests/                        # Pytest suite & feature contract assertions
├── README.md
└── requirements.txt
```

---

## 📊 Pipeline Stages & Methodology

1. **Stage 1: Data Audit & Validation**
   * Inspects incoming sensor telemetry schemas, missing values, and data distributions.
2. **Stage 2: Cleaning & Feature Engineering**
   * Handles missing value imputation, creates indicator flags, and builds aggregated rolling sensor metrics (vibration, oil condition, temperature trends).
3. **Stage 3: Machine Learning Baselines**
   * Trains Random Forest, Gradient Boosting, and Logistic Regression models across dual targets: binary failure window (`failure_within_24h`) and continuous RUL (days).
4. **Stage 4: Deep Learning Benchmarking**
   * Builds and evaluates Multi-Layer Perceptron (MLP) and LSTM architectures to capture non-linear temporal dynamics.
5. **Stage 5: SHAP Explainability**
   * Calculates global SHAP values across 5,000 holdout samples to identify key failure drivers.
6. **Stage 6: Champion Selection & Deployment**
   * Evaluates validation metrics, selects champion models, and exports lightweight binary deployment artifacts.

---

## 🏆 Current Champion Models

| Task | Champion Model | Target Metric | Holdout ROC-AUC / RMSE | Key Predictors |
| :--- | :--- | :--- | :--- | :--- |
| **Classification** | Deep Learning MLP | `failure_within_24h` | **0.9487 ROC-AUC** (0.2677 PR-AUC) | Vibration, Oil Particles (PPM) |
| **RUL Regression** | Gradient Boosting Regressor | `remaining_useful_life` | **83.79 Days RMSE** | Overall Vibration, Temperature |

---

## ⚡ Quick Start

### 1. Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/golfsong707/Predictive_Maintenance.git
cd Predictive_Maintenance
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

Execute all six stages end-to-end:

```bash
python scripts/run_predictive_maintenance_pipeline.py
```

*Or run directly as a module:*

```bash
python -m predictive_maintenance
```

### 3. Run Test Suite

Run unit tests and verify feature contract specs:

```bash
python -m pytest -v
```

---

## 💡 Real-Time Inference Example

To score incoming telemetry using the trained production artifacts:

```python
import joblib
import tensorflow as tf

# Load binary classifier artifact
classifier = tf.keras.models.load_model("data/04_corrected_pipeline/stage_4_deep_learning/dl_classifier_mlp.keras")

# Load RUL regressor artifact
rul_model = joblib.load("data/04_corrected_pipeline/stage_3_ml/gradient_boosting_regressor.pkl")

# Predict failure probability & estimated remaining life
failure_prob = classifier.predict(X_new_scaled)
estimated_rul = rul_model.predict(X_new)
```

---

## ⚙️ Configuration

Modify paths, split ratios, and hyperparameters without changing core code by editing `configs/pipeline_config.json`.
