
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        BASE_DIR
    )
)

MODELS_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

CLASSIFICATION_DIR = os.path.join(
    MODELS_DIR,
    "classification"
)

REGRESSION_DIR = os.path.join(
    MODELS_DIR,
    "regression"
)

PREPROCESSING_DIR = os.path.join(
    MODELS_DIR,
    "preprocessing"
)


MLP_MODEL_PATH = os.path.join(
    CLASSIFICATION_DIR,
    "dl_classifier_mlp.keras"
)

MLP_IMPUTER_PATH = os.path.join(
    PREPROCESSING_DIR,
    "dl_feature_imputer.pkl"
)

MLP_SCALER_PATH = os.path.join(
    PREPROCESSING_DIR,
    "dl_feature_scaler.pkl"
)

GB_REGRESSOR_PATH = os.path.join(
    REGRESSION_DIR,
    "gradient_boosting_regressor.pkl"
)


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


class ModelService:

    def __init__(self):

        self.classifier = None
        self.feature_imputer = None
        self.feature_scaler = None
        self.regressor = None
        self.loaded = False

        self.load_models()


    def load_models(self):

        self.classifier = tf.keras.models.load_model(
            MLP_MODEL_PATH,
            compile=False
        )

        self.feature_imputer = joblib.load(
            MLP_IMPUTER_PATH
        )

        self.feature_scaler = joblib.load(
            MLP_SCALER_PATH
        )

        self.regressor = joblib.load(
            GB_REGRESSOR_PATH
        )

        self.loaded = True


    def validate_features(self, data):

        missing = [
            feature
            for feature in FEATURE_COLUMNS
            if feature not in data
        ]

        if missing:

            raise ValueError(
                "Missing required features: "
                + ", ".join(missing)
            )


    def prepare_classifier_input(self, data):

        self.validate_features(data)

        X = pd.DataFrame(
            [[
                data[feature]
                for feature in FEATURE_COLUMNS
            ]],
            columns=FEATURE_COLUMNS
        )

        X_imputed = self.feature_imputer.transform(X)

        X_scaled = self.feature_scaler.transform(
            X_imputed
        )

        return X_scaled.astype(
            np.float32
        )


    def predict_failure(self, data):

        X_scaled = self.prepare_classifier_input(
            data
        )

        probability = float(
            self.classifier.predict(
                X_scaled,
                verbose=0
            ).reshape(-1)[0]
        )

        probability = max(
            0.0,
            min(
                1.0,
                probability
            )
        )

        if probability >= 0.70:
            risk_level = "CRITICAL"

        elif probability >= 0.30:
            risk_level = "HIGH"

        elif probability >= 0.10:
            risk_level = "MODERATE"

        else:
            risk_level = "LOW"

        return {
            "failure_probability_24h": probability,
            "failure_risk": risk_level
        }


    def predict_rul(self, data):

        self.validate_features(data)

        X = pd.DataFrame(
            [[
                data[feature]
                for feature in FEATURE_COLUMNS
            ]],
            columns=FEATURE_COLUMNS
        )

        prediction = float(
            np.asarray(
                self.regressor.predict(X)
            ).reshape(-1)[0]
        )

        prediction = max(
            0.0,
            prediction
        )

        return {
            "rul_days": prediction
        }


    def predict(self, data):

        failure_result = self.predict_failure(
            data
        )

        rul_result = self.predict_rul(
            data
        )

        return {
            **failure_result,
            **rul_result,
            "classification_model":
                "Deep Learning MLP",
            "regression_model":
                "Gradient Boosting Regressor"
        }


model_service = ModelService()
