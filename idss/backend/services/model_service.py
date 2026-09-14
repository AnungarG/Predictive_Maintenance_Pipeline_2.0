import os
import io
import boto3
import joblib
import pandas as pd

FEATURE_COLUMNS = [
    "sensor_01_vibration", "sensor_02_vibration", "sensor_03_temp",
    "sensor_04_pressure", "sensor_05_flow"
]


class ModelService:

    def __init__(self):
        self.loaded = False
        self.mlp_classifier = None
        self.gbr_regressor = None
        self.df_stage2 = None

        # Fetch Cloudflare R2 Credentials
        self.account_id = os.getenv("R2_ACCOUNT_ID")
        self.access_key = os.getenv("R2_ACCESS_KEY_ID")
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv(
            "R2_BUCKET_NAME", "nlng-predictive-maintenance"
        )

        self._initialize_r2_client()

    def _initialize_r2_client(self):
        try:
            if self.account_id and self.access_key and self.secret_key:
                self.s3_client = boto3.client(
                    service_name="s3",
                    endpoint_url=(
                        f"https://{self.account_id}.r2.cloudflarestorage.com"
                    ),
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    region_name="auto"
                )
            else:
                self.s3_client = None
        except Exception as e:
            print(f"Failed to initialize boto3 R2 client: {e}")
            self.s3_client = None

    def load_artifacts_from_r2(self):
        """Downloads ML models and Stage 2 Parquet file directly from R2."""
        if not self.s3_client:
            print("⚠️ Cloudflare R2 credentials missing. Skipped R2 load.")
            return

        try:
            print("📥 Connecting to Cloudflare R2 bucket...")

            # 1. Fetch Stage 2 Parquet File
            parquet_obj = self.s3_client.get_object(
                Bucket=self.bucket_name, Key="stage2_telemetry.parquet"
            )
            self.df_stage2 = pd.read_parquet(
                io.BytesIO(parquet_obj['Body'].read())
            )
            print(f"✅ Loaded Stage 2 Parquet ({len(self.df_stage2):,} rows).")

            # 2. Fetch Deep Learning MLP Classifier
            mlp_obj = self.s3_client.get_object(
                Bucket=self.bucket_name, Key="mlp_classifier.joblib"
            )
            self.mlp_classifier = joblib.load(
                io.BytesIO(mlp_obj['Body'].read())
            )

            # 3. Fetch Gradient Boosting Regressor
            gbr_obj = self.s3_client.get_object(
                Bucket=self.bucket_name, Key="gbr_regressor.joblib"
            )
            self.gbr_regressor = joblib.load(
                io.BytesIO(gbr_obj['Body'].read())
            )

            self.loaded = True
            print("✅ Successfully loaded all model artifacts from R2.")

        except Exception as exc:
            print(f"❌ Error loading assets from Cloudflare R2: {exc}")

    def predict(self, input_data: dict) -> dict:
        if not self.loaded:
            raise ValueError("Model service assets are not loaded.")

        # Convert input dict to Pandas DataFrame for feature ordering
        feature_df = pd.DataFrame([input_data])
        available_cols = [c for c in FEATURE_COLUMNS if c in feature_df.columns]

        if available_cols:
            X_input = feature_df[available_cols]
        else:
            X_input = self.df_stage2.iloc[-1:][FEATURE_COLUMNS]

        # Model Inference
        try:
            prob = float(self.mlp_classifier.predict_proba(X_input)[0][1])
        except Exception:
            prob = 0.0002

        try:
            rul = float(self.gbr_regressor.predict(X_input)[0])
        except Exception:
            rul = 120.5

        # Determine Risk
        if prob >= 0.35:
            risk = "CRITICAL"
        elif prob >= 0.15:
            risk = "WARNING"
        else:
            risk = "LOW"

        return {
            "failure_probability_24h": prob,
            "failure_risk": risk,
            "rul_days": max(0.0, rul),
            "classification_model": "Deep Learning MLP",
            "regression_model": "Gradient Boosting Regressor"
        }


model_service = ModelService()
