import io
import os
import boto3
import joblib
import pandas as pd
import tensorflow as tf

class ModelService:
    def __init__(self):
        self.s3_client = None
        self.bucket_name = os.getenv("R2_BUCKET_NAME", "nlng-predictive-maintenance")
        self.df_stage2 = None
        self.mlp_classifier = None
        self.lstm_regressor = None
        self.feature_imputer = None
        self.feature_scaler = None
        self.rul_target_scaler = None
        self.loaded = False
        
        self._init_r2_client()

    def _init_r2_client(self):
        account_id = os.getenv("R2_ACCOUNT_ID")
        access_key = os.getenv("R2_ACCESS_KEY_ID")
        secret_key = os.getenv("R2_SECRET_ACCESS_KEY")

        if account_id and access_key and secret_key:
            endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name="auto"
            )

    def load_artifacts_from_r2(self):
        """Downloads ML/DL models and parquet dataset directly from R2."""
        if not self.s3_client:
            print("⚠️ Cloudflare R2 credentials missing. Skipped R2 load.")
            return

        try:
            print("📥 Connecting to Cloudflare R2 bucket...")

            # 1. Fetch Parquet File (Root Level)
            parquet_obj = self.s3_client.get_object(
                Bucket=self.bucket_name, Key="eaned_leakage_controlled.parquet"
            )
            self.df_stage2 = pd.read_parquet(
                io.BytesIO(parquet_obj['Body'].read())
            )
            print(f"✅ Loaded Stage 2 Parquet ({len(self.df_stage2):,} rows).")

            # 2. Load Deep Learning MLP Classifier (.keras format)
            mlp_obj = self.s3_client.get_object(
                Bucket=self.bucket_name, Key="stage_4_deep_learning/dl_classifier_mlp.keras"
            )
            mlp_bytes = io.BytesIO(mlp_obj['Body'].read())
            # Save bytes temporarily or load directly via h5py/keras memory buffer
            with open("/tmp/dl_classifier_mlp.keras", "wb") as f:
                f.write(mlp_bytes.getvalue())
            self.mlp_classifier = tf.keras.models.load_model("/tmp/dl_classifier_mlp.keras")

            # 3. Load Deep Learning LSTM Regressor (.keras format)
            lstm_obj = self.s3_client.get_object(
                Bucket=self.bucket_name, Key="stage_4_deep_learning/dl_regressor_lstm.keras"
            )
            lstm_bytes = io.BytesIO(lstm_obj['Body'].read())
            with open("/tmp/dl_regressor_lstm.keras", "wb") as f:
                f.write(lstm_bytes.getvalue())
            self.lstm_regressor = tf.keras.models.load_model("/tmp/dl_regressor_lstm.keras")

            # 4. Load Scalers and Imputers (.pkl format)
            imputer_obj = self.s3_client.get_object(
                Bucket=self.bucket_name, Key="stage_4_deep_learning/dl_feature_imputer.pkl"
            )
            self.feature_imputer = joblib.load(io.BytesIO(imputer_obj['Body'].read()))

            scaler_obj = self.s3_client.get_object(
                Bucket=self.bucket_name, Key="stage_4_deep_learning/dl_feature_scaler.pkl"
            )
            self.feature_scaler = joblib.load(io.BytesIO(scaler_obj['Body'].read()))

            target_scaler_obj = self.s3_client.get_object(
                Bucket=self.bucket_name, Key="stage_4_deep_learning/dl_rul_target_scaler.pkl"
            )
            self.rul_target_scaler = joblib.load(io.BytesIO(target_scaler_obj['Body'].read()))

            self.loaded = True
            print("✅ Successfully loaded all deep learning artifacts from R2.")

        except Exception as exc:
            print(f"❌ Error loading assets from Cloudflare R2: {exc}")

model_service = ModelService()
