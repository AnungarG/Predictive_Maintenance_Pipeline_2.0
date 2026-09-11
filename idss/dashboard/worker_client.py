import os
import requests
import io
import pandas as pd
import streamlit as st

def get_cloudflare_config():
    """Retrieve Worker configuration from Streamlit secrets or environment variables."""
    if "cloudflare" in st.secrets:
        worker_url = st.secrets["cloudflare"].get("worker_url", "https://r2-worker.anungar.workers.dev")
        dataset_token = st.secrets["cloudflare"].get("dataset_token", "")
    else:
        worker_url = st.secrets.get("WORKER_URL", os.getenv("WORKER_URL", "https://r2-worker.anungar.workers.dev"))
        dataset_token = st.secrets.get("DATASET_TOKEN", os.getenv("DATASET_TOKEN", ""))
    
    worker_url = worker_url.rstrip("/")
    return worker_url, dataset_token

def get_api_url():
    """Retrieve FastAPI Backend URL from secrets or environment."""
    return st.secrets.get("API_URL", os.getenv("API_URL", "https://nlng-predictive-maintenance-api.onrender.com")).rstrip("/")

def get_headers():
    """Return authorization headers for Worker requests."""
    _, token = get_cloudflare_config()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def check_worker_health():
    """Check Worker health endpoint."""
    worker_url, _ = get_cloudflare_config()
    try:
        response = requests.get(f"{worker_url}/health", timeout=10)
        if response.status_code == 200:
            return True, response.json()
        return False, f"Status code: {response.status_code}"
    except Exception as exc:
        return False, str(exc)

def fetch_dataset_bytes():
    """Fetch the dataset Parquet file from R2 via Cloudflare Worker."""
    worker_url, _ = get_cloudflare_config()
    headers = get_headers()
    
    url = f"{worker_url}/dataset"
    response = requests.get(url, headers=headers, stream=True, timeout=120)
    
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch dataset from Worker ({response.status_code}): {response.text}"
        )
    
    return response.content

def fetch_model_bytes(model_path: str):
    """Fetch model file from Worker (/models/...)."""
    worker_url, _ = get_cloudflare_config()
    headers = get_headers()
    
    clean_path = model_path.lstrip("/")
    url = f"{worker_url}/models/{clean_path}"
    
    response = requests.get(url, headers=headers, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch model from Worker ({response.status_code}): {response.text}"
        )
    return response.content