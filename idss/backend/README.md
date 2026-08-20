# NLNG Predictive Maintenance IDSS - FastAPI

## Models

Classification:
Deep Learning MLP

Regression:
Gradient Boosting Regressor

## Endpoints

GET /
GET /health
GET /model-info
POST /predict
POST /predict/batch

## Start

python -m uvicorn app:app --host 127.0.0.1 --port 8000

## Documentation

http://127.0.0.1:8000/docs
