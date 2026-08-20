
from typing import List

from fastapi import FastAPI, HTTPException

from schemas.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse
)

from services.model_service import (
    model_service,
    FEATURE_COLUMNS
)


app = FastAPI(
    title="NLNG Predictive Maintenance IDSS API",
    description=(
        "REST model-serving API for the NLNG predictive maintenance "
        "decision-support system."
    ),
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "application": "NLNG Predictive Maintenance IDSS",
        "status": "running",
        "classification_model": "Deep Learning MLP",
        "regression_model": "Gradient Boosting Regressor",
        "feature_count": len(FEATURE_COLUMNS)
    }


@app.get(
    "/health",
    response_model=HealthResponse
)
def health_check():

    if not model_service.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model service is not ready."
        )

    return {
        "status": "healthy",
        "classification_model": "Deep Learning MLP",
        "regression_model": "Gradient Boosting Regressor",
        "feature_count": len(FEATURE_COLUMNS)
    }


@app.get("/model-info")
def model_info():

    return {
        "classification": {
            "model": "Deep Learning MLP",
            "task": "24-hour failure classification",
            "output": "failure probability"
        },
        "regression": {
            "model": "Gradient Boosting Regressor",
            "task": "Remaining Useful Life regression",
            "output": "rul_days"
        },
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):

    try:

        input_data = request.model_dump()

        equipment_id = input_data.pop(
            "equipment_id",
            None
        )

        train = input_data.pop(
            "train",
            None
        )

        result = model_service.predict(
            input_data
        )

        return {
            "equipment_id": equipment_id,
            "train": train,
            "failure_probability_24h":
                result["failure_probability_24h"],
            "failure_risk":
                result["failure_risk"],
            "rul_days":
                result["rul_days"],
            "classification_model":
                result["classification_model"],
            "regression_model":
                result["regression_model"]
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Prediction service error: " + str(exc)
        )


@app.post("/predict/batch")
def predict_batch(
    requests: List[PredictionRequest]
):

    if not requests:

        raise HTTPException(
            status_code=400,
            detail="No prediction records supplied."
        )

    results = []

    for request in requests:

        try:

            input_data = request.model_dump()

            equipment_id = input_data.pop(
                "equipment_id",
                None
            )

            train = input_data.pop(
                "train",
                None
            )

            result = model_service.predict(
                input_data
            )

            results.append({
                "equipment_id": equipment_id,
                "train": train,
                "failure_probability_24h":
                    result["failure_probability_24h"],
                "failure_risk":
                    result["failure_risk"],
                "rul_days":
                    result["rul_days"],
                "classification_model":
                    result["classification_model"],
                "regression_model":
                    result["regression_model"]
            })

        except Exception as exc:

            raise HTTPException(
                status_code=422,
                detail="Batch record failed: " + str(exc)
            )

    return {
        "count": len(results),
        "predictions": results
    }
