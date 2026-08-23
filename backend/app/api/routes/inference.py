"""
Individual Prediction API Endpoints.

Executes single-record inference using the canonical V3
feature engineering and XGBoost model pipeline.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pymongo.database import Database

from backend.app.api.dependencies import (
    get_current_user,
    get_db_dep,
)
from backend.app.db.repositories.prediction_repository import (
    PredictionRepository,
)
from backend.app.ml.inference import predict_single_employee
from backend.app.schemas.inference import (
    PredictionDetailResponse,
    PredictionResult,
    RawEmployeeInput,
)

router = APIRouter(
    prefix="/inference",
    tags=["ML Inference"],
)


@router.post(
    "/predict",
    response_model=PredictionResult,
)
def predict_individual_employee(
    payload: RawEmployeeInput,
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user),
):
    """
    Execute single employee attrition prediction.

    The original employee attributes are persisted as
    raw_features so the Analytics engine can later
    filter/group predictions by HR dimensions.
    """

    raw_dict = payload.model_dump()

    result = predict_single_employee(raw_dict)

    prediction_doc = {
        "prediction_id": result.prediction_id,
        "batch_id": None,
        "mode": "individual",

        # IMPORTANT:
        # Preserve the original employee attributes.
        "raw_features": raw_dict,

        # Model output.
        "attrition_probability": result.attrition_probability,
        "attrition_prediction": result.attrition_prediction,
        "selected_threshold": result.selected_threshold,
        "risk_recommendation": result.risk_recommendation,

        # Model metadata.
        "model_version": result.model_version,
        "feature_version": result.feature_version,
        "latency_ms": result.latency_ms,

        # V3 engineered features.
        "engineered_features": result.engineered_features,

        # Audit information.
        "created_by": current_user["email"],
    }

    repository = PredictionRepository(database)

    repository.insert_one(prediction_doc)

    return result


@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionDetailResponse,
)
def get_prediction(
    prediction_id: str,
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user),
):
    """
    Retrieve a previously generated prediction by ID.
    """

    repository = PredictionRepository(database)

    prediction = repository.get_by_prediction_id(
        prediction_id
    )

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="Prediction not found.",
        )

    return PredictionDetailResponse(
        **prediction
    )