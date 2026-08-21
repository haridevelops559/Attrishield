"""
Individual Prediction API Endpoints.
Executes single-record inference using canonical V3 feature engineering and XGBoost model pipeline.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends
from pymongo.database import Database
from backend.app.schemas.inference import RawEmployeeInput, PredictionResult
from backend.app.ml.inference import predict_single_employee
from backend.app.db.repositories.prediction_repository import PredictionRepository
from backend.app.api.dependencies import get_db_dep, get_current_user

router = APIRouter(prefix="/inference", tags=["ML Inference"])


@router.post("/predict", response_model=PredictionResult)
def predict_individual_employee(
    payload: RawEmployeeInput,
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user)
):
    """
    Executes single employee attrition prediction.
    Enforces canonical V3 feature engineering, XGBoost inference, and threshold evaluation (0.15).
    Persists prediction log in MongoDB.
    """
    raw_dict = payload.model_dump()
    result = predict_single_employee(raw_dict)
    
    # Store prediction log in database
    pred_doc = {
        "prediction_id": result.prediction_id,
        "batch_id": None,
        "mode": "individual",
        "attrition_probability": result.attrition_probability,
        "attrition_prediction": result.attrition_prediction,
        "selected_threshold": result.selected_threshold,
        "risk_recommendation": result.risk_recommendation,
        "model_version": result.model_version,
        "feature_version": result.feature_version,
        "latency_ms": result.latency_ms,
        "engineered_features": result.engineered_features,
        "created_by": current_user["email"]
    }
    repo = PredictionRepository(database)
    repo.insert_one(pred_doc)

    return result
