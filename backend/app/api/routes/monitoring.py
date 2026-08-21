"""
Prediction Monitoring API Endpoints.
Replicates and enhances original prediction monitoring metrics (counts, review rates, latency, risk distribution).
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends
from pymongo.database import Database

from backend.app.db.repositories.prediction_repository import PredictionRepository
from backend.app.db.repositories.batch_repository import BatchRepository
from backend.app.ml.model_loader import model_manager
from backend.app.api.dependencies import get_db_dep, get_current_user

router = APIRouter(prefix="/monitoring", tags=["Monitoring & Operations"])


@router.get("/metrics")
def get_monitoring_metrics(
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user)
):
    """
    Returns high-level system monitoring metrics:
    - Total prediction count
    - Batch count
    - Attrition review rate
    - Average prediction probability
    - Latency distribution
    - Model & threshold metadata
    """
    pred_repo = PredictionRepository(database)
    batch_repo = BatchRepository(database)
    
    pred_summary = pred_repo.get_monitoring_summary()
    batches = batch_repo.list_all(limit=100)
    
    _, metadata = model_manager.get_model_and_metadata()

    return {
        "prediction_summary": pred_summary,
        "batch_count": len(batches),
        "active_model_version": metadata.get("model_version", "v3_engineered_without_raw_overtime"),
        "active_feature_version": metadata.get("feature_version", "engineered_features_without_raw_overtime"),
        "selected_threshold": metadata.get("selected_threshold", 0.15),
        "cv_roc_auc": metadata.get("cv_metrics", {}).get("roc_auc_mean", 0.8162),
        "test_roc_auc": metadata.get("test_metrics", {}).get("roc_auc", 0.8058),
        "brier_score": metadata.get("test_metrics", {}).get("brier_score", 0.1049)
    }
