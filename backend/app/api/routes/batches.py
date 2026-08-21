"""
Batch Inference API Endpoints.
Handles CSV upload, batch execution pipeline, feature materialization, and status polling.
"""

import io
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import pandas as pd
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from pymongo.database import Database

from backend.app.schemas.batch import BatchJobCreateResponse, BatchJobStatusResponse
from backend.app.ml.inference import predict_batch_dataframe
from backend.app.feature_store.service import FeatureStoreService
from backend.app.db.repositories.batch_repository import BatchRepository
from backend.app.db.repositories.prediction_repository import PredictionRepository
from backend.app.api.dependencies import get_db_dep, get_current_user
from backend.app.core.logging import logger

router = APIRouter(prefix="/batches", tags=["Batch Pipelines"])


@router.post("", response_model=BatchJobStatusResponse)
async def create_and_run_batch(
    file: UploadFile = File(...),
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user)
):
    """
    Uploads a CSV file of employee records and triggers the batch pipeline:
    CSV Upload -> Schema Validation -> Canonical Feature Engineering -> Feature Store Materialization -> XGBoost Inference -> MongoDB Persistence.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file format: {e}")

    row_count = len(df)
    if row_count == 0:
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    batch_id = f"batch_{uuid.uuid4().hex[:10]}"
    now_str = datetime.now(timezone.utc).isoformat()

    batch_repo = BatchRepository(database)
    pred_repo = PredictionRepository(database)
    fs_service = FeatureStoreService(database)

    # Initial batch record
    batch_doc = {
        "batch_id": batch_id,
        "filename": file.filename,
        "status": "PROCESSING",
        "row_count": row_count,
        "high_risk_count": 0,
        "review_rate": 0.0,
        "average_latency_ms": 0.0,
        "model_version": "v3_engineered_without_raw_overtime",
        "feature_version": "engineered_features_without_raw_overtime",
        "threshold": 0.15,
        "created_at": now_str,
        "created_by": current_user["email"]
    }
    batch_repo.create(batch_doc)

    try:
        # 1. Feature Store Materialization
        fs_service.materializer.materialize_batch(df, batch_id=batch_id, feature_version="v3")

        # 2. Vectorized Batch Prediction
        result_df, summary = predict_batch_dataframe(df, batch_id=batch_id)

        # 3. Store Prediction Records in MongoDB
        prediction_docs = []
        for idx, row in result_df.iterrows():
            eng_dict = {
                "IncomePerJobLevel": float(row.get("IncomePerJobLevel", 0.0)),
                "PromotionStagnationRatio": float(row.get("PromotionStagnationRatio", 0.0)),
                "ManagerTenureRatio": float(row.get("ManagerTenureRatio", 0.0)),
                "RoleTenureRatio": float(row.get("RoleTenureRatio", 0.0)),
                "OverTimeBinary": int(row.get("OverTimeBinary", 0)),
                "CommuteOvertimeBurden": float(row.get("CommuteOvertimeBurden", 0.0)),
                "EarlyCareerFlag": int(row.get("EarlyCareerFlag", 0))
            }
            pred_doc = {
                "prediction_id": f"pred_{batch_id}_{idx}",
                "batch_id": batch_id,
                "mode": "batch",
                "attrition_probability": float(row["attrition_probability"]),
                "attrition_prediction": int(row["attrition_prediction"]),
                "selected_threshold": summary.threshold_used,
                "risk_recommendation": str(row["risk_recommendation"]),
                "model_version": summary.model_version,
                "feature_version": "engineered_features_without_raw_overtime",
                "latency_ms": summary.average_latency_ms,
                "engineered_features": eng_dict,
                "raw_features": row.to_dict(),
                "created_by": current_user["email"]
            }
            prediction_docs.append(pred_doc)

        pred_repo.insert_many(prediction_docs)

        # 4. Update Batch Status
        completed_str = datetime.now(timezone.utc).isoformat()
        updated_batch = batch_repo.update_status(
            batch_id=batch_id,
            status="COMPLETED",
            completed_at=completed_str,
            high_risk_count=summary.high_risk_count,
            average_latency_ms=summary.average_latency_ms
        )
        updated_batch["review_rate"] = summary.review_rate

        return BatchJobStatusResponse(**updated_batch)

    except Exception as e:
        logger.error(f"Batch processing error for {batch_id}: {e}")
        batch_repo.update_status(batch_id=batch_id, status="FAILED", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch pipeline failure: {e}")


@router.get("/{batch_id}", response_model=BatchJobStatusResponse)
def get_batch_status(
    batch_id: str,
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user)
):
    """Retrieves status and summary statistics for a batch job."""
    repo = BatchRepository(database)
    batch = repo.get_by_id(batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch job not found.")
    return BatchJobStatusResponse(**batch)


@router.get("", response_model=List[BatchJobStatusResponse])
def list_batches(
    limit: int = 50,
    skip: int = 0,
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user)
):
    """Lists recent batch jobs."""
    repo = BatchRepository(database)
    batches = repo.list_all(limit=limit, skip=skip)
    return [BatchJobStatusResponse(**b) for b in batches]
