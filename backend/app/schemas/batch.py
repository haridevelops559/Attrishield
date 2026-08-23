"""
Batch Job Schemas.
"""

from typing import Optional, List, Dict, Any

from pydantic import BaseModel


class BatchJobCreateResponse(BaseModel):
    batch_id: str
    status: str
    filename: str
    row_count: int
    created_at: str


class BatchJobStatusResponse(BaseModel):
    batch_id: str
    filename: str
    status: str
    row_count: int
    high_risk_count: Optional[int] = 0
    review_rate: Optional[float] = 0.0
    average_latency_ms: Optional[float] = 0.0
    model_version: str
    feature_version: str
    threshold: float
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    prediction_id: str
    batch_id: str
    mode: str

    attrition_probability: float
    attrition_prediction: int
    selected_threshold: float
    risk_recommendation: str

    model_version: str
    feature_version: str
    latency_ms: float

    engineered_features: Dict[str, Any]
    raw_features: Optional[Dict[str, Any]] = None

    created_by: Optional[str] = None
    created_at: Optional[str] = None