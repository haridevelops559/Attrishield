"""
Batch Job Schemas.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


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
