"""
Feature Store Domain Schemas.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class FeatureDefinitionModel(BaseModel):
    feature_name: str
    data_type: str
    entity_type: str = "employee"
    description: str
    formula: Optional[str] = None
    feature_version: str = "v3"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class FeatureGroupModel(BaseModel):
    group_name: str
    description: str
    features: List[str]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class FeatureValueModel(BaseModel):
    entity_id: str
    feature_name: str
    feature_value: Any
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    feature_version: str = "v3"


class FeatureMaterializationRecord(BaseModel):
    materialization_id: str
    batch_id: str
    feature_version: str
    features_materialized: List[str]
    entity_count: int
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "SUCCESS"


class PointInTimeFeatureRequest(BaseModel):
    entity_ids: List[str]
    features: List[str]
    as_of_timestamp: Optional[str] = None
