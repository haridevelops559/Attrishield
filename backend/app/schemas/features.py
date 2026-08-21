"""
Feature Store Schemas.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class FeatureDefinitionSchema(BaseModel):
    feature_name: str
    data_type: str
    entity_type: str = "employee"
    description: str
    formula: Optional[str] = None
    feature_version: str = "v3"
    created_at: str


class FeatureGroupSchema(BaseModel):
    group_name: str
    description: str
    features: List[str]
    created_at: str


class FeatureValueSchema(BaseModel):
    entity_id: str
    feature_name: str
    feature_value: Any
    timestamp: str
    feature_version: str


class MaterializationRequest(BaseModel):
    batch_id: str
    feature_version: str = "v3"


class FeatureLineageSchema(BaseModel):
    feature_name: str
    source_columns: List[str]
    transformation_logic: str
    downstream_consumers: List[str]


class PointInTimeFeatureRequest(BaseModel):
    entity_ids: List[str]
    features: List[str]
    as_of_timestamp: Optional[str] = None
