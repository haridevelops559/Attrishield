"""
Analytics Request & Response Schemas.
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field


class FilterCondition(BaseModel):
    field: str
    operator: str = Field(..., example="eq", description="eq, neq, gt, gte, lt, lte, in, contains, between")
    value: Any


class AnalyticsQueryRequest(BaseModel):
    batch_id: Optional[str] = None
    filters: Optional[List[FilterCondition]] = None
    group_by: Optional[List[str]] = None
    pivot_rows: Optional[List[str]] = None
    pivot_cols: Optional[List[str]] = None
    pivot_values: Optional[str] = None
    pivot_aggfunc: Optional[str] = "mean"


class GroupByResult(BaseModel):
    group_keys: Dict[str, Any]
    record_count: int
    high_risk_count: int
    review_rate: float
    average_attrition_probability: float
    additional_aggregations: Dict[str, Any] = {}


class AnalyticsQueryResponse(BaseModel):
    total_records: int
    filtered_records: int
    group_by_results: Optional[List[GroupByResult]] = None
    pivot_table: Optional[Dict[str, Any]] = None
    summary_kpis: Dict[str, Any]
