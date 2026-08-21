"""
LLM Insight Schemas.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class OllamaInsightRequest(BaseModel):
    batch_id: Optional[str] = None
    aggregated_statistics: Dict[str, Any]
    department_summary: Optional[List[Dict[str, Any]]] = None
    custom_prompt_notes: Optional[str] = None


class RetentionRecommendation(BaseModel):
    category: str
    action_item: str
    target_segment: str
    priority: str = Field(..., example="HIGH", description="HIGH, MEDIUM, LOW")


class OllamaInsightResponse(BaseModel):
    executive_summary: str
    key_findings: List[str]
    department_insights: List[Dict[str, Any]]
    recommendations: List[RetentionRecommendation]
    limitations_disclaimer: str
    model_used: str
    is_fallback: bool = False
