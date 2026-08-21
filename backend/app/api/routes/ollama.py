"""
Ollama AI Insights Endpoints.
Triggers grounded retention insight generation using local Ollama model (Qwen).
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends
from pymongo.database import Database

from backend.app.schemas.analytics import AnalyticsQueryRequest
from backend.app.llm.schemas import OllamaInsightRequest, OllamaInsightResponse
from backend.app.llm.insight_engine import InsightEngine
from backend.app.db.repositories.prediction_repository import PredictionRepository
from backend.app.analytics.metrics import compute_summary_kpis
from backend.app.analytics.groupby import execute_group_by
from backend.app.api.dependencies import get_db_dep, get_current_user
import pandas as pd

router = APIRouter(prefix="/ollama", tags=["GenAI & Insights"])


@router.post("/insights", response_model=OllamaInsightResponse)
async def generate_ai_retention_insights(
    payload: OllamaInsightRequest,
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user)
):
    """
    Generates executive retention insights using local Ollama (Qwen).
    Input must consist of pre-aggregated statistical summaries (never raw employee records).
    """
    engine = InsightEngine()
    
    # If stats not provided, calculate from DB or payload
    stats = payload.aggregated_statistics
    dept_summary = payload.department_summary
    
    if not stats and payload.batch_id:
        repo = PredictionRepository(database)
        preds = repo.get_by_batch_id(payload.batch_id)
        if preds:
            records = []
            for p in preds:
                row = p.get("raw_features", {}).copy()
                row.update(p.get("engineered_features", {}))
                row["attrition_probability"] = p.get("attrition_probability", 0.0)
                row["attrition_prediction"] = p.get("attrition_prediction", 0)
                records.append(row)
            df = pd.DataFrame(records)
            stats = compute_summary_kpis(df)
            
            # Dept breakdown
            dept_res = execute_group_by(df, ["Department"])
            dept_summary = [
                {
                    "department": str(g.group_keys.get("Department")),
                    "total": g.record_count,
                    "high_risk": g.high_risk_count,
                    "avg_probability": g.average_attrition_probability
                }
                for g in dept_res
            ]

    res = await engine.generate_insights(
        aggregated_stats=stats,
        dept_summary=dept_summary,
        custom_notes=payload.custom_prompt_notes
    )
    return res
