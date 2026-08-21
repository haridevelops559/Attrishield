"""
Analytics API Endpoints.
Executes dynamic Pandas/NumPy filtering, group-by, pivot tables, and KPI generation.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from fastapi import APIRouter, Depends
from pymongo.database import Database

from backend.app.schemas.analytics import AnalyticsQueryRequest, AnalyticsQueryResponse
from backend.app.analytics.filters import apply_dynamic_filters
from backend.app.analytics.groupby import execute_group_by
from backend.app.analytics.pivot import execute_pivot_table
from backend.app.analytics.metrics import compute_summary_kpis
from backend.app.analytics.charts import generate_chart_data
from backend.app.db.repositories.prediction_repository import PredictionRepository
from backend.app.api.dependencies import get_db_dep, get_current_user

router = APIRouter(prefix="/analytics", tags=["Analytics Engine"])


@router.post("/query", response_model=AnalyticsQueryResponse)
def run_analytics_query(
    request: AnalyticsQueryRequest,
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user)
):
    """
    Executes Pandas/NumPy analytics query across stored predictions or batch records.
    Supports dynamic filtering, group-by, pivot table, and KPI generation.
    """
    repo = PredictionRepository(database)
    
    if request.batch_id:
        predictions = repo.get_by_batch_id(request.batch_id)
    else:
        # Load recent predictions
        if repo.collection is not None:
            predictions = list(repo.collection.find({}, {"_id": 0}).limit(1000))
        else:
            predictions = repo.get_by_batch_id("")  # Fallback

    if not predictions:
        return AnalyticsQueryResponse(
            total_records=0,
            filtered_records=0,
            summary_kpis=compute_summary_kpis(pd.DataFrame())
        )

    # Flatten raw_features and engineered_features for analytical querying
    records = []
    for p in predictions:
        row = {}
        if "raw_features" in p and isinstance(p["raw_features"], dict):
            row.update(p["raw_features"])
        if "engineered_features" in p and isinstance(p["engineered_features"], dict):
            row.update(p["engineered_features"])
        row["attrition_probability"] = p.get("attrition_probability", 0.0)
        row["attrition_prediction"] = p.get("attrition_prediction", 0)
        row["risk_recommendation"] = p.get("risk_recommendation", "")
        records.append(row)

    df = pd.DataFrame(records)
    total_records = len(df)

    # 1. Dynamic Filtering
    filtered_df = apply_dynamic_filters(df, request.filters) if request.filters else df
    filtered_records = len(filtered_df)

    # 2. Group By Aggregations
    group_by_res = execute_group_by(filtered_df, request.group_by) if request.group_by else None

    # 3. Pivot Table Generation
    pivot_res = None
    if request.pivot_rows:
        pivot_res = execute_pivot_table(
            filtered_df,
            index_cols=request.pivot_rows,
            columns_cols=request.pivot_cols or [],
            values_col=request.pivot_values or "attrition_probability",
            aggfunc=request.pivot_aggfunc or "mean"
        )

    # 4. Summary KPIs
    kpis = compute_summary_kpis(filtered_df)

    return AnalyticsQueryResponse(
        total_records=total_records,
        filtered_records=filtered_records,
        group_by_results=group_by_res,
        pivot_table=pivot_res,
        summary_kpis=kpis
    )


@router.get("/charts")
def get_dashboard_charts(
    batch_id: Optional[str] = None,
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user)
):
    """Generates JSON chart specifications for frontend dashboard rendering."""
    repo = PredictionRepository(database)
    if batch_id:
        predictions = repo.get_by_batch_id(batch_id)
    else:
        if repo.collection is not None:
            predictions = list(repo.collection.find({}, {"_id": 0}).limit(1000))
        else:
            predictions = []

    records = []
    for p in predictions:
        row = {}
        if "raw_features" in p and isinstance(p["raw_features"], dict):
            row.update(p["raw_features"])
        row["attrition_probability"] = p.get("attrition_probability", 0.0)
        row["attrition_prediction"] = p.get("attrition_prediction", 0)
        row["risk_recommendation"] = p.get("risk_recommendation", "")
        records.append(row)

    df = pd.DataFrame(records)
    return generate_chart_data(df)
