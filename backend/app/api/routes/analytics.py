"""
Analytics API Endpoints.
Executes dynamic Pandas/NumPy filtering, group-by,
pivot tables, and KPI generation.
"""

from typing import List, Dict, Any, Optional

import pandas as pd
from fastapi import APIRouter, Depends
from pymongo.database import Database

from backend.app.schemas.analytics import (
    AnalyticsQueryRequest,
    AnalyticsQueryResponse,
)
from backend.app.analytics.filters import apply_dynamic_filters
from backend.app.analytics.groupby import execute_group_by
from backend.app.analytics.pivot import execute_pivot_table
from backend.app.analytics.metrics import compute_summary_kpis
from backend.app.analytics.charts import generate_chart_data
from backend.app.db.repositories.prediction_repository import (
    PredictionRepository,
)
from backend.app.api.dependencies import (
    get_db_dep,
    get_current_user,
)

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics Engine"],
)


@router.post(
    "/query",
    response_model=AnalyticsQueryResponse,
)
def run_analytics_query(
    request: AnalyticsQueryRequest,
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user),
):
    """
    Executes Pandas/NumPy analytics query across stored
    individual and batch predictions.

    Supports:
    - Dynamic filtering
    - Group-by
    - Pivot tables
    - Summary KPIs
    """

    repo = PredictionRepository(database)

    # --------------------------------------------------
    # 1. Load prediction records
    # --------------------------------------------------

    if request.batch_id:
        predictions = repo.get_by_batch_id(
            request.batch_id
        )
    else:
        predictions = repo.get_all(
            limit=1000
        )

    # --------------------------------------------------
    # 2. Handle empty dataset
    # --------------------------------------------------

    if not predictions:
        return AnalyticsQueryResponse(
            total_records=0,
            filtered_records=0,
            group_by_results=None,
            pivot_table=None,
            summary_kpis=compute_summary_kpis(
                pd.DataFrame()
            ),
        )

    # --------------------------------------------------
    # 3. Flatten prediction records
    # --------------------------------------------------

    records = []

    for prediction in predictions:

        row = {}

        raw_features = prediction.get(
            "raw_features"
        )

        if isinstance(
            raw_features,
            dict,
        ):
            row.update(
                raw_features
            )

        engineered_features = prediction.get(
            "engineered_features"
        )

        if isinstance(
            engineered_features,
            dict,
        ):
            row.update(
                engineered_features
            )

        # Prediction outputs.
        row["prediction_id"] = prediction.get(
            "prediction_id"
        )

        row["batch_id"] = prediction.get(
            "batch_id"
        )

        row["mode"] = prediction.get(
            "mode"
        )

        row["attrition_probability"] = prediction.get(
            "attrition_probability",
            0.0,
        )

        row["attrition_prediction"] = prediction.get(
            "attrition_prediction",
            0,
        )

        row["selected_threshold"] = prediction.get(
            "selected_threshold",
            0.15,
        )

        row["risk_recommendation"] = prediction.get(
            "risk_recommendation",
            "",
        )

        row["model_version"] = prediction.get(
            "model_version"
        )

        row["feature_version"] = prediction.get(
            "feature_version"
        )

        row["latency_ms"] = prediction.get(
            "latency_ms",
            0.0,
        )

        row["created_by"] = prediction.get(
            "created_by"
        )

        row["created_at"] = prediction.get(
            "created_at"
        )

        records.append(row)

    df = pd.DataFrame(records)

    total_records = len(df)

    # --------------------------------------------------
    # 4. Dynamic filtering
    # --------------------------------------------------

    if request.filters:
        filtered_df = apply_dynamic_filters(
            df,
            request.filters,
        )
    else:
        filtered_df = df

    filtered_records = len(
        filtered_df
    )

    # --------------------------------------------------
    # 5. Group-by
    # --------------------------------------------------

    group_by_res = None

    if request.group_by:
        group_by_res = execute_group_by(
            filtered_df,
            request.group_by,
        )

    # --------------------------------------------------
    # 6. Pivot table
    # --------------------------------------------------

    pivot_res = None

    if request.pivot_rows:

        pivot_res = execute_pivot_table(
            filtered_df,
            index_cols=request.pivot_rows,
            columns_cols=request.pivot_cols or [],
            values_col=(
                request.pivot_values
                or "attrition_probability"
            ),
            aggfunc=(
                request.pivot_aggfunc
                or "mean"
            ),
        )

    # --------------------------------------------------
    # 7. Summary KPIs
    # --------------------------------------------------

    kpis = compute_summary_kpis(
        filtered_df
    )

    # --------------------------------------------------
    # 8. Return analytics response
    # --------------------------------------------------

    return AnalyticsQueryResponse(
        total_records=total_records,
        filtered_records=filtered_records,
        group_by_results=group_by_res,
        pivot_table=pivot_res,
        summary_kpis=kpis,
    )


@router.get("/charts")
def get_dashboard_charts(
    batch_id: Optional[str] = None,
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user),
):
    """
    Generates JSON chart specifications for
    frontend analytics/dashboard rendering.
    """

    repo = PredictionRepository(database)

    # --------------------------------------------------
    # 1. Load records
    # --------------------------------------------------

    if batch_id:
        predictions = repo.get_by_batch_id(
            batch_id
        )
    else:
        predictions = repo.get_all(
            limit=1000
        )

    # --------------------------------------------------
    # 2. Flatten records
    # --------------------------------------------------

    records = []

    for prediction in predictions:

        row = {}

        raw_features = prediction.get(
            "raw_features"
        )

        if isinstance(
            raw_features,
            dict,
        ):
            row.update(
                raw_features
            )

        engineered_features = prediction.get(
            "engineered_features"
        )

        if isinstance(
            engineered_features,
            dict,
        ):
            row.update(
                engineered_features
            )

        row["prediction_id"] = prediction.get(
            "prediction_id"
        )

        row["batch_id"] = prediction.get(
            "batch_id"
        )

        row["attrition_probability"] = prediction.get(
            "attrition_probability",
            0.0,
        )

        row["attrition_prediction"] = prediction.get(
            "attrition_prediction",
            0,
        )

        row["selected_threshold"] = prediction.get(
            "selected_threshold",
            0.15,
        )

        row["risk_recommendation"] = prediction.get(
            "risk_recommendation",
            "",
        )

        records.append(row)

    df = pd.DataFrame(
        records
    )

    # --------------------------------------------------
    # 3. Generate chart data
    # --------------------------------------------------

    return generate_chart_data(
        df
    )