"""
Unit Tests for Pandas & NumPy Analytics Engine.
"""

import pandas as pd
from backend.app.schemas.analytics import FilterCondition
from backend.app.analytics.filters import apply_dynamic_filters
from backend.app.analytics.groupby import execute_group_by
from backend.app.analytics.pivot import execute_pivot_table
from backend.app.analytics.metrics import compute_summary_kpis


def test_analytics_filtering():
    data = pd.DataFrame([
        {"Department": "Sales", "Age": 30, "MonthlyIncome": 4000},
        {"Department": "R&D", "Age": 45, "MonthlyIncome": 8000},
        {"Department": "Sales", "Age": 25, "MonthlyIncome": 3000}
    ])
    
    filters = [FilterCondition(field="Department", operator="eq", value="Sales")]
    res = apply_dynamic_filters(data, filters)
    assert len(res) == 2
    assert set(res["Department"].unique()) == {"Sales"}


def test_analytics_groupby():
    data = pd.DataFrame([
        {"Department": "Sales", "attrition_prediction": 1, "attrition_probability": 0.45},
        {"Department": "Sales", "attrition_prediction": 0, "attrition_probability": 0.05},
        {"Department": "R&D", "attrition_prediction": 0, "attrition_probability": 0.02}
    ])
    
    groups = execute_group_by(data, ["Department"])
    assert len(groups) == 2
    sales_grp = [g for g in groups if g.group_keys.get("Department") == "Sales"][0]
    assert sales_grp.record_count == 2
    assert sales_grp.high_risk_count == 1
    assert sales_grp.review_rate == 0.5


def test_analytics_pivot_table():
    data = pd.DataFrame([
        {"Department": "Sales", "JobLevel": 1, "attrition_probability": 0.30},
        {"Department": "Sales", "JobLevel": 2, "attrition_probability": 0.10},
        {"Department": "R&D", "JobLevel": 1, "attrition_probability": 0.05}
    ])
    
    pivot = execute_pivot_table(data, index_cols=["Department"], columns_cols=["JobLevel"], values_col="attrition_probability")
    assert "data" in pivot
    assert len(pivot["data"]) == 2
