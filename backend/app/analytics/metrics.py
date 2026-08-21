"""
Summary KPI & Metrics Aggregation Engine.
Calculates high-level statistical summaries across prediction datasets.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np


def compute_summary_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates summary KPIs across a dataset."""
    if df.empty:
        return {
            "total_employees": 0,
            "high_risk_count": 0,
            "low_risk_count": 0,
            "review_rate": 0.0,
            "avg_attrition_probability": 0.0,
            "avg_monthly_income": 0.0,
            "avg_tenure_years": 0.0
        }

    total_records = len(df)
    
    if "attrition_prediction" in df.columns:
        high_risk_count = int((df["attrition_prediction"] == 1).sum())
    else:
        high_risk_count = 0

    low_risk_count = total_records - high_risk_count
    review_rate = round(high_risk_count / total_records, 4) if total_records > 0 else 0.0

    avg_prob = round(float(df["attrition_probability"].mean()), 4) if "attrition_probability" in df.columns else 0.0
    avg_income = round(float(df["MonthlyIncome"].mean()), 2) if "MonthlyIncome" in df.columns else 0.0
    avg_tenure = round(float(df["YearsAtCompany"].mean()), 2) if "YearsAtCompany" in df.columns else 0.0

    return {
        "total_employees": total_records,
        "high_risk_count": high_risk_count,
        "low_risk_count": low_risk_count,
        "review_rate": review_rate,
        "avg_attrition_probability": avg_prob,
        "avg_monthly_income": avg_income,
        "avg_tenure_years": avg_tenure
    }
