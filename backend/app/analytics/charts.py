"""
Chart Data Formatting Engine.
Generates structured JSON chart specifications for frontend dashboard rendering.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def generate_chart_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates chart-ready JSON structures for:
    - Department Attrition Risk Distribution
    - Risk Distribution (High Risk vs Low Risk)
    - OverTime vs Attrition Probability
    - Monthly Income vs Job Level Risk Distribution
    """
    if df.empty:
        return {
            "department_risk": [],
            "risk_distribution": [],
            "overtime_impact": [],
            "income_vs_risk": []
        }

    # 1. Department Risk Breakdown
    dept_chart = []
    if "Department" in df.columns and "attrition_probability" in df.columns:
        dept_grp = df.groupby("Department").agg(
            total_count=("Department", "count"),
            high_risk_count=("attrition_prediction", lambda x: int((x == 1).sum())),
            avg_probability=("attrition_probability", "mean")
        ).reset_index()
        for _, r in dept_grp.iterrows():
            dept_chart.append({
                "department": str(r["Department"]),
                "total": int(r["total_count"]),
                "high_risk": int(r["high_risk_count"]),
                "avg_probability": round(float(r["avg_probability"]), 4)
            })

    # 2. Risk Category Distribution
    risk_chart = []
    if "risk_recommendation" in df.columns:
        risk_counts = df["risk_recommendation"].value_counts().to_dict()
        for cat, cnt in risk_counts.items():
            risk_chart.append({"category": str(cat), "count": int(cnt)})

    # 3. OverTime Impact
    overtime_chart = []
    ot_col = "OverTime" if "OverTime" in df.columns else ("OverTimeBinary" if "OverTimeBinary" in df.columns else None)
    if ot_col and "attrition_probability" in df.columns:
        ot_grp = df.groupby(ot_col).agg(
            avg_probability=("attrition_probability", "mean"),
            count=(ot_col, "count")
        ).reset_index()
        for _, r in ot_grp.iterrows():
            label = "Overtime" if str(r[ot_col]) in ["1", "Yes", "True"] else "No Overtime"
            overtime_chart.append({
                "overtime_status": label,
                "avg_probability": round(float(r["avg_probability"]), 4),
                "employee_count": int(r["count"])
            })

    return {
        "department_risk": dept_chart,
        "risk_distribution": risk_chart,
        "overtime_impact": overtime_chart
    }
