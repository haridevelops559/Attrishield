"""
Pandas Group-By Aggregation Engine.
Executes dynamic group-by operations and calculates attrition statistics.
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np
from backend.app.schemas.analytics import GroupByResult


def execute_group_by(df: pd.DataFrame, group_cols: List[str]) -> List[GroupByResult]:
    """
    Groups a DataFrame by specified categorical columns and calculates key attrition metrics.
    """
    if df.empty or not group_cols:
        return []

    # Filter group columns that actually exist in the DataFrame
    valid_cols = [c for c in group_cols if c in df.columns]
    if not valid_cols:
        return []

    results: List[GroupByResult] = []

    grouped = df.groupby(valid_cols, dropna=False)

    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
            
        group_keys_dict = {col: (None if pd.isna(val) else val) for col, val in zip(valid_cols, keys)}
        
        record_count = len(group)
        
        if "attrition_prediction" in group.columns:
            high_risk_count = int((group["attrition_prediction"] == 1).sum())
        else:
            high_risk_count = 0

        review_rate = round(high_risk_count / record_count, 4) if record_count > 0 else 0.0

        if "attrition_probability" in group.columns:
            avg_prob = round(float(group["attrition_probability"].mean()), 4)
        else:
            avg_prob = 0.0

        add_aggs = {}
        for num_col in ["MonthlyIncome", "TotalWorkingYears", "YearsAtCompany", "Age"]:
            if num_col in group.columns:
                add_aggs[f"mean_{num_col}"] = round(float(group[num_col].mean()), 2)

        results.append(
            GroupByResult(
                group_keys=group_keys_dict,
                record_count=record_count,
                high_risk_count=high_risk_count,
                review_rate=review_rate,
                average_attrition_probability=avg_prob,
                additional_aggregations=add_aggs
            )
        )

    return results
