"""
Pandas Dynamic Query Filtering Engine.
Applies dynamic filters to Pandas DataFrames using vectorized NumPy masks.
"""

from typing import List, Dict, Any, Union
import pandas as pd
import numpy as np
from backend.app.schemas.analytics import FilterCondition
from backend.app.core.logging import logger


def apply_dynamic_filters(df: pd.DataFrame, filters: List[FilterCondition]) -> pd.DataFrame:
    """
    Applies a series of dynamic filter conditions against a DataFrame.
    Supported operators: eq, neq, gt, gte, lt, lte, in, contains, between, is_null, not_null.
    """
    if df.empty or not filters:
        return df

    filtered_df = df.copy()

    for cond in filters:
        col = cond.field
        op = str(cond.operator).lower().strip()
        val = cond.value

        if col not in filtered_df.columns:
            logger.warning(f"Filter column '{col}' not found in DataFrame. Skipping filter.")
            continue

        try:
            if op in ["eq", "=="]:
                filtered_df = filtered_df[filtered_df[col] == val]
            elif op in ["neq", "!="]:
                filtered_df = filtered_df[filtered_df[col] != val]
            elif op in ["gt", ">"]:
                filtered_df = filtered_df[filtered_df[col] > float(val)]
            elif op in ["gte", ">="]:
                filtered_df = filtered_df[filtered_df[col] >= float(val)]
            elif op in ["lt", "<"]:
                filtered_df = filtered_df[filtered_df[col] < float(val)]
            elif op in ["lte", "<="]:
                filtered_df = filtered_df[filtered_df[col] <= float(val)]
            elif op == "in":
                val_list = val if isinstance(val, list) else [val]
                filtered_df = filtered_df[filtered_df[col].isin(val_list)]
            elif op in ["contains", "like"]:
                filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(str(val), case=False, na=False)]
            elif op == "between":
                if isinstance(val, (list, tuple)) and len(val) == 2:
                    filtered_df = filtered_df[(filtered_df[col] >= float(val[0])) & (filtered_df[col] <= float(val[1]))]
            elif op in ["is_null", "isnull"]:
                filtered_df = filtered_df[filtered_df[col].isna()]
            elif op in ["not_null", "notnull"]:
                filtered_df = filtered_df[filtered_df[col].notna()]
            else:
                logger.warning(f"Unsupported filter operator '{op}'. Skipping.")
        except Exception as e:
            logger.error(f"Error applying filter condition ({col} {op} {val}): {e}")

    return filtered_df
