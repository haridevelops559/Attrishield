"""
Pandas Pivot Table Engine.
Generates dynamic multidimensional pivot tables.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from backend.app.core.logging import logger


def execute_pivot_table(
    df: pd.DataFrame,
    index_cols: List[str],
    columns_cols: List[str],
    values_col: str = "attrition_probability",
    aggfunc: str = "mean"
) -> Dict[str, Any]:
    """
    Executes a dynamic pivot table operation over a DataFrame.
    """
    if df.empty or not index_cols:
        return {"rows": [], "columns": [], "data": []}

    valid_indices = [c for c in index_cols if c in df.columns]
    valid_columns = [c for c in columns_cols if c in df.columns] if columns_cols else None
    
    if not valid_indices:
        return {"rows": [], "columns": [], "data": []}

    val_col = values_col if values_col in df.columns else ("attrition_probability" if "attrition_probability" in df.columns else df.columns[0])

    try:
        pivot_df = pd.pivot_table(
            df,
            values=val_col,
            index=valid_indices,
            columns=valid_columns,
            aggfunc=aggfunc,
            fill_value=0.0
        )

        # Format into clean JSON-serializable dictionary
        if isinstance(pivot_df.columns, pd.MultiIndex):
            col_names = ["_".join(map(str, c)) for c in pivot_df.columns]
        else:
            col_names = [str(c) for c in pivot_df.columns]

        rows_data = []
        for idx, row in pivot_df.iterrows():
            row_dict = {}
            if isinstance(idx, tuple):
                for i_col, i_val in zip(valid_indices, idx):
                    row_dict[i_col] = str(i_val)
            else:
                row_dict[valid_indices[0]] = str(idx)

            for col_n, val in zip(col_names, row.values):
                row_dict[str(col_n)] = round(float(val), 4)

            rows_data.append(row_dict)

        return {
            "index_fields": valid_indices,
            "column_fields": col_names,
            "values_field": val_col,
            "aggfunc": aggfunc,
            "data": rows_data
        }
    except Exception as e:
        logger.error(f"Pivot table execution error: {e}")
        return {"error": str(e), "rows": [], "columns": [], "data": []}
