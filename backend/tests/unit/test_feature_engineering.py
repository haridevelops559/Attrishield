"""
Unit Tests for Canonical V3 Feature Engineering Engine.
"""

import pandas as pd
from backend.app.ml.feature_engineering import apply_v3_feature_engineering, ENGINEERED_FEATURE_NAMES


def test_v3_feature_engineering_calculations(sample_employee_raw):
    df = pd.DataFrame([sample_employee_raw])
    engineered_df = apply_v3_feature_engineering(df, retain_raw_overtime=False)

    # 1. IncomePerJobLevel = 5000 / 2 = 2500.0
    assert "IncomePerJobLevel" in engineered_df.columns
    assert float(engineered_df["IncomePerJobLevel"].iloc[0]) == 2500.0

    # 2. PromotionStagnationRatio = 3 / (1 + 1) = 1.5
    assert "PromotionStagnationRatio" in engineered_df.columns
    assert float(engineered_df["PromotionStagnationRatio"].iloc[0]) == 1.5

    # 3. ManagerTenureRatio = 2 / (10 + 1) = 2/11 = ~0.1818
    assert "ManagerTenureRatio" in engineered_df.columns
    assert abs(float(engineered_df["ManagerTenureRatio"].iloc[0]) - (2 / 11)) < 1e-4

    # 4. RoleTenureRatio = 3 / (10 + 1) = 3/11 = ~0.2727
    assert "RoleTenureRatio" in engineered_df.columns
    assert abs(float(engineered_df["RoleTenureRatio"].iloc[0]) - (3 / 11)) < 1e-4

    # 5. OverTimeBinary = 1 (OverTime == 'Yes')
    assert "OverTimeBinary" in engineered_df.columns
    assert int(engineered_df["OverTimeBinary"].iloc[0]) == 1

    # 6. CommuteOvertimeBurden = 10 * 1 = 10.0
    assert "CommuteOvertimeBurden" in engineered_df.columns
    assert float(engineered_df["CommuteOvertimeBurden"].iloc[0]) == 10.0

    # 7. EarlyCareerFlag = 0 (TotalWorkingYears 10 > 3)
    assert "EarlyCareerFlag" in engineered_df.columns
    assert int(engineered_df["EarlyCareerFlag"].iloc[0]) == 0

    # Verify raw OverTime was dropped
    assert "OverTime" not in engineered_df.columns


def test_early_career_flag():
    df = pd.DataFrame([{"TotalWorkingYears": 2, "OverTime": "No", "DistanceFromHome": 5}])
    res = apply_v3_feature_engineering(df)
    assert int(res["EarlyCareerFlag"].iloc[0]) == 1
    assert int(res["OverTimeBinary"].iloc[0]) == 0
    assert float(res["CommuteOvertimeBurden"].iloc[0]) == 0.0
