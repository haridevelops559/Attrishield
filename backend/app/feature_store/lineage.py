"""
Feature Lineage Registry.
Defines explicit dependency graphs between raw input attributes and engineered features.
"""

from typing import List, Dict, Any

CANONICAL_FEATURE_LINEAGE: List[Dict[str, Any]] = [
    {
        "feature_name": "IncomePerJobLevel",
        "feature_version": "v3",
        "source_columns": ["MonthlyIncome", "JobLevel"],
        "transformation_logic": "MonthlyIncome / JobLevel",
        "description": "Calculates monthly compensation relative to job hierarchy level.",
        "downstream_consumers": ["attrishield_pipeline_v3.joblib", "batch_analytics_engine"]
    },
    {
        "feature_name": "PromotionStagnationRatio",
        "feature_version": "v3",
        "source_columns": ["YearsInCurrentRole", "YearsSinceLastPromotion"],
        "transformation_logic": "YearsInCurrentRole / (YearsSinceLastPromotion + 1)",
        "description": "Measures career progression stagnation by comparing role tenure against promotion recency.",
        "downstream_consumers": ["attrishield_pipeline_v3.joblib", "batch_analytics_engine"]
    },
    {
        "feature_name": "ManagerTenureRatio",
        "feature_version": "v3",
        "source_columns": ["YearsWithCurrManager", "TotalWorkingYears"],
        "transformation_logic": "YearsWithCurrManager / (TotalWorkingYears + 1)",
        "description": "Quantifies reporting manager relationship tenure relative to overall career length.",
        "downstream_consumers": ["attrishield_pipeline_v3.joblib", "batch_analytics_engine"]
    },
    {
        "feature_name": "RoleTenureRatio",
        "feature_version": "v3",
        "source_columns": ["YearsInCurrentRole", "TotalWorkingYears"],
        "transformation_logic": "YearsInCurrentRole / (TotalWorkingYears + 1)",
        "description": "Measures specialization or stagnation in current position relative to total experience.",
        "downstream_consumers": ["attrishield_pipeline_v3.joblib", "batch_analytics_engine"]
    },
    {
        "feature_name": "OverTimeBinary",
        "feature_version": "v3",
        "source_columns": ["OverTime"],
        "transformation_logic": "1 if OverTime.lower() in ['yes', '1', 'true'] else 0",
        "description": "Converts categorical overtime indicator into binary integer.",
        "downstream_consumers": ["attrishield_pipeline_v3.joblib", "CommuteOvertimeBurden"]
    },
    {
        "feature_name": "CommuteOvertimeBurden",
        "feature_version": "v3",
        "source_columns": ["DistanceFromHome", "OverTimeBinary"],
        "transformation_logic": "DistanceFromHome * OverTimeBinary",
        "description": "Quantifies combined physical commute distance and overtime working strain.",
        "downstream_consumers": ["attrishield_pipeline_v3.joblib", "batch_analytics_engine"]
    },
    {
        "feature_name": "EarlyCareerFlag",
        "feature_version": "v3",
        "source_columns": ["TotalWorkingYears"],
        "transformation_logic": "1 if TotalWorkingYears <= 3 else 0",
        "description": "Binary flag isolating high-turnover early career stage employees.",
        "downstream_consumers": ["attrishield_pipeline_v3.joblib", "batch_analytics_engine"]
    }
]


def get_all_feature_lineage() -> List[Dict[str, Any]]:
    """Returns complete list of canonical feature lineage definitions."""
    return CANONICAL_FEATURE_LINEAGE


def get_feature_lineage(feature_name: str) -> Dict[str, Any]:
    """Retrieves lineage metadata for a specific feature."""
    for item in CANONICAL_FEATURE_LINEAGE:
        if item["feature_name"] == feature_name:
            return item
    return {
        "feature_name": feature_name,
        "feature_version": "v3",
        "source_columns": [],
        "transformation_logic": "Unknown / Raw Input Attribute",
        "description": "Raw employee attribute",
        "downstream_consumers": ["attrishield_pipeline_v3.joblib"]
    }
