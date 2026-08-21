"""
Feature Materializer Engine.
Materializes engineered features from raw input DataFrames into the feature store.
"""

import uuid
from typing import Dict, Any, List
import pandas as pd
from backend.app.ml.feature_engineering import apply_v3_feature_engineering, ENGINEERED_FEATURE_NAMES
from backend.app.feature_store.repository import FeatureStoreRepository
from backend.app.core.logging import logger


class FeatureMaterializer:
    def __init__(self, repo: FeatureStoreRepository):
        self.repo = repo

    def materialize_batch(self, df: pd.DataFrame, batch_id: str, feature_version: str = "v3") -> Dict[str, Any]:
        """
        Materializes engineered features for a DataFrame batch.
        1. Ensures V3 engineered features exist.
        2. Upserts online feature values into feature store collection per entity.
        3. Records materialization metadata log.
        """
        engineered_df = apply_v3_feature_engineering(df, retain_raw_overtime=False)
        
        entity_count = len(engineered_df)
        materialized_features = [col for col in ENGINEERED_FEATURE_NAMES if col in engineered_df.columns]

        logger.info(f"Materializing {entity_count} entities for batch {batch_id}...")

        for idx, row in engineered_df.iterrows():
            entity_id = str(row.get("EmployeeNumber", row.get("EmployeeID", f"emp_{idx + 1}")))
            feature_dict = {col: float(row[col]) for col in materialized_features if pd.notnull(row[col])}
            self.repo.upsert_online_features(entity_id=entity_id, feature_dict=feature_dict, feature_version=feature_version)

        mat_record = {
            "materialization_id": f"mat_{uuid.uuid4().hex[:10]}",
            "batch_id": batch_id,
            "feature_version": feature_version,
            "features_materialized": materialized_features,
            "entity_count": entity_count,
            "status": "SUCCESS"
        }
        self.repo.record_materialization(mat_record)

        return mat_record
