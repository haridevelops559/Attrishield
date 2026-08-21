"""
Feature Store Service Layer.
High-level service interface for querying feature definitions, groups, lineage, and online feature retrieval.
"""

from typing import Optional, List, Dict, Any
from pymongo.database import Database
from backend.app.feature_store.repository import FeatureStoreRepository
from backend.app.feature_store.materializer import FeatureMaterializer
from backend.app.feature_store.lineage import get_all_feature_lineage, get_feature_lineage


class FeatureStoreService:
    def __init__(self, db: Optional[Database] = None):
        self.repo = FeatureStoreRepository(db)
        self.repo.init_default_definitions()
        self.materializer = FeatureMaterializer(self.repo)

    def get_definitions(self) -> List[Dict[str, Any]]:
        return self.repo.list_definitions()

    def get_groups(self) -> List[Dict[str, Any]]:
        return self.repo.list_groups()

    def get_lineage(self, feature_name: Optional[str] = None) -> Any:
        if feature_name:
            return get_feature_lineage(feature_name)
        return get_all_feature_lineage()

    def get_online_features(self, entity_id: str, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        return self.repo.get_online_features(entity_id=entity_id, feature_names=feature_names)

    def get_materialization_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self.repo.list_materializations(limit=limit)
