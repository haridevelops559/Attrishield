"""
Feature Store Service Layer.

Provides the high-level interface for:
- Feature definitions
- Feature groups
- Feature lineage
- Online feature retrieval
- Point-in-time feature retrieval
- Materialization history
"""

from typing import Optional, List, Dict, Any

from pymongo.database import Database

from backend.app.feature_store.repository import (
    FeatureStoreRepository,
)
from backend.app.feature_store.materializer import (
    FeatureMaterializer,
)
from backend.app.feature_store.lineage import (
    get_all_feature_lineage,
    get_feature_lineage,
)


class FeatureStoreService:
    """
    High-level Feature Store service.
    """

    def __init__(
        self,
        db: Optional[Database] = None,
    ):
        self.repo = FeatureStoreRepository(db)

        self.repo.init_default_definitions()

        self.materializer = FeatureMaterializer(
            self.repo
        )

    def get_definitions(
        self,
        feature_version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns feature definitions.
        """

        return self.repo.list_definitions(
            feature_version=feature_version
        )

    def get_groups(
        self,
        feature_version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns feature groups.
        """

        return self.repo.list_groups(
            feature_version=feature_version
        )

    def get_lineage(
        self,
        feature_name: Optional[str] = None,
    ) -> Any:
        """
        Returns lineage for one feature
        or all features.
        """

        if feature_name:
            return get_feature_lineage(
                feature_name
            )

        return get_all_feature_lineage()

    def get_online_features(
        self,
        entity_id: str,
        feature_names: Optional[
            List[str]
        ] = None,
        feature_version: str = "v3",
    ) -> Dict[str, Any]:
        """
        Retrieves online features for an entity
        and feature version.
        """

        return self.repo.get_online_features(
            entity_id=entity_id,
            feature_names=feature_names,
            feature_version=feature_version,
        )

    def get_point_in_time_features(
        self,
        entity_ids: List[str],
        features: List[str],
        feature_version: str = "v3",
        as_of_timestamp: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves features for one or more entities
        at a requested point in time.
        """

        return self.repo.get_point_in_time_features(
            entity_ids=entity_ids,
            feature_names=features,
            feature_version=feature_version,
            as_of_timestamp=as_of_timestamp,
        )

    def get_materialization_history(
        self,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Returns recent materialization runs.
        """

        return self.repo.list_materializations(
            limit=limit
        )