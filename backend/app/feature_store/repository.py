"""
Feature Store Repository Layer.
Handles MongoDB persistence for feature definitions, values, groups, and materializations.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from pymongo.database import Database
from backend.app.core.logging import logger
from backend.app.feature_store.lineage import CANONICAL_FEATURE_LINEAGE

_IN_MEMORY_FEATURE_DEFS: Dict[str, Dict[str, Any]] = {}
_IN_MEMORY_FEATURE_GROUPS: Dict[str, Dict[str, Any]] = {}
_IN_MEMORY_FEATURE_VALUES: Dict[str, Dict[str, Any]] = {}
_IN_MEMORY_MATERIALIZATIONS: List[Dict[str, Any]] = []


class FeatureStoreRepository:
    def __init__(self, db: Optional[Database]):
        self.db = db
        self.defs_col = db["feature_definitions"] if db is not None else None
        self.groups_col = db["feature_groups"] if db is not None else None
        self.values_col = db["feature_values"] if db is not None else None
        self.mat_col = db["feature_materializations"] if db is not None else None

    def init_default_definitions(self):
        """Initializes canonical V3 feature definitions in database."""
        now_str = datetime.now(timezone.utc).isoformat()
        for item in CANONICAL_FEATURE_LINEAGE:
            fname = item["feature_name"]
            doc = {
                "feature_name": fname,
                "data_type": "float" if "Ratio" in fname or "Burden" in fname or "Income" in fname else "int",
                "entity_type": "employee",
                "description": item["description"],
                "formula": item["transformation_logic"],
                "feature_version": item["feature_version"],
                "created_at": now_str
            }
            if self.defs_col is not None:
                try:
                    self.defs_col.update_one({"feature_name": fname}, {"$set": doc}, upsert=True)
                except Exception as e:
                    logger.error(f"Error seeding feature definition {fname}: {e}")
            _IN_MEMORY_FEATURE_DEFS[fname] = doc

        # Seed default V3 Feature Group
        group_doc = {
            "group_name": "v3_engineered_attrition_features",
            "description": "Canonical 7 engineered features required by XGBoost V3 model contract",
            "features": [item["feature_name"] for item in CANONICAL_FEATURE_LINEAGE],
            "created_at": now_str
        }
        if self.groups_col is not None:
            try:
                self.groups_col.update_one({"group_name": group_doc["group_name"]}, {"$set": group_doc}, upsert=True)
            except Exception as e:
                logger.error(f"Error seeding feature group: {e}")
        _IN_MEMORY_FEATURE_GROUPS[group_doc["group_name"]] = group_doc

    def list_definitions(self) -> List[Dict[str, Any]]:
        """Returns all feature definitions."""
        if self.defs_col is not None:
            try:
                return list(self.defs_col.find({}, {"_id": 0}))
            except Exception as e:
                logger.error(f"Error listing feature definitions from MongoDB: {e}")
        return list(_IN_MEMORY_FEATURE_DEFS.values())

    def list_groups(self) -> List[Dict[str, Any]]:
        """Returns all feature groups."""
        if self.groups_col is not None:
            try:
                return list(self.groups_col.find({}, {"_id": 0}))
            except Exception as e:
                logger.error(f"Error listing feature groups from MongoDB: {e}")
        return list(_IN_MEMORY_FEATURE_GROUPS.values())

    def upsert_online_features(self, entity_id: str, feature_dict: Dict[str, Any], feature_version: str = "v3") -> bool:
        """Upserts online feature values for a specific entity."""
        now_str = datetime.now(timezone.utc).isoformat()
        docs = []
        for fname, val in feature_dict.items():
            key = f"{entity_id}:{fname}"
            doc = {
                "entity_id": str(entity_id),
                "feature_name": fname,
                "feature_value": val,
                "timestamp": now_str,
                "feature_version": feature_version
            }
            docs.append(doc)
            _IN_MEMORY_FEATURE_VALUES[key] = doc

        if self.values_col is not None:
            try:
                for doc in docs:
                    self.values_col.update_one(
                        {"entity_id": doc["entity_id"], "feature_name": doc["feature_name"]},
                        {"$set": doc},
                        upsert=True
                    )
            except Exception as e:
                logger.error(f"Error upserting online feature values: {e}")
        return True

    def get_online_features(self, entity_id: str, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Retrieves point-in-time online feature values for an entity."""
        result = {}
        if self.values_col is not None:
            try:
                query: Dict[str, Any] = {"entity_id": str(entity_id)}
                if feature_names:
                    query["feature_name"] = {"$in": feature_names}
                cursor = self.values_col.find(query, {"_id": 0})
                for doc in cursor:
                    result[doc["feature_name"]] = doc["feature_value"]
                return result
            except Exception as e:
                logger.error(f"Error fetching online feature values from MongoDB: {e}")

        for key, doc in _IN_MEMORY_FEATURE_VALUES.items():
            if doc["entity_id"] == str(entity_id):
                fname = doc["feature_name"]
                if not feature_names or fname in feature_names:
                    result[fname] = doc["feature_value"]
        return result

    def record_materialization(self, mat_record: Dict[str, Any]) -> Dict[str, Any]:
        """Records a materialization run."""
        c = mat_record.copy()
        if "timestamp" not in c:
            c["timestamp"] = datetime.now(timezone.utc).isoformat()
            
        if self.mat_col is not None:
            try:
                self.mat_col.insert_one(c)
            except Exception as e:
                logger.error(f"Error logging materialization record: {e}")

        _IN_MEMORY_MATERIALIZATIONS.append(c)
        return c

    def list_materializations(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns recent materialization logs."""
        if self.mat_col is not None:
            try:
                cursor = self.mat_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
                return list(cursor)
            except Exception as e:
                logger.error(f"Error listing materializations from MongoDB: {e}")

        res = sorted(_IN_MEMORY_MATERIALIZATIONS, key=lambda x: x.get("timestamp", ""), reverse=True)
        return res[:limit]
