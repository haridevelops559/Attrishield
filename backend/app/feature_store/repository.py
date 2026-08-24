"""
Feature Store Repository Layer.

Handles MongoDB persistence for:
- Feature definitions
- Feature groups
- Online feature values
- Materialization history
- Point-in-time feature retrieval

Includes an in-memory fallback for local/offline development.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from pymongo.database import Database

from backend.app.core.logging import logger
from backend.app.feature_store.lineage import (
    CANONICAL_FEATURE_LINEAGE,
)


_IN_MEMORY_FEATURE_DEFS: Dict[str, Dict[str, Any]] = {}
_IN_MEMORY_FEATURE_GROUPS: Dict[str, Dict[str, Any]] = {}
_IN_MEMORY_FEATURE_VALUES: Dict[str, Dict[str, Any]] = {}
_IN_MEMORY_MATERIALIZATIONS: List[Dict[str, Any]] = []


class FeatureStoreRepository:
    """
    Repository responsible for Feature Store persistence.

    MongoDB is used when a database connection is available.
    Otherwise, an in-memory fallback is used.
    """

    def __init__(self, db: Optional[Database]):
        self.db = db

        self.defs_col = (
            db["feature_definitions"]
            if db is not None
            else None
        )

        self.groups_col = (
            db["feature_groups"]
            if db is not None
            else None
        )

        self.values_col = (
            db["feature_values"]
            if db is not None
            else None
        )

        self.mat_col = (
            db["feature_materializations"]
            if db is not None
            else None
        )

    # ------------------------------------------------------------------
    # Feature Definitions
    # ------------------------------------------------------------------

    def init_default_definitions(self):
        """
        Initializes canonical V3 feature definitions.
        """

        now_str = datetime.now(
            timezone.utc
        ).isoformat()

        for item in CANONICAL_FEATURE_LINEAGE:
            feature_name = item["feature_name"]
            feature_version = item["feature_version"]

            doc = {
                "feature_name": feature_name,
                "data_type": (
                    "float"
                    if (
                        "Ratio" in feature_name
                        or "Burden" in feature_name
                        or "Income" in feature_name
                    )
                    else "int"
                ),
                "entity_type": "employee",
                "description": item["description"],
                "formula": item["transformation_logic"],
                "feature_version": feature_version,
                "created_at": now_str,
            }

            if self.defs_col is not None:
                try:
                    self.defs_col.update_one(
                        {
                            "feature_name": feature_name,
                            "feature_version": feature_version,
                        },
                        {
                            "$set": doc
                        },
                        upsert=True,
                    )

                except Exception as e:
                    logger.error(
                        "Error seeding feature definition "
                        f"{feature_name}: {e}"
                    )

            definition_key = (
                f"{feature_name}:{feature_version}"
            )

            _IN_MEMORY_FEATURE_DEFS[
                definition_key
            ] = doc

        # Default V3 Feature Group
        group_doc = {
            "group_name": (
                "v3_engineered_attrition_features"
            ),
            "description": (
                "Canonical 7 engineered features "
                "required by XGBoost V3 model contract"
            ),
            "features": [
                item["feature_name"]
                for item in CANONICAL_FEATURE_LINEAGE
            ],
            "feature_version": "v3",
            "created_at": now_str,
        }

        if self.groups_col is not None:
            try:
                self.groups_col.update_one(
                    {
                        "group_name": group_doc[
                            "group_name"
                        ],
                        "feature_version": group_doc[
                            "feature_version"
                        ],
                    },
                    {
                        "$set": group_doc
                    },
                    upsert=True,
                )

            except Exception as e:
                logger.error(
                    "Error seeding feature group: "
                    f"{e}"
                )

        group_key = (
            f"{group_doc['group_name']}:"
            f"{group_doc['feature_version']}"
        )

        _IN_MEMORY_FEATURE_GROUPS[
            group_key
        ] = group_doc

    # ------------------------------------------------------------------
    # Feature Definition Queries
    # ------------------------------------------------------------------

    def list_definitions(
        self,
        feature_version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns feature definitions.

        If feature_version is provided, only definitions
        belonging to that version are returned.
        """

        if self.defs_col is not None:
            try:
                query: Dict[str, Any] = {}

                if feature_version:
                    query[
                        "feature_version"
                    ] = feature_version

                return list(
                    self.defs_col.find(
                        query,
                        {"_id": 0},
                    )
                )

            except Exception as e:
                logger.error(
                    "Error listing feature definitions "
                    f"from MongoDB: {e}"
                )

        definitions = list(
            _IN_MEMORY_FEATURE_DEFS.values()
        )

        if feature_version:
            definitions = [
                definition
                for definition in definitions
                if definition.get(
                    "feature_version"
                ) == feature_version
            ]

        return definitions

    # ------------------------------------------------------------------
    # Feature Groups
    # ------------------------------------------------------------------

    def list_groups(
        self,
        feature_version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns feature groups.

        If feature_version is provided, only groups
        belonging to that version are returned.
        """

        if self.groups_col is not None:
            try:
                query: Dict[str, Any] = {}

                if feature_version:
                    query[
                        "feature_version"
                    ] = feature_version

                return list(
                    self.groups_col.find(
                        query,
                        {"_id": 0},
                    )
                )

            except Exception as e:
                logger.error(
                    "Error listing feature groups "
                    f"from MongoDB: {e}"
                )

        groups = list(
            _IN_MEMORY_FEATURE_GROUPS.values()
        )

        if feature_version:
            groups = [
                group
                for group in groups
                if group.get(
                    "feature_version"
                ) == feature_version
            ]

        return groups

    # ------------------------------------------------------------------
    # Online Feature Upsert
    # ------------------------------------------------------------------

    def upsert_online_features(
        self,
        entity_id: str,
        feature_dict: Dict[str, Any],
        feature_version: str = "v3",
    ) -> bool:
        """
        Upserts online feature values for an entity.

        Logical identity:

            entity_id + feature_name + feature_version

        This prevents V3 and future V4 feature values
        from overwriting each other.
        """

        now_str = datetime.now(
            timezone.utc
        ).isoformat()

        docs: List[Dict[str, Any]] = []

        for feature_name, feature_value in (
            feature_dict.items()
        ):
            doc = {
                "entity_id": str(entity_id),
                "feature_name": feature_name,
                "feature_value": feature_value,
                "timestamp": now_str,
                "feature_version": feature_version,
            }

            key = (
                f"{entity_id}:"
                f"{feature_name}:"
                f"{feature_version}"
            )

            docs.append(doc)

            _IN_MEMORY_FEATURE_VALUES[
                key
            ] = doc

        if self.values_col is not None:
            try:
                for doc in docs:
                    self.values_col.update_one(
                        {
                            "entity_id":
                                doc["entity_id"],
                            "feature_name":
                                doc["feature_name"],
                            "feature_version":
                                doc["feature_version"],
                        },
                        {
                            "$set": doc
                        },
                        upsert=True,
                    )

            except Exception as e:
                logger.error(
                    "Error upserting online feature "
                    f"values: {e}"
                )

        return True

    # ------------------------------------------------------------------
    # Online Feature Retrieval
    # ------------------------------------------------------------------

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
        and a specific feature version.
        """

        result: Dict[str, Any] = {}

        if self.values_col is not None:
            try:
                query: Dict[str, Any] = {
                    "entity_id": str(entity_id),
                    "feature_version": feature_version,
                }

                if feature_names:
                    query[
                        "feature_name"
                    ] = {
                        "$in": feature_names
                    }

                cursor = self.values_col.find(
                    query,
                    {"_id": 0},
                )

                for doc in cursor:
                    result[
                        doc["feature_name"]
                    ] = doc["feature_value"]

                return result

            except Exception as e:
                logger.error(
                    "Error fetching online feature "
                    f"values from MongoDB: {e}"
                )

        for doc in (
            _IN_MEMORY_FEATURE_VALUES.values()
        ):
            if (
                doc.get("entity_id")
                != str(entity_id)
            ):
                continue

            if (
                doc.get("feature_version")
                != feature_version
            ):
                continue

            feature_name = doc[
                "feature_name"
            ]

            if (
                feature_names
                and feature_name
                not in feature_names
            ):
                continue

            result[
                feature_name
            ] = doc["feature_value"]

        return result

    # ------------------------------------------------------------------
    # Point-in-Time Feature Retrieval
    # ------------------------------------------------------------------

    def get_point_in_time_features(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        feature_version: str = "v3",
        as_of_timestamp: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves the latest feature values that were available
        at or before the requested point in time.

        Supports timestamps stored as either:
        - native MongoDB datetime values
        - ISO-8601 strings

        For every entity + feature combination, the latest
        valid value at or before the requested timestamp is used.
        """

        if not entity_ids:
            return {}

        if not feature_names:
            return {}

        # --------------------------------------------------------------
        # Parse requested point-in-time timestamp
        # --------------------------------------------------------------

        if as_of_timestamp is None:
            as_of_datetime = datetime.now(
                timezone.utc
            )

        else:
            try:
                normalized_timestamp = (
                    as_of_timestamp.replace(
                        "Z",
                        "+00:00",
                    )
                )

                as_of_datetime = (
                    datetime.fromisoformat(
                        normalized_timestamp
                    )
                )

                if (
                    as_of_datetime.tzinfo
                    is None
                ):
                    as_of_datetime = (
                        as_of_datetime.replace(
                            tzinfo=timezone.utc
                        )
                    )

                as_of_datetime = (
                    as_of_datetime.astimezone(
                        timezone.utc
                    )
                )

            except (
                ValueError,
                TypeError,
            ):
                logger.error(
                    "Invalid point-in-time timestamp: "
                    f"{as_of_timestamp}"
                )
                return {}

        result: Dict[
            str,
            Dict[str, Any],
        ] = {}

        requested_entities = {
            str(entity_id)
            for entity_id in entity_ids
        }

        requested_features = set(
            feature_names
        )

        # --------------------------------------------------------------
        # MongoDB path
        # --------------------------------------------------------------

        if self.values_col is not None:
            try:
                query: Dict[str, Any] = {
                    "entity_id": {
                        "$in": list(
                            requested_entities
                        )
                    },
                    "feature_name": {
                        "$in": list(
                            requested_features
                        )
                    },
                    "feature_version":
                        feature_version,
                }

                cursor = self.values_col.find(
                    query,
                    {"_id": 0},
                )

                # Keep the latest valid record for
                # every entity + feature pair.
                latest: Dict[
                    tuple,
                    tuple,
                ] = {}

                for doc in cursor:
                    entity_id = str(
                        doc.get(
                            "entity_id"
                        )
                    )

                    feature_name = doc.get(
                        "feature_name"
                    )

                    if (
                        entity_id
                        not in requested_entities
                    ):
                        continue

                    if (
                        feature_name
                        not in requested_features
                    ):
                        continue

                    if (
                        doc.get(
                            "feature_version"
                        )
                        != feature_version
                    ):
                        continue

                    raw_timestamp = doc.get(
                        "timestamp"
                    )

                    if raw_timestamp is None:
                        continue

                    # --------------------------------------------------
                    # Normalize stored timestamp
                    # --------------------------------------------------

                    if isinstance(
                        raw_timestamp,
                        datetime,
                    ):
                        doc_datetime = (
                            raw_timestamp
                        )

                        if (
                            doc_datetime.tzinfo
                            is None
                        ):
                            doc_datetime = (
                                doc_datetime.replace(
                                    tzinfo=timezone.utc
                                )
                            )

                        doc_datetime = (
                            doc_datetime.astimezone(
                                timezone.utc
                            )
                        )

                    elif isinstance(
                        raw_timestamp,
                        str,
                    ):
                        try:
                            normalized_stored_timestamp = (
                                raw_timestamp.replace(
                                    "Z",
                                    "+00:00",
                                )
                            )

                            doc_datetime = (
                                datetime.fromisoformat(
                                    normalized_stored_timestamp
                                )
                            )

                            if (
                                doc_datetime.tzinfo
                                is None
                            ):
                                doc_datetime = (
                                    doc_datetime.replace(
                                        tzinfo=timezone.utc
                                    )
                                )

                            doc_datetime = (
                                doc_datetime.astimezone(
                                    timezone.utc
                                )
                            )

                        except (
                            ValueError,
                            TypeError,
                        ):
                            logger.warning(
                                "Skipping feature record "
                                "with invalid timestamp: "
                                f"{raw_timestamp}"
                            )
                            continue

                    else:
                        logger.warning(
                            "Skipping feature record "
                            "with unsupported timestamp "
                            "type: "
                            f"{type(raw_timestamp)}"
                        )
                        continue

                    # --------------------------------------------------
                    # Point-in-time condition
                    # --------------------------------------------------

                    if (
                        doc_datetime
                        > as_of_datetime
                    ):
                        continue

                    key = (
                        entity_id,
                        feature_name,
                    )

                    previous = latest.get(
                        key
                    )

                    if (
                        previous is None
                        or doc_datetime
                        > previous[0]
                    ):
                        latest[key] = (
                            doc_datetime,
                            doc.get(
                                "feature_value"
                            ),
                        )

                # ------------------------------------------------------
                # Build response
                # ------------------------------------------------------

                for (
                    entity_id,
                    feature_name,
                ), (
                    _timestamp,
                    feature_value,
                ) in latest.items():

                    if (
                        entity_id
                        not in result
                    ):
                        result[
                            entity_id
                        ] = {}

                    result[
                        entity_id
                    ][
                        feature_name
                    ] = feature_value

                return result

            except Exception as e:
                logger.error(
                    "Error retrieving point-in-time "
                    f"features from MongoDB: {e}"
                )

        # --------------------------------------------------------------
        # In-memory fallback
        # --------------------------------------------------------------

        latest: Dict[
            tuple,
            tuple,
        ] = {}

        for doc in (
            _IN_MEMORY_FEATURE_VALUES.values()
        ):
            entity_id = str(
                doc.get("entity_id")
            )

            feature_name = doc.get(
                "feature_name"
            )

            if (
                entity_id
                not in requested_entities
            ):
                continue

            if (
                feature_name
                not in requested_features
            ):
                continue

            if (
                doc.get("feature_version")
                != feature_version
            ):
                continue

            raw_timestamp = doc.get(
                "timestamp"
            )

            if raw_timestamp is None:
                continue

            try:
                if isinstance(
                    raw_timestamp,
                    datetime,
                ):
                    doc_datetime = (
                        raw_timestamp
                    )

                    if (
                        doc_datetime.tzinfo
                        is None
                    ):
                        doc_datetime = (
                            doc_datetime.replace(
                                tzinfo=timezone.utc
                            )
                        )

                    doc_datetime = (
                        doc_datetime.astimezone(
                            timezone.utc
                        )
                    )

                else:
                    normalized_timestamp = (
                        str(
                            raw_timestamp
                        ).replace(
                            "Z",
                            "+00:00",
                        )
                    )

                    doc_datetime = (
                        datetime.fromisoformat(
                            normalized_timestamp
                        )
                    )

                    if (
                        doc_datetime.tzinfo
                        is None
                    ):
                        doc_datetime = (
                            doc_datetime.replace(
                                tzinfo=timezone.utc
                            )
                        )

                    doc_datetime = (
                        doc_datetime.astimezone(
                            timezone.utc
                        )
                    )

            except (
                ValueError,
                TypeError,
            ):
                continue

            if (
                doc_datetime
                > as_of_datetime
            ):
                continue

            key = (
                entity_id,
                feature_name,
            )

            previous = latest.get(
                key
            )

            if (
                previous is None
                or doc_datetime
                > previous[0]
            ):
                latest[key] = (
                    doc_datetime,
                    doc.get(
                        "feature_value"
                    ),
                )

        for (
            entity_id,
            feature_name,
        ), (
            _timestamp,
            feature_value,
        ) in latest.items():

            if (
                entity_id
                not in result
            ):
                result[
                    entity_id
                ] = {}

            result[
                entity_id
            ][
                feature_name
            ] = feature_value

        return result

    # ------------------------------------------------------------------
    # Materialization History
    # ------------------------------------------------------------------

    def record_materialization(
        self,
        mat_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Records a Feature Store materialization run.
        """

        record = mat_record.copy()

        if "timestamp" not in record:
            record["timestamp"] = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

        if self.mat_col is not None:
            try:
                self.mat_col.insert_one(
                    record
                )

            except Exception as e:
                logger.error(
                    "Error logging materialization "
                    f"record: {e}"
                )

        _IN_MEMORY_MATERIALIZATIONS.append(
            record
        )

        return record

    # ------------------------------------------------------------------
    # Materialization History Query
    # ------------------------------------------------------------------

    def list_materializations(
        self,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Returns recent materialization runs.
        """

        if self.mat_col is not None:
            try:
                cursor = (
                    self.mat_col
                    .find(
                        {},
                        {"_id": 0},
                    )
                    .sort(
                        "timestamp",
                        -1,
                    )
                    .limit(limit)
                )

                return list(cursor)

            except Exception as e:
                logger.error(
                    "Error listing materializations "
                    f"from MongoDB: {e}"
                )

        results = sorted(
            _IN_MEMORY_MATERIALIZATIONS,
            key=lambda x: x.get(
                "timestamp",
                "",
            ),
            reverse=True,
        )

        return results[:limit]