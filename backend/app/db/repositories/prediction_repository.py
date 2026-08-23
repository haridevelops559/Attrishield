"""
Prediction Logs Repository Layer.

Manages individual and batch inference prediction records
in MongoDB with in-memory fallback.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

from pymongo.database import Database

from backend.app.core.logging import logger


_IN_MEMORY_PREDICTIONS: List[Dict[str, Any]] = []


class PredictionRepository:
    def __init__(self, db: Optional[Database]):
        self.db = db
        self.collection = (
            db["predictions"]
            if db is not None
            else None
        )

    def insert_many(
        self,
        prediction_docs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Stores a batch of prediction records."""

        if not prediction_docs:
            return []

        docs_copy = []

        now_str = datetime.now(
            timezone.utc
        ).isoformat()

        for doc in prediction_docs:
            copied_doc = doc.copy()

            if "created_at" not in copied_doc:
                copied_doc["created_at"] = now_str

            docs_copy.append(copied_doc)

        if self.collection is not None:
            try:
                self.collection.insert_many(
                    docs_copy
                )
            except Exception as e:
                logger.error(
                    "Error bulk inserting prediction "
                    f"documents: {e}"
                )

        # Always keep an in-memory copy so the
        # application can operate without MongoDB.
        _IN_MEMORY_PREDICTIONS.extend(
            docs_copy
        )

        return docs_copy

    def insert_one(
        self,
        prediction_doc: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stores a single prediction record."""

        if not prediction_doc:
            return prediction_doc

        result = self.insert_many(
            [prediction_doc]
        )

        return (
            result[0]
            if result
            else prediction_doc
        )

    def get_by_prediction_id(
        self,
        prediction_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieves a single prediction by ID."""

        if self.collection is not None:
            try:
                result = self.collection.find_one(
                    {
                        "prediction_id": prediction_id
                    },
                    {
                        "_id": 0
                    },
                )

                if result:
                    return result

            except Exception as e:
                logger.error(
                    "Error querying prediction "
                    f"by ID: {e}"
                )

        for prediction in _IN_MEMORY_PREDICTIONS:
            if (
                prediction.get("prediction_id")
                == prediction_id
            ):
                prediction.pop("_id", None)
                return prediction

        return None

    def get_by_batch_id(
        self,
        batch_id: str,
    ) -> List[Dict[str, Any]]:
        """Retrieves prediction records for a batch."""

        if self.collection is not None:
            try:
                cursor = self.collection.find(
                    {
                        "batch_id": batch_id
                    },
                    {
                        "_id": 0
                    },
                )

                return list(cursor)

            except Exception as e:
                logger.error(
                    "Error querying predictions "
                    f"by batch ID: {e}"
                )

        results = [
            prediction
            for prediction in _IN_MEMORY_PREDICTIONS
            if prediction.get("batch_id")
            == batch_id
        ]

        for prediction in results:
            prediction.pop("_id", None)

        return results

    def get_all(
        self,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves recent prediction records across
        both individual and batch inference.
        """

        if self.collection is not None:
            try:
                cursor = (
                    self.collection
                    .find(
                        {},
                        {
                            "_id": 0
                        },
                    )
                    .sort(
                        "created_at",
                        -1,
                    )
                    .limit(limit)
                )

                return list(cursor)

            except Exception as e:
                logger.error(
                    "Error retrieving all predictions "
                    f"from MongoDB: {e}"
                )

        # Offline / in-memory fallback.
        predictions = list(
            reversed(
                _IN_MEMORY_PREDICTIONS
            )
        )

        for prediction in predictions:
            prediction.pop("_id", None)

        return predictions[:limit]

    def get_monitoring_summary(
        self,
    ) -> Dict[str, Any]:
        """Calculates aggregated prediction monitoring metrics."""

        if self.collection is not None:
            try:
                total_count = (
                    self.collection.count_documents({})
                )

                high_risk_count = (
                    self.collection.count_documents(
                        {
                            "risk_recommendation":
                                "High Risk - Review Required"
                        }
                    )
                )

                pipeline = [
                    {
                        "$group": {
                            "_id": None,
                            "avg_probability": {
                                "$avg":
                                    "$attrition_probability"
                            },
                            "avg_latency": {
                                "$avg":
                                    "$latency_ms"
                            },
                        }
                    }
                ]

                aggregation_result = list(
                    self.collection.aggregate(
                        pipeline
                    )
                )

                if aggregation_result:
                    avg_probability = (
                        aggregation_result[0]
                        .get(
                            "avg_probability"
                        )
                        or 0.0
                    )

                    avg_latency = (
                        aggregation_result[0]
                        .get(
                            "avg_latency"
                        )
                        or 0.0
                    )
                else:
                    avg_probability = 0.0
                    avg_latency = 0.0

                return {
                    "total_predictions":
                        total_count,

                    "high_risk_count":
                        high_risk_count,

                    "review_rate": (
                        high_risk_count
                        / total_count
                        if total_count > 0
                        else 0.0
                    ),

                    "average_attrition_probability":
                        round(
                            float(
                                avg_probability
                            ),
                            4,
                        ),

                    "average_latency_ms":
                        round(
                            float(
                                avg_latency
                            ),
                            2,
                        ),
                }

            except Exception as e:
                logger.error(
                    "Error calculating prediction "
                    f"monitoring summary from MongoDB: {e}"
                )

        # Offline / in-memory fallback.
        total_count = len(
            _IN_MEMORY_PREDICTIONS
        )

        high_risk_count = sum(
            1
            for prediction
            in _IN_MEMORY_PREDICTIONS
            if prediction.get(
                "risk_recommendation"
            )
            == "High Risk - Review Required"
        )

        avg_probability = (
            sum(
                prediction.get(
                    "attrition_probability",
                    0.0,
                )
                for prediction
                in _IN_MEMORY_PREDICTIONS
            )
            / total_count
            if total_count > 0
            else 0.0
        )

        avg_latency = (
            sum(
                prediction.get(
                    "latency_ms",
                    0.0,
                )
                for prediction
                in _IN_MEMORY_PREDICTIONS
            )
            / total_count
            if total_count > 0
            else 0.0
        )

        return {
            "total_predictions":
                total_count,

            "high_risk_count":
                high_risk_count,

            "review_rate": round(
                high_risk_count
                / total_count,
                4,
            )
            if total_count > 0
            else 0.0,

            "average_attrition_probability":
                round(
                    float(
                        avg_probability
                    ),
                    4,
                ),

            "average_latency_ms":
                round(
                    float(
                        avg_latency
                    ),
                    2,
                ),
        }