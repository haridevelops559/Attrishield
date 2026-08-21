"""
Prediction Logs Repository Layer.
Manages individual and batch inference prediction records in MongoDB with in-memory fallback.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from pymongo.database import Database
from backend.app.core.logging import logger

_IN_MEMORY_PREDICTIONS: List[Dict[str, Any]] = []


class PredictionRepository:
    def __init__(self, db: Optional[Database]):
        self.db = db
        self.collection = db["predictions"] if db is not None else None

    def insert_many(self, prediction_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stores a batch of prediction records."""
        if not prediction_docs:
            return []
            
        docs_copy = []
        now_str = datetime.now(timezone.utc).isoformat()
        for doc in prediction_docs:
            c = doc.copy()
            if "created_at" not in c:
                c["created_at"] = now_str
            docs_copy.append(c)

        if self.collection is not None:
            try:
                self.collection.insert_many(docs_copy)
            except Exception as e:
                logger.error(f"Error bulk inserting prediction documents: {e}")

        _IN_MEMORY_PREDICTIONS.extend(docs_copy)
        return docs_copy

    def insert_one(self, prediction_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Stores a single prediction record."""
        res = self.insert_many([prediction_doc]) if prediction_doc else []
        return res[0] if res else prediction_doc

    def get_by_batch_id(self, batch_id: str) -> List[Dict[str, Any]]:
        """Retrieves prediction records associated with a specific batch."""
        if self.collection is not None:
            try:
                cursor = self.collection.find({"batch_id": batch_id}, {"_id": 0})
                return list(cursor)
            except Exception as e:
                logger.error(f"Error querying predictions by batch ID: {e}")

        res = [p for p in _IN_MEMORY_PREDICTIONS if p.get("batch_id") == batch_id]
        for p in res:
            p.pop("_id", None)
        return res

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Calculates aggregated prediction monitoring metrics."""
        if self.collection is not None:
            try:
                total_count = self.collection.count_documents({})
                high_risk_count = self.collection.count_documents({"risk_recommendation": "High Risk - Review Required"})
                
                pipeline = [
                    {"$group": {
                        "_id": None,
                        "avg_probability": {"$avg": "$attrition_probability"},
                        "avg_latency": {"$avg": "$latency_ms"}
                    }}
                ]
                agg_res = list(self.collection.aggregate(pipeline))
                avg_prob = agg_res[0]["avg_probability"] if agg_res else 0.0
                avg_lat = agg_res[0]["avg_latency"] if agg_res else 0.0

                return {
                    "total_predictions": total_count,
                    "high_risk_count": high_risk_count,
                    "review_rate": (high_risk_count / total_count) if total_count > 0 else 0.0,
                    "average_attrition_probability": round(float(avg_prob), 4),
                    "average_latency_ms": round(float(avg_lat), 2)
                }
            except Exception as e:
                logger.error(f"Error calculating prediction monitoring summary from MongoDB: {e}")

        total_count = len(_IN_MEMORY_PREDICTIONS)
        high_risk_count = sum(1 for p in _IN_MEMORY_PREDICTIONS if p.get("risk_recommendation") == "High Risk - Review Required")
        avg_prob = (sum(p.get("attrition_probability", 0.0) for p in _IN_MEMORY_PREDICTIONS) / total_count) if total_count > 0 else 0.0
        avg_lat = (sum(p.get("latency_ms", 0.0) for p in _IN_MEMORY_PREDICTIONS) / total_count) if total_count > 0 else 0.0

        return {
            "total_predictions": total_count,
            "high_risk_count": high_risk_count,
            "review_rate": round(high_risk_count / total_count, 4) if total_count > 0 else 0.0,
            "average_attrition_probability": round(float(avg_prob), 4),
            "average_latency_ms": round(float(avg_lat), 2)
        }
