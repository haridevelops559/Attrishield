"""
Batch Repository Layer.
Manages batch job persistence in MongoDB with in-memory fallback.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from pymongo.database import Database
from backend.app.core.logging import logger

_IN_MEMORY_BATCHES: Dict[str, Dict[str, Any]] = {}


class BatchRepository:
    def __init__(self, db: Optional[Database]):
        self.db = db
        self.collection = db["batches"] if db is not None else None

    def create(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Stores a new batch record."""
        batch_doc = batch_data.copy()
        batch_id = batch_doc["batch_id"]
        if "created_at" not in batch_doc:
            batch_doc["created_at"] = datetime.now(timezone.utc).isoformat()
            
        if self.collection is not None:
            try:
                self.collection.insert_one(batch_doc)
            except Exception as e:
                logger.error(f"Error creating batch in MongoDB: {e}")
                
        _IN_MEMORY_BATCHES[batch_id] = batch_doc
        return batch_doc

    def get_by_id(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Finds a batch record by batch_id."""
        if self.collection is not None:
            try:
                res = self.collection.find_one({"batch_id": batch_id})
                if res:
                    res.pop("_id", None)
                    return res
            except Exception as e:
                logger.error(f"Error querying batch by ID: {e}")
        doc = _IN_MEMORY_BATCHES.get(batch_id)
        if doc and "_id" in doc:
            doc.pop("_id", None)
        return doc

    def list_all(self, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        """Lists recent batch jobs sorted by creation date descending."""
        if self.collection is not None:
            try:
                cursor = self.collection.find({}, {"_id": 0}).sort("created_at", -1).skip(skip).limit(limit)
                return list(cursor)
            except Exception as e:
                logger.error(f"Error listing batches from MongoDB: {e}")
                
        sorted_batches = sorted(_IN_MEMORY_BATCHES.values(), key=lambda x: x.get("created_at", ""), reverse=True)
        res = sorted_batches[skip:skip + limit]
        for b in res:
            b.pop("_id", None)
        return res

    def update_status(self, batch_id: str, status: str, completed_at: Optional[str] = None, error: Optional[str] = None, row_count: Optional[int] = None, high_risk_count: Optional[int] = None, average_latency_ms: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Updates the status and summary metrics of an existing batch."""
        update_fields: Dict[str, Any] = {"status": status}
        if completed_at:
            update_fields["completed_at"] = completed_at
        if error:
            update_fields["error"] = error
        if row_count is not None:
            update_fields["row_count"] = row_count
        if high_risk_count is not None:
            update_fields["high_risk_count"] = high_risk_count
        if average_latency_ms is not None:
            update_fields["average_latency_ms"] = average_latency_ms

        if self.collection is not None:
            try:
                self.collection.update_one({"batch_id": batch_id}, {"$set": update_fields})
            except Exception as e:
                logger.error(f"Error updating batch status in MongoDB: {e}")

        if batch_id in _IN_MEMORY_BATCHES:
            _IN_MEMORY_BATCHES[batch_id].update(update_fields)

        return self.get_by_id(batch_id)
