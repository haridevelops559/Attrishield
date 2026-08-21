"""
MongoDB Collection Indexes Initialization.
Defines required database indexes for fast query performance and uniqueness constraints.
"""

from typing import Optional
from pymongo.database import Database
from pymongo import ASCENDING, DESCENDING
from backend.app.core.logging import logger


def init_indexes(database: Optional[Database]):
    """Creates database indexes across all AttriShield collections."""
    if database is None:
        logger.info("Skipping MongoDB index creation (database instance not available).")
        return
        
    try:
        # Users Collection
        database.users.create_index([("email", ASCENDING)], unique=True)
        
        # Batches Collection
        database.batches.create_index([("batch_id", ASCENDING)], unique=True)
        database.batches.create_index([("created_at", DESCENDING)])
        
        # Predictions Collection
        database.predictions.create_index([("prediction_id", ASCENDING)], unique=True)
        database.predictions.create_index([("batch_id", ASCENDING)])
        database.predictions.create_index([("created_at", DESCENDING)])
        database.predictions.create_index([("risk_recommendation", ASCENDING)])
        
        # Feature Definitions Collection
        database.feature_definitions.create_index([("feature_name", ASCENDING)], unique=True)
        
        # Feature Groups Collection
        database.feature_groups.create_index([("group_name", ASCENDING)], unique=True)
        
        # Feature Values Collection (Online Feature Store)
        database.feature_values.create_index([("entity_id", ASCENDING), ("feature_name", ASCENDING)], unique=True)
        
        # Feature Materializations Collection
        database.feature_materializations.create_index([("materialization_id", ASCENDING)], unique=True)
        database.feature_materializations.create_index([("timestamp", DESCENDING)])

        # Feature Lineage Collection
        database.feature_lineage.create_index([("feature_name", ASCENDING)], unique=True)
        
        logger.info("MongoDB database indexes successfully verified/created.")
    except Exception as e:
        logger.error(f"Error creating MongoDB indexes: {e}")
