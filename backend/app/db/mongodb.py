"""
MongoDB Database Connection Management.
Provides database access via PyMongo with thread-safe client lifecycle management.
"""

from typing import Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure
from backend.app.core.config import settings
from backend.app.core.logging import logger


class MongoDB:
    client: Optional[MongoClient] = None
    db: Optional[Database] = None


db = MongoDB()


def connect_to_mongo() -> Optional[Database]:
    """Establishes database connection to MongoDB."""
    try:
        logger.info(f"Connecting to MongoDB at {settings.MONGODB_URI}...")
        db.client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=2000)
        # Verify connection
        db.client.admin.command('ping')
        db.db = db.client[settings.MONGODB_DB_NAME]
        logger.info(f"Successfully connected to MongoDB database: {settings.MONGODB_DB_NAME}")
        return db.db
    except Exception as e:
        logger.warning(f"MongoDB connection failed: {e}. Operating in graceful offline mode for non-persistent API operations.")
        db.client = None
        db.db = None
        return None


def close_mongo_connection():
    """Closes MongoDB database connection on server shutdown."""
    if db.client is not None:
        logger.info("Closing MongoDB client connection...")
        db.client.close()
        db.client = None
        db.db = None


def get_database() -> Optional[Database]:
    """Dependency getter returning active MongoDB database instance or None."""
    return db.db
