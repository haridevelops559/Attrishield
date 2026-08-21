"""
User Repository Layer.
Handles CRUD operations for user entities with MongoDB persistence and in-memory fallback.
"""

from typing import Optional, Dict, Any
from datetime import datetime, timezone
from pymongo.database import Database
from backend.app.core.config import settings
from backend.app.core.security import get_password_hash
from backend.app.core.logging import logger

# In-memory user store fallback for testing or offline mode
_IN_MEMORY_USERS: Dict[str, Dict[str, Any]] = {}


class UserRepository:
    def __init__(self, db: Optional[Database]):
        self.db = db
        self.collection = db["users"] if db is not None else None

    def get_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Retrieves a user document by email address."""
        if self.collection is not None:
            try:
                return self.collection.find_one({"email": email})
            except Exception as e:
                logger.error(f"Error querying user by email: {e}")
        return _IN_MEMORY_USERS.get(email.lower())

    def create(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Creates a new user record."""
        user_doc = user_data.copy()
        user_doc["email"] = user_doc["email"].lower()
        if "created_at" not in user_doc:
            user_doc["created_at"] = datetime.now(timezone.utc).isoformat()
            
        if self.collection is not None:
            try:
                self.collection.insert_one(user_doc)
            except Exception as e:
                logger.error(f"Error inserting user document: {e}")
                
        _IN_MEMORY_USERS[user_doc["email"]] = user_doc
        return user_doc

    def ensure_seed_admin(self) -> Dict[str, Any]:
        """Ensures the default HR Admin user exists on startup."""
        admin_email = settings.SEED_ADMIN_EMAIL.lower()
        existing = self.get_by_email(admin_email)
        if existing:
            return existing
            
        admin_data = {
            "email": admin_email,
            "full_name": settings.SEED_ADMIN_NAME,
            "hashed_password": get_password_hash(settings.SEED_ADMIN_PASSWORD),
            "role": "HR_ADMIN",
            "is_active": True,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        logger.info(f"Seeding default HR Admin user: {admin_email}")
        return self.create(admin_data)
