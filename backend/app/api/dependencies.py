"""
FastAPI Dependencies Module.
Provides reusable dependency injectors for DB session, authentication, and RBAC authorization.
"""

from typing import Optional, Generator, Dict, Any, Callable
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pymongo.database import Database
from backend.app.db.mongodb import db, get_database
from backend.app.db.repositories.user_repository import UserRepository
from backend.app.core.security import decode_access_token
from backend.app.core.exceptions import UnauthorizedError, ForbiddenError
from backend.app.core.logging import logger

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_db_dep() -> Optional[Database]:
    """Dependency for injecting active MongoDB database instance."""
    return get_database()


def get_current_user(token: str = Depends(oauth2_scheme), database: Optional[Database] = Depends(get_db_dep)) -> Dict[str, Any]:
    """
    Validates JWT token from Bearer authorization header and returns user dict.
    """
    payload = decode_access_token(token)
    if not payload:
        raise UnauthorizedError("Invalid or expired authentication token")

    email: str = payload.get("sub")
    if not email:
        raise UnauthorizedError("Token missing user subject claim")

    repo = UserRepository(database)
    user = repo.get_by_email(email)
    if not user:
        raise UnauthorizedError("User associated with token no longer exists")

    if not user.get("is_active", True):
        raise ForbiddenError("User account is inactive")

    return user


def require_role(required_role: str) -> Callable:
    """
    RBAC dependency factory enforcing required user role (e.g. 'HR_ADMIN').
    """
    def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        user_role = current_user.get("role", "").upper()
        if user_role != required_role.upper() and user_role != "SUPER_ADMIN":
            logger.warning(f"Access denied for user {current_user.get('email')}: Required role '{required_role}', user has '{user_role}'")
            raise ForbiddenError(f"Action requires '{required_role}' permissions")
        return current_user

    return role_checker
