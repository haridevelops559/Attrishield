"""
Authentication API Endpoints.
Handles OAuth2 login, JWT issuance, and current user profile retrieval.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pymongo.database import Database
from backend.app.schemas.auth import Token, UserResponse, LoginRequest
from backend.app.db.repositories.user_repository import UserRepository
from backend.app.api.dependencies import get_db_dep, get_current_user
from backend.app.core.security import verify_password, create_access_token
from backend.app.core.exceptions import UnauthorizedError

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), database: Optional[Database] = Depends(get_db_dep)):
    """
    OAuth2 compatible token login endpoint. Accepts username (email) and password.
    """
    repo = UserRepository(database)
    user = repo.get_by_email(form_data.username)
    if not user or not verify_password(form_data.password, user.get("hashed_password", "")):
        raise UnauthorizedError("Incorrect email or password")

    access_token = create_access_token(data={"sub": user["email"], "role": user.get("role", "HR_ADMIN")})
    return Token(
        access_token=access_token,
        token_type="bearer",
        user_role=user.get("role", "HR_ADMIN"),
        user_email=user["email"]
    )


@router.post("/login/json", response_model=Token)
def login_json(request: LoginRequest, database: Optional[Database] = Depends(get_db_dep)):
    """JSON payload login endpoint for frontend requests."""
    repo = UserRepository(database)
    user = repo.get_by_email(request.email)
    if not user or not verify_password(request.password, user.get("hashed_password", "")):
        raise UnauthorizedError("Incorrect email or password")

    access_token = create_access_token(data={"sub": user["email"], "role": user.get("role", "HR_ADMIN")})
    return Token(
        access_token=access_token,
        token_type="bearer",
        user_role=user.get("role", "HR_ADMIN"),
        user_email=user["email"]
    )


@router.get("/me", response_model=UserResponse)
def read_current_user(current_user: dict = Depends(get_current_user)):
    """Retrieves profile info for currently authenticated user."""
    return UserResponse(
        email=current_user["email"],
        full_name=current_user.get("full_name", "HR User"),
        role=current_user.get("role", "HR_ADMIN"),
        is_active=current_user.get("is_active", True),
        created_at=current_user.get("created_at", "")
    )
