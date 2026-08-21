"""
Model Information API Endpoints.
Exposes loaded model metadata, threshold, CV/test metrics, and feature list.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends
from backend.app.ml.model_loader import model_manager
from backend.app.api.dependencies import get_current_user

router = APIRouter(prefix="/model", tags=["ML Model"])


@router.get("/info")
def get_model_information(current_user: dict = Depends(get_current_user)):
    """Returns active model metadata, decision threshold, metrics, and feature contract."""
    _, metadata = model_manager.get_model_and_metadata()
    return metadata
