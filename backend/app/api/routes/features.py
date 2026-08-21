"""
Feature Store API Endpoints.
Exposes Feast-like feature definitions, groups, lineage, and online feature store queries.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pymongo.database import Database

from backend.app.schemas.features import FeatureDefinitionSchema, FeatureGroupSchema, PointInTimeFeatureRequest
from backend.app.feature_store.service import FeatureStoreService
from backend.app.api.dependencies import get_db_dep, get_current_user

router = APIRouter(prefix="/features", tags=["Feature Store"])


@router.get("/definitions", response_model=List[FeatureDefinitionSchema])
def get_feature_definitions(
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user)
):
    """Lists all registered feature definitions in the feature store."""
    service = FeatureStoreService(database)
    return service.get_definitions()


@router.get("/groups", response_model=List[FeatureGroupSchema])
def get_feature_groups(
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user)
):
    """Lists feature groups."""
    service = FeatureStoreService(database)
    return service.get_groups()


@router.get("/lineage")
def get_feature_lineage(
    feature_name: Optional[str] = None,
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user)
):
    """Returns canonical feature lineage dependency graphs."""
    service = FeatureStoreService(database)
    return service.get_lineage(feature_name)


@router.get("/online/{entity_id}")
def get_online_features(
    entity_id: str,
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user)
):
    """Retrieves latest online feature values for a specific entity ID."""
    service = FeatureStoreService(database)
    res = service.get_online_features(entity_id=entity_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"No online features found for entity '{entity_id}'.")
    return {"entity_id": entity_id, "features": res}


@router.get("/materializations")
def list_materialization_history(
    limit: int = 20,
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user)
):
    """Lists recent feature materialization job logs."""
    service = FeatureStoreService(database)
    return service.get_materialization_history(limit=limit)
