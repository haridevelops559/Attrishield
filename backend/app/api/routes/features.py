"""
Feature Store API Endpoints.

Exposes Feast-like feature definitions, groups, lineage,
online feature retrieval, point-in-time retrieval,
and materialization history.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pymongo.database import Database

from backend.app.schemas.features import (
    FeatureDefinitionSchema,
    FeatureGroupSchema,
    PointInTimeFeatureRequest,
)
from backend.app.feature_store.service import FeatureStoreService
from backend.app.api.dependencies import (
    get_db_dep,
    get_current_user,
)


router = APIRouter(
    prefix="/features",
    tags=["Feature Store"],
)


@router.get(
    "/definitions",
    response_model=List[FeatureDefinitionSchema],
)
def get_feature_definitions(
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user),
):
    """
    Lists all registered feature definitions
    in the feature store.
    """

    service = FeatureStoreService(database)

    return service.get_definitions()


@router.get(
    "/groups",
    response_model=List[FeatureGroupSchema],
)
def get_feature_groups(
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user),
):
    """
    Lists registered feature groups.
    """

    service = FeatureStoreService(database)

    return service.get_groups()


@router.get("/lineage")
def get_feature_lineage(
    feature_name: Optional[str] = None,
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user),
):
    """
    Returns canonical feature lineage dependency graphs.

    If feature_name is provided, lineage for that
    individual feature is returned.
    """

    service = FeatureStoreService(database)

    return service.get_lineage(
        feature_name=feature_name
    )


@router.get("/online/{entity_id}")
def get_online_features(
    entity_id: str,
    feature_version: str = "v3",
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user),
):
    """
    Retrieves online feature values for an entity
    and a specific feature version.
    """

    service = FeatureStoreService(database)

    result = service.get_online_features(
        entity_id=entity_id,
        feature_version=feature_version,
    )

    if not result:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No online features found for "
                f"entity '{entity_id}' with "
                f"feature version '{feature_version}'."
            ),
        )

    return {
        "entity_id": entity_id,
        "feature_version": feature_version,
        "features": result,
    }


@router.post("/point-in-time")
def get_point_in_time_features(
    request: PointInTimeFeatureRequest,
    feature_version: str = "v3",
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user),
):
    """
    Retrieves feature values available at or before
    the requested point in time.

    This endpoint supports historical feature lookup
    for point-in-time-correct ML workflows.
    """

    service = FeatureStoreService(database)

    result = service.get_point_in_time_features(
        entity_ids=request.entity_ids,
        features=request.features,
        feature_version=feature_version,
        as_of_timestamp=request.as_of_timestamp,
    )

    return {
        "feature_version": feature_version,
        "as_of_timestamp": request.as_of_timestamp,
        "entities": result,
    }


@router.get("/materializations")
def list_materialization_history(
    limit: int = 20,
    database: Optional[Database] = Depends(get_db_dep),
    current_user: dict = Depends(get_current_user),
):
    """
    Lists recent feature materialization job logs.
    """

    service = FeatureStoreService(database)

    return service.get_materialization_history(
        limit=limit
    )