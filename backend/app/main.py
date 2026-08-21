"""
AttriShield FastAPI Application Entry Point.
Sets up middleware, route handlers, database initialization, model loading, and health checks.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pymongo.database import Database

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.db.mongodb import connect_to_mongo, close_mongo_connection, get_database
from backend.app.db.indexes import init_indexes
from backend.app.db.repositories.user_repository import UserRepository
from backend.app.ml.model_loader import model_manager
from backend.app.feature_store.service import FeatureStoreService
from backend.app.llm.ollama_client import OllamaClient

from backend.app.api.routes.auth import router as auth_router
from backend.app.api.routes.inference import router as inference_router
from backend.app.api.routes.batches import router as batches_router
from backend.app.api.routes.analytics import router as analytics_router
from backend.app.api.routes.features import router as features_router
from backend.app.api.routes.monitoring import router as monitoring_router
from backend.app.api.routes.model import router as model_router
from backend.app.api.routes.ollama import router as ollama_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup and shutdown lifecycle management."""
    logger.info("Initializing AttriShield backend service...")
    
    # 1. Connect to MongoDB
    database = connect_to_mongo()
    if database is not None:
        init_indexes(database)
        user_repo = UserRepository(database)
        user_repo.ensure_seed_admin()
        fs_service = FeatureStoreService(database)
    else:
        # Fallback seed admin for offline mode
        user_repo = UserRepository(None)
        user_repo.ensure_seed_admin()
        fs_service = FeatureStoreService(None)

    # 2. Pre-load ML Model Artifacts
    try:
        model_manager.load_artifacts()
    except Exception as e:
        logger.error(f"Failed to load ML artifacts during startup: {e}")

    yield

    # Shutdown
    logger.info("Shutting down AttriShield backend...")
    close_mongo_connection()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-grade 2026 Resume Full-Stack Employee Attrition ML Platform API",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# CORS Middleware Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers under /api/v1
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(inference_router, prefix=settings.API_V1_STR)
app.include_router(batches_router, prefix=settings.API_V1_STR)
app.include_router(analytics_router, prefix=settings.API_V1_STR)
app.include_router(features_router, prefix=settings.API_V1_STR)
app.include_router(monitoring_router, prefix=settings.API_V1_STR)
app.include_router(model_router, prefix=settings.API_V1_STR)
app.include_router(ollama_router, prefix=settings.API_V1_STR)


# Health Check Endpoints
@app.get("/health", tags=["Health"])
def health_check():
    """Overall system health check."""
    return {"status": "healthy", "service": settings.PROJECT_NAME, "version": settings.VERSION}


@app.get("/health/mongodb", tags=["Health"])
def mongodb_health(database: Optional[Database] = Depends(get_database)):
    """MongoDB connection health check."""
    if database is not None:
        try:
            database.command("ping")
            return {"status": "healthy", "database": settings.MONGODB_DB_NAME}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    return {"status": "offline_mode", "message": "Operating in-memory fallback mode"}


@app.get("/health/ml", tags=["Health"])
def ml_health():
    """ML Model artifacts health check."""
    is_loaded = model_manager.is_loaded
    _, metadata = model_manager.get_model_and_metadata()
    return {
        "status": "healthy" if is_loaded else "unhealthy",
        "model_version": metadata.get("model_version"),
        "threshold": metadata.get("selected_threshold")
    }


@app.get("/health/ollama", tags=["Health"])
async def ollama_health():
    """Ollama local LLM service health check."""
    client = OllamaClient()
    healthy = await client.check_health()
    return {
        "status": "healthy" if healthy else "unreachable",
        "ollama_url": settings.OLLAMA_BASE_URL,
        "configured_model": settings.OLLAMA_MODEL
    }


@app.get("/health/feature-store", tags=["Health"])
def feature_store_health(database: Optional[Database] = Depends(get_database)):
    """Feature store health check."""
    service = FeatureStoreService(database)
    defs = service.get_definitions()
    return {
        "status": "healthy",
        "registered_features": len(defs),
        "feature_version": "v3"
    }
