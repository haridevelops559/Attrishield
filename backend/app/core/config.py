"""
System Configuration Module.
Loads environment variables and application settings using Pydantic BaseSettings.
"""

from pathlib import Path
from typing import List, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    PROJECT_NAME: str = "AttriShield HR Attrition Platform"
    VERSION: str = "3.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = "attri_shield_super_secret_jwt_key_change_in_production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480
    
    # MongoDB
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "attrishield"
    
    # ML Model Artifacts
    MODEL_PATH: str = "artifacts/attrishield_pipeline_v3.joblib"
    METADATA_PATH: str = "artifacts/model_metadata_v3.json"
    
    # Ollama Local LLM
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"
    OLLAMA_TIMEOUT_SECONDS: int = 60
    
    # Seed Admin Credentials
    SEED_ADMIN_EMAIL: str = "admin@attrishield.com"
    SEED_ADMIN_PASSWORD: str = "riskbeda12@"
    SEED_ADMIN_NAME: str = "HR Administrator"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    @property
    def absolute_model_path(self) -> Path:
        p = Path(self.MODEL_PATH)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent.parent.parent / p
        return p

    @property
    def absolute_metadata_path(self) -> Path:
        p = Path(self.METADATA_PATH)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent.parent.parent / p
        return p


settings = Settings()
