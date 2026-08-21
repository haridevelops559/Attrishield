"""
Model & Metadata Loader Module.
Loads the executable `.joblib` model pipeline and `model_metadata_v3.json` companion artifact.
"""

import json
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import joblib
import sklearn.base
import xgboost as xgb

# Compatibility patch for XGBoost 2.0.x with scikit-learn 1.6.x
class DummyTags:
    requires_fit = True

def _compat_sklearn_tags(self):
    try:
        from sklearn.utils._tags import Tags
        return Tags()
    except Exception:
        return DummyTags()

xgb.XGBClassifier.__sklearn_tags__ = _compat_sklearn_tags
xgb.XGBModel.__sklearn_tags__ = _compat_sklearn_tags
if not hasattr(sklearn.base.BaseEstimator, "__sklearn_tags__"):
    sklearn.base.BaseEstimator.__sklearn_tags__ = _compat_sklearn_tags

from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.core.exceptions import ModelNotFoundError


class ModelManager:
    _instance: Optional['ModelManager'] = None
    
    def __init__(self):
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self.is_loaded: bool = False

    @classmethod
    def get_instance(cls) -> 'ModelManager':
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def load_artifacts(self) -> Tuple[Any, Dict[str, Any]]:
        """Loads model pipeline and companion metadata from configured artifact paths."""
        model_path = settings.absolute_model_path
        metadata_path = settings.absolute_metadata_path

        if not model_path.exists():
            logger.error(f"Model joblib file not found at: {model_path}")
            raise ModelNotFoundError(f"Model file missing: {model_path}")

        if not metadata_path.exists():
            logger.error(f"Model metadata JSON file not found at: {metadata_path}")
            raise ModelNotFoundError(f"Metadata file missing: {metadata_path}")

        try:
            logger.info(f"Loading metadata companion artifact from {metadata_path}...")
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

            logger.info(f"Loading executable ML model pipeline from {model_path}...")
            self.model = joblib.load(model_path)
            self.is_loaded = True
            
            logger.info(
                f"Successfully loaded ML model '{self.metadata.get('model_name')}' "
                f"(Version: {self.metadata.get('model_version')}, Threshold: {self.metadata.get('selected_threshold')})"
            )
            return self.model, self.metadata
        except Exception as e:
            logger.error(f"Failed loading ML model artifacts: {e}")
            raise ModelNotFoundError(f"Artifact loading error: {str(e)}")

    def get_model_and_metadata(self) -> Tuple[Any, Dict[str, Any]]:
        if not self.is_loaded or self.model is None:
            return self.load_artifacts()
        return self.model, self.metadata

    def get_threshold(self) -> float:
        """Returns the decision threshold defined in metadata (default 0.15)."""
        if not self.metadata:
            _, meta = self.get_model_and_metadata()
            return float(meta.get("selected_threshold", 0.15))
        return float(self.metadata.get("selected_threshold", 0.15))


model_manager = ModelManager.get_instance()
