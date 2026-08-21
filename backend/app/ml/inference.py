"""
ML Inference Service Module.
Executes end-to-end individual and vectorized batch predictions.
Enforces canonical V3 feature engineering parity and threshold evaluation.
"""

import time
import uuid
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from backend.app.ml.feature_engineering import apply_v3_feature_engineering
from backend.app.ml.model_loader import model_manager
from backend.app.core.logging import logger
from backend.app.schemas.inference import PredictionResult, BatchPredictionSummary


def predict_single_employee(raw_input: Dict[str, Any]) -> PredictionResult:
    """
    Runs individual employee inference.
    1. Transforms raw inputs with canonical V3 feature engineering.
    2. Runs XGBoost pipeline predict_proba.
    3. Evaluates probability against selected_threshold (0.15).
    4. Measures latency and logs result.
    """
    start_time = time.perf_counter()
    
    # 1. Convert to DataFrame
    raw_df = pd.DataFrame([raw_input])
    
    # 2. Apply Canonical V3 Feature Engineering
    engineered_df = apply_v3_feature_engineering(raw_df, retain_raw_overtime=False)
    
    # 3. Load Model Artifacts
    model, metadata = model_manager.get_model_and_metadata()
    threshold = float(metadata.get("selected_threshold", 0.15))
    model_version = str(metadata.get("model_version", "v3_engineered_without_raw_overtime"))
    feature_version = str(metadata.get("feature_version", "engineered_features_without_raw_overtime"))

    # 4. Execute Prediction
    try:
        probabilities = model.predict_proba(engineered_df)
        prob = float(probabilities[0][1])
    except Exception as e:
        logger.error(f"Inference execution error: {e}")
        raise RuntimeError(f"Prediction failed: {e}")

    # 5. Apply Threshold Decision Rule
    prediction_class = 1 if prob >= threshold else 0
    recommendation = "High Risk - Review Required" if prediction_class == 1 else "Low Risk - Monitor"
    
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Extract engineered feature values dict
    eng_features_dict = {}
    for col in [
        "IncomePerJobLevel", "PromotionStagnationRatio", "ManagerTenureRatio",
        "RoleTenureRatio", "OverTimeBinary", "CommuteOvertimeBurden", "EarlyCareerFlag"
    ]:
        if col in engineered_df.columns:
            eng_features_dict[col] = float(engineered_df.iloc[0][col])

    return PredictionResult(
        prediction_id=f"pred_{uuid.uuid4().hex[:10]}",
        attrition_probability=round(prob, 4),
        attrition_prediction=prediction_class,
        selected_threshold=threshold,
        risk_recommendation=recommendation,
        model_version=model_version,
        feature_version=feature_version,
        latency_ms=round(latency_ms, 2),
        engineered_features=eng_features_dict
    )


def predict_batch_dataframe(df: pd.DataFrame, batch_id: str) -> Tuple[pd.DataFrame, BatchPredictionSummary]:
    """
    Executes vectorized batch inference across a pandas DataFrame.
    Returns transformed DataFrame with predictions attached, alongside summary statistics.
    """
    start_time = time.perf_counter()
    
    # 1. Canonical Feature Engineering
    engineered_df = apply_v3_feature_engineering(df, retain_raw_overtime=False)
    
    # 2. Load Model Artifacts
    model, metadata = model_manager.get_model_and_metadata()
    threshold = float(metadata.get("selected_threshold", 0.15))
    model_version = str(metadata.get("model_version", "v3_engineered_without_raw_overtime"))

    # 3. Vectorized Prediction
    probabilities = model.predict_proba(engineered_df)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    recommendations = np.where(predictions == 1, "High Risk - Review Required", "Low Risk - Monitor")

    total_records = len(df)
    high_risk_count = int(np.sum(predictions))
    low_risk_count = total_records - high_risk_count
    review_rate = round(high_risk_count / total_records, 4) if total_records > 0 else 0.0
    avg_probability = round(float(np.mean(probabilities)), 4) if total_records > 0 else 0.0

    total_latency_ms = (time.perf_counter() - start_time) * 1000.0
    avg_latency_ms = round(total_latency_ms / total_records, 2) if total_records > 0 else 0.0

    # Attach prediction outputs back to engineered DataFrame
    result_df = engineered_df.copy()
    result_df["batch_id"] = batch_id
    result_df["attrition_probability"] = np.round(probabilities, 4)
    result_df["attrition_prediction"] = predictions
    result_df["risk_recommendation"] = recommendations

    summary = BatchPredictionSummary(
        batch_id=batch_id,
        total_records=total_records,
        high_risk_count=high_risk_count,
        low_risk_count=low_risk_count,
        review_rate=review_rate,
        average_probability=avg_probability,
        average_latency_ms=avg_latency_ms,
        model_version=model_version,
        threshold_used=threshold
    )

    return result_df, summary
