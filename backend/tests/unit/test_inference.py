"""
Unit Tests for Model Loading and Inference Execution.
"""

from backend.app.ml.model_loader import model_manager
from backend.app.ml.inference import predict_single_employee, predict_batch_dataframe
import pandas as pd


def test_model_manager_loads_artifacts():
    model, metadata = model_manager.get_model_and_metadata()
    assert model is not None
    assert metadata is not None
    assert "selected_threshold" in metadata
    assert metadata["selected_threshold"] == 0.15


def test_predict_single_employee(sample_employee_raw):
    res = predict_single_employee(sample_employee_raw)
    assert res.prediction_id.startswith("pred_")
    assert 0.0 <= res.attrition_probability <= 1.0
    assert res.selected_threshold == 0.15
    assert res.risk_recommendation in ["High Risk - Review Required", "Low Risk - Monitor"]
    assert "IncomePerJobLevel" in res.engineered_features


def test_predict_batch_dataframe(sample_employee_raw):
    df = pd.DataFrame([sample_employee_raw, sample_employee_raw])
    res_df, summary = predict_batch_dataframe(df, batch_id="batch_test_123")
    
    assert summary.total_records == 2
    assert summary.threshold_used == 0.15
    assert "attrition_probability" in res_df.columns
    assert "risk_recommendation" in res_df.columns
