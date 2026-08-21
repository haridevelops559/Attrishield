# 01 - Current System Audit

## Executive Summary
This document presents a comprehensive technical audit of the existing Employee Attrition Prediction ML system ("AttriShield"). The prototype consists of a trained XGBoost classifier pipeline saved as a `.joblib` artifact, a companion metadata JSON file (`model_metadata_v3.json`), an MLflow experiment tracking history embedded in a Google Colab Jupyter Notebook (`EAPMPmlflowfe(3).ipynb`), and a baseline Streamlit web application.

---

## 1. System Architecture Overview

```text
                     ┌───────────────────────────────────┐
                     │    Google Colab Training Notebook │
                     │       (EAPMPmlflowfe(3).ipynb)    │
                     └─────────────────┬─────────────────┘
                                       │
                      ┌────────────────┴────────────────┐
                      │                                 │
           ┌──────────▼───────────┐          ┌──────────▼───────────┐
           │ Model Artifact       │          │ Metadata Artifact    │
           │ (.joblib Pipeline)   │          │ (model_metadata_v3) │
           └──────────┬───────────┘          └──────────┬───────────┘
                      │                                 │
                      └────────────────┬────────────────┘
                                       │
                     ┌─────────────────▼─────────────────┐
                     │     Legacy Streamlit Prototype    │
                     │  (Individual & Batch Inference)   │
                     └───────────────────────────────────┘
```

---

## 2. Model & Companion Artifact Analysis

### Model Artifact (`attrishield_pipeline_v3.joblib`)
- **Primary Executable Artifact**: Scikit-learn / Imblearn Pipeline incorporating preprocessing, imbalanced data handling (SMOTE), and an XGBoost binary classifier.
- **Lineage**: Trained on the IBM HR Employee Attrition dataset using 5-fold Stratified Cross-Validation with SMOTE applied strictly inside cross-validation folds.

### Companion Metadata Artifact (`model_metadata_v3.json`)
The `.joblib` model alone cannot be deployed safely without its companion metadata. Key parameters stored include:
- `model_version`: `"v3_engineered_without_raw_overtime"`
- `feature_version`: `"engineered_features_without_raw_overtime"`
- `selected_threshold`: `0.15` (Selected via 5-fold out-of-fold cost minimization: False Negative Cost = $5x, False Positive Cost = $1x)
- `test_metrics`:
  - ROC-AUC: `0.8058`
  - PR-AUC: `0.5440`
  - Recall (at 0.15 threshold): `0.6383`
  - Precision (at 0.15 threshold): `0.3846`
  - Brier Score: `0.1049`
  - Expected Calibration Error (ECE): `0.0564`

---

## 3. Canonical V3 Feature Engineering Contract

The V3 feature engineering pipeline transforms raw HR data into 7 domain-specific engineered features:

| Engineered Feature | Calculation / Definition | Purpose |
| :--- | :--- | :--- |
| `IncomePerJobLevel` | `MonthlyIncome / JobLevel` | Detects compensation disparities relative to hierarchy level |
| `PromotionStagnationRatio` | `YearsInCurrentRole / (YearsSinceLastPromotion + 1)` | Measures career trajectory stagnation |
| `ManagerTenureRatio` | `YearsWithCurrManager / (TotalWorkingYears + 1)` | Captures manager relationship longevity relative to total career |
| `RoleTenureRatio` | `YearsInCurrentRole / (TotalWorkingYears + 1)` | Identifies role specialization vs. stagnation |
| `OverTimeBinary` | `1 if OverTime == 'Yes' else 0` | Standardizes categorical overtime input |
| `CommuteOvertimeBurden` | `DistanceFromHome * OverTimeBinary` | Quantifies combined commute and overtime strain |
| `EarlyCareerFlag` | `1 if TotalWorkingYears <= 3 else 0` | Isolates high-volatility early career employees |

> [!IMPORTANT]
> Raw `OverTime` is omitted from final feature input to prevent collinearity with `OverTimeBinary` and `CommuteOvertimeBurden` (`raw_overtime_retained: false`).

---

## 4. Inference & Monitoring Flows

### Individual Inference
1. User provides raw employee features via form.
2. Canonical V3 feature transformations are calculated.
3. Transformed dataset is fed into `attrishield_pipeline_v3.joblib`.
4. Output probability `P(Attrition=1)` is evaluated against `selected_threshold = 0.15`.
5. Recommendation rendered: `"High Risk - Review Required"` if `P >= 0.15` else `"Low Risk - Monitor"`.

### Batch Inference
1. User uploads a CSV file of employee records.
2. Canonical feature engineering is applied across all rows.
3. Vectorized batch inference produces prediction probabilities and risk classifications.
4. Summary distributions (high risk count, review rate, latency) are displayed.

---

## 5. Existing Limitations & Migration Risks

| Area | Current Limitation | Migration Strategy / Target |
| :--- | :--- | :--- |
| **Backend** | Streamlit couples UI and ML logic tightly in a single process | Decouple into high-performance FastAPI service layer with Pydantic validation |
| **Data Persistence** | Prediction logs written to transient local files or lost | Implement MongoDB database with structured repositories for batches, predictions, and feature store |
| **Feature Store** | Features engineered dynamically on every request with no versioning | Build a Feast-inspired MongoDB Feature Store supporting feature definitions, versioning, lineage, and materialization |
| **Analytics Engine** | Basic Streamlit summary metrics | Pandas/NumPy driven analytics engine providing dynamic filters, group-by, pivot tables, and KPI aggregations |
| **GenAI / LLM** | None | Integrate local Ollama (Qwen) LLM for grounded retention insights from aggregated analytics |
| **Authentication** | None | Implement JWT authentication with role-based access control (HR Admin) |
| **Frontend** | Streamlit UI | Rebuild UI in modern React with Tailwind CSS and responsive design |

---

## 6. Conclusion
The V3 ML artifacts and feature engineering contracts are solid and scientifically validated. The primary goal of this project is to build a production-grade full-stack platform around this core model without altering the underlying ML inference contract.
