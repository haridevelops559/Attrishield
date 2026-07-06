# AttriShield — Employee Attrition Risk Decision-Support System

AttriShield is an end-to-end machine-learning decision-support prototype for prioritizing employee attrition-risk reviews and generating explainable retention insights. Rather than treating attrition prediction as an accuracy-only task, the project frames it as an imbalanced, cost-sensitive classification problem.

> **Responsible use:** This is a portfolio prototype for decision support only. It must not be used to make automated employment decisions.

## Key Features

* Leakage-safe machine-learning pipeline with stratified train/test split and stratified 5-fold cross-validation
* SMOTE applied only within training folds to prevent synthetic-data leakage
* Controlled V1 → V2 → V3 feature-pipeline experiments
* Model benchmarking using PR-AUC, ROC-AUC, precision, recall, and F1-score
* Cost-sensitive threshold optimization for recall-oriented risk prioritization
* Calibration reliability analysis using Brier Score and Expected Calibration Error
* Global and local SHAP explanations for model transparency
* Structured false-positive and false-negative error analysis
* Feature-ablation studies to measure feature-group contribution
* MLflow experiment tracking, artifact logging, and model versioning
* FastAPI single/batch inference, Streamlit dashboard, and Gemini-powered retention insights

## Dataset

**IBM HR Analytics Employee Attrition Dataset**

* 1,470 employee records
* 35+ HR attributes
* Features include demographics, compensation, role, satisfaction, travel, tenure, and work-life indicators
* Binary target: employee attrition (`Stay` / `Leave`)

## From the Original Version to V3

The original project used manual preprocessing, changing train/test splits, accuracy-led comparison, and a basic Streamlit demo.

The upgraded workflow uses fixed seeds, an untouched test set, stratified cross-validation, leakage-safe SMOTE, PR-AUC-based model selection, controlled feature experiments, calibration checks, error analysis, SHAP explanations, ablation studies, MLflow tracking, and a threshold-aware deployment workflow.

| Version | Pipeline                             |                                   Result |
| ------- | ------------------------------------ | ---------------------------------------: |
| V1      | Baseline/raw feature pipeline        |       Established leakage-safe benchmark |
| V2      | Engineered features + raw `OverTime` |     CV PR-AUC: 0.621; Test PR-AUC: 0.523 |
| V3      | Engineered features − raw `OverTime` | **CV PR-AUC: 0.625; Test PR-AUC: 0.544** |

## Why V3 Was Selected

V3 retained the engineered overtime and commute signals while removing the raw `OverTime` feature. It was selected because it produced better cross-validation and untouched-test PR-AUC than V2 while using a simpler feature set.

The final V3 model uses XGBoost with stratified 5-fold cross-validation and SMOTE inside each training fold.

## Model Benchmarking

Models were evaluated using metrics appropriate for imbalanced classification.

| Model               | CV Accuracy | CV Precision | CV Recall | CV F1 | CV ROC-AUC | CV PR-AUC |
| ------------------- | ----------: | -----------: | --------: | ----: | ---------: | --------: |
| Logistic Regression |       0.777 |        0.398 |     0.711 | 0.507 |  **0.831** |     0.623 |
| XGBoost             |   **0.876** |    **0.724** |     0.389 | 0.499 |      0.816 | **0.625** |
| Random Forest       |       0.875 |        0.720 |     0.358 | 0.467 |      0.817 |     0.587 |
| Decision Tree       |       0.790 |        0.364 |     0.389 | 0.392 |      0.691 |     0.333 |

XGBoost was selected as the final candidate because it delivered the strongest PR-AUC, balancing ranking quality with a deployable tree-based model.

## Threshold Optimization

The default probability threshold of `0.50` was not assumed to be operationally optimal.

A cost-sensitive threshold analysis used:

```text
False Negative Cost = 5
False Positive Cost = 1
```

The selected operating threshold was **0.15** because missed attrition cases were treated as more costly than additional HR review alerts.

### Final V3 Untouched-Test Results

| Metric             | Score |
| ------------------ | ----: |
| Accuracy           | 0.779 |
| Precision          | 0.385 |
| Recall             | 0.638 |
| F1-score           | 0.480 |
| ROC-AUC            | 0.806 |
| PR-AUC             | 0.544 |
| Selected threshold |  0.15 |

## Confusion-Matrix Analysis

At the recall-oriented threshold of `0.15`, the final V3 model produced:

| Actual / Predicted | Stay | Leave |
| ------------------ | ---: | ----: |
| Stay               |  199 |    48 |
| Leave              |   17 |    30 |

The model identified **30 of 47** employees who actually left. The trade-off was **48 false-positive review alerts**, which is expected because the selected threshold favors recall over precision.

## Calibration and Reliability

The project evaluates whether predicted probabilities are meaningful risk estimates, not only ranking scores.

| Metric                     |  Score |
| -------------------------- | -----: |
| Brier Score                | 0.1049 |
| Expected Calibration Error | 0.0564 |

Reliability-bin analysis was also performed to compare predicted attrition risk against observed attrition frequency.

## Explainability and Error Analysis

### SHAP Explainability

The final model includes:

* Global SHAP feature importance
* Local SHAP explanation for the highest-risk untouched-test employee
* Feature-contribution analysis for individual predictions

Important signals included overtime-derived features, satisfaction, stock options, job level, commute burden, age, and job-role indicators.

### Structured Error Analysis

Predictions were grouped into:

* True Positives
* True Negatives
* False Positives
* False Negatives

False-positive and false-negative cases were reviewed across overtime, business travel, department, job role, age, income, satisfaction, and tenure to identify failure patterns and limitations.

## Feature Ablation Study

Feature groups were removed one at a time and evaluated using cross-validated PR-AUC.

| Removed Feature Group    | PR-AUC Change vs. Full Model | Interpretation                                 |
| ------------------------ | ---------------------------: | ---------------------------------------------- |
| Satisfaction signals     |                       -0.069 | Largest contribution to predictive performance |
| Compensation / seniority |                       -0.035 | Meaningful contribution                        |
| Travel / commute         |                       -0.014 | Smaller but measurable contribution            |
| Raw `OverTime`           |                        0.000 | Redundant after engineered overtime features   |

The ablation study supports the V3 decision: engineered overtime signals retained useful information while the raw overtime field did not add measurable value.

## Experiment Tracking

MLflow tracks:

* Feature version and experiment configuration
* Random seed, split strategy, CV folds, and SMOTE configuration
* Model parameters and evaluation metrics
* Threshold and business-cost assumptions
* Calibration metrics, plots, and diagnostic artifacts
* Final V3 model version

## Deployment

The existing application supports:

* FastAPI inference for single and batch predictions
* Streamlit interface for HR decision support
* SHAP-based explanations
* Gemini-powered retention insights
* Threshold-aware attrition-risk recommendations

## How to Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `imbalanced-learn` · `SHAP` · `MLflow` · `FastAPI` · `Streamlit` · `Gemini API`

## Limitations and Future Work

* The IBM dataset is a public benchmark and may not represent a real organization.
* Attrition labels may reflect factors unavailable in the dataset.
* Predictions should support human review, not automate employment decisions.
* Future work includes fairness analysis, drift monitoring, feedback collection, and scheduled retraining.

