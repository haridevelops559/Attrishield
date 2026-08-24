# AttriShield — Employee Attrition Risk AI Platform

> **End-to-end AI engineering platform for explainable attrition-risk prediction, combining leakage-safe ML experimentation, cost-sensitive XGBoost inference, versioned features, and production AI serving.**

AttriShield demonstrates the **full ML lifecycle**: leakage-safe feature engineering, stratified cross-validation with fold-level SMOTE, PR-AUC-driven model benchmarking, **cost-aware threshold optimization**, calibration and error analysis, SHAP explainability, feature ablation, MLflow experiment tracking, and a **versioned Feature Store with point-in-time retrieval**.

The production system serves these capabilities through **FastAPI + Pydantic** APIs for single/batch inference, analytics, monitoring and feature access, with a **React + Vite + TanStack Query + Zod** frontend for dashboard, individual-risk, batch, monitoring and AI-insight workflows. **Ollama with deterministic analytics fallback** provides the optional LLM insight layer.

> ⚠️ **Responsible AI:** Portfolio decision-support prototype only. Model outputs must not be used to make automated employment decisions.

---

## ⚡ End-to-End System

| System Area | Frontend | Backend / AI | Engineering Responsibility |
|---|---|---|---|
| **Dashboard** | React · TanStack Query | FastAPI · Analytics APIs | Risk KPIs, attrition trends, model performance and system metrics |
| **Individual Risk** | React · Zod · TanStack Query | FastAPI · XGBoost · SHAP | Single-employee prediction, probability, risk level and explanations |
| **Batch Prediction** | React · File/Data Upload | FastAPI · Batch Inference | Bulk scoring, validation, prediction results and review prioritization |
| **Monitoring** | React · TanStack Query | FastAPI · Monitoring APIs | Model/runtime health, prediction activity and operational metrics |
| **Feature Store** | React Feature Views | FastAPI · MongoDB | Feature definitions, versions, materialization and point-in-time retrieval |
| **AI Insights** | React Insight UI | Ollama · Deterministic Fallback | Retention insights grounded in model predictions and analytics |
| **Authentication** | React Auth Context | FastAPI Auth APIs · JWT | Login, token handling and protected application routes |
| **Model Serving** | API-driven UI | FastAPI · Uvicorn · XGBoost | Production model loading and single/batch inference |
| **Explainability** | Risk + explanation views | SHAP | Global feature importance and local prediction explanations |
| **Validation** | Zod | Pydantic | Typed request/response validation across frontend and backend |
| **Server State** | TanStack Query | REST APIs | API caching, loading/error states and query synchronization |
| **Data Layer** | React data views | MongoDB · Feature Store | Persistent feature values, metadata and versioned data |
| **ML Pipeline** | — | Scikit-learn · XGBoost · SMOTE | Leakage-safe training, CV, imbalance handling and benchmarking |
| **Model Evaluation** | — | PR-AUC · ROC-AUC · Calibration | Threshold optimization, error analysis and reliability evaluation |
| **MLOps** | — | MLflow · Model Artifacts | Experiment tracking, metrics, configurations and model metadata |
| **Deployment** | React/Vite build | FastAPI service | Railway-based frontend/backend deployment and environment configuration |

---
## 🏗️ End-to-End Application + ML Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                              REACT FRONTEND                                  │
│                         React + Vite + TypeScript                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Login                                                                       │
│    │                                                                         │
│    ▼                                                                         │
│  Auth Context ── JWT / Session ── Protected Routes                          │
│    │                                                                         │
│    ├────────────── Dashboard ──────────────┐                                │
│    │                                        │                                │
│    ├──────── Individual Risk ───────────────┤                                │
│    │                                        │                                │
│    ├──────── Batch Prediction ──────────────┤                                │
│    │                                        │                                │
│    ├──────── Monitoring ────────────────────┤                                │
│    │                                        │                                │
│    ├──────── Feature Store ─────────────────┤                                │
│    │                                        │                                │
│    └──────── AI Insights ───────────────────┘                                │
│                                                                              │
│  TanStack Query → API fetching / caching / synchronization                   │
│  Zod           → client-side validation                                      │
│  React Router  → application routing / protected views                       │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                              HTTPS / REST
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              FASTAPI BACKEND                                 │
│                    FastAPI + Pydantic + Uvicorn                              │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Authentication API                                                          │
│       │                                                                      │
│       ├── Login / Token Validation                                           │
│       └── Protected API Routes                                               │
│                                                                              │
│  Dashboard / Analytics API                                                   │
│       │                                                                      │
│       ├── Risk KPIs                                                         │
│       ├── Attrition Analytics                                                │
│       └── Model / Prediction Statistics                                      │
│                                                                              │
│  Inference API                                                               │
│       │                                                                      │
│       ├── Single Employee Prediction                                         │
│       └── Batch Prediction                                                   │
│                                                                              │
│  Explanation API                                                             │
│       │                                                                      │
│       └── SHAP / Feature Contributions                                        │
│                                                                              │
│  Feature Store API                                                           │
│       │                                                                      │
│       ├── Feature Definitions                                                │
│       ├── Feature Versions                                                   │
│       ├── Materialization                                                    │
│       └── Point-in-Time Retrieval                                             │
│                                                                              │
│  Monitoring API                                                              │
│       │                                                                      │
│       └── Model / Runtime / Prediction Metrics                               │
│                                                                              │
│  AI Insights API                                                             │
│       │                                                                      │
│       └── Grounded Retention Insights                                        │
│                                                                              │
│  Pydantic → request / response validation                                    │
│  FastAPI  → typed REST API + OpenAPI                                         │
│  Uvicorn  → ASGI application runtime                                         │
└───────────────┬───────────────────────┬───────────────────────┬──────────────┘
                │                       │                       │
                ▼                       ▼                       ▼
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│    ML INFERENCE      │   │    FEATURE STORE     │   │     AI INSIGHTS      │
├──────────────────────┤   ├──────────────────────┤   ├──────────────────────┤
│                      │   │                      │   │                      │
│  XGBoost V3          │   │  MongoDB             │   │  Ollama              │
│       │              │   │       │              │   │       │              │
│       ▼              │   │       ▼              │   │       ▼              │
│  Risk Probability    │   │  Versioned Features  │   │  LLM Insight         │
│       │              │   │       │              │   │                      │
│       ▼              │   │       ▼              │   │  If unavailable      │
│  Threshold = 0.15    │   │  Point-in-Time       │   │       ↓              │
│       │              │   │  Retrieval            │   │  Deterministic       │
│       ▼              │   │                      │   │  Analytics Fallback  │
│  Risk Classification │   │  Feature Lineage     │   │                      │
│                      │   │  + Materialization   │   │  Grounded by model   │
│  SHAP Explanations   │   │                      │   │  + analytics data    │
└──────────────────────┘   └──────────────────────┘   └──────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              ML PIPELINE                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  IBM HR Analytics Dataset                                                    │
│             │                                                                │
│             ▼                                                                │
│  Feature Engineering                                                        │
│             │                                                                │
│             ▼                                                                │
│  Stratified Train / Test Split                                               │
│             │                                                                │
│             ▼                                                                │
│  Stratified 5-Fold Cross Validation                                          │
│             │                                                                │
│             ▼                                                                │
│  SMOTE inside training folds only                                            │
│             │                                                                │
│             ▼                                                                │
│  Model Benchmarking                                                          │
│      ┌────────────┬──────────────┬──────────────┐                            │
│      │ Logistic   │ XGBoost      │ Random       │                            │
│      │ Regression │              │ Forest       │                            │
│      └────────────┴──────────────┴──────────────┘                            │
│             │                                                                │
│             ▼                                                                │
│       XGBoost V3                                                              │
│             │                                                                │
│      ┌──────┼───────────┬────────────┐                                       │
│      ▼      ▼           ▼            ▼                                       │
│  PR-AUC  Calibration  SHAP       Ablation                                    │
│      │      │           │            │                                       │
│      └──────┴───────────┴────────────┘                                       │
│                     │                                                        │
│                     ▼                                                        │
│          Cost-Sensitive Threshold                                            │
│              FN Cost = 5                                                     │
│              FP Cost = 1                                                     │
│              Threshold = 0.15                                                │
│                     │                                                        │
│                     ▼                                                        │
│              Model Artifact                                                   │
└─────────────────────┬────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              MLOps                                           │
│                                 MLflow                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Experiments · Parameters · Feature Versions · CV Results · PR-AUC           │
│  ROC-AUC · Threshold · Calibration · Artifacts · Model Metadata              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

````
## 🧠 ML Engineering — Implementation Nuances

| Area | Implementation | Engineering Signal |
|---|---|---|
| Data Split | Stratified train/test split + fixed seed | Reproducible evaluation |
| Cross-Validation | Stratified 5-fold CV | Robust model comparison |
| Imbalance | SMOTE **inside training folds only** | Prevents synthetic-data leakage |
| Feature Engineering | Controlled V1 → V2 → V3 pipelines | Reproducible feature experiments |
| Model Selection | Logistic Regression · XGBoost · Random Forest · Decision Tree | Comparative benchmarking |
| Primary Metric | PR-AUC | Appropriate for imbalanced attrition target |
| Thresholding | Cost-sensitive threshold = `0.15` | Recall-oriented business trade-off |
| Calibration | Brier Score + ECE + reliability analysis | Probability quality |
| Explainability | Global + local SHAP | Model transparency |
| Error Analysis | FP / FN segmentation | Failure-mode analysis |
| Ablation | Remove feature groups individually | Feature contribution measurement |
| Model Artifact | Versioned V3 pipeline/model | Reproducible inference |
| Experiment Tracking | MLflow | Parameters · metrics · artifacts |
| Feature Lifecycle | Versioned Feature Store + PIT retrieval | Training/serving consistency |

---

## ⚛️ Frontend Engineering

| Area | Implementation | Purpose |
|---|---|---|
| UI | React | Component-based application |
| Build | Vite | Fast development/build pipeline |
| Routing | React Router | Dashboard + protected application routes |
| Authentication | Auth Context + JWT/session handling | Application authentication state |
| Server State | TanStack Query | API caching · fetching · synchronization |
| Validation | Zod | Runtime input/data validation |
| Dashboard | React + API queries | KPIs · attrition analytics · model metrics |
| Individual Risk | React + FastAPI | Employee risk · probability · SHAP explanations |
| Batch | React upload/data workflow | Bulk prediction and result review |
| Monitoring | React + API data | Runtime/model/prediction monitoring |
| Feature Store | React feature views | Feature versions · metadata · retrieval |
| AI Insights | React insight interface | LLM/fallback retention insights |
| API Errors | Query/mutation error states | Loading · failure · retry UX |

---

## ⚙️ Backend & API Engineering

| Area | Implementation | Purpose |
|---|---|---|
| API Framework | FastAPI | REST API + service orchestration |
| Runtime | Uvicorn | ASGI production runtime |
| Schemas | Pydantic | Typed request/response validation |
| Configuration | Pydantic Settings | Environment-based configuration |
| Authentication | JWT-based API authentication | Protected endpoints |
| API Modules | Auth · Inference · Batch · Analytics · Monitoring · Features · AI | Domain separation |
| Single Inference | FastAPI → XGBoost | Real-time risk prediction |
| Batch Inference | FastAPI → model pipeline | Bulk scoring |
| Explainability | FastAPI → SHAP | Prediction-level explanations |
| Analytics | FastAPI service layer | Aggregated risk/model analytics |
| Feature APIs | FastAPI → Feature Store | Versioned feature access |
| Monitoring | FastAPI monitoring endpoints | Operational/model visibility |
| AI Layer | FastAPI → Ollama/fallback | Grounded AI insights |
| API Contract | OpenAPI | Discoverable typed API surface |

---

## 🗄️ Database & Feature Store

| Area | Implementation | Purpose |
|---|---|---|
| Database | MongoDB | Persistent application/feature data |
| Feature Definitions | Versioned metadata | Reproducible feature contracts |
| Feature Groups | Logical grouping | Manage related features |
| Feature Versions | V1/V2/V3 | Model-compatible feature evolution |
| Materialization | Persist computed features | Serving-ready feature values |
| Lineage | Feature metadata | Trace feature origin/version |
| Point-in-Time Retrieval | `timestamp <= requested_time` | Prevent future-data leakage |
| Version-Aware Querying | Feature version + timestamp | Reconstruct historical inputs |
| Serving Consistency | Same feature definitions for training/inference | Reduce training-serving skew |


## 🧪 ML Results

### V1 → V2 → V3

| Version | Feature Pipeline            | CV PR-AUC | Test PR-AUC |
| ------- | --------------------------- | --------: | ----------: |
| V1      | Baseline/raw                |         — |           — |
| V2      | Engineered + raw `OverTime` |     0.621 |       0.523 |
| **V3**  | Engineered − raw `OverTime` | **0.625** |   **0.544** |

**V3** was selected for improved cross-validation and untouched-test PR-AUC while removing a redundant raw feature.

### Model Benchmark

| Model               | CV ROC-AUC | CV PR-AUC |
| ------------------- | ---------: | --------: |
| Logistic Regression |  **0.831** |     0.623 |
| **XGBoost**         |      0.816 | **0.625** |
| Random Forest       |      0.817 |     0.587 |
| Decision Tree       |      0.691 |     0.333 |

**Selected model: XGBoost**

---

## 🎚️ Threshold Optimization

The default `0.50` threshold was not assumed to be operationally optimal.

| Parameter           |     Value |
| ------------------- | --------: |
| False Negative Cost |         5 |
| False Positive Cost |         1 |
| Selected Threshold  |  **0.15** |
| Test Accuracy       |     0.779 |
| Test Precision      |     0.385 |
| Test Recall         | **0.638** |
| Test F1             |     0.480 |
| Test ROC-AUC        |     0.806 |
| Test PR-AUC         |     0.544 |

At threshold `0.15`, the model identified **30/47 actual attrition cases** with **48 false-positive review alerts**.

---

## 🔍 Explainability & Reliability

| Component      | Purpose                                       |
| -------------- | --------------------------------------------- |
| SHAP           | Global and individual prediction explanations |
| Calibration    | Probability reliability                       |
| Error Analysis | False-positive / false-negative analysis      |
| Ablation       | Feature-group contribution                    |

| Calibration Metric         | Result |
| -------------------------- | -----: |
| Brier Score                | 0.1049 |
| Expected Calibration Error | 0.0564 |

### Feature Ablation

| Removed Feature Group    | PR-AUC Change |
| ------------------------ | ------------: |
| Satisfaction             |    **-0.069** |
| Compensation / Seniority |        -0.035 |
| Travel / Commute         |        -0.014 |
| Raw `OverTime`           |         0.000 |

---

## 🗃️ Feature Store

The platform includes a versioned Feature Store supporting:

| Capability              | Purpose                     |
| ----------------------- | --------------------------- |
| Feature definitions     | Central feature metadata    |
| Feature groups          | Logical organization        |
| Feature versions        | V1/V2/V3 management         |
| Materialization         | Persist feature values      |
| Lineage                 | Track feature origins       |
| Point-in-Time retrieval | Prevent future-data leakage |
| Version-aware queries   | Reproduce model inputs      |

Example:

```text
Feature Version: v3
Employee: emp_1

IncomePerJobLevel = 2539.4
OverTimeBinary    = 0
```

---




---



---

## 📁 Repository Structure

```text
AttriShield/
├── backend/
│   ├── app/
│   │   ├── api/routes/
│   │   ├── core/
│   │   ├── feature_store/
│   │   └── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── context/
│   │   ├── hooks/
│   │   ├── pages/
│   │   ├── routes/
│   │   └── services/
│   └── package.json
│
├── notebooks/
│   └── attrishield_ml_experiment_v3.ipynb
│
├── artifacts/
│   ├── attrishield_pipeline_v3.joblib
│   └── model_metadata_v3.json
│
├── docs/
└── README.md
```

---

## ▶️ Run Locally

| Service | Commands | URL |
|---|---|---|
| **Backend** | `cd backend` → `pip install -r requirements.txt` → `uvicorn app.main:app --reload` | http://localhost:8000 |
| **API Docs** | — | http://localhost:8000/docs |
| **Frontend** | `cd frontend` → `npm install` → `npm run dev` | http://localhost:5173 |

---

## 🌐 Live Deployment

| Service | Platform | URL |
|---|---|---|
| **Frontend** | Railway | https://attrishield-production-5133.up.railway.app |
| **Backend API** | Railway | https://attrishield-production.up.railway.app |
| **API Docs** | Railway | https://attrishield-production.up.railway.app/docs |
| **Production API Base** | Railway | `https://attrishield-production.up.railway.app/api/v1` |

---

## 🔐 Environment & Credentials

| Environment | Configuration |
|---|---|
| **Local** | `.env` / local environment variables |
| **Production** | Railway service variables |
| **Frontend API** | `VITE_API_BASE_URL` |
| **Backend Secrets** | JWT, database, model and AI configuration via environment variables |
| **Credentials** | Not committed to GitHub |

> **Security:** `.env` files, passwords, API keys, database credentials and production secrets must never be committed to the repository.

## ⚠️ Limitations

| Area           | Limitation                                                                  |
| -------------- | --------------------------------------------------------------------------- |
| Dataset        | IBM HR Analytics is a public benchmark                                      |
| Generalization | Results may not represent real organizations                                |
| Bias           | Predictions may contain statistical/data biases                             |
| Threshold      | Business costs are illustrative                                             |
| LLM            | AI insights are optional and not authoritative                              |
| Production     | Requires organization-specific validation, fairness, privacy and governance |

---

## 🔮 Future Work

| Area           | Planned Work                         |
| -------------- | ------------------------------------ |
| MLOps          | Model registry + automated promotion |
| Monitoring     | Data / feature / model drift         |
| Responsible AI | Fairness evaluation                  |
| Retraining     | Automated retraining pipeline        |
| Testing        | API contracts + integration tests    |
| Deployment     | CI/CD                                |
| AI             | Production-grade managed LLM         |
| Governance     | Feedback + model monitoring          |

---

## Responsible AI

AttriShield is a **decision-support prototype**, not an automated employment decision system.

Predictions should be treated as statistical signals requiring qualified human review and appropriate organizational context.

```
```
