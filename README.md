
# AttriShield — Employee Attrition Risk AI Platform

> **End-to-end AI/ML engineering platform for explainable attrition-risk prediction, cost-sensitive XGBoost inference, versioned features, and production AI serving.**

AttriShield demonstrates the full ML lifecycle: leakage-safe feature engineering, stratified cross-validation with fold-level SMOTE, PR-AUC-driven model benchmarking, cost-aware threshold optimization, calibration, SHAP explainability, error analysis, feature ablation, MLflow tracking, and a versioned Feature Store with point-in-time retrieval. :contentReference[oaicite:0]{index=0}

The production application exposes these capabilities through **FastAPI + Pydantic** APIs and a **React + Vite + TanStack Query + Zod** frontend covering authentication, dashboards, individual and batch inference, monitoring, Feature Store workflows, and AI insights. :contentReference[oaicite:1]{index=1}

> ⚠️ **Responsible AI:** Portfolio decision-support prototype only. Predictions must not be used to make automated employment decisions.

---

## ⚡ 3-Second Scan

| Engineering Area | Implementation | Hiring Signal |
|---|---|---|
| **ML Engineering** | Python · Scikit-learn · XGBoost · SMOTE | Leakage-safe, imbalanced classification |
| **Model Evaluation** | PR-AUC · ROC-AUC · Calibration · Error Analysis | Metric-driven model selection |
| **Decision Thresholding** | Cost-sensitive threshold `0.15` | Business-aware ML decisions |
| **Explainability** | SHAP · Local + Global Explanations | Interpretable ML |
| **Feature Engineering** | Controlled V1 → V2 → V3 pipelines | Reproducible experimentation |
| **MLOps** | MLflow · Model Artifacts · Metadata | Experiment/model tracking |
| **Feature Store** | MongoDB · Versions · Lineage · PIT Retrieval | Training/serving consistency |
| **AI Engineering** | Ollama · Deterministic Fallback | Grounded AI application design |
| **Backend** | FastAPI · Pydantic · Uvicorn | Typed production API serving |
| **Frontend** | React · Vite · React Router | Component-based SPA architecture |
| **Server State** | TanStack Query | API caching and synchronization |
| **Validation** | Zod + Pydantic | Frontend/backend contract validation |
| **Deployment** | Railway | Deployed frontend + backend services |

---

## 🎯 Engineering Highlights

| Layer | What Was Engineered |
|---|---|
| **ML** | Leakage-safe training, stratified CV, fold-level SMOTE, benchmarking and threshold optimization |
| **Model** | XGBoost V3 with versioned inference artifact |
| **Evaluation** | PR-AUC, ROC-AUC, precision, recall, F1, calibration and error analysis |
| **Explainability** | Global/local SHAP and feature-ablation analysis |
| **MLOps** | MLflow experiments, metrics, configurations and artifacts |
| **Serving** | FastAPI single + batch inference APIs |
| **Data** | MongoDB-backed versioned Feature Store with point-in-time retrieval |
| **AI** | Ollama insights with deterministic analytics fallback |
| **Frontend** | React workflows for prediction, analytics, monitoring, Feature Store and AI insights |
| **Deployment** | Independent frontend/backend deployment on Railway |

---

## 🏗️ System Architecture

```text
                         ATTRISHIELD
                              │
                ┌─────────────┴─────────────┐
                │                           │
         REACT FRONTEND                ML PIPELINE
        React + Vite                  Python / sklearn
                │                           │
       ┌────────┼────────┐                  │
       │        │        │                  ▼
    Router   Hooks    Components      Feature Engineering
       │        │        │                  │
       │   TanStack Query                  ▼
       │     + Zod                 Stratified 5-Fold CV
       │        │                         + SMOTE
       ▼        ▼                          │
    Auth     API Service                  ▼
       │        │                     Benchmarking
       └────────┤                          │
                │                          ▼
                │                     XGBoost V3
                │                          │
                │             ┌────────────┼────────────┐
                │             ▼            ▼            ▼
                │          SHAP       Calibration    Ablation
                │             │            │            │
                │             └────────────┼────────────┘
                │                          ▼
                │                Cost-Sensitive Threshold
                │                     FN Cost = 5
                │                     FP Cost = 1
                │                     Threshold = 0.15
                │                          │
                ▼                          ▼
        HTTPS / REST              Versioned Model Artifact
                │                          │
                ▼                          │
        ┌──────────────────────────────────┴──────┐
        │             FASTAPI BACKEND              │
        │        FastAPI + Pydantic + Uvicorn      │
        ├──────────────────────────────────────────┤
        │ Auth · Inference · Batch · Analytics     │
        │ Monitoring · Features · AI Insights      │
        └───────────────┬──────────────┬───────────┘
                        │              │
                ┌───────┴──────┐   ┌───┴────────────┐
                ▼              ▼   ▼                ▼
             MongoDB        XGBoost              Ollama
          Feature Store      + SHAP           + Fallback
                │              │                  │
                ▼              ▼                  ▼
          Versioned/PIT    Predictions       Grounded AI
             Features       + Explanations      Insights

                        MLflow
                           │
             Experiments · Metrics · Artifacts
````

---

## 🧠 ML Engineering

| Area                 | Implementation                                                | Engineering Decision                  |
| -------------------- | ------------------------------------------------------------- | ------------------------------------- |
| **Dataset**          | IBM HR Analytics                                              | Imbalanced attrition classification   |
| **Split**            | Stratified train/test + fixed seed                            | Reproducible evaluation               |
| **Cross-Validation** | Stratified 5-fold CV                                          | Robust model comparison               |
| **Imbalance**        | SMOTE inside training folds only                              | Prevent synthetic-data leakage        |
| **Features**         | Controlled V1 → V2 → V3                                       | Isolated feature experiments          |
| **Benchmarking**     | Logistic Regression · XGBoost · Random Forest · Decision Tree | Comparative model evaluation          |
| **Primary Metric**   | PR-AUC                                                        | Better suited to imbalanced ranking   |
| **Thresholding**     | Cost-sensitive optimization                                   | Recall-oriented operational trade-off |
| **Calibration**      | Brier Score · ECE · reliability                               | Probability-quality assessment        |
| **Explainability**   | SHAP                                                          | Global + local explanations           |
| **Error Analysis**   | FP/FN segmentation                                            | Failure-mode analysis                 |
| **Ablation**         | Feature-group removal                                         | Contribution analysis                 |
| **Artifact**         | Versioned V3 pipeline/model                                   | Reproducible serving                  |
| **Tracking**         | MLflow                                                        | Parameters · metrics · artifacts      |

---

## 📊 ML Results

### V1 → V2 → V3

| Version | Pipeline                        | CV PR-AUC | Test PR-AUC |
| ------- | ------------------------------- | --------: | ----------: |
| V1      | Baseline/raw                    |         — |           — |
| V2      | Engineered + raw `OverTime`     |     0.621 |       0.523 |
| **V3**  | **Engineered − raw `OverTime`** | **0.625** |   **0.544** |

**Selected model: XGBoost V3**

### Model Benchmark

| Model               | CV ROC-AUC | CV PR-AUC |
| ------------------- | ---------: | --------: |
| Logistic Regression |  **0.831** |     0.623 |
| **XGBoost**         |      0.816 | **0.625** |
| Random Forest       |      0.817 |     0.587 |
| Decision Tree       |      0.691 |     0.333 |

### Cost-Aware Threshold

| Metric              |    Result |
| ------------------- | --------: |
| False Negative Cost |         5 |
| False Positive Cost |         1 |
| Selected Threshold  |  **0.15** |
| Test Accuracy       |     0.779 |
| Test Precision      |     0.385 |
| Test Recall         | **0.638** |
| Test F1             |     0.480 |
| Test ROC-AUC        |     0.806 |
| Test PR-AUC         | **0.544** |

---

## 🔍 Explainability & Reliability

| Component          | Purpose                                                        |
| ------------------ | -------------------------------------------------------------- |
| **SHAP**           | Global feature importance + individual prediction explanations |
| **Calibration**    | Probability reliability                                        |
| **Error Analysis** | False-positive / false-negative analysis                       |
| **Ablation**       | Feature-group contribution                                     |

| Calibration Metric         | Result |
| -------------------------- | -----: |
| Brier Score                | 0.1049 |
| Expected Calibration Error | 0.0564 |

---

## ⚙️ Backend Engineering

| Area                 | Implementation                | Responsibility                     |
| -------------------- | ----------------------------- | ---------------------------------- |
| **Framework**        | FastAPI                       | REST API and service orchestration |
| **Runtime**          | Uvicorn                       | ASGI application runtime           |
| **Schemas**          | Pydantic                      | Typed request/response validation  |
| **Configuration**    | Pydantic Settings             | Environment-based configuration    |
| **Authentication**   | JWT                           | Protected API access               |
| **Routing**          | Domain-specific route modules | Separation of API responsibilities |
| **Single Inference** | FastAPI → XGBoost             | Real-time prediction               |
| **Batch Inference**  | FastAPI → model pipeline      | Bulk scoring                       |
| **Analytics**        | Analytics services/routes     | Risk and model statistics          |
| **Monitoring**       | Monitoring routes             | Operational visibility             |
| **Feature Store**    | Feature routes/services       | Versioned feature access           |
| **Explainability**   | SHAP service                  | Prediction explanations            |
| **AI Layer**         | Ollama + fallback             | Grounded retention insights        |
| **API Contract**     | OpenAPI                       | Discoverable API interface         |

### Backend Structure

```text
backend/
├── app/
│   ├── api/
│   │   ├── dependencies.py
│   │   └── routes/
│   │       ├── auth.py
│   │       ├── inference.py
│   │       ├── batches.py
│   │       ├── analytics.py
│   │       ├── monitoring.py
│   │       ├── features.py
│   │       ├── model.py
│   │       └── ollama.py
│   │
│   ├── analytics/
│   ├── core/
│   ├── db/
│   ├── feature_store/
│   ├── llm/
│   ├── ml/
│   ├── schemas/
│   └── main.py
│
└── tests/
```

---

## ⚛️ Frontend Engineering

| Area                | Implementation              | Responsibility                         |
| ------------------- | --------------------------- | -------------------------------------- |
| **UI Architecture** | React components            | Reusable presentation layer            |
| **Build**           | Vite                        | Development/build pipeline             |
| **Routing**         | React Router                | SPA + protected routes                 |
| **Authentication**  | Auth Context                | Shared authentication state            |
| **Server State**    | TanStack Query              | Fetching · caching · synchronization   |
| **Validation**      | Zod                         | Runtime client-side validation         |
| **API Layer**       | Central `api.js`            | HTTP abstraction + auth headers        |
| **Custom Hooks**    | Domain-specific hooks       | Separate data-fetching logic from UI   |
| **Dashboard**       | React + API queries         | Risk KPIs and analytics                |
| **Individual Risk** | React + FastAPI             | Prediction + probability + explanation |
| **Batch**           | React + FastAPI             | Bulk inference workflow                |
| **Monitoring**      | React + API                 | Operational/model visibility           |
| **Feature Store**   | React views                 | Feature versions and metadata          |
| **AI Insights**     | React + AI API              | Retention insight workflow             |
| **UX States**       | Loading/error/empty/success | Resilient async workflows              |

### Frontend Structure

```text
frontend/src/
├── components/
│   ├── layout/
│   └── prediction/
│
├── context/
│   └── AuthContext.jsx
│
├── hooks/
│   ├── useAnalytics.js
│   ├── useBatchPredictions.js
│   ├── useFeatureStore.js
│   ├── useMonitoring.js
│   ├── usePredictEmployee.js
│   └── ...
│
├── pages/
│   ├── Dashboard.jsx
│   ├── IndividualPrediction.jsx
│   ├── BatchInference.jsx
│   ├── BatchResults.jsx
│   ├── Analytics.jsx
│   ├── Monitoring.jsx
│   ├── FeatureStore.jsx
│   ├── AIInsights.jsx
│   └── PredictionDetail.jsx
│
├── routes/
│   └── ProtectedRoute.jsx
│
├── schemas/
│   └── apiSchemas.js
│
└── services/
    └── api.js
```

---

## 🗄️ Data & Feature Store

| Capability              | Implementation             | Purpose                             |
| ----------------------- | -------------------------- | ----------------------------------- |
| Database                | MongoDB                    | Persistent application/feature data |
| Feature Definitions     | Versioned metadata         | Reproducible feature contracts      |
| Feature Versions        | V1/V2/V3                   | Model-compatible feature evolution  |
| Materialization         | Persisted feature values   | Serving-ready features              |
| Lineage                 | Feature metadata           | Trace feature origin/version        |
| Point-in-Time Retrieval | Timestamp-aware retrieval  | Prevent future-data leakage         |
| Version-Aware Queries   | Version + timestamp        | Reconstruct historical inputs       |
| Serving Consistency     | Shared feature definitions | Reduce training-serving skew        |

---

## 🤖 AI Engineering

| Area           | Implementation                 | Role                                           |
| -------------- | ------------------------------ | ---------------------------------------------- |
| Model          | XGBoost V3                     | Attrition-risk prediction                      |
| Explainability | SHAP                           | Prediction reasoning                           |
| LLM            | Ollama                         | Optional retention insights                    |
| Grounding      | Prediction + analytics context | Reduce unsupported AI output                   |
| Fallback       | Deterministic analytics        | Application remains useful when LLM is offline |

---

## 📈 MLOps

| Capability          | Implementation                                     |
| ------------------- | -------------------------------------------------- |
| Experiment Tracking | MLflow                                             |
| Parameters          | Feature versions · seed · CV · SMOTE configuration |
| Metrics             | PR-AUC · ROC-AUC · Precision · Recall · F1         |
| Threshold Tracking  | Business costs + selected threshold                |
| Calibration         | Brier Score + ECE                                  |
| Artifacts           | Versioned model/pipeline                           |
| Metadata            | Model configuration and evaluation information     |
| Reproducibility     | Fixed seeds + controlled pipelines                 |

---

## 📁 Repository Structure

```text
AttriShield/
│
├── backend/
│   ├── app/
│   │   ├── api/routes/
│   │   ├── analytics/
│   │   ├── core/
│   │   ├── db/
│   │   ├── feature_store/
│   │   ├── llm/
│   │   ├── ml/
│   │   └── schemas/
│   └── tests/
│
├── frontend/
│   └── src/
│       ├── components/
│       ├── context/
│       ├── hooks/
│       ├── pages/
│       ├── routes/
│       ├── schemas/
│       └── services/
│
├── notebooks/
│   ├── attrishield_ml_experiment_v3.ipynb
│   └── historical/
│       ├── EAPMPmlflowfe.ipynb
│       └── final-mp-eap (5).ipynb
│
├── artifacts/
│   ├── attrishield_pipeline_v3.joblib
│   └── model_metadata_v3.json
│
├── docs/
├── .env.example
├── Procfile
└── README.md
```

---

## ▶️ Run Locally

| Service      | Commands                                                                           | URL                          |
| ------------ | ---------------------------------------------------------------------------------- | ---------------------------- |
| **Backend**  | `cd backend` → `pip install -r requirements.txt` → `uvicorn app.main:app --reload` | `http://localhost:8000`      |
| **API Docs** | —                                                                                  | `http://localhost:8000/docs` |
| **Frontend** | `cd frontend` → `npm install` → `npm run dev`                                      | `http://localhost:5173`      |

---

## 🌐 Live Deployment

| Service               | Platform | URL                                                                                                      |
| --------------------- | -------- | -------------------------------------------------------------------------------------------------------- |
| **Frontend**          | Railway  | [https://attrishield-production-5133.up.railway.app](https://attrishield-production-5133.up.railway.app) |
| **Backend API**       | Railway  | [https://attrishield-production.up.railway.app](https://attrishield-production.up.railway.app)           |
| **Swagger / OpenAPI** | Railway  | [https://attrishield-production.up.railway.app/docs](https://attrishield-production.up.railway.app/docs) |
| **API Base**          | Railway  | `https://attrishield-production.up.railway.app/api/v1`                                                   |

---

## 🔐 Environment & Security

| Environment  | Configuration                            |
| ------------ | ---------------------------------------- |
| Local        | `.env` / local environment variables     |
| Production   | Railway service variables                |
| Frontend API | `VITE_API_BASE_URL`                      |
| Backend      | JWT · MongoDB · model · AI configuration |
| Credentials  | Never committed to Git                   |

> **Security:** Never commit passwords, API keys, database credentials, JWT secrets or `.env` files.

---

## ⚠️ Limitations

| Area           | Limitation                                                                  |
| -------------- | --------------------------------------------------------------------------- |
| Dataset        | IBM HR Analytics is a public benchmark                                      |
| Generalization | Results may not represent real organizations                                |
| Bias           | Predictions can contain statistical/data biases                             |
| Threshold      | Business costs are illustrative                                             |
| LLM            | AI insights are optional and non-authoritative                              |
| Production     | Requires organization-specific validation, fairness, privacy and governance |

---

## 🔮 Future Engineering Work

| Area           | Planned Direction                         |
| -------------- | ----------------------------------------- |
| MLOps          | Automated model registry and promotion    |
| Monitoring     | Data, feature and model drift             |
| Responsible AI | Fairness evaluation                       |
| Retraining     | Automated retraining pipeline             |
| Testing        | Expanded API/integration/contract testing |
| Deployment     | CI/CD                                     |
| AI             | Production-grade managed LLM              |
| Governance     | Feedback and model monitoring             |

---

## Responsible AI

AttriShield is a **decision-support prototype**, not an automated employment decision system.

Predictions are statistical signals intended to support qualified human review and should be evaluated within appropriate organizational, privacy, fairness and governance requirements.

```
```
