# 03 - 2026 Resume Skills

This document maps the architectural capabilities implemented in AttriShield to industry-standard 2026 Machine Learning Engineering, MLOps, Data Engineering, Backend, GenAI, and Full-Stack competencies.

---

## 1. Skill Categorization Matrix

### ML Engineering
- **XGBoost Inference**: Loading and serving pre-trained tree-based pipeline models via `joblib`.
- **Feature Engineering Parity**: Maintaining exact contract alignment between offline training and online serving.
- **Threshold Tuning**: Utilizing cost-optimized Decision Thresholds (`0.15`) based on asymmetric business cost matrices ($5 FN vs. $1 FP).
- **Probability Calibration**: Tracking Brier score (`0.1049`) and Expected Calibration Error (`0.0564`) for reliable risk scores.
- **Batch & Real-time Processing**: Vectorized batch inference pipelines alongside single-record low-latency endpoints.

### MLOps
- **Model Lineage**: Tracking model parameters, metrics, and artifact versions linked with MLflow metadata (`model_metadata_v3.json`).
- **Feature Store Engineering**: Building a MongoDB-backed Feast-inspired feature store supporting feature definitions, materialization, freshness, and lineage.
- **Prediction Drift Monitoring**: Recording inference throughput, risk distribution shifts, and latency percentiles.
- **Artifact Management**: Versioned management of model binary (`attrishield_pipeline_v3.joblib`) and deployment metadata.

### Data Engineering
- **Pandas & NumPy Analytics**: High-performance vectorized operations for dynamic filtering, custom aggregations, and pivot table generation.
- **Schema Enforcement**: Input/output contract validation using Pydantic models.
- **NoSQL Data Modeling**: MongoDB database architecture for document persistence across batches, features, and monitoring logs.
- **Batch Processing**: Scalable batch inference ingestion pipeline handling CSV parsing, validation, transformation, and database storage.

### Backend Engineering
- **FastAPI Framework**: Modular RESTful backend using APIRouter, dependency injection, and async request handling.
- **Authentication & Security**: JWT-based authentication with bcrypt password hashing and Role-Based Access Control (`HR_ADMIN`).
- **Clean Architecture**: Decoupled Layered Architecture (API -> Service -> Repository -> Database).
- **OpenAPI Standards**: Self-documenting Swagger/ReDoc schemas automatically generated via Pydantic response models.

### GenAI & LLM Integration
- **Local LLM Inference**: Integration with Ollama serving local open-weights LLMs (Qwen).
- **Grounded Prompting**: Constructing deterministic statistical context prompts to prevent hallucinations.
- **LLM Safety Controls**: Restricting automated decision-making and enforcing advice-only retention recommendations.
- **Resilient AI Pipelines**: Graceful fallback handling when LLM services are offline or unreachable.

### Modern Frontend & UI
- **React 18 Architecture**: Clean component-driven architecture using functional components, hooks, and context API.
- **Tailwind CSS**: Utility-first responsive design system for administrative HR dashboards.
- **Data Visualization**: Reusable interactive charts (risk distribution, department breakdown, tenure vs income).

### DevOps & Software Quality
- **Docker Orchestration**: Containerized multi-container setup via Docker & Docker Compose (`backend`, `frontend`, `mongodb`).
- **Automated Testing**: Comprehensive `pytest` suite covering unit, API, feature store, and integration workflows.
- **Clean Code Standards**: Type hints, docstrings, structured logging, and SOLID code design.
