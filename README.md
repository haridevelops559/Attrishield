# AttriShield — Employee Attrition Prediction System

AttriShield is an end-to-end machine learning system designed to predict employee attrition
and provide actionable, explainable insights for HR decision-making.

## Key Features
- End-to-end ML pipeline on IBM HR Analytics dataset
- Multiple model benchmarking (RF, DT, SVM, LightGBM, XGBoost)
- Class imbalance handling using SMOTE
- Explainable AI using SHAP
- Interactive Streamlit dashboard
- LLM-powered retention insights using Gemini API

## Dataset
- IBM HR Analytics Dataset
- 1,470 employee records
- 35+ features (demographics, role, compensation, work-life balance)

## Model Performance
- Baseline accuracy: ~86%
- Optimized XGBoost accuracy: 92.5%
- Balanced precision-recall after threshold tuning

## How to Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
