"""
Inference Request & Response Schemas.
Defines input employee attributes and prediction output contracts.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class RawEmployeeInput(BaseModel):
    Age: int = Field(..., ge=18, le=80, example=35)
    BusinessTravel: str = Field(..., example="Travel_Rarely")
    DailyRate: int = Field(..., ge=100, le=2000, example=800)
    Department: str = Field(..., example="Research & Development")
    DistanceFromHome: int = Field(..., ge=1, le=100, example=10)
    Education: int = Field(..., ge=1, le=5, example=3)
    EducationField: str = Field(..., example="Life Sciences")
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4, example=3)
    Gender: str = Field(..., example="Male")
    HourlyRate: int = Field(..., ge=10, le=200, example=65)
    JobInvolvement: int = Field(..., ge=1, le=4, example=3)
    JobLevel: int = Field(..., ge=1, le=5, example=2)
    JobRole: str = Field(..., example="Research Scientist")
    JobSatisfaction: int = Field(..., ge=1, le=4, example=4)
    MaritalStatus: str = Field(..., example="Single")
    MonthlyIncome: int = Field(..., ge=1000, le=50000, example=5000)
    MonthlyRate: int = Field(..., ge=1000, le=50000, example=15000)
    NumCompaniesWorked: int = Field(..., ge=0, le=20, example=2)
    OverTime: str = Field(..., example="Yes")
    PercentSalaryHike: int = Field(..., ge=0, le=50, example=15)
    PerformanceRating: int = Field(..., ge=1, le=4, example=3)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4, example=3)
    StockOptionLevel: int = Field(..., ge=0, le=3, example=1)
    TotalWorkingYears: int = Field(..., ge=0, le=50, example=10)
    TrainingTimesLastYear: int = Field(..., ge=0, le=10, example=2)
    WorkLifeBalance: int = Field(..., ge=1, le=4, example=3)
    YearsAtCompany: int = Field(..., ge=0, le=40, example=5)
    YearsInCurrentRole: int = Field(..., ge=0, le=30, example=3)
    YearsSinceLastPromotion: int = Field(..., ge=0, le=30, example=1)
    YearsWithCurrManager: int = Field(..., ge=0, le=30, example=2)


class PredictionResult(BaseModel):
    prediction_id: str
    attrition_probability: float
    attrition_prediction: int
    selected_threshold: float
    risk_recommendation: str
    model_version: str
    feature_version: str
    latency_ms: float
    engineered_features: Dict[str, Any]


class BatchPredictionSummary(BaseModel):
    batch_id: str
    total_records: int
    high_risk_count: int
    low_risk_count: int
    review_rate: float
    average_probability: float
    average_latency_ms: float
    model_version: str
    threshold_used: float
