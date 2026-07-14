from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import pickle
import io

# ================== App ==================
app = FastAPI(title="Employee Attrition Prediction API")

# ================== Load Model ==================
with open("eapsnew2.pkl", "rb") as f:
    model = pickle.load(f)

MODEL_FEATURES = model.get_booster().feature_names

# ================== Input Schema ==================
class EmployeeFeatures(BaseModel):

    # Numeric
    Age: int
    DailyRate: int
    DistanceFromHome: int
    EnvironmentSatisfaction: int
    JobInvolvement: int
    JobLevel: int
    JobSatisfaction: int
    RelationshipSatisfaction: int
    WorkLifeBalance: int
    MonthlyIncome: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int
    OverTime: int

    # One-hot
    BusinessTravel_Travel_Frequently: int
    BusinessTravel_Travel_Rarely: int
    BusinessTravel_Non_Travel: int

    Department_Research_and_Development: int
    Department_Sales: int
    Department_Human_Resources: int

    EducationField_Life_Sciences: int
    EducationField_Medical: int
    EducationField_Marketing: int
    EducationField_Technical_Degree: int
    EducationField_Other: int

    JobRole_Human_Resources: int
    JobRole_Laboratory_Technician: int
    JobRole_Manager: int
    JobRole_Manufacturing_Director: int
    JobRole_Research_Director: int
    JobRole_Research_Scientist: int
    JobRole_Sales_Executive: int
    JobRole_Sales_Representative: int


# ================== Recommendation Logic ==================
def generate_recommendations(features, probability):
    recs = []

    if features.OverTime == 1:
        recs.append("Reduce overtime workload or introduce flexible schedules")

    if features.WorkLifeBalance <= 2:
        recs.append("Improve work-life balance policies")

    if features.JobSatisfaction <= 2:
        recs.append("Increase employee engagement and satisfaction initiatives")

    if features.EnvironmentSatisfaction <= 2:
        recs.append("Enhance workplace environment and culture")

    if features.MonthlyIncome < 4000:
        recs.append("Review compensation and benefits")

    if features.YearsSinceLastPromotion > 3:
        recs.append("Provide career growth and promotion opportunities")

    if not recs:
        recs.append("Maintain current engagement and retention strategies")

    return recs[:3]


# ================== Single Prediction ==================
@app.post("/predict")
def predict_attrition(features: EmployeeFeatures):

    df = pd.DataFrame([features.dict()])

    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    df = df[MODEL_FEATURES]

    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    recommendations = generate_recommendations(features, probability)

    return {
        "prediction": "ATTRITION" if prediction == 1 else "RETENTION",
        "attrition_probability": round(probability, 4),
        "risk_level": (
            "HIGH" if probability > 0.7 else
            "MEDIUM" if probability > 0.4 else
            "LOW"
        ),
        "recommendations": recommendations
    }


# ================== Batch Prediction ==================
latest_df = None  # 🔥 add at top of file


@app.post("/batch_predict")
def batch_predict(file: UploadFile = File(...)):

    global latest_df

    try:
        contents = file.file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # 🔹 Clean column names
        df.columns = df.columns.str.strip()

        # 🔹 KEEP ORIGINAL DATA
        original_df = df.copy()

        # 🔹 Create model input separately
        model_df = df.copy()

        for col in MODEL_FEATURES:
            if col not in model_df.columns:
                model_df[col] = 0

        model_df = model_df[MODEL_FEATURES]

        # 🔹 Convert to numeric
        model_df = model_df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # 🔹 Predictions
        predictions = model.predict(model_df)
        probabilities = model.predict_proba(model_df)[:, 1]

        # 🔹 Attach results to ORIGINAL DATA
        original_df["prediction"] = ["ATTRITION" if p == 1 else "RETENTION" for p in predictions]
        original_df["attrition_probability"] = probabilities.round(4)

        # 🔥 STORE FULL DATA FOR FILTERING
        latest_df = original_df.copy()

        return {
            "total_records": len(original_df),
            "attrition_count": int((predictions == 1).sum()),
            "retention_count": int((predictions == 0).sum()),
            "results": original_df.to_dict(orient="records")  # 🔥 FULL 40 RECORDS
        }

    except Exception as e:
        return {"error": str(e)}


# ================== Insights / Filtering ==================



from typing import Optional
from fastapi import Query
import pandas as pd





@app.post("/insights")
def get_insights(

    # 🔹 RANGE FILTERS (NUMERIC)
    Age_min: Optional[int] = Query(None), Age_max: Optional[int] = Query(None),
    DailyRate_min: Optional[int] = Query(None), DailyRate_max: Optional[int] = Query(None),
    DistanceFromHome_min: Optional[int] = Query(None), DistanceFromHome_max: Optional[int] = Query(None),

    EnvironmentSatisfaction_min: Optional[int] = Query(None), EnvironmentSatisfaction_max: Optional[int] = Query(None),
    JobInvolvement_min: Optional[int] = Query(None), JobInvolvement_max: Optional[int] = Query(None),
    JobLevel_min: Optional[int] = Query(None), JobLevel_max: Optional[int] = Query(None),
    JobSatisfaction_min: Optional[int] = Query(None), JobSatisfaction_max: Optional[int] = Query(None),
    RelationshipSatisfaction_min: Optional[int] = Query(None), RelationshipSatisfaction_max: Optional[int] = Query(None),
    WorkLifeBalance_min: Optional[int] = Query(None), WorkLifeBalance_max: Optional[int] = Query(None),

    MonthlyIncome_min: Optional[int] = Query(None), MonthlyIncome_max: Optional[int] = Query(None),
    StockOptionLevel_min: Optional[int] = Query(None), StockOptionLevel_max: Optional[int] = Query(None),

    TotalWorkingYears_min: Optional[int] = Query(None), TotalWorkingYears_max: Optional[int] = Query(None),
    TrainingTimesLastYear_min: Optional[int] = Query(None), TrainingTimesLastYear_max: Optional[int] = Query(None),

    YearsAtCompany_min: Optional[int] = Query(None), YearsAtCompany_max: Optional[int] = Query(None),
    YearsInCurrentRole_min: Optional[int] = Query(None), YearsInCurrentRole_max: Optional[int] = Query(None),
    YearsSinceLastPromotion_min: Optional[int] = Query(None), YearsSinceLastPromotion_max: Optional[int] = Query(None),
    YearsWithCurrManager_min: Optional[int] = Query(None), YearsWithCurrManager_max: Optional[int] = Query(None),

    # 🔹 EXACT FILTERS (BINARY / ONE-HOT)
    OverTime: Optional[int] = Query(None),

    BusinessTravel_Travel_Frequently: Optional[int] = Query(None),
    BusinessTravel_Travel_Rarely: Optional[int] = Query(None),
    BusinessTravel_Non_Travel: Optional[int] = Query(None),

    Department_Research_and_Development: Optional[int] = Query(None),
    Department_Sales: Optional[int] = Query(None),
    Department_Human_Resources: Optional[int] = Query(None),

    EducationField_Life_Sciences: Optional[int] = Query(None),
    EducationField_Medical: Optional[int] = Query(None),
    EducationField_Marketing: Optional[int] = Query(None),
    EducationField_Technical_Degree: Optional[int] = Query(None),
    EducationField_Other: Optional[int] = Query(None),

    JobRole_Human_Resources: Optional[int] = Query(None),
    JobRole_Laboratory_Technician: Optional[int] = Query(None),
    JobRole_Manager: Optional[int] = Query(None),
    JobRole_Manufacturing_Director: Optional[int] = Query(None),
    JobRole_Research_Director: Optional[int] = Query(None),
    JobRole_Research_Scientist: Optional[int] = Query(None),
    JobRole_Sales_Executive: Optional[int] = Query(None),
    JobRole_Sales_Representative: Optional[int] = Query(None),
):

    global latest_df

    try:
        if latest_df is None:
            return {"error": "Run /batch_predict first"}

        df = latest_df.copy()

        # 🔥 MASK INITIALIZATION
        mask = pd.Series(True, index=df.index)

        # 🔹 RANGE FILTER HELPER
        def apply_range(col, min_val, max_val):
            nonlocal mask
            if col in df.columns:
                if min_val is not None:
                    mask &= df[col] >= min_val
                if max_val is not None:
                    mask &= df[col] <= max_val

        # 🔹 EXACT FILTER HELPER
        def apply_exact(col, val):
            nonlocal mask
            if val is not None and col in df.columns:
                mask &= df[col] == val

        # 🔥 APPLY RANGE FILTERS
        apply_range("Age", Age_min, Age_max)
        apply_range("DailyRate", DailyRate_min, DailyRate_max)
        apply_range("DistanceFromHome", DistanceFromHome_min, DistanceFromHome_max)
        apply_range("EnvironmentSatisfaction", EnvironmentSatisfaction_min, EnvironmentSatisfaction_max)
        apply_range("JobInvolvement", JobInvolvement_min, JobInvolvement_max)
        apply_range("JobLevel", JobLevel_min, JobLevel_max)
        apply_range("JobSatisfaction", JobSatisfaction_min, JobSatisfaction_max)
        apply_range("RelationshipSatisfaction", RelationshipSatisfaction_min, RelationshipSatisfaction_max)
        apply_range("WorkLifeBalance", WorkLifeBalance_min, WorkLifeBalance_max)
        apply_range("MonthlyIncome", MonthlyIncome_min, MonthlyIncome_max)
        apply_range("StockOptionLevel", StockOptionLevel_min, StockOptionLevel_max)
        apply_range("TotalWorkingYears", TotalWorkingYears_min, TotalWorkingYears_max)
        apply_range("TrainingTimesLastYear", TrainingTimesLastYear_min, TrainingTimesLastYear_max)
        apply_range("YearsAtCompany", YearsAtCompany_min, YearsAtCompany_max)
        apply_range("YearsInCurrentRole", YearsInCurrentRole_min, YearsInCurrentRole_max)
        apply_range("YearsSinceLastPromotion", YearsSinceLastPromotion_min, YearsSinceLastPromotion_max)
        apply_range("YearsWithCurrManager", YearsWithCurrManager_min, YearsWithCurrManager_max)

        # 🔥 APPLY EXACT FILTERS
        apply_exact("OverTime", OverTime)

        apply_exact("BusinessTravel_Travel_Frequently", BusinessTravel_Travel_Frequently)
        apply_exact("BusinessTravel_Travel_Rarely", BusinessTravel_Travel_Rarely)
        apply_exact("BusinessTravel_Non_Travel", BusinessTravel_Non_Travel)

        apply_exact("Department_Research_and_Development", Department_Research_and_Development)
        apply_exact("Department_Sales", Department_Sales)
        apply_exact("Department_Human_Resources", Department_Human_Resources)

        apply_exact("EducationField_Life_Sciences", EducationField_Life_Sciences)
        apply_exact("EducationField_Medical", EducationField_Medical)
        apply_exact("EducationField_Marketing", EducationField_Marketing)
        apply_exact("EducationField_Technical_Degree", EducationField_Technical_Degree)
        apply_exact("EducationField_Other", EducationField_Other)

        apply_exact("JobRole_Human_Resources", JobRole_Human_Resources)
        apply_exact("JobRole_Laboratory_Technician", JobRole_Laboratory_Technician)
        apply_exact("JobRole_Manager", JobRole_Manager)
        apply_exact("JobRole_Manufacturing_Director", JobRole_Manufacturing_Director)
        apply_exact("JobRole_Research_Director", JobRole_Research_Director)
        apply_exact("JobRole_Research_Scientist", JobRole_Research_Scientist)
        apply_exact("JobRole_Sales_Executive", JobRole_Sales_Executive)
        apply_exact("JobRole_Sales_Representative", JobRole_Sales_Representative)

        # 🔥 APPLY FILTER ONCE
        df_filtered = df[mask]

        if len(df_filtered) == 0:
            return {"message": "No data after filtering"}

        attrition_mask = df_filtered["prediction"] == "ATTRITION"

        return {
            "filtered_count": len(df_filtered),
            "attrition_count": int(attrition_mask.sum()),
            "retention_count": int((~attrition_mask).sum()),
            "attrition_rate": round(attrition_mask.mean(), 3),
            "avg_probability": round(df_filtered["attrition_probability"].mean(), 3),
            "records": df_filtered.to_dict(orient="records")
        }

    except Exception as e:
        print("FINAL ERROR:", e)
        return {"error": str(e)}



# ================== Health Check ==================
@app.get("/health")
def health():
    return {"status": "API running successfully"}


# ================== Root ==================
@app.get("/")
def root():
    return {"message": "Employee Attrition Prediction API is running"}