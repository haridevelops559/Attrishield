import shap
import joblib
import pandas as pd

model = joblib.load("attrishield_model.pkl")
df = pd.read_csv("data/IBM_HR_Attrition.csv")

X = df.drop("Attrition", axis=1)

explainer = shap.Explainer(model, X)
shap_values = explainer(X)

shap.summary_plot(shap_values, X)
