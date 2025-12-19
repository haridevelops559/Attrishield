import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="AttriShield", layout="wide")

st.title("AttriShield — Employee Attrition Prediction")

model = joblib.load("attrishield_model.pkl")

st.sidebar.header("Employee Features")

def user_input():
    age = st.sidebar.slider("Age", 18, 60, 30)
    income = st.sidebar.slider("Monthly Income", 1000, 20000, 5000)
    overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
    overtime = 1 if overtime == "Yes" else 0
    return np.array([[age, income, overtime]])

input_data = user_input()

if st.button("Predict Attrition"):
    prob = model.predict_proba(input_data)[0][1]
    st.metric("Attrition Probability", f"{prob*100:.2f}%")

    if prob > 0.5:
        st.warning("High attrition risk detected")
    else:
        st.success("Low attrition risk")
