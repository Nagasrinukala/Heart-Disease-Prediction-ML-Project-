import streamlit as st
import joblib
import numpy as np
import pandas as pd

# IMPORTANT: custom function MUST exist before loading model
def safe_log_transform(x):
    return np.log1p(np.clip(x, a_min=0, a_max=None))

# PAGE CONFIG
st.set_page_config(
    page_title="Heart Failure Prediction",
    layout="centered"
)

st.title("Heart Failure Prediction App")
st.write("Enter patient details to predict risk of death event.")

# LOAD MODEL (joblib only)
@st.cache_resource
def load_model():
    return joblib.load("dt_pipeline.pkl")

model = load_model()

# USER INPUTS
st.subheader("Patient Information")

age = st.number_input("Age", 1, 120, 60)
anaemia = st.selectbox("Anaemia", [0, 1])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", 0)
diabetes = st.selectbox("Diabetes", [0, 1])
ejection_fraction = st.number_input("Ejection Fraction", 1, 100, 35)
high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1])
platelets = st.number_input("Platelets", 0)
serum_creatinine = st.number_input("Serum Creatinine", 0.0, format="%.2f")
serum_sodium = st.number_input("Serum Sodium", 0)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
smoking = st.selectbox("Smoking", [0, 1])
time = st.number_input("Follow-up Period (days)", 1)

# INPUT DATAFRAME
input_data = pd.DataFrame([{
    "age": age,
    "anaemia": anaemia,
    "creatinine_phosphokinase": creatinine_phosphokinase,
    "diabetes": diabetes,
    "ejection_fraction": ejection_fraction,
    "high_blood_pressure": high_blood_pressure,
    "platelets": platelets,
    "serum_creatinine": serum_creatinine,
    "serum_sodium": serum_sodium,
    "sex": sex,
    "smoking": smoking,
    "time": time
}])

# PREDICTION
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("High Risk of Death Event")
    else:
        st.success("Low Risk of Death Event")

    st.write(f"**Probability of Death Event:** {probability:.2f}")
