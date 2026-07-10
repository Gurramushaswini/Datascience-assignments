import streamlit as st
import pickle
import numpy as np
import os

st.title("🩺 Diabetes Prediction App")

# Current Folder
st.write("Current Working Directory:")
st.code(os.getcwd())

# Check files
st.write("Model Exists:", os.path.exists("diabetes_model.pkl"))
st.write("Scaler Exists:", os.path.exists("scaler.pkl"))

# Load model & scaler
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Inputs
pregnancies = st.number_input("Pregnancies", value=1)
glucose = st.number_input("Glucose", value=90)
bloodpressure = st.number_input("Blood Pressure", value=72)
skinthickness = st.number_input("Skin Thickness", value=25)
insulin = st.number_input("Insulin", value=80)
bmi = st.number_input("BMI", value=23.5)
dpf = st.number_input("Diabetes Pedigree Function", value=0.25)
age = st.number_input("Age", value=28)

if st.button("Predict"):

    features = np.array([[pregnancies,
                          glucose,
                          bloodpressure,
                          skinthickness,
                          insulin,
                          bmi,
                          dpf,
                          age]])

    st.write("Original Features")
    st.write(features)

    features_scaled = scaler.transform(features)

    st.write("Scaled Features")
    st.write(features_scaled)

    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)

    st.write("Prediction =", prediction)
    st.write("Probability =", probability)

    if prediction[0] == 1:
        st.error("⚠️ Person is Diabetic")
    else:
        st.success("✅ Person is Not Diabetic")