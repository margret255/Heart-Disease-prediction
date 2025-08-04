import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

# Set page config
st.set_page_config(page_title="💓 Heart Disease Predictor", layout="centered")

# Header
st.markdown("<h1 style='text-align: center; color: crimson;'>💓 Heart Disease Risk Assessment</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>A real-world machine learning app powered by a Random Forest classifier.</p>", unsafe_allow_html=True)
st.markdown("---")

# Collect user input
st.markdown("### 👤 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

# Preprocess user input
sex = 1 if sex == "Male" else 0
cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
restecg_map = {"Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2}
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "cp": [cp_map[cp]],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg_map[restecg]],
    "thalach": [thalach],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope_map[slope]]
})

# Prediction button
st.markdown("---")
if st.button("🩺 Predict Heart Disease Risk"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({prob:.2f}%)")
        st.markdown("💡 *Please consult a cardiologist as soon as possible.*")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({prob:.2f}%)")
        st.markdown("🎉 *You're likely heart-healthy. Keep up with regular checkups!*")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 13px;'>Made with ❤️ for health awareness</p>", unsafe_allow_html=True)
