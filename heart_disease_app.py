import streamlit as st
import pandas as pd


# Load trained model

import joblib

model = joblib.load('model.pkl')
 
 

# Page configuration
st.set_page_config(page_title="Heart Disease Predictor 💓", page_icon="💓", layout="centered")

# App title
st.markdown("""
    <h1 style="color: #DC143C; text-align: center;">💓 Heart Disease Risk Assessment</h1>
    <h4 style="text-align: center; color: gray;">
        A real-world machine learning app powered by a Random Forest classifier.
    </h4> 
    <br>
""", unsafe_allow_html=True)

st.markdown("### 👤 Patient Information")

# Form inputs
with st.form(key='heart_form'):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 100, 40)
        bp = st.number_input("Blood Pressure (mmHg)", min_value=50, max_value=200, value=120)
        chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
        hr = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=80)
        blood_sugar = st.selectbox("Blood Sugar Level", ['Normal', 'High'])

    with col2:
        gender = st.selectbox("Gender", ['Male', 'Female'])
        cp_type = st.selectbox("Chest Pain Type", ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
        exercise_hours = st.slider("Weekly Exercise Hours", 0, 20, 3)
        stress = st.slider("Stress Level (1-10)", 1, 10, 5)
        smoking = st.selectbox("Smoking Status", ['Never', 'Former', 'Current'])

    st.markdown("### 🩺 Medical History")
    col3, col4, col5 = st.columns(3)
    with col3:
        alcohol = st.selectbox("Alcohol Intake", ['None', 'Moderate', 'High'])
    with col4:
        family_history = st.selectbox("Family History of Heart Disease", ['No', 'Yes'])
    with col5:
        diabetes = st.selectbox("Diabetes", ['No', 'Yes'])

    col6, col7 = st.columns(2)
    with col6:
        obesity = st.selectbox("Obesity", ['No', 'Yes'])
    with col7:
        angina = st.selectbox("Exercise Induced Angina", ['No', 'Yes'])

    submitted = st.form_submit_button("🔍 Predict")

# If submitted
if submitted:
    # Prepare input data
    data = {
        'Age': age,
        'Cholesterol': chol,
        'Blood Pressure': bp,
        'Heart Rate': hr,
        'Exercise Hours': exercise_hours,
        'Stress Level': stress,
        'Blood Sugar': 1 if blood_sugar == 'High' else 0,

        'Gender_Male': 1 if gender == 'Male' else 0,
        'Smoking_Former': 1 if smoking == 'Former' else 0,
        'Smoking_Never': 1 if smoking == 'Never' else 0,
        'Alcohol Intake_Moderate': 1 if alcohol == 'Moderate' else 0,
        'Family History_Yes': 1 if family_history == 'Yes' else 0,
        'Diabetes_Yes': 1 if diabetes == 'Yes' else 0,
        'Obesity_Yes': 1 if obesity == 'Yes' else 0,
        'Exercise Induced Angina_Yes': 1 if angina == 'Yes' else 0,

        'Chest Pain Type_Atypical Angina': 1 if cp_type == 'Atypical Angina' else 0,
        'Chest Pain Type_Non-anginal Pain': 1 if cp_type == 'Non-anginal Pain' else 0,
        'Chest Pain Type_Typical Angina': 1 if cp_type == 'Typical Angina' else 0,
    }

    input_df = pd.DataFrame([data])

    # Prediction
    prediction = model.predict(input_df)[0]
    risk = "High Risk 💔" if prediction == 1 else "Low Risk 💖"

    st.markdown("---")
    st.success(f"### 🧠 Prediction: **{risk}**")
