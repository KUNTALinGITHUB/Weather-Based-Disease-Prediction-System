import streamlit as st
import joblib
import numpy as np
import os

# Load trained model dynamically from the current directory
model_path = r"D:\Kuntal\project_idea\my_Projects\Weather_based_disease_outbreak_prediction_system\Final_app\xgboost_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("🚨 Model file not found! Please ensure 'xgboost_model.pkl' is in the same directory.")

# Define feature names as per your training model
feature_names = [
    "Age", "Temperature (C)", "Humidity", "Wind Speed (km/h)", "nausea", "joint_pain", "abdominal_pain",
    "high_fever", "chills", "fatigue", "runny_nose", "pain_behind_the_eyes", "dizziness", "headache",
    "chest_pain", "vomiting", "cough", "shivering", "asthma_history", "high_cholesterol", "diabetes",
    "obesity", "hiv_aids", "nasal_polyps", "asthma", "high_blood_pressure", "severe_headache", "weakness",
    "trouble_seeing", "fever", "body_aches", "sore_throat", "sneezing", "diarrhea", "rapid_breathing",
    "rapid_heart_rate", "pain_behind_eyes", "swollen_glands", "rashes", "sinus_headache", "facial_pain",
    "shortness_of_breath", "reduced_smell_and_taste", "skin_irritation", "itchiness", "throbbing_headache",
    "confusion", "back_pain", "knee_ache", "Gender_1", "Weather_Severity"
]

# Define disease labels (adjust based on your dataset)
disease_labels = {
    0: "Heart Attack",
    1: "Influenza",
    2: "Dengue",
    3: "Sinusitis",
    4: "Asthma",
    5: "Diabetes",
    6: "Hypertension",
    7: "Pneumonia",
    8: "COVID-19",
    9: "Common Cold",
    10: "Malaria"
}

st.title("🌦️ Weather-Based Disease Prediction System")

# Sidebar input for weather conditions & symptoms
st.sidebar.header("Enter Weather Data & Symptoms")

# Collect weather-related inputs
age = st.sidebar.slider("Age", 1, 100, 50)
temperature = st.sidebar.slider("Temperature (°C)", -15, 45, 20)
humidity = st.sidebar.slider("Humidity (%)", 30, 100, 70)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 50, 10)

# Collect symptom inputs (binary values: 1 if checked, 0 if unchecked)
symptoms = {feature: st.sidebar.checkbox(feature.replace('_', ' ').title()) for feature in feature_names[4:-2]}  

# Gender input (assuming Gender_1 is 1 for male, 0 for female)
gender = st.sidebar.radio("Gender", ["Male", "Female"])
gender_value = 1 if gender == "Male" else 0

# Calculate Weather Severity
weather_severity = temperature * humidity * wind_speed

# Create input array
features = np.array([
    [
        age, temperature, humidity, wind_speed, *symptoms.values(), gender_value, weather_severity
    ]
])

# Prediction button
if st.sidebar.button("Predict"):
    prediction = model.predict(features)[0]  # Get the first prediction
    disease_name = disease_labels.get(prediction, "Unknown Disease")  # Map to disease name
    
    # Display in sidebar
    st.sidebar.success(f"🔍 Predicted Disease: **{disease_name}**")

    # Display in footer with red color
    st.markdown(f"### <span style='color:red'>🔍 Predicted Disease: **{disease_name}**</span>", unsafe_allow_html=True)

def reset_selections():
    for key in feature_names[4:-2]:  # Reset symptom checkboxes only
        if key in st.session_state:
            st.session_state[key] = False  # Ensure checkboxes are unchecked
    st.session_state["age"] = 50
    st.session_state["temperature"] = 20
    st.session_state["humidity"] = 70
    st.session_state["wind_speed"] = 10
    st.session_state["gender"] = "Male"  # Reset gender selection
    st.rerun()  # Correct method to refresh UI

# Reset button
if st.sidebar.button("Reset"):
    reset_selections()

# Footer
st.markdown("---")
st.markdown("🔬 **Powered by AI & Machine Learning** | Made by Kuntal ")
st.markdown("📝 **Disclaimer:** This tool is for informational purposes only and should not be used as a substitute for professional medical advice. Always consult a healthcare provider for medical concerns.")
st.markdown("---")
# Add the line specifying predicted diseases
st.markdown("⚠️ **Note:** This system can only predict the following diseases:")
st.markdown("**Heart Attack, Influenza, Dengue, Sinusitis, Asthma, Diabetes, Hypertension, Pneumonia, COVID-19, Common Cold, Malaria**")
