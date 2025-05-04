

https://github.com/user-attachments/assets/bd11ed5a-b2ee-420a-b2f4-cbd65ea7ec8b

# Weather-Based-Disease-Prediction-System
 Powered by AI &amp; Machine Learning | Made by Kuntal





**Weather-Based Disease Prediction System**  
📌 _Predicts potential diseases based on weather conditions and symptoms using machine learning._

## 🔍 Overview
This system leverages weather data and symptoms to predict diseases using **advanced machine learning models**. The primary model, **XGBoost**, achieves an impressive **93.8% accuracy**, significantly outperforming the baseline **Random Forest model**.

## 🚀 Features
- **High-Accuracy Prediction** using XGBoost (93.8%).
- Streamlit-based UI for easy interaction.
- Hyperparameter tuning for optimized performance.
- Disease predictions based on weather and symptom data.

## 🏆 Why XGBoost?
XGBoost is the preferred model due to its **higher accuracy, better generalization, and faster computation speed** compared to Random Forest. It leverages:
- **Gradient boosting** for improved learning.
- **Regularization** to prevent overfitting.
- **Optimized performance** on structured datasets.

## 🛠️ Installation & Usage

### 1️⃣ Install dependencies
```sh
pip install pandas seaborn matplotlib xgboost scikit-learn streamlit joblib numpy
```

### 2️⃣ Run the web application
```sh
streamlit run app.py
```
_Ensure `xgboost_model.pkl` is available in the project directory._

## 🎯 How It Works
1. Users provide **age, weather conditions, and symptoms** through the Streamlit UI.
2. The system applies the **XGBoost model** to predict diseases.
3. Predicted diseases include:
   - Heart Attack
   - Influenza
   - Dengue
   - Sinusitis
   - Asthma
   - Diabetes
   - Hypertension
   - Pneumonia
   - COVID-19
   - Common Cold
   - Malaria

## 📌 Use Cases
- **Healthcare:** Early detection & risk assessment based on weather trends.
- **Epidemiology:** Weather-driven disease outbreak prediction.
- **Remote Diagnosis:** Helpful for individuals without immediate medical access.

---


