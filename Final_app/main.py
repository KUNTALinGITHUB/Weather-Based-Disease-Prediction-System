# Import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv(r"D:\Kuntal\project_idea\my_Projects\Weather_based_disease_outbreak_prediction_system\Dataset\Weather-related disease prediction.csv")

# Step 1: Understanding the Dataset
print("\nDataset Overview:\n")
print(df.head())

print("\nColumn Info:\n")
print(df.info())

print("\nMissing Values:\n")
missing_values = df.isnull().sum()
print(missing_values)

# Visualize missing data
plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False, cmap="coolwarm")
plt.title("Missing Data Heatmap")
plt.show()

# Summary statistics
print("\nSummary Statistics:\n")
print(df.describe())

# Visualize Age distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Correlation heatmap (excluding non-numeric columns like 'prognosis')
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(12, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 2: Data Preprocessing

# Convert Gender column to categorical
df["Gender"] = df["Gender"].astype("category")
df = pd.get_dummies(df, columns=["Gender"], drop_first=True)

# Normalize numerical data
num_cols = ["Age", "Temperature (C)", "Humidity", "Wind Speed (km/h)"]
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Step 3: Feature Selection & Engineering

# Identify highly correlated features
corr_matrix = numeric_df.corr()
high_corr_features = [column for column in corr_matrix if any(corr_matrix[column] > 0.85) and column != "prognosis"]
print("\nHighly correlated features:\n", high_corr_features)

# Creating a weather severity feature (example)
df["Weather_Severity"] = df["Temperature (C)"] * df["Humidity"] * df["Wind Speed (km/h)"]

# Display processed dataset info after Step 3
print("\nFinal Processed Data Overview:\n")
print(df.head())
print(df.info())

# Step 4: Model Selection & Training

# Define features & target
X = df.drop(columns=["prognosis"])
y = df["prognosis"]

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nData split completed! Training samples:", X_train.shape[0], "Testing samples:", X_test.shape[0])

# Print expected feature names and number of features before training
print("\nExpected Feature Names in Training Data:\n", X_train.columns.tolist())
print("Number of features expected:", X_train.shape[1])

# Train a baseline Random Forest model
baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_model.fit(X_train, y_train)

# Make predictions
y_pred = baseline_model.predict(X_test)

# Evaluate baseline performance
baseline_accuracy = accuracy_score(y_test, y_pred)
print("\nBaseline Model Accuracy:", baseline_accuracy)
print("\nBaseline Classification Report:\n", classification_report(y_test, y_pred))

# Step 5: Hyperparameter Tuning for Random Forest

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("\nBest Random Forest Parameters:", best_params)

best_model = RandomForestClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)

y_pred_tuned = best_model.predict(X_test)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
print("\nTuned Random Forest Model Accuracy:", tuned_accuracy)
print("\nTuned Classification Report:\n", classification_report(y_test, y_pred_tuned))

# Step 6: Implementing XGBoost

xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("\nXGBoost Model Accuracy:", accuracy_xgb)
print("\nXGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))

# Hyperparameter Tuning for XGBoost

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

grid_search_xgb = GridSearchCV(XGBClassifier(random_state=42), xgb_param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search_xgb.fit(X_train, y_train)

best_xgb_params = grid_search_xgb.best_params_
print("\nBest XGBoost Parameters:", best_xgb_params)

best_xgb_model = XGBClassifier(**best_xgb_params, random_state=42)
best_xgb_model.fit(X_train, y_train)

y_pred_xgb_tuned = best_xgb_model.predict(X_test)
accuracy_xgb_tuned = accuracy_score(y_test, y_pred_xgb_tuned)

print("\nTuned XGBoost Model Accuracy:", accuracy_xgb_tuned)
print("\nTuned XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb_tuned))

# Step 7: Save the models

joblib.dump(best_model, "random_forest_model.pkl")
joblib.dump(best_xgb_model, "xgboost_model.pkl")

print("\nModels saved successfully!")