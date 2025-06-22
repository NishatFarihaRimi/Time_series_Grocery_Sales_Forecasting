"""
command: 
>> mlflow ui                                  # Initailize server from another terminal
>> python xgboost_mlflow.py                   # From another terminal run mlflow python file
http://127.0.0.1:5000/                        # Check the experimant from browser 
"""


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import mlflow
import mlflow.sklearn
import os
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# --- Paths ---
DATA_PATH = "Data/train_Guayas_final.csv"# adjust to your actual path
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load Data ---
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df['store_nbr'] = df['store_nbr'].astype('category')
df['item_nbr'] = df['item_nbr'].astype('category')
df['unit_sales'] = pd.to_numeric(df['unit_sales'], errors='coerce')

# --- Filter a store/item pair or use all data ---
df = df.dropna(subset=['unit_sales'])  # ensure target is clean

# --- Feature Selection ---
excluded = ['date', 'unit_sales']
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
features = [col for col in numeric_cols if col not in excluded]

X = df[features]
y = df['unit_sales']

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Scale Data ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train Model ---
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

# --- Predict & Evaluate ---
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# --- MLflow Logging ---
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("features", features)

    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)


    # Save model and scaler as artifacts
    model_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    #mlflow.log_artifact(model_path, artifact_path="model")
   # mlflow.log_artifact(scaler_path, artifact_path="scaler")

    # Log model using MLflow's native XGBoost flavor
    mlflow.sklearn.log_model(model, "xgboost_model")

print(f"Model trained and logged to MLflow. MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")