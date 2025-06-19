import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb

# --- Load Data ---
data_path = "../Data/train_Guayas_final.csv"
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    st.error(f"Could not find file at {data_path}. Make sure the path is correct.")
    st.stop()

df['date'] = pd.to_datetime(df['date'])
df['store_nbr'] = df['store_nbr'].astype('category')
df['item_nbr'] = df['item_nbr'].astype('category')

# Show data overview
st.subheader("Dataset Overview")
st.write(df.head())
st.write(f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
st.write(f"Shape: {df.shape}")

# --- Correlation Plot ---
with st.expander("📊 Show Feature Correlation with Unit Sales"):
    excluded_columns = ['date', 'store_nbr', 'item_nbr', 'unit_sales']
    correlation_columns = [col for col in df.columns if col not in excluded_columns]
    correlation = df[correlation_columns + ['unit_sales']].corr()['unit_sales'].drop('unit_sales')

    fig, ax = plt.subplots(figsize=(10, 6))
    correlation.sort_values(ascending=False).plot.barh(ax=ax, color='teal')
    ax.set_title("Feature Correlation with Unit Sales")
    ax.set_xlabel("Correlation")
    ax.grid(True)
    st.pyplot(fig)

# --- Train-Test Split ---
split_date = pd.to_datetime("2014-01-01")
train = df[df['date'] < split_date]
test = df[df['date'] >= split_date]

# Select highly correlated features
high_corr_columns = correlation[correlation.abs() > 0.1].index.tolist()
model_columns = high_corr_columns + ['date', 'store_nbr', 'item_nbr', 'unit_sales']

train_filtered = train[model_columns]
test_filtered = test[model_columns]

# Feature-target split
X_train = train_filtered.drop(['date', 'unit_sales'], axis=1)
X_test = test_filtered.drop(['date', 'unit_sales'], axis=1)
y_train = train_filtered['unit_sales']
y_test = test_filtered['unit_sales']

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model Training ---
st.subheader("⚙️ Train XGBoost Model")
if st.button("Train Model"):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        enable_categorical=True
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # --- Evaluation Metrics ---
    st.subheader("📏 Evaluation Metrics")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAPE:** {mape:.2f}")
    st.write(f"**R² Score:** {r2:.4f}")

    # --- Prediction Plot ---
    st.subheader("📉 Actual vs Predicted Sales")
    i = 90  # number of days to plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_test.index[:i], y_test.values[:i], label='Actual Sales')
    ax.plot(y_test.index[:i], y_pred[:i], label='Predicted Sales')
    ax.set_xlabel('Index')
    ax.set_ylabel('Unit Sales')
    ax.set_title('Actual vs Predicted Sales using XGBoost')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
