import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.model_utils import load_model
from app.config import DATA_PATH, MODEL_PATH

# --- Setup ---
st.set_page_config(layout="wide")
st.title("🧠 XGBoost Sales Forecasting App")

# --- Load Data ---
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(f"Could not find file at {DATA_PATH}. Make sure the path is correct.")
    st.stop()

df['date'] = pd.to_datetime(df['date'])
df['store_nbr'] = df['store_nbr'].astype('category')
df['item_nbr'] = df['item_nbr'].astype('category')

# Ensure 'unit_sales' is numeric
df['unit_sales'] = pd.to_numeric(df['unit_sales'], errors='coerce')

# Step 1: Calculate total unit sales per store and item
sales_summary = df.groupby(['store_nbr', 'item_nbr'])['unit_sales'].sum().reset_index()

# Step 2: Sort by store and descending unit_sales
sales_summary_sorted = sales_summary.sort_values(['store_nbr', 'unit_sales'], ascending=[True, False])

# Step 3: Get top 10 items per store
top_10_per_store = sales_summary_sorted.groupby('store_nbr').head(10)

# Step 4: For each store, get the list of item_nbrs
top_items_dict = top_10_per_store.groupby('store_nbr')['item_nbr'].apply(list).to_dict()

# Now top_items_dict looks like: {store_1: [item_a, item_b, ...], store_2: [...], ...}



# #--- UI Controls ---
# st.sidebar.header("🔧 Select Store and Item")
# store_ids = df['store_nbr'].cat.categories.tolist()
# selected_store = st.sidebar.selectbox("Store Number", store_ids)

# item_ids = df['item_nbr'].cat.categories.tolist()
# selected_item = st.sidebar.selectbox("Item Number", item_ids)

selected_store = st.sidebar.selectbox("Select Store", list(top_items_dict.keys()))
selected_item = st.sidebar.selectbox("Select Item", top_items_dict[selected_store])

submit = st.sidebar.button("🔍 Predict")

if submit:
    # --- Filter Data ---
    split_date = pd.to_datetime("2014-01-01")
    df_filtered = df[(df['store_nbr'] == selected_store) & (df['item_nbr'] == selected_item)].copy()

    if df_filtered.empty:
        st.warning("No data available for this store and item combination.")
        st.stop()

    train = df_filtered[df_filtered['date'] < split_date]
    test = df_filtered[df_filtered['date'] >= split_date]

    # --- Features ---
    excluded_columns = ['date', 'unit_sales']
    correlation_columns = [col for col in df.columns if col not in excluded_columns + ['store_nbr', 'item_nbr']]
    correlation = df[correlation_columns + ['unit_sales']].corr()['unit_sales'].drop('unit_sales')
    high_corr_columns = correlation[correlation.abs() > 0.1].index.tolist()
    #features = high_corr_columns + ['store_nbr']
    features= ['transactions', 'lag_1', 'lag_7', 'lag_30', 'lag_14', 'lag_21',
       'lag_60', 'rolling_mean_7', 'rolling_std_7', 'rolling_mean_14',
       'rolling_std_14', 'rolling_mean_30', 'rolling_std_30', 'ewm_mean_7',
       'ewm_std_7', 'ewm_mean_14', 'ewm_std_14', 'ewm_mean_30', 'ewm_std_30',
       'unit_sales_7d_avg', 'unit_sales_30d_avg', 'store_nbr', 'item_nbr']
    #print(features)
    # --- Scaling ---
    scaler = StandardScaler()
    X_train = train[features]
    y_train = train['unit_sales']
    X_test = test[features]
    y_test = test['unit_sales']
    scaler.fit(X_train)

    # --- Load Model ---
    model = load_model(MODEL_PATH)

    # --- Prediction ---
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    # --- Evaluation ---
    st.subheader(f"📊 Model Evaluation for Store {selected_store}, Item {selected_item}")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAPE:** {mape:.2f}")
    st.write(f"**R² Score:** {r2:.4f}")

    # --- Plot ---
    st.subheader("📉 Actual vs Predicted Sales")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test['date'], y_test.values, label='Actual')
    ax.plot(test['date'], y_pred, label='Predicted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Unit Sales')
    ax.set_title(f'Store {selected_store}, Item {selected_item} — Actual vs Predicted')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
