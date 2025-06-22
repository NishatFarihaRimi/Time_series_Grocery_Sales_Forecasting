"""
command
>> source .venv/bin/activate                   # Activate the virtual environment
>> deactivate                                  # Deactivate the virtual environment
>> python3 -m venv .venv                       # Create a virtual environment
>> python3 -m pip install -r requirements.txt  # Install dependencies from requirements.txt
>> python3 app.py                               # Run the app locally
>> streamlit run app.py                         # Run the app with Streamlit
>> python3 -m pip install streamlit             # Install Streamlit
>> python3 -m pip freeze > requirements.txt     # Save dependencies to requirements.txt
"""

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
from model.model_utils import load_scaler
from app.config import DATA_PATH, MODEL_PATH, SCALER_PATH 
#st.write("Looking for file at:", os.path.abspath(DATA_PATH))

# --- Setup ---
st.set_page_config(layout="wide")
st.title("Corporación Favorita Sales Forecasting")
st.markdown("## 🧠 XGBoost Forecasting App")
st.markdown("---")


# --- Load Data ---
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(f"Could not find file at {DATA_PATH}. Make sure the path is correct.")
    st.stop()

df['date'] = pd.to_datetime(df['date'])
df['store_nbr'] = df['store_nbr'].astype('category')
df['item_nbr'] = df['item_nbr'].astype('category')
df['unit_sales'] = pd.to_numeric(df['unit_sales'], errors='coerce')

# Step 1: Calculate total unit sales per store and item
sales_summary = df.groupby(['store_nbr', 'item_nbr'])['unit_sales'].sum().reset_index()

# --- Top 10 Items per Store ---
sales_summary = df.groupby(['store_nbr', 'item_nbr'])['unit_sales'].sum().reset_index()
sales_summary_sorted = sales_summary.sort_values(['store_nbr', 'unit_sales'], ascending=[True, False])
top_10_per_store = sales_summary_sorted.groupby('store_nbr').head(10)
top_items_dict = top_10_per_store.groupby('store_nbr')['item_nbr'].apply(list).to_dict()



# #--- UI Controls ---
# st.sidebar.header("🔧 Select Store and Item")
st.sidebar.write("*How it works:*")
st.sidebar.write(
    "Please select a store and item to forecast the sales for the selected store and item."
)
st.sidebar.write(
    "*Note:* All Stores in Guayas Region are included in the dataset to provide forecast for top-10 items of selected store."
)
st.sidebar.markdown("---")
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
    features = high_corr_columns + ['store_nbr', 'item_nbr']
    print(features)

    X_train = train[features]
    y_train = train['unit_sales']
    X_test = test[features]
    y_test = test['unit_sales']

    # --- Load Scaler ---
    scaler= load_scaler(SCALER_PATH)

    # --- Load Model ---
    model = load_model(MODEL_PATH)

    # --- Prediction ---
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    # --- Evaluation ---
    st.subheader(f" Forecasting Sales for Store {selected_store}, Item {selected_item}")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    #mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    #col1, col2 = st.columns(2)
    col1, col2 = st.columns([3, 1])  # col1 is 3 times wider than col2

    with col1:
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
        

    with col2:
        st.subheader(f"📊 Performance metrices")
        st.write(f"**MSE:** {mse:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")
        # st.write(f"**MAPE:** {mape:.2f}")
        st.write(f"**R² Score:** {r2:.4f}")
        #st.write(f"**MSE:** {mse:.2f}")
        #st.write(f"**RMSE:** {rmse:.2f}")
        #st.write(f"**MAPE:** {mape:.2f}")
        #st.write(f"**R² Score:** {r2:.4f}")

    # --- Plot ---
    # st.subheader("📉 Actual vs Predicted Sales")
    # fig, ax = plt.subplots(figsize=(14, 6))
    # ax.plot(test['date'], y_test.values, label='Actual')
    # ax.plot(test['date'], y_pred, label='Predicted')
    # ax.set_xlabel('Date')
    # ax.set_ylabel('Unit Sales')
    # ax.set_title(f'Store {selected_store}, Item {selected_item} — Actual vs Predicted')
    # ax.legend()
    # ax.grid(True)
    # st.pyplot(fig)
