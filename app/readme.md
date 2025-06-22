# App

## 🧠 XGBoost Sales Forecasting App
A Streamlit-based interactive web app for time series sales forecasting using a pre-trained XGBoost model. This application allows users to select a store and one of its top 10 best-selling items to view predicted vs. actual unit sales, along with evaluation metrics and trend visualizations.

## Features

### 📋 Sidebar Controls

Store & Item Selection: Choose a store and one of its top 10 best-selling items.

- Predict Button: Trigger the forecast for the selected combination.

### 📈 Sales Trend Visualization
Compare actual vs. predicted unit sales via an interactive line chart.
Visualizes predicted vs. actual unit sales trends from January to March 2014.

- Helps in comparing the forecast performance over time for selected store and item. .

### 🧾 Tabular Output

Actual vs. predicted unit sales
- Displays detailed prediction data (date, predicted sales) for the selected store and item.

## Model Evaluation Metrics
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score (Coefficient of Determination)

## ⚠️ Error & Warning Handling
-Provides intuitive messages if:
-No data is found for the selected store-item pair.
-Required model or data files are missing or misconfigured.


### Pipeline
- Data Loading: Sales data is read from a CSV file.
- Top Sellers: Identifies the top 10 items by sales volume for each store.
- User Selection: Users select a store and item to predict.
- Feature Selection: Features with correlation to unit sales are selected dynamically.
- Scaling: Data is scaled using a pre-saved scaler.
- Prediction: Forecasts are generated using a pre-trained XGBoost model.
- Evaluation & Visualization: Outputs include error metrics and visual comparison charts.

## 🧱 Project Structure

```text
├── app/
│   └── config.py              # File paths for model, scaler, and data
├── model/
│   └── model_utils.py         # Utility functions: load_model(), load_scaler()
├── streamlit_app.py           # Main Streamlit app
├── models/                    # Folder containing saved model files
├── data/                      # Folder containing input CSV data
├── Used Notebooks For Time Series Analysis/
│   └── 7_XGboost_Model.ipynb  # Model development notebook
└── README.md                  # This file 
```
## Model Training
The model was trained in Jupyter Notebook using VS Code.

Main notebook:
Used Notebooks For 7_XGboost_Model.ipynb.ipynb

Additional notebooks support preprocessing and exploratory analysis.

## Reproducing the Model (Google Colab)
To retrain or experiment in Google Colab:

1. Download the training dataset from manually from this Google Drive [link](https://drive.google.com/file/d/1lcXGfg32fbnm8_12WaAWejymi0cu2DXP/view?usp=sharing):
train_Guayas_final.csv

2. Upload it to your Colab environment.

3. Run the notebooks in the project to retrain the XGBoost model.

## Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the dependencies:
pip install -r requirements.txt
Run the Streamlit app:

streamlit run streamlit_app.py
The app will open in your default browser.

📌 Note: On the first run, the app will download train_Guayas_final.csv (~905.9 MB) into the /data directory. On first run, this file will be downloaded to the /data folder. Please allow a few minutes for this process to complete.

## 📂  Configuration
Ensure the following paths are correctly set in app/config.py:
DATA_PATH = "path/to/your/train_final.csv"
MODEL_PATH = "path/to/your/xgboost_model.pkl"
SCALER_PATH = "path/to/your/scaler.pkl"


## 📦 Requirements

Minimum: Python 3.8+

Main packages used:

Streamlit

Pandas

NumPy

scikit-learn

XGBoost

Matplotlib

Install all dependencies using:

pip install -r requirements.txt.

---

## License

This project is licensed under the [MIT License](LICENSE).
