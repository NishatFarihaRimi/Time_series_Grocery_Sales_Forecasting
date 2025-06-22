# Corporación Favorita Grocery Sales Forecasting

This project focuses on building time series models to forecast item-level sales for **Corporación Favorita** grocery stores across Ecuador. Accurate sales forecasting helps optimize inventory management, prevent stockouts, and enhance promotional strategies.

## Objective
The primary goal is to predict future sales of items sold in different store locations using historical sales data. Reliable forecasts support better decisions for:

- Inventory and logistics
- Promotional planning
- Business strategy

## 📊 Dataset
**Source:** [Kaggle: Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting)

**Download:** Via Kaggle API script (`kaggle competitions download`)

You can also reproduce the dataset by following the step-by-step data download and extraction instructions provided in the notebook:

📓 Note_books/1_Kaggle_API_Data_download.ipynb

This notebook uses the Kaggle API to fetch the dataset directly from the competition page. Make sure you have your kaggle.json API token ready, and follow the notebook in Google Colab or locally. Ensure the .csv files are located inside the Data/ directory before running the notebooks or app. This folder is used as the data source throughout the project.


The dataset includes:
| Dataset | Description    |
| :-------- | :------- |
| train.csv:| Daily item-level sales (unit_sales) per store. Includes onpromotion. Only non-zero sales are recorded. Negative values indicate returns.|
| stores.csv:| city, state, type, and cluster.|
| items.csv: | Item metadata – family, class, and perishable. Perishable items have a score weight of 1.25; others, 1.0.|
| transactions.csv:| Number of transactions per store/date (training period only).|
| oil.csv: | Daily oil prices. Important due to Ecuador's oil-driven economy.|
| holidays_events.csv: | National/local holidays and special events, including transferred, bridge, workday, and additional holiday types.|

### Downloading Additional File
Download the zipped data files manually from this Google Drive [link](https://drive.google.com/file/d/1-OZfY3-VOYt44nThkkuhO5z_QbXs1e4e/view?usp=sharing). Extract the contents and
place all extracted .csv files into the Data/ folder in the root of the project directory.
Your folder should then look like:
```text
Grocery_Sales_Forecasting/
├── Data/
│   ├── train_Guayas_final.csv
│   ├── ...
```
Alternatively, Generate Data via Kaggle + Notebooks
If you prefer to generate the data yourself, first download the original dataset from Kaggle. Then, execute the notebooks in sequence (2_ → 5_) to preprocess the data and produce the final processed CSV files used in this project.

## 📁 Project Structure

```text

Grocery_Sales_Forecasting//
│
├── Data/                           # Raw and processed data files
│   ├── __init__.py                 # Data module initialization
│   └── data_utils.py               # Utility functions for data handling
│
├── Note_books/                     # Jupyter notebooks for analysis and modeling
│   ├── 1_Kaggle_API_Data_download.ipynb
│   ├── 2_Filter_Train_Data_for_Guayas.ipynb
│   ├── 3_EDA.ipynb                 # Exploratory Data Analysis
│   ├── 4_Data_Preprocessing.ipynb
│   ├── 5_Feature_Engineering.ipynb
│   ├── 6_SARIMAX_Holtwinters_Model.ipynb
│   └── 7_XGboost_Model.ipynb      # Model training & tuning with XGBoost
│
├── app/                           # Main Streamlit app and related files
│   ├── __pycache__/               # Python cache files (usually gitignored)
│   ├── __init__.py                # App module initialization
│   ├── config.py                  # Config variables for file paths
│   ├── main.py                    # Main Streamlit app script
│   └── README.md                  # App-specific README (optional)
│
├── mlartifacts/0/models/          # ML artifacts saved by MLflow or similar
│
├── mlruns/0/                     # MLflow tracking files and experiment runs
│   ├── 15c00c7676e64ac0a59b532674c9a443/
│   ├── 15c9652540c842d3ade62b438cc211fe/
│   ├── 542eff3e68ba4e4e807fddbcec4b1421/
│   └── bcb006db19234a23a3b955d04fec50c9/
│
├── models/                       # Folder for model metadata and saved models
│   ├── meta.yaml                 # Model metadata/configuration file
│
├── model/                       # Python module for model-related utilities
│   ├── __pycache__/             # Python cache files (gitignored)
│   ├── __init__.py              # Model module initialization
│   ├── model_utils.py           # Functions to load model and scaler
│   └── xgb_model.pkl            # Pre-trained XGBoost model file
│
├── .gitignore                   # Git ignore rules
├── README.md                    # Project-level README file
├── mlflow.db                   # MLflow database file (tracking experiments)
├── requirements.txt            # Python dependencies
└── xgboost_mlflow.py           # Script for training/logging with MLflow & XGBoost

```
🔬 Project Pipeline
Notebook	Description
| Notebook | Description    |
| :-------- | :------- |
| 1_Kaggle_API_Data_download.ipynb | Kaggle API to download the dataset for project, including steps to authenticate, fetch, and extract the data|
| 2_Filter_Train_Data_for_Guayas.ipynb | Focused on filtering train data for Guayas region and top 3 item families|
| 3_EDA.ipynb | Exploratory Data Analysis (EDA): trends, seasonality, outliers, and missing values|
| 4_Data_Preprocessing.ipynb | Handling missing values, handling outliers, formatting dates, cleaning and basic preprocessing steps|
| 5_Feature_Engineering.ipynb | Creation of time-based features, lags, rolling stats, exponential smoothing, holiday indicators, etc|
| 6_SARIMAX_Holtwinters_Model.ipynb | Time series forecasting using SARIMAX and Holt-Winters models|
| 7_XGboost_Model.ipynb | Gradient boosting model (XGBoost) for forecasting with hyperpamrameter tunning|


## 🔧 Key Techniques Used
* Time-based train-test split

* Feature scaling (StandardScaler)

* Lag and rolling window feature engineering

* Exogenous variable selection via correlation analysis


## Forecasting Models

SARIMAX: Seasonal AutoRegressive Integrated Moving Average with exogenous variables

Holt-Winters: Triple exponential smoothing

XGBoost: Machine learning model with custom time features

## 📈  Evaluation Metrics
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)
* R² Score


## Getting Started
#### Step 1: Clone the GitHub Repository 
* Clone the repository to your local machine using Git.
* Open command prompt and run the following command:
```bash
git clone https://github.com/NishatFarihaRimi/Time_series_Grocery_Sales_Forecasting.git
cd Time_series_Grocery_Sales_Forecasting

```
#### Step 2: (Optional) Set up a Virtual Environment
Using conda:
```bash
conda create -n time_series_env python=3.8
conda activate time_series_env
```
#### Step 3: Set Up a Virtual Environment (Optional)
* Creating a venv with conda and activating it
```bash
conda create -n time_series_env python==3.8.0
conda activate time_series_env
```
#### Step 4: Install dependencies
 Installing the packages listed in *'requirements.txt'** file
```bash
pip install -r requirements.txt
```
#### Step 4: Run the Notebooks
Execute notebooks in order (1_ → 7_) to reproduce the full workflow: from raw data to forecasting models.


## ✨ Future Work
* Hyperparameter tuning using GridSearchCV
* Add LSTM/DeepAR models
* Deploy model via Streamlit or Flask

## Acknowledgments
Dataset from Kaggle: Corporación Favorita Grocery Sales Forecasting

XGBoost: https://xgboost.readthedocs.io/

MLflow: For model tracking and experimentation












