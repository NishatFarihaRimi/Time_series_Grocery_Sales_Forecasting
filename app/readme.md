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

## Model Training

The XGBoost model used for forecasting was trained in Jupyter Notebook (VS Code). The full training pipeline, including hyperparameter tuning and feature selection, is documented in the notebook:

### Pipeline
- Data Loading: Sales data is read from a CSV file.
- Top Sellers: Identifies the top 10 items by sales volume for each store.
- User Selection: Users select a store and item to predict.
- Feature Selection: Features with correlation to unit sales are selected dynamically.
- Scaling: Data is scaled using a pre-saved scaler.
- Prediction: Forecasts are generated using a pre-trained XGBoost model.
- Evaluation & Visualization: Outputs include error metrics and visual comparison charts.

## Project Structure

.
├── app/
│   └── config.py              # File paths for model, scaler, and data
├── model/
│   └── model_utils.py         # Utility functions: load_model(), load_scaler()
├── streamlit_app.py           # Main Streamlit app
├── models/                    # Folder containing saved model files
├── data/                      # Folder containing input CSV data
├── Used Notebooks For Time Series Analysis/
│   └── Hyperparameter_Tuning.ipynb  # Model development notebook
└── README.md                  # This file

---

## Reproducing the Model

To retrain or validate the model in Google Colab, follow these steps:

Download the training dataset from the following link:
train_final.csv

Upload the file to your Colab environment.

Open and run the Jupyter notebooks from the project to reproduce the model training and forecasting steps.

Alternatively, you can test the app locally by following these steps:

You can also run the app on your local machine:

Clone this repository and navigate to the project folder.

Create a virtual environment (optional but recommended).

Install dependencies using:

pip install -r requirements.txt
Launch the Streamlit app with:

streamlit run streamlit_app.py
The app will open in your default browser, allowing you to generate predictions for Jan–Mar 2014.

The app uses a large dataset (train_final.csv, approx. 905.9 MB). On first run, this file will be downloaded to the /data folder. Please allow a few minutes for this process to complete.
---

## Requirements

Key dependencies include:

Streamlit

Pandas

NumPy

Scikit-learn

Matplotlib

XGBoost

To view all dependencies, refer to the requirements.txt file in the project root.

---

## License

This project is licensed under the [MIT License](LICENSE).
