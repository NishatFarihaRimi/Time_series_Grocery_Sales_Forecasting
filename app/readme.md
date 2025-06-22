# App

- **App — XGBoost Sales Forecasting** 
The Forecaster App is an interactive Streamlit application designed to predict and visualize sales for selected store-item combinations over time. Powered by a trained XGBoost model, this app helps users analyze sales trends and evaluate model performance with ease.

## Features

### Sidebar Controls

Store & Item Selection: Choose a store and one of its top 10 best-selling items.

Predict Button: Trigger the forecast for the selected combination.

### Line Chart
Visualizes predicted vs. actual unit sales trends from January to March 2014.

Helps in comparing the forecast performance over time for selected store and item. .

### Tabular Output

Actual vs. predicted unit sales
- Displays detailed prediction data (date, predicted sales) for the selected store and item.


## Model Training

The XGBoost model used for forecasting was trained in Jupyter Notebook (VS Code). The full training pipeline, including hyperparameter tuning and feature selection, is documented in the notebook:

Used Notebooks For Time Series Analysis/Hyperparameter_Tuning.ipynb

Other supporting notebooks for preprocessing and exploration are also available in that folder.

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