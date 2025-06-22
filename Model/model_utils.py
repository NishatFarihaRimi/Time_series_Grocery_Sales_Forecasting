import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

def load_scaler(path):
    with open(path, 'rb') as file:
        scaler = pickle.load(file)
    return scaler

def preprocess_data(df, features, scaler):
    df = df.copy()
    df['store_nbr'] = df['store_nbr'].astype('category')
    df['item_nbr'] = df['item_nbr'].astype('category')
    df['date'] = pd.to_datetime(df['date'])
    X = df[features]
    X_scaled = scaler.transform(X)
    return X_scaled