import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error
import warnings

warnings.filterwarnings("ignore")

# Load data
_input_filename = os.getenv("BIKE_INPUT_FILE", "train.csv")
train = pd.read_csv(f"./input/{_input_filename}")


# Feature engineering
def fe(df):
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    return df


train = fe(train)

# Define features and target
features = [
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "year",
    "month",
    "hour",
    "weekday",
]
X = train[features]
y = train["count"]


# RMSLE function
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred.clip(0, None)))


# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    # log1p transform
    y_tr_log = np.log1p(y_tr)
    model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr_log)
    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)
    scores.append(rmsle(y_val, preds))
print(f"CV RMSLE: {np.mean(scores):.5f}")
