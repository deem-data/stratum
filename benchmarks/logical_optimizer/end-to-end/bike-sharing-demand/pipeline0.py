import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

# Load data
_input_filename = os.getenv("BIKE_INPUT_FILE", "train.csv")
train = pd.read_csv(f"./input/{_input_filename}", parse_dates=["datetime"])


# Feature engineering
def add_datetime_features(df):
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    return df


train = add_datetime_features(train)

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
    "dayofweek",
    "hour",
]
X = train[features]
y = train["count"]


# RMSLE scorer
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, 0, None)))


# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    scores.append(rmsle(y_val, preds))

print(f"5-fold RMSLE: {np.mean(scores):.5f}")
