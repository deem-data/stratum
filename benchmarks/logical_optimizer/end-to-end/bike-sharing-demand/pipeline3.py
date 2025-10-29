import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load data
_input_filename = os.getenv("BIKE_INPUT_FILE", "train.csv")
train = pd.read_csv(f"./input/{_input_filename}")


# Feature engineering
def preprocess(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    return df[
        [
            "year",
            "month",
            "dayofweek",
            "hour",
            "season",
            "weather",
            "temp",
            "atemp",
            "humidity",
            "windspeed",
            "workingday",
            "holiday",
        ]
    ]


X = preprocess(train)
y = np.log1p(train["count"])

# 5-fold CV for RMSLE on log scale
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []
for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = Ridge()
    model.fit(X_tr, y_tr)
    y_pred_log = model.predict(X_val)
    rmsle = np.sqrt(mean_squared_error(y_val, y_pred_log))
    rmsle_scores.append(rmsle)

cv_rmsle = np.mean(rmsle_scores)
print(f"CV RMSLE: {cv_rmsle:.5f}")
