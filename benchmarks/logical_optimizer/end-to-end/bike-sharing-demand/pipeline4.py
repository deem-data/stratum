import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import xgboost as xgb

# Load data
_input_filename = os.getenv("BIKE_INPUT_FILE", "train.csv")
train = pd.read_csv(f"./input/{_input_filename}")

# Identify target from sample submission
target_col = "count"


# Feature engineering
def prepare(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    return df


train_p = prepare(train)

features = [
    "season",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "workingday",
    "holiday",
    "year",
    "month",
    "day_of_week",
    "hour",
]
X = train_p[features]
y = np.log1p(train_p[target_col])

# 5-fold CV evaluation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmsle_scores = []
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    score = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(y_pred)))
    rmsle_scores.append(score)

print(f"CV RMSLE: {np.mean(rmsle_scores):.5f}")
