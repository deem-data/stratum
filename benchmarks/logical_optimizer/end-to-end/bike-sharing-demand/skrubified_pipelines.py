import skrub
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error, make_scorer
import time

from stratum.logical_optimizer import optimize

t0 = time.time()

# Load data
train = pd.read_csv("./input/train_augmented_3x.csv", parse_dates=["datetime"])

# Skrub DataOps plan
data = skrub.var("data", train).skb.subsample(n=1000)
X = data.drop("count", axis=1).skb.mark_as_X()
y = data["count"].skb.mark_as_y()
mode = skrub.eval_mode()

# Pipeline 0
datetime_col = X["datetime"].dt
X_feat_pipe0 = X.assign(
    year=datetime_col.year,
    month=datetime_col.month,
    dayofweek=datetime_col.dayofweek,
    hour=datetime_col.hour)

X_feat_pipe0 = X_feat_pipe0.drop(["datetime", "casual", "registered"], axis=1, errors="ignore")
model_pipe0 = lgb.LGBMRegressor(random_state=42)
pred_pipe0 = X_feat_pipe0.skb.apply(model_pipe0, y=y).skb.set_name("Pipeline 0")

# Pipeline 1
X_feat_pipe1 = X.assign(
    year=datetime_col.year,
    month=datetime_col.month,
    weekday=datetime_col.weekday,
    hour=datetime_col.hour)

X_feat_pipe1 = X_feat_pipe1.drop(["datetime", "casual", "registered"], axis=1, errors="ignore")

model_pipe1 = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
pred_pipe1 = X_feat_pipe1.skb.apply(model_pipe1, y=y.skb.apply_func(np.log1p)).skb.set_name("Pipeline 1")
pred_final_pipe1 = pred_pipe1.skb.apply_func(
    lambda a,b: np.expm1(a) if b=="predict" else a,
    mode).skb.set_name("Reverse log for prediction1")

# Pipeline 2
X_feat_pipe2 = X.assign(
    year=datetime_col.year,
    month=datetime_col.month,
    dayofweek=datetime_col.dayofweek,
    hour=datetime_col.hour)
X_feat_pipe2 = X_feat_pipe2.drop(["datetime", "casual", "registered"], axis=1, errors="ignore")

model_pipe2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
pred_pipe2 = X_feat_pipe2.skb.apply(model_pipe2, y=y.skb.apply_func(np.log1p)).skb.set_name("Pipeline 2")
pred_final_pipe2 = pred_pipe2.skb.apply_func(
    lambda a,b: np.expm1(a) if b=="predict" else a,
    mode).skb.set_name("Reverse log for prediction2")

# Pipeline 3
X_feat_pipe3 = X.assign(
    year=datetime_col.year,
    month=datetime_col.month,
    dayofweek=datetime_col.dayofweek,
    hour=datetime_col.hour)
X_feat_pipe3 = X_feat_pipe3.drop(["datetime", "casual", "registered"], axis=1, errors="ignore")

model_pipe3 = Ridge()
pred_pipe3 = X_feat_pipe3.skb.apply(model_pipe3, y=y.skb.apply_func(np.log1p)).skb.set_name("Pipeline 3")
pred_final_pipe3 = pred_pipe3.skb.apply_func(
    lambda a,b: np.expm1(a) if b=="predict" else a,
    mode).skb.set_name("Reverse log for prediction3")

# Pipeline 4
X_feat_pipe4 = X.assign(
    year=datetime_col.year,
    month=datetime_col.month,
    dayofweek=datetime_col.dayofweek,
    hour=datetime_col.hour)
X_feat_pipe4 = X_feat_pipe4.drop(["datetime", "casual", "registered"], axis=1, errors="ignore")

model_pipe4 = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
)
pred_pipe4 = X_feat_pipe4.skb.apply(model_pipe4, y=y.skb.apply_func(np.log1p)).skb.set_name("Pipeline 4")
pred_final_pipe4 = pred_pipe4.skb.apply_func(
    lambda a,b: np.expm1(a) if b=="predict" else a,
    mode).skb.set_name("Reverse log for prediction4")

merged_pipelines = skrub.choose_from({
    "pipeline0": pred_pipe0,
    "pipeline1": pred_final_pipe1,
    "pipeline2": pred_final_pipe2,
    "pipeline3": pred_final_pipe3,
    "pipeline4": pred_final_pipe4,
}, name="merged pipelines").as_data_op().skb.set_name("GridSearchCV")

# merged_pipelines.skb.draw_graph().open()
merged_pipelines = optimize(merged_pipelines)
# merged_pipelines.skb.draw_graph().open()

# RMSLE scorer
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, 0, None)))
scorer = make_scorer(rmsle)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
t0_ = time.time()
search = merged_pipelines.skb.make_grid_search(fitted=True, cv=cv, scoring=scorer, n_jobs=-1)
print(search.results_)

t1 = time.time()
print(t1 - t0)
print(t1 - t0_)
