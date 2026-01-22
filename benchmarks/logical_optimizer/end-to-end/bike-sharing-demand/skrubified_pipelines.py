import stratum as skrub
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error, make_scorer
import time

from stratum.logical_optimizer import apply_cse_on_skrub_ir

t0 = time.time()

# Skrub DataOps plan
data = skrub.var("data_file", "./input/train_augmented_3x.csv").skb.apply_func(pd.read_csv).skb.subsample(n=1000)
X = data.drop("count", axis=1).skb.mark_as_X()
y = data["count"].skb.mark_as_y()
mode = skrub.eval_mode()

# Pipeline 0
datetime_col0 = X["datetime"].skb.apply_func(pd.to_datetime).dt
X_feat_pipe0 = X.assign(
    year=datetime_col0.year,
    month=datetime_col0.month,
    dayofweek=datetime_col0.dayofweek,
    hour=datetime_col0.hour)

X_feat_pipe0 = X_feat_pipe0.drop(["datetime", "casual", "registered"], axis=1, errors="ignore")
model_pipe0 = lgb.LGBMRegressor(random_state=42)
pred_pipe0 = X_feat_pipe0.skb.apply(model_pipe0, y=y).skb.set_name("Pipeline 0")

# Pipeline 1
datetime_col1 = X["datetime"].skb.apply_func(pd.to_datetime).dt
X_feat_pipe1 = X.assign(
    year=datetime_col1.year,
    month=datetime_col1.month,
    weekday=datetime_col1.weekday,
    hour=datetime_col1.hour)

X_feat_pipe1 = X_feat_pipe1.drop(["datetime", "casual", "registered"], axis=1, errors="ignore")

model_pipe1 = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
pred_pipe1 = X_feat_pipe1.skb.apply(model_pipe1, y=y.skb.apply_func(np.log1p)).skb.set_name("Pipeline 1")
pred_final_pipe1 = pred_pipe1.skb.apply_func(
    lambda a,b: np.expm1(a) if b=="predict" else a,
    mode).skb.set_name("Reverse log for prediction1")

# Pipeline 2
datetime_col2 = X["datetime"].skb.apply_func(pd.to_datetime).dt
X_feat_pipe2 = X.assign(
    year=datetime_col2.year,
    month=datetime_col2.month,
    dayofweek=datetime_col2.dayofweek,
    hour=datetime_col2.hour)
X_feat_pipe2 = X_feat_pipe2.drop(["datetime", "casual", "registered"], axis=1, errors="ignore")

model_pipe2 = ElasticNet() #RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
pred_pipe2 = X_feat_pipe2.skb.apply(model_pipe2, y=y.skb.apply_func(np.log1p)).skb.set_name("Pipeline 2")
pred_final_pipe2 = pred_pipe2.skb.apply_func(
    lambda a,b: np.expm1(a) if b=="predict" else a,
    mode).skb.set_name("Reverse log for prediction2")

# Pipeline 3
datetime_col3 = X["datetime"].skb.apply_func(pd.to_datetime).dt
X_feat_pipe3 = X.assign(
    year=datetime_col3.year,
    month=datetime_col3.month,
    dayofweek=datetime_col3.dayofweek,
    hour=datetime_col3.hour)
X_feat_pipe3 = X_feat_pipe3.drop(["datetime", "casual", "registered"], axis=1, errors="ignore")

model_pipe3 = Ridge()
pred_pipe3 = X_feat_pipe3.skb.apply(model_pipe3, y=y.skb.apply_func(np.log1p)).skb.set_name("Pipeline 3")
pred_final_pipe3 = pred_pipe3.skb.apply_func(
    lambda a,b: np.expm1(a) if b=="predict" else a,
    mode).skb.set_name("Reverse log for prediction3")

# Pipeline 4
datetime_col4 = X["datetime"].skb.apply_func(pd.to_datetime).dt
X_feat_pipe4 = X.assign(
    year=datetime_col4.year,
    month=datetime_col4.month,
    dayofweek=datetime_col4.dayofweek,
    hour=datetime_col4.hour)
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

merged_pipelines.skb.draw_graph().open()
# merged_pipelines.skb.draw_graph().open()
# merged_pipelines = apply_cse_on_skrub_ir(merged_pipelines)
# merged_pipelines.skb.draw_graph().open()

# RMSLE scorer
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, 0, None)))
scorer = make_scorer(rmsle)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
t0_ = time.time()
with skrub.config(scheduler=False, stats=20, force_polars=True, DEBUG=True):
    search = merged_pipelines.skb.make_grid_search(fitted=True, cv=cv, scoring=scorer, n_jobs=1)
print(search.results_)

t1 = time.time()
print(t1 - t0)
print(t1 - t0_)
