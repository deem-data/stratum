import stratum as skrub
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
from time import perf_counter
from xgboost import XGBRegressor
import numpy as np

t0 = perf_counter()
file_path = "input/price_paid_records_1M.csv"
df = skrub.as_data_op(file_path).skb.apply_func(pd.read_csv).skb.subsample(n=1000)

df = df.rename(columns={"Town/City": "Town"}, inplace=False)
y = df["Price"].skb.mark_as_y()
X = df.drop(columns=["Price", "Transaction unique identifier"],axis=1)

date_col = X["Date of Transfer"].skb.apply_func(pd.to_datetime)
X = X.assign(
    year=date_col.dt.year, 
    month=date_col.dt.month, 
    day=date_col.dt.day, 
    dayofweek=date_col.dt.dayofweek, 
    hour=date_col.dt.hour)
X = X.assign(
    month_sin=(date_col.dt.month * (2 * np.pi / 12)).apply(np.sin),
    month_cos=(date_col.dt.month * (2 * np.pi / 12)).apply(np.cos),
    day_sin=(date_col.dt.day * (2 * np.pi / 30)).apply(np.sin),
    day_cos=(date_col.dt.day * (2 * np.pi / 30)).apply(np.cos),
    dayofweek_sin=(date_col.dt.dayofweek * (2 * np.pi / 7)).apply(np.sin),
    dayofweek_cos=(date_col.dt.dayofweek * (2 * np.pi / 7)).apply(np.cos),
    hour_sin=(date_col.dt.hour * (2 * np.pi / 24)).apply(np.sin),
    hour_cos=(date_col.dt.hour * (2 * np.pi / 24)).apply(np.cos),
)
X = X.drop(["Date of Transfer",
    "Duration", 
    "PPDCategory Type", 
    "Record Status - monthly file only"
], axis=1).skb.mark_as_X()

X_num = X.skb.select(skrub.selectors.numeric())
X_cat = X.skb.select(~skrub.selectors.numeric())
X_cat_enc = X_cat.skb.apply( OneHotEncoder(drop='if_binary', dtype='float32', handle_unknown='ignore', sparse_output=False))
X_num_scaled = X_num.skb.apply(StandardScaler())
X_enc = X_num_scaled.skb.concat([X_cat_enc], axis=1)

models = [
    LGBMRegressor(random_state=42),
    Ridge(random_state=42),
    XGBRegressor(random_state=42),
    ElasticNet(random_state=42),
]

preds = {f"{i}": X_enc.skb.apply(model, y=y) for i, model in enumerate(models)}
preds = skrub.choose_from(preds, name="model").as_data_op()
scorer = make_scorer(r2_score)
t1 = perf_counter()
cv = KFold(n_splits=3, shuffle=True, random_state=42)
with skrub.config(scheduler=True, stats=20, rust_backend=True, scheduler_parallelism="auto",force_polars=True,):
    search_stratum = preds.skb.make_grid_search(cv=cv, n_jobs=1, fitted=True, scoring=scorer)
t2 = perf_counter()
print(f"Preview time taken: {t1 - t0} seconds")
print(f"Search time taken: {t2 - t1} seconds")
print(search_stratum.results_)