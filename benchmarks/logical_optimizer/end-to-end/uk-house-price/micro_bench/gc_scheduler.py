from time import perf_counter
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMRegressor
import argparse
import stratum as skrub
from memory_tracker import MemoryTracker

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=str, default="100K")
parser.add_argument("--stratum", action="store_true")
parser.add_argument("--njobs", type=int, default=1)
args = parser.parse_args()

size = args.size
if len(size) > 0:
    size = "_" + size

# Load data
data_path = f"../input/price_paid_records{size}.csv"
data = skrub.var("data_path", data_path).skb.apply_func(pd.read_csv).skb.subsample(n=1000)

y = data["Price"].skb.mark_as_y()
X = data.drop(columns=["Price"]).skb.mark_as_X()
drop_cols = ["Date of Transfer", "PPDCategory Type", "Record Status - monthly file only",'Transaction unique identifier']


y_log = y.skb.apply_func(np.log1p)
date_col = X["Date of Transfer"].skb.apply_func(pd.to_datetime, errors="coerce")
X_dropped = X.drop(columns=drop_cols)
X2 = X_dropped.assign(
    year=date_col.dt.year, 
    month=date_col.dt.month, 
    day=date_col.dt.day, 
    dayofweek=date_col.dt.dayofweek, 
    dayofyear=date_col.dt.dayofyear, 
    quarter=date_col.dt.quarter, 
)

X2_num_cols = X2.skb.select(skrub.selectors.numeric())

enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X2_cat_cols2 = X2.skb.select(~skrub.selectors.numeric())
X3 = X2_cat_cols2.skb.apply(enc).skb.concat([X2_num_cols.skb.apply(StandardScaler())], axis=1)

X3_cat_cols = X2_cat_cols2.skb.apply(SimpleImputer(strategy="most_frequent")).skb.apply(OneHotEncoder(handle_unknown="ignore", sparse_output=False))
X3_num_cols = X2_num_cols.skb.apply(SimpleImputer(strategy="median")).skb.apply(StandardScaler())
X4 = X3_cat_cols.skb.concat([X3_num_cols], axis=1)

X = skrub.choose_from({"no_impute": X3, "impute": X4}, name="preprocess").as_data_op()
model1 = LGBMRegressor(random_state=42, verbose=-1)
model2 = Ridge(alpha=1.0, random_state=42)
preds_log = skrub.choose_from({"lgb": X.skb.apply(model1, y=y_log), "ridge": X.skb.apply(model2, y=y_log)}, name="model").as_data_op()
preds = preds_log.skb.apply_func(lambda x,m: np.expm1(x) if m != "fit" else 0, m=skrub.eval_mode())


cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

tracker = MemoryTracker(
    mode="system" if args.njobs != 1 else "process",
)
tracker.start()

t0 = perf_counter()
try:
    with skrub.config(scheduler=args.stratum, stats=20, rust_backend=False, scheduler_parallelism=None, force_polars=False, cse=False, DEBUG=False):
        search = preds.skb.make_grid_search(cv=cv, n_jobs=args.njobs, fitted=True, refit=False)
finally:
    tracker.stop()
t1 = perf_counter()

tracker.write_csv("memory_usage.csv")

print(f"Time taken: {t1 - t0} seconds")
print(search.results_)


