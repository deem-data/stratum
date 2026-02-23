from time import perf_counter
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import logging
logging.basicConfig(level=logging.DEBUG)
import stratum as skrub
import pandas as pd
import argparse
from lightgbm import LGBMRegressor
import numpy as np
from memory_tracker import MemoryTracker
import polars as pl

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=str, default="100K")
parser.add_argument("--polars", action="store_true")
args = parser.parse_args()

size = args.size
if len(size) > 0:
    size = "_" + size
file_path = f"../input/price_paid_records{size}.csv"
df = skrub.as_data_op(file_path).skb.apply_func(pd.read_csv).skb.subsample(n=1000)
df = df.rename(columns={"Town/City": "Town"}, inplace=False)

y = df["Price"].skb.mark_as_y()
X = df.drop("Price", axis=1).skb.mark_as_X()

date = X["Date of Transfer"].skb.apply_func(pd.to_datetime)
X = X.assign(
    year=date.dt.year, 
    month=date.dt.month, 
    day=date.dt.day, 
    dayofweek=date.dt.dayofweek, 
    hour=date.dt.hour)
X = X.assign(
    month_sin=(date.dt.month * (2 * np.pi / 12)).apply(np.sin),
    month_cos=(date.dt.month * (2 * np.pi / 12)).apply(np.cos),
    day_sin=(date.dt.day * (2 * np.pi / 30)).apply(np.sin),
    day_cos=(date.dt.day * (2 * np.pi / 30)).apply(np.cos),
    dayofweek_sin=(date.dt.dayofweek * (2 * np.pi / 7)).apply(np.sin),
    dayofweek_cos=(date.dt.dayofweek * (2 * np.pi / 7)).apply(np.cos),
    hour_sin=(date.dt.hour * (2 * np.pi / 24)).apply(np.sin),
    hour_cos=(date.dt.hour * (2 * np.pi / 24)).apply(np.cos),
)
X = X.drop([
    "Date of Transfer", 
    'Duration', 
    'Transaction unique identifier', 
    'PPDCategory Type', 
    'Record Status - monthly file only'], axis=1)

X_cat = X.skb.select(~skrub.selectors.numeric())
X_cat_ord = X_cat.skb.apply(OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
X_cat_ohe = X_cat.skb.apply(skrub.OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
X_cat = skrub.choose_from({"ordinal": X_cat_ord, "one-hot": X_cat_ohe}, name="encoder").as_data_op()

X_num = X.skb.select(skrub.selectors.numeric())
X_num = X_num.skb.apply(StandardScaler())
X_vec = X_cat.skb.concat([X_num],axis=1)

def to_pandas(df):
    if type(df) == pl.DataFrame or type(df) == pl.Series:
        out = df.to_pandas()
        del df
        return out
    return df

X_vec = X_vec.skb.apply_func(to_pandas)
y = y.skb.apply_func(to_pandas)
pred1 = X_vec.skb.apply(LGBMRegressor(verbose=-1, random_state=42), y=y)
# pred2 = X_vec.skb.apply(ElasticNet(random_state=42, max_iter=300), y=y)
# #pred2 = X_vec.skb.apply(XGBRegressor(random_state=42), y=y)
# pred = skrub.choose_from({
#     "LGBM": pred1, 
#     "ElasticNet": pred2
# }, name="model").as_data_op()
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
scorer = make_scorer(r2_score)

tracker = MemoryTracker()
tracker.start()
t0 = perf_counter()
try:
    with skrub.config(scheduler=True, force_polars=args.polars, stats=20, DEBUG=True):
        search = pred1.skb.make_grid_search(fitted=True, scoring="r2", cv=cv)
finally:
    tracker.stop()
t1 = perf_counter()

tracker.write_csv(f"memory_usage_{'polars' if args.polars else 'pandas'}{size}.csv")
print(f"Time taken: {t1 - t0:.2f} seconds")
print(search.results_)