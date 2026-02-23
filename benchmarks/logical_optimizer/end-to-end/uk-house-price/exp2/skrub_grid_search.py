import pandas as pd
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import make_scorer, r2_score
import logging
logging.basicConfig(level=logging.DEBUG)
import stratum as skrub
from time import perf_counter
import argparse
import polars as pl
from pipeline_definitions import *
parser = argparse.ArgumentParser()
parser.add_argument("--size", type=str, default="100K")
parser.add_argument("--stratum", action="store_true")
args = parser.parse_args()

size = args.size
if len(size) > 0:
    size = "_" + size

# Load data
data_path = f"../input/price_paid_records{size}.csv"
data = skrub.var("data_path", data_path).skb.apply_func(pd.read_csv).skb.subsample(n=1000)

y = data["Price"].skb.mark_as_y()
X = data.drop(columns=["Price"]).skb.mark_as_X()
#X = X.skb.apply_func(lambda a: (a, print("RUNNING"))[0])

pred = skrub.choose_from(
    {
        "1_targ_lgb":           pipeline1(X, y),
        "2_targ_com_lgb":       pipeline2(X, y),
        "4_targ_comp_cyc_lgb":  pipeline4(X, y),
        "5_targ_comp2_cyc_lgb": pipeline5(X, y),
        "6_town_cluster":       pipeline6(X, y),
        "7_targ_ridge_1.0":     pipeline7(X, y),
        "8_targ_ridge_0.1":     pipeline8(X, y),
        "9_targ_ridge_0.01":    pipeline9(X, y),
        # "10_tv_ridge_1.0":      pipeline_10(X, y),
        # "11_tv_ridge_0.1":      pipeline_11(X, y),
        # "12_tv_ridge_0.01":     pipeline_12(X, y),
        # "13_tv_lgb":            pipeline_13(X, y),
        # "14_tv_elasticnet":     pipeline_14(X, y),
        "20_ohe_lgb":           pipeline20(X, y),
        "21_ohe_ridge_1.0":     pipeline21(X, y),
        "22_ohe_ridge_0.1":     pipeline22(X, y),
        "23_ohe_ridge_0.01":    pipeline23(X, y),
        "30_ohe_lasso_1.0":     pipeline30(X, y),
        "31_ohe_lasso_0.1":     pipeline31(X, y),
        "32_ohe_lasso_0.01":    pipeline32(X, y),
        "33_ohe_lasso_0.001":   pipeline33(X, y),
    },
    name="pipe_nr",
).as_data_op()


cv = 3
cv = ShuffleSplit(n_splits=cv, test_size=0.2, random_state=42) if cv == 1 else KFold(n_splits=cv, shuffle=True, random_state=42)
scoring = make_scorer(r2_score)

t0 = perf_counter()
with skrub.config(scheduler=args.stratum, stats=20, force_polars=True, rust_backend=True, scheduler_parallelism="threading", DEBUG=True):
    search =pred.skb.make_grid_search(fitted=True, refit=False,scoring=scoring, cv=cv, n_jobs=1)
t1 = perf_counter()
print(f"Time taken: {t1 - t0} seconds")
if isinstance(search.results_, pl.DataFrame):
    search.results_.show(limit=100)
else:
    print(search.results_.to_string())