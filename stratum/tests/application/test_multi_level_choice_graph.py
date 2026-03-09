import os
import pickle
import tempfile
import unittest
import uuid

from skrub import TableVectorizer
import stratum as skrub
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from stratum.logical_optimizer._optimize import optimize
import polars as pl
import logging
logging.basicConfig(level=logging.DEBUG)
from stratum.runtime._scheduler import SchedulerFlags
class TargetEncoder(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        print("fit target encoder")
        self.global_mean_ = y.mean()
        tmp = pd.concat([X, y], axis=1)
        self.cols = X.columns
        self.means = {}
        for col in self.cols:
            self.means[col] = tmp.groupby(col)[tmp.columns[-1]].mean()
        return self

    def transform(self, X):
        print("transform target encoder")
        X_out = X.copy()
        for col in self.cols:
            X_out[col] = X_out[col].map(self.means[col]).fillna(self.global_mean_)
        return X_out

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self):
        return self.cols


def define_pipeline(file_path):
    df = skrub.as_data_op(file_path).skb.apply_func(pd.read_csv)
    df = df.rename(columns={"Town/City": "Town"}, inplace=False)
    y = df["Price"].skb.mark_as_y()
    X = df.drop("Price", axis=1).skb.mark_as_X()

    def df1(X, y):
        date = X["Date of Transfer"].skb.apply_func(pd.to_datetime)
        X = X.assign(
            month_sin=(date.dt.month * (2 * np.pi / 12)).apply(np.sin),
            month_cos=(date.dt.month * (2.0 * np.pi / 12)).apply(np.cos),
            day_sin=(date.dt.day * (2.0 * np.pi / 30)).apply(np.sin),
            day_cos=(date.dt.day * (2.0 * np.pi / 30)).apply(np.cos),
            dayofweek_sin=(date.dt.dayofweek * (2.0 * np.pi / 7)).apply(np.sin),
            dayofweek_cos=(date.dt.dayofweek * (2.0 * np.pi / 7)).apply(np.cos),
            hour_sin=(date.dt.hour * (2.0 * np.pi / 24)).apply(np.sin),
            hour_cos=(date.dt.hour * (2.0 * np.pi / 24)).apply(np.cos),
        )
        X = X.drop([
            "Date of Transfer", 
            'Duration', 
            'Transaction unique identifier', 
            'PPDCategory Type', 
            'Record Status - monthly file only'], axis=1)

        def is_string_column(col):
            return col.dtype == "object"

        def is_numeric_column(col):
            return col.dtype != "object"
        cat_selector = skrub.selectors.filter(is_string_column)
        X_cat = X.skb.select(cat_selector)
        X_cat_enc = X_cat.skb.apply(skrub.StringEncoder())
        num_selector = skrub.selectors.filter(is_numeric_column)

        X_te = X[["District", "County", "Town"]].skb.apply(TargetEncoder(), y=y)
        X_te = X_te.rename(columns={"District": "district_te", "County": "county_te", "Town": "town_te"})
        X_num = X.skb.select(num_selector)
        X_vec = X_num.skb.concat([X_te, X_cat_enc], axis=1)

        return X_vec

    def df2(X):
        return X.skb.apply(TableVectorizer())

    X_vec = skrub.choose_from({"1": df1(X,y), "2": df2(X)}, name = "pre").as_data_op()
    models = {
        "Ridge": Ridge(random_state=42),
        "xgb": XGBRegressor(random_state=42),
        "lgbm": LGBMRegressor(random_state=42),
        "elastic": ElasticNet(random_state=42),
    }
    preds = {name: X_vec.skb.apply(m, y=y) for name, m in models.items()}
    return skrub.choose_from(preds, name="m").as_data_op()

def make_data(n: int = 1000, seed: int = 42):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Transaction unique identifier": [str(uuid.uuid4()) for _ in range(n)],
        "Price": rng.integers(50000, 2_000_000, size=n),
        "Date of Transfer": pd.to_datetime(
            rng.choice(pd.date_range("2010-01-01", "2024-12-31"), size=n)
        ).astype(str),
        "Property Type": rng.choice(list("DSTFO"), size=n),
        "Old/New": rng.choice(["Y", "N"], size=n),
        "Duration": rng.choice(["F", "L"], size=n),
        "Town/City": rng.choice(
            ["London", "Manchester", "Birmingham", "Leeds", "Bristol"], size=n
        ),
        "District": rng.choice(
            ["District A", "District B", "District C"], size=n
        ),
        "County": rng.choice(
            ["Greater London", "West Midlands", "Greater Manchester"], size=n
        ),
        "PPDCategory Type": rng.choice(["A", "B"], size=n),
        "Record Status - monthly file only": rng.choice(["A", "C"], size=n),
    })
    return df

class TestMultiLevelChoiceGraph(unittest.TestCase):
    expected_results = pl.DataFrame({
        "id": [
            "m:elastic, pre:2",
            "m:elastic, pre:1",
            "m:Ridge, pre:2",
            "m:Ridge, pre:1",
            "m:xgb, pre:2",
            "m:lgbm, pre:2",
            "m:lgbm, pre:1",
            "m:xgb, pre:1"
        ],
        "scores": [
            -0.000779,
            -0.028774,
            -0.021469,
            -0.040625,
            -0.156263,
            -0.174555,
            -0.172825,
            -0.251869
        ]
    })

    def test_application(self):
        tmp_path = tempfile.mkdtemp()
        df = make_data()
        print(df.dtypes)
        df.to_csv(os.path.join(tmp_path, "data.csv"), index=False)
        preds = define_pipeline(os.path.join(tmp_path, "data.csv"))
        scorer = make_scorer(r2_score)
        cv = KFold(n_splits=2, shuffle=True, random_state=42)
        with skrub.config(DEBUG=True, open_graph=False, scheduler=True, rust_backend=False, scheduler_parallelism=None, stats=20):
            search = preds.skb.make_grid_search(fitted=True, cv = cv, scoring=scorer)
        print(search.results_)


    def run_application(self, sched_par: str = None):
        tmp_path = tempfile.mkdtemp()
        df = make_data()
        df.to_csv(os.path.join(tmp_path, "data.csv"), index=False)
        preds = define_pipeline(os.path.join(tmp_path, "data.csv"))
        preds = preds.skb.apply_func(lambda a, m: a, m=skrub.eval_mode())
        scorer = make_scorer(r2_score)
        cv = KFold(n_splits=2, shuffle=True, random_state=42)
        with skrub.config(DEBUG=True, open_graph=False, scheduler=True, rust_backend=False, scheduler_parallelism=sched_par, stats=20):
            search = preds.skb.make_grid_search(fitted=True, cv = cv, scoring=scorer)
        print(search.results_)
        return search.results_

    def test_application_no_parallelism(self):
        actual_results = self.run_application()
        # Convert to pandas for comparison
        # TODO: pre:2 is non-deterministic right now, so we need to filter it out
        filter_expr = pl.col("id").str.contains("pre:1") & ~pl.col("id").str.contains("xgb|lgbm")
        actual_df = actual_results.sort("id").filter(filter_expr).to_pandas()
        expected_df = self.expected_results.sort("id").filter(filter_expr).to_pandas()
        print(actual_df)
        pd.testing.assert_frame_equal(
            actual_df,
            expected_df,
            atol=1e-6,
            check_dtype=False
        )

    def test_application_threading(self):
        SchedulerFlags.stratum_gc = False
        actual_results = self.run_application(sched_par="threading")
        # Convert to pandas for comparison
        # TODO: pre:2 is non-deterministic right now, so we need to filter it out
        filter_expr = pl.col("id").str.contains("pre:1") & ~pl.col("id").str.contains("xgb|lgbm")
        actual_df = actual_results.sort("id").filter(filter_expr).to_pandas()
        expected_df = self.expected_results.sort("id").filter(filter_expr).to_pandas()
        print(actual_df)
        pd.testing.assert_frame_equal(
            actual_df,
            expected_df,
            atol=1e-6,
            check_dtype=False
        )
        SchedulerFlags.stratum_gc = True
    
    def test_application_process(self):
        SchedulerFlags.stratum_gc = False
        actual_results = self.run_application(sched_par="process")
        # Convert to pandas for comparison
        # TODO: pre:2 is non-deterministic right now, so we need to filter it out
        filter_expr = pl.col("id").str.contains("pre:1") & ~pl.col("id").str.contains("xgb|lgbm")
        actual_df = actual_results.sort("id").filter(filter_expr).to_pandas()
        expected_df = self.expected_results.sort("id").filter(filter_expr).to_pandas()
        print(actual_df)
        pd.testing.assert_frame_equal(
            actual_df,
            expected_df,
            atol=1e-6,
            check_dtype=False
        )
        SchedulerFlags.stratum_gc = True

    def test_application_auto(self):
        SchedulerFlags.stratum_gc = False
        actual_results = self.run_application(sched_par="auto")
        # Convert to pandas for comparison
        # TODO: pre:2 is non-deterministic right now, so we need to filter it out
        filter_expr = pl.col("id").str.contains("pre:1") & ~pl.col("id").str.contains("xgb|lgbm")
        actual_df = actual_results.sort("id").filter(filter_expr).to_pandas()
        expected_df = self.expected_results.sort("id").filter(filter_expr).to_pandas()
        print(actual_df)
        pd.testing.assert_frame_equal(
            actual_df,
            expected_df,
            atol=1e-6,
            check_dtype=False
        )
        SchedulerFlags.stratum_gc = True
if __name__ == "__main__":
    unittest.main()