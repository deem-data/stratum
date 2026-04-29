import os
import tempfile
import unittest
import uuid

from skrub import TableVectorizer
import stratum as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, r2_score


class TargetEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.global_mean_ = y.mean()
        tmp = pd.concat([X, y], axis=1)
        self.cols = X.columns
        self.means = {}
        for col in self.cols:
            self.means[col] = tmp.groupby(col)[tmp.columns[-1]].mean()
        return self

    def transform(self, X):
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
    df = st.as_data_op(file_path).skb.apply_func(pd.read_csv)
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

        X_cat = X.skb.select(~st.selectors.numeric())
        X_cat_enc = X_cat.skb.apply(st.StringEncoder())

        X_te = X[["District", "County", "Town"]].skb.apply(TargetEncoder(), y=y)
        X_te = X_te.rename(columns={"District": "district_te", "County": "county_te", "Town": "town_te"})
        X_num = X.skb.select(st.selectors.numeric())
        X_vec = X_num.skb.concat([X_te, X_cat_enc], axis=1)

        return X_vec

    def df2(X):
        return X.skb.apply(TableVectorizer())

    X_vec = st.choose_from({"1": df1(X, y), "2": df2(X)}, name ="data engineering").as_data_op()
    models = {
        "Ridge": Ridge(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "LightGBM": LGBMRegressor(random_state=42),
        "ElasticNet": ElasticNet(random_state=42),
    }
    preds = {name: X_vec.skb.apply(m, y=y) for name, m in models.items()}
    return st.choose_from(preds, name="models").as_data_op()

def make_data(n: int = 1000):
    df = pd.DataFrame({
        "Transaction unique identifier": [str(uuid.uuid4()) for _ in range(n)],
        "Price": np.random.randint(50000, 2_000_000, size=n),
        "Date of Transfer": pd.to_datetime(
            np.random.choice(pd.date_range("2010-01-01", "2024-12-31"), size=n)
        ).astype(str),
        "Property Type": np.random.choice(list("DSTFO"), size=n),
        "Old/New": np.random.choice(["Y", "N"], size=n),
        "Duration": np.random.choice(["F", "L"], size=n),
        "Town/City": np.random.choice(
            ["London", "Manchester", "Birmingham", "Leeds", "Bristol"], size=n
        ),
        "District": np.random.choice(
            ["District A", "District B", "District C"], size=n
        ),
        "County": np.random.choice(
            ["Greater London", "West Midlands", "Greater Manchester"], size=n
        ),
        "PPDCategory Type": np.random.choice(["A", "B"], size=n),
        "Record Status - monthly file only": np.random.choice(["A", "C"], size=n),
    })
    return df

class TestMultiLevelChoiceGraph(unittest.TestCase):

    def test_application(self):
        tmp_path = tempfile.mkdtemp()
        df = make_data()
        df.to_csv(os.path.join(tmp_path, "data.csv"), index=False)
        preds = define_pipeline(os.path.join(tmp_path, "data.csv"))
        scorer = make_scorer(r2_score)
        with st.config(DEBUG=True, open_graph=False, scheduler=True, rust_backend=False):
            search = preds.skb.make_grid_search(fitted=True, cv = 2, scoring=scorer)
            self.assertIsNotNone(search.results_)
            self.assertGreater(len(search.results_), 0)
