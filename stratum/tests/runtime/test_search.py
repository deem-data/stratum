from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from stratum import config
from sklearn.model_selection import KFold
from stratum.tests.runtime.runtime_test_utils import RuntimeTest, datetime_pipeline1, datetime_pipeline2
from contextlib import redirect_stdout
from io import StringIO
import time
import unittest
import pandas as pd
import numpy as np
import stratum as st
import logging

logging.basicConfig(level=logging.INFO)

class InputCheckEstimator(BaseEstimator):
    def fit(self, X, y):
        self.cols = X.columns
        self.my_id = f"train {time.time()}"
        return self
    def predict(self, X):
        if not set(X.columns) == set(self.cols):
            raise ValueError(f"Columns mismatch: {set(X.columns)} != {set(self.cols)}")
        return X[self.cols[0]]

class SearchTest(RuntimeTest):
    def test_search(self):
        data = st.as_data_op(self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        y1 = datetime_pipeline1(X, y)
        y2 = datetime_pipeline2(X, y)
        y = st.choose_from({"pipeline 1": y1, "pipeline 2": y2}).as_data_op()

        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        search_stratum, preds = st._api.grid_search(y, cv=cv, scoring="neg_mean_squared_error", return_predictions=True)

        search = y.skb.make_grid_search(cv=cv, fitted=True,scoring="neg_mean_squared_error")
        assert(np.allclose(search.results_["mean_test_score"]*-1, search_stratum.results_["scores"]))



    def test_search_with_no_X(self):
        start = st.as_data_op(True)
        end = start.skb.apply_func(lambda a: a).skb.mark_as_y()

        try:
            with st.config(stats=True):
                st._api.grid_search(end, return_predictions=True)
            self.fail("Expected RuntimeError")
        except RuntimeError as e:
            self.assertEqual("X and y nodes not found in the DAG",str(e))

    def test_search_with_no_y(self):
        start = st.as_data_op(True)
        end = start.skb.apply_func(lambda a: a).skb.mark_as_X()

        try:
            with st.config(stats=True):
                st._api.grid_search(end, return_predictions=True)
            self.fail("Expected RuntimeError")
        except RuntimeError as e:
            self.assertEqual("X and y nodes not found in the DAG",str(e))


    def test_search_choice_not_at_the_end1(self):
        data = st.as_data_op(self.df)
        X = data[["x"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        X = X + st.choose_from([0,1]).as_data_op()
        pred = X.skb.apply(DummyRegressor(), y=y)
        st._api.grid_search(pred)

    def test_search_choice_not_at_the_end2(self):
        data = st.as_data_op(self.df)
        X = data[["x"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        X1 = X.assign(x_a= X["x"] + 1)
        X2 = X.assign(x_b = X["x"] - 1)
        X = 4 + st.choose_from([X1,X2]).as_data_op()
        pred = X.skb.apply(DummyRegressor(), y=y)
        with config(scheduler=True):
            pred.skb.make_grid_search()

    def test_search_choice_not_at_the_end3(self):
        data = st.as_data_op(self.df)
        X = data[["x"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        X1 = X.assign(x_a= X["x"] + 1)
        X2 = X.assign(x_b = X["x"] - 1)
        X = 4 + st.choose_from([X1,X2]).as_data_op()
        pred = X.skb.apply(InputCheckEstimator(), y=y)
        st._api.grid_search(pred)

    def test_search_error_during_dataop_processing(self):
        data = st.as_data_op(self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        y = y.skb.apply_func(lambda a, m: (a, print(m))[0] if m != 'predict' else int("grr"), st.eval_mode())
        pred = X.skb.apply(DummyRegressor(), y=y)
        try:
            st._api.grid_search(pred)
            self.fail("Expected RunTimeError")
        except RuntimeError as e:
            self.assertTrue(e.args[0].startswith("[predict] Error processing 'CallOp(<lambda>)': invalid literal for int() with base 10: 'grr'"))



    def test_search_with_stats(self):
        data = st.as_data_op(self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        X2 = X.skb.apply_func(lambda a: (a, time.sleep(0.01))[0])
        pred = X2.skb.apply(DummyRegressor(), y=y)
        # capture stdout
        with redirect_stdout(StringIO()) as stdout, st.config(stats=True, stats_top_k=20):
            st._api.grid_search(pred, return_predictions=False)
        out = stdout.getvalue()
        out = out.split("\n")
        self.assertIn("Heavy hitters", out[2])
        self.assertIn("CallOp(<lambda>)", out[5])
        assert(out[5].split(" ")[-1] == "10")


    def test_fused_attr(self):
        data = st.as_data_op(self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        date = X["datetime"].skb.apply_func(pd.to_datetime, format="%Y-%m-%d %H:%M:%S")
        X = X.assign(year=date.dt.year)
        X = X.drop(columns=["datetime"])
        pred = X.skb.apply(DummyRegressor(), y=y)
        st._api.grid_search(pred)

if __name__ == "__main__":
    unittest.main()
