from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from stratum import config
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
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

class CountingKFold(KFold):
    """KFold that records how often it was asked for splits."""

    def __init__(self, n_splits=3):
        super().__init__(n_splits=n_splits)
        self.calls = 0

    def split(self, X, y=None, groups=None):
        self.calls += 1
        return super().split(X, y, groups)


def _classification_frame(n=90, n_classes=3):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "t": rng.integers(0, n_classes, size=n),
    })


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
        # Header exposes the runtime-distribution column.
        self.assertIn("%", out[4])
        # Row: Op, Count, Time, %  (the lambda sleeps 10x so it dominates).
        self.assertIn("CallOp(<lambda>)", out[5])
        fields = out[5].split()
        self.assertEqual(fields[1], "10")          # invocation count
        self.assertTrue(fields[-1].endswith("%"))  # share of total runtime


    def test_fused_attr(self):
        data = st.as_data_op(self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        date = X["datetime"].skb.apply_func(pd.to_datetime, format="%Y-%m-%d %H:%M:%S")
        X = X.assign(year=date.dt.year)
        X = X.drop(columns=["datetime"])
        pred = X.skb.apply(DummyRegressor(), y=y)
        st._api.grid_search(pred)


class CrossValidationSplitterTest(unittest.TestCase):
    """Regression tests for issue #199."""

    def setUp(self):
        self.df = _classification_frame()

    def _classification_pipeline(self, **mark_as_x_kwargs):
        data = st.as_data_op(self.df)
        y = data["t"].skb.mark_as_y()
        X = data[["a", "b"]].skb.mark_as_X(**mark_as_x_kwargs)
        return X.skb.apply(DummyClassifier(strategy="most_frequent"), y=y)

    def test_stratified_cv_gets_y(self):
        """A stratified splitter needs the labels; passing X only raised TypeError."""
        cv = StratifiedKFold(n_splits=3)
        sched = st._api.grid_search(self._classification_pipeline(),
                                    cv=cv, scoring="accuracy")

        search = self._classification_pipeline().skb.make_grid_search(
            cv=StratifiedKFold(n_splits=3), fitted=True, scoring="accuracy")
        assert np.allclose(search.results_["mean_test_score"], sched.results_["scores"])

    def test_declared_cv_drives_the_folds(self):
        """`mark_as_X(cv=...)` used to be ignored in favour of check_cv(None)."""
        cv = CountingKFold(n_splits=3)
        pred = self._classification_pipeline(cv=cv, split_kwargs={})
        with config(scheduler=True):
            pred.skb.make_grid_search(scoring="accuracy")
        self.assertGreater(cv.calls, 0)

    def test_declared_cv_without_split_kwargs(self):
        """`split_kwargs` defaults to None and must not reach the splitter as **None."""
        cv = CountingKFold(n_splits=3)
        pred = self._classification_pipeline(cv=cv)
        with config(scheduler=True):
            pred.skb.make_grid_search(scoring="accuracy")
        self.assertGreater(cv.calls, 0)

    def test_explicit_cv_overrides_declared_cv(self):
        """skrub's precedence: an explicit splitter wins over the declared one."""
        declared = CountingKFold(n_splits=3)
        pred = self._classification_pipeline(cv=declared, split_kwargs={})
        st._api.grid_search(pred, cv=StratifiedKFold(n_splits=2), scoring="accuracy")
        self.assertEqual(declared.calls, 0)

    def test_declared_split_kwargs_reach_the_splitter(self):
        """`groups` for GroupKFold travels through split_kwargs."""
        groups = np.arange(len(self.df)) % 3
        pred = self._classification_pipeline(cv=GroupKFold(n_splits=3),
                                             split_kwargs={"groups": groups})
        sched = st._api.grid_search(pred, scoring="accuracy")

        search = self._classification_pipeline(
            cv=GroupKFold(n_splits=3), split_kwargs={"groups": groups},
        ).skb.make_grid_search(fitted=True, scoring="accuracy")
        assert np.allclose(search.results_["mean_test_score"], sched.results_["scores"])


if __name__ == "__main__":
    unittest.main()
