"""Execution of an Apply's per-method kwargs.

skrub keys ``.skb.apply()``'s kwargs by the method they belong to and evaluates
only the group for the method it is about to call. These tests pin both halves of
that: the kwargs reach the right call, and the fit group is left alone in predict
mode.
"""
import unittest

import numpy as np
import pandas as pd
import stratum as st
from lightgbm import LGBMRegressor, early_stopping
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from stratum._api import grid_search


class RecordingTransformer(TransformerMixin, BaseEstimator):
    """Records the kwargs each call receives.

    The log lives on the class because the op fits a clone of the estimator it was
    handed, so per-instance state would not come back to the test.
    """

    CALLS: list = []

    def fit_transform(self, X, y=None, **kwargs):
        self.CALLS.append(("fit_transform", kwargs))
        return X

    def fit(self, X, y=None, **kwargs):
        self.CALLS.append(("fit", kwargs))
        return self

    def transform(self, X, **kwargs):
        self.CALLS.append(("transform", kwargs))
        return X


class SplitOffEvalSet(TransformerMixin, BaseEstimator):
    """Carve an early-stopping set out of the training fold.

    In predict mode there is no eval set to carve, so the extra keys are None:
    the fit kwargs that read them are not evaluated in that mode anyway.
    """

    def fit(self, X, y):
        return self

    def fit_transform(self, X, y):
        X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.25, random_state=0)
        return {"X": X_fit, "X_val": X_val, "y": y_fit, "y_val": y_val}

    def transform(self, X):
        return {"X": X, "X_val": None, "y": None, "y_val": None}


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["w"] = np.where(df["a"] > 0, 100.0, 0.01)  # weights that visibly bend the fit
    df["y"] = 2.0 * df["a"] + df["b"] + rng.normal(scale=0.1, size=n)
    return df


class ApplyKwargsRuntimeTest(unittest.TestCase):
    SCORING = "neg_root_mean_squared_error"

    def setUp(self):
        self.enterContext(st.config_context(eager_data_ops=False))
        self.df = _frame()

    def assert_matches_skrub(self, pipeline, cv=2):
        """Both engines must fit the same model, so their fold scores must agree.

        stratum reports the raw scorer value, skrub the signed one.
        """
        ours = grid_search(pipeline, cv=cv, scoring=self.SCORING)
        theirs = pipeline.skb.make_grid_search(cv=cv, fitted=True, refit=False,
                                               scoring=self.SCORING)
        np.testing.assert_allclose(theirs.results_["mean_test_score"] * -1,
                                   ours.results_["scores"], rtol=1e-9)


class TestFitKwargsReachFit(ApplyKwargsRuntimeTest):
    def _weighted_pipeline(self, with_weights):
        data = st.as_data_op(self.df)
        y = data["y"].skb.mark_as_y()
        X = data.drop(columns=["y"]).skb.mark_as_X()
        weights = X["w"]
        fit_kwargs = {"sample_weight": weights} if with_weights else None
        return X.drop(columns=["w"]).skb.apply(
            LinearRegression(), y=y, fit_kwargs=fit_kwargs)

    def test_sample_weight_changes_the_fitted_model(self):
        # The kwargs used to be dropped, which left both variants identical.
        weighted = grid_search(self._weighted_pipeline(True), cv=2, scoring=self.SCORING)
        plain = grid_search(self._weighted_pipeline(False), cv=2, scoring=self.SCORING)
        self.assertNotAlmostEqual(weighted.results_["scores"][0],
                                  plain.results_["scores"][0], places=6)

    def test_matches_skrub_for_a_graph_fed_fit_kwarg(self):
        self.assert_matches_skrub(self._weighted_pipeline(True))

    def test_unknown_fit_kwarg_reaches_the_estimator(self):
        # Proves the kwargs are splatted rather than swallowed somewhere.
        data = st.as_data_op(self.df)
        y = data["y"].skb.mark_as_y()
        X = data.drop(columns=["y", "w"]).skb.mark_as_X()
        pipeline = X.skb.apply(LinearRegression(), y=y, fit_kwargs={"nonexistent": 1})
        with self.assertRaises(RuntimeError) as ctx:
            grid_search(pipeline, cv=2)
        self.assertIn("unexpected keyword argument 'nonexistent'", str(ctx.exception))


class TestKwargsAreRoutedPerMethod(ApplyKwargsRuntimeTest):
    def setUp(self):
        super().setUp()
        RecordingTransformer.CALLS = []

    def _run_recording_pipeline(self, **apply_kwargs):
        data = st.as_data_op(self.df)
        y = data["y"].skb.mark_as_y()
        X = data.drop(columns=["y", "w"]).skb.mark_as_X()
        transformed = X.skb.apply(RecordingTransformer(), how="no_wrap", **apply_kwargs)
        grid_search(transformed.skb.apply(LinearRegression(), y=y), cv=2)
        return dict(RecordingTransformer.CALLS)

    def test_transform_kwargs_only_reach_transform(self):
        calls = self._run_recording_pipeline(transform_kwargs={"flag": "t"})
        self.assertEqual(calls["fit_transform"], {})
        self.assertEqual(calls["transform"], {"flag": "t"})

    def test_fit_transform_kwargs_only_reach_fit_transform(self):
        calls = self._run_recording_pipeline(fit_transform_kwargs={"flag": "ft"})
        self.assertEqual(calls["fit_transform"], {"flag": "ft"})
        self.assertEqual(calls["transform"], {})


class TestEvalSetPipeline(ApplyKwargsRuntimeTest):
    """The reported case: an early-stopping eval set built inside the pipeline."""

    def _pipeline(self):
        data = st.as_data_op(self.df)
        y = data["y"].skb.mark_as_y()
        X = data.drop(columns=["y", "w"]).skb.mark_as_X()

        parts = X.skb.apply(SplitOffEvalSet(), y=y, how="no_wrap")
        model = LGBMRegressor(n_estimators=200, learning_rate=0.1,
                              random_state=0, verbose=-1)
        return parts["X"].skb.apply(
            model,
            y=parts["y"],
            fit_kwargs={"eval_set": [(parts["X_val"], parts["y_val"])],
                        "callbacks": [early_stopping(stopping_rounds=5, verbose=False)]},
        )

    def test_runs_and_matches_skrub(self):
        self.assert_matches_skrub(self._pipeline())

    def test_early_stopping_actually_triggers(self):
        # If the eval set never arrived, LightGBM would run all 200 rounds.
        pipeline = self._pipeline()
        sched = grid_search(pipeline, cv=2, scoring=self.SCORING)
        estimators = [op.estimator for op in sched.linearized_dag
                      if isinstance(getattr(op, "estimator", None), LGBMRegressor)]
        self.assertTrue(estimators)
        for est in estimators:
            self.assertLess(est.best_iteration_, 200)


if __name__ == "__main__":
    unittest.main()
