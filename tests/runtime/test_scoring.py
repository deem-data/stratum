"""Regression tests for issue #200: `scoring=` must be honoured, not reinvented.

The scheduler used to unwrap a scorer's `_score_func` and fall back to
`mean_squared_error` for anything it did not recognise, so a callable scorer, a
`make_scorer` with kwargs, a `neg_*` sign and `scoring=None` all came out wrong -- some
of them silently. Every test here pins one score against what plain skrub reports for
the same plan.
"""
import unittest

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, fbeta_score, make_scorer
from sklearn.model_selection import KFold, StratifiedKFold

import stratum as st


def _classification_frame(n=150):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["t"] = (df["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return df


def _regression_frame(n=150):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["t"] = 2.0 * df["a"] - df["b"] + rng.normal(scale=0.1, size=n)
    return df


def _my_accuracy(y_true, y_pred):
    """A metric as Stratum defines one: a pure function of labels and values."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def _sklearn_style_scorer(estimator, X, y_true):
    """What scikit-learn calls a scorer. Stratum has no estimator to pass it."""
    return accuracy_score(y_true, estimator.predict(X))


class ScoringTest(unittest.TestCase):
    def setUp(self):
        self.clf_df = _classification_frame()
        self.reg_df = _regression_frame()

    def _classification_pipeline(self, estimator=None, cv=None):
        cv = cv if cv is not None else StratifiedKFold(n_splits=3)
        data = st.as_data_op(self.clf_df)
        y = data["t"].skb.mark_as_y()
        X = data[["a", "b"]].skb.mark_as_X(cv=cv, split_kwargs={})
        estimator = estimator or RandomForestClassifier(n_estimators=5, random_state=0)
        return X.skb.apply(estimator, y=y)

    def _mixed_kind_pipeline(self):
        """A batch whose candidates are not the same kind of estimator (ADR 0004)."""
        return self._classification_pipeline(estimator=st.choose_from(
            {"clf": RandomForestClassifier(n_estimators=5, random_state=0),
             "reg": DummyRegressor(strategy="mean")}, name="model"))

    def _regression_pipeline(self):
        data = st.as_data_op(self.reg_df)
        y = data["t"].skb.mark_as_y()
        X = data[["a", "b"]].skb.mark_as_X(cv=KFold(n_splits=3), split_kwargs={})
        return X.skb.apply(LinearRegression(), y=y)

    def assert_matches_skrub(self, make_pipeline, scoring):
        """The same plan, scored the same way, must report the same numbers as skrub."""
        ours = st._api.grid_search(make_pipeline(), scoring=scoring)
        theirs = make_pipeline().skb.make_grid_search(fitted=True, scoring=scoring)
        np.testing.assert_allclose(theirs.results_["mean_test_score"],
                                   ours.results_["scores"], rtol=1e-9)
        return ours.results_["scores"].to_list()

    def test_string_scorer(self):
        self.assert_matches_skrub(self._classification_pipeline, "accuracy")

    def test_user_metric_is_called(self):
        """A callable is a metric, `f(y_true, y_pred)`, and gets the plan's values."""
        ours = st._api.grid_search(self._classification_pipeline(), scoring=_my_accuracy)
        theirs = st._api.grid_search(self._classification_pipeline(), scoring="accuracy")
        np.testing.assert_allclose(ours.results_["scores"], theirs.results_["scores"])

    def test_sklearn_style_scorer_is_refused(self):
        """`f(estimator, X, y)` needs a fitted estimator; a candidate is a sub-DAG."""
        with self.assertRaisesRegex(ValueError, "no estimator to pass one"):
            st._api.grid_search(self._classification_pipeline(),
                                scoring=_sklearn_style_scorer)

    def test_scorer_kwargs_are_applied(self):
        """`make_scorer` kwargs used to be dropped, which raised or silently changed
        the metric's defaults."""
        beta2 = self.assert_matches_skrub(self._classification_pipeline,
                                          make_scorer(fbeta_score, beta=2))
        beta1 = st._api.grid_search(self._classification_pipeline(),
                                    scoring=make_scorer(fbeta_score, beta=1))
        self.assertNotAlmostEqual(beta2[0], beta1.results_["scores"][0], places=6)

    def test_neg_scorer_keeps_its_sign(self):
        """`_sign` was dropped, so every `neg_*` metric was reported positive."""
        scores = self.assert_matches_skrub(self._regression_pipeline,
                                           "neg_root_mean_squared_error")
        self.assertLess(scores[0], 0)

    def test_scoring_is_required(self):
        """`scoring=None` would measure each candidate with its own estimator's `score`,
        so a batch mixing estimator kinds would rank an accuracy against an R²."""
        for make in (self._regression_pipeline, self._mixed_kind_pipeline):
            with self.assertRaisesRegex(ValueError, "requires scoring="):
                st._api.grid_search(make(), scoring=None)

    def test_make_grid_search_without_scoring_falls_back_to_skrub(self):
        """ADR 0002: a call real skrub accepts must not start failing under Stratum."""
        with st.config(scheduler=True):
            search = self._regression_pipeline().skb.make_grid_search(fitted=True)
        # skrub's own results table, so the call was handed back rather than refused.
        self.assertIn("mean_test_score", list(search.results_.columns))

    def test_probability_scorer_reaches_predict_proba(self):
        """A scorer that needs more than `predict` gets it from the fitted estimator."""
        self.assert_matches_skrub(self._classification_pipeline, "roc_auc")
        self.assert_matches_skrub(self._classification_pipeline, "neg_log_loss")

    def test_candidates_are_ranked_best_first(self):
        """A scorer returns a utility whatever its metric, so the ranking is by value."""
        def make_pipeline():
            return self._classification_pipeline(estimator=st.choose_from(
                {"forest": RandomForestClassifier(n_estimators=5, random_state=0),
                 "constant": DummyClassifier(strategy="constant", constant=1)},
                name="model"))

        scores = self.assert_matches_skrub(make_pipeline, "neg_log_loss")
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_multimetric_scoring_is_rejected(self):
        """One score per candidate is all the results table holds; say so."""
        with self.assertRaisesRegex(ValueError, "several metrics"):
            st._api.grid_search(self._classification_pipeline(),
                                scoring=["accuracy", "roc_auc"])

    def test_unknown_scorer_raises(self):
        with self.assertRaises(Exception) as caught:
            st._api.grid_search(self._classification_pipeline(), scoring="not_a_metric")
        self.assertIn("not_a_metric", str(caught.exception))

    def test_bare_metric_function_is_accepted(self):
        """`accuracy_score(y_true, y_pred)` is exactly what Stratum scores with. sklearn
        rejects it as "not a scorer"; here it is the plain case."""
        ours = st._api.grid_search(self._classification_pipeline(), scoring=accuracy_score)
        theirs = st._api.grid_search(self._classification_pipeline(), scoring="accuracy")
        np.testing.assert_allclose(ours.results_["scores"], theirs.results_["scores"])

    def test_probability_metric_reaches_a_post_processed_plan(self):
        """The pass runs the response the scorer asked for, so a tail after the
        predictor sees probabilities and the metric is computed on what the plan
        produced. This is what skrub does, and used to raise here."""
        self.assert_matches_skrub(
            lambda: self._classification_pipeline().skb.apply_func(np.asarray), "roc_auc")

    def test_metric_no_candidate_can_feed_is_refused_before_fitting(self):
        """A probability metric over a regressor is refused when the plan is built, not
        after a fold has been fitted."""
        with self.assertRaisesRegex(ValueError, "no single response provides"):
            st._api.grid_search(self._regression_pipeline(), scoring="neg_log_loss")

    def test_native_metrics_match_sklearn(self):
        """Equivalence for every metric Stratum computes itself. Moving a metric into
        `_NATIVE` is only safe while this holds."""
        from stratum.optimizer.logical._scoring import _NATIVE, _decompose
        from sklearn.metrics import get_scorer

        for name, mine in sorted(_NATIVE.items()):
            with self.subTest(metric=name):
                theirs = _decompose(get_scorer(name), name=name)
                pipeline = (self._classification_pipeline if name == "accuracy"
                            else self._regression_pipeline)
                a = st._api.grid_search(pipeline(), scoring=name)
                self.assertEqual(mine.source, "stratum")
                b = st._api.grid_search(pipeline(), scoring=theirs.fn)
                np.testing.assert_allclose(a.results_["scores"],
                                           [theirs.sign * v for v in b.results_["scores"]],
                                           rtol=1e-12)

    def test_return_predictions_refuses_a_non_predict_metric(self):
        """The values a `roc_auc` pass produces are probabilities, not predictions."""
        with self.assertRaisesRegex(ValueError, "responses, not predictions"):
            st._api.grid_search(self._classification_pipeline(), scoring="roc_auc",
                                return_predictions=True)

    def test_score_is_taken_after_post_processing(self):
        """The score measures what the plan produced, not what the predictor emitted.

        The tail flips every label, so the plan's accuracy is `1 -` the predictor's:
        scoring at the wrong point in the plan is visible in the number.
        """
        flipped = self._classification_pipeline().skb.apply_func(lambda p: 1 - np.asarray(p))
        at_model = st._api.grid_search(self._classification_pipeline(), scoring="accuracy")
        after_tail = st._api.grid_search(flipped, scoring="accuracy")
        self.assertAlmostEqual(after_tail.results_["scores"][0],
                               1 - at_model.results_["scores"][0], places=9)


    def _stacked_pipeline(self, inner):
        """`inner` feeds a second estimator, so it is not the last step of the plan."""
        data = st.as_data_op(self.clf_df)
        y = data["t"].skb.mark_as_y()
        X = data[["a", "b"]].skb.mark_as_X(cv=StratifiedKFold(n_splits=3), split_kwargs={})
        features = X.skb.apply(inner, y=y).skb.apply_func(
            lambda p: pd.DataFrame(np.asarray(p).reshape(len(p), -1)).add_prefix("f"))
        return features.skb.apply(LogisticRegression(), y=y)

    def test_inner_predictor_predicts_in_a_probability_pass(self):
        """Only the last estimator answers the scorer: one feeding another predicts in
        every pass, so its test-fold features are labels as they were while fitting."""
        self.assert_matches_skrub(
            lambda: self._stacked_pipeline(LogisticRegression()), "roc_auc")

    def test_inner_estimator_with_transform_transforms(self):
        """An inner estimator that can transform does so, in skrub: a KMeans hands on
        its distances, not its labels."""
        self.assert_matches_skrub(
            lambda: self._stacked_pipeline(KMeans(n_clusters=3, n_init=1, random_state=0)),
            "roc_auc")


    # --- metrics declared with `.skb.with_scoring()` ---------------------------------

    def _declared_pipeline(self, scoring, sample_weight=None):
        """A two-candidate search whose metric the DataOp declares itself.

        ``sample_weight(X, y)`` builds the scorer's kwargs from the plan's own data, so
        they are computed per test fold.
        """
        data = st.as_data_op(self.clf_df)
        y = data["t"].skb.mark_as_y()
        X = data[["a", "b"]].skb.mark_as_X(cv=StratifiedKFold(n_splits=3), split_kwargs={})
        pred = X.skb.apply(LogisticRegression(C=st.choose_from([0.01, 1.0], name="C")), y=y)
        kwargs = None if sample_weight is None else {"sample_weight": sample_weight(X, y)}
        return pred.skb.with_scoring(scoring, kwargs=kwargs)

    def assert_declared_matches_skrub(self, make_pipeline):
        ours = st._api.grid_search(make_pipeline())
        theirs = make_pipeline().skb.make_grid_search(fitted=True)
        np.testing.assert_allclose(theirs.results_["mean_test_score"],
                                   ours.results_["scores"], rtol=1e-9)

    def test_declared_metric_is_used(self):
        self.assert_declared_matches_skrub(lambda: self._declared_pipeline("accuracy"))

    def test_declared_kwargs_are_computed_on_each_test_fold(self):
        weights = lambda X, y: X["a"].abs()
        for scoring in ("accuracy", "roc_auc", "neg_log_loss"):
            with self.subTest(scoring=scoring):
                self.assert_declared_matches_skrub(
                    lambda: self._declared_pipeline(scoring, weights))

    def test_declared_kwargs_follow_a_choice_per_candidate(self):
        """A choice in the kwargs is a candidate dimension of its own: each candidate is
        scored with the weights of its own branch."""
        def make():
            data = st.as_data_op(self.clf_df)
            y = data["t"].skb.mark_as_y()
            X = data[["a", "b"]].skb.mark_as_X(cv=StratifiedKFold(n_splits=3),
                                               split_kwargs={})
            column = st.as_data_op(st.choose_from(["a", "b"], name="weights"))
            return X.skb.apply(LogisticRegression(), y=y).skb.with_scoring(
                "accuracy", kwargs={"sample_weight": X[column].abs()})
        self.assert_declared_matches_skrub(make)

    def test_explicit_scoring_overrides_the_declared_one(self):
        """As in skrub: the declared metric and its kwargs are ignored."""
        make = lambda: self._declared_pipeline("accuracy", lambda X, y: X["a"].abs())
        ours = st._api.grid_search(make(), scoring="roc_auc")
        theirs = make().skb.make_grid_search(fitted=True, scoring="roc_auc")
        np.testing.assert_allclose(theirs.results_["mean_test_score"],
                                   ours.results_["scores"], rtol=1e-9)

    def test_declarations_stratum_cannot_score_with_are_refused(self):
        cases = {
            "2 scorers": self._declared_pipeline("accuracy").skb.with_scoring("roc_auc"),
            # The estimator's own `score`, which a search never uses (ADR 0004).
            "would score each candidate with its own": self._declared_pipeline(None),
        }
        for message, dag in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                st._api.grid_search(dag)

    def test_make_grid_search_uses_the_declared_metric(self):
        with st.config(scheduler=True):
            search = self._declared_pipeline(
                "accuracy", lambda X, y: X["a"].abs()).skb.make_grid_search(fitted=True)
        # Stratum's results table, not skrub's.
        self.assertIn("scores", list(search.results_.columns))

    def test_make_grid_search_hands_an_unusable_declaration_to_skrub(self):
        with st.config(scheduler=True):
            search = self._declared_pipeline(None).skb.make_grid_search(fitted=True)
        self.assertIn("mean_test_score", list(search.results_.columns))

    def test_a_declared_metric_does_not_change_evaluate(self):
        """Without a search the declaration is dropped, kwargs subgraph included."""
        plain = st._api.evaluate(self._declared_pipeline("accuracy"))
        declared = st._api.evaluate(
            self._declared_pipeline("accuracy", lambda X, y: X["a"].abs()))
        np.testing.assert_array_equal(np.asarray(plain), np.asarray(declared))


if __name__ == "__main__":
    unittest.main()
