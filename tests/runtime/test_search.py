from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from stratum import config
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from tests.runtime.runtime_test_utils import RuntimeTest, datetime_pipeline1, datetime_pipeline2
from contextlib import redirect_stdout
from io import StringIO
import time
import unittest
import pandas as pd
import numpy as np
import pytest
import stratum as st
import logging
import re

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
        assert(np.allclose(search.results_["mean_test_score"], search_stratum.results_["scores"]))



    def test_search_with_no_X(self):
        start = st.as_data_op(True)
        end = start.skb.apply_func(lambda a: a).skb.mark_as_y()

        try:
            with st.config(stats=True):
                st._api.grid_search(end, scoring="neg_mean_squared_error", return_predictions=True)
            self.fail("Expected RuntimeError")
        except RuntimeError as e:
            self.assertEqual("X and y nodes not found in the DAG",str(e))

    def test_search_with_no_y(self):
        start = st.as_data_op(True)
        end = start.skb.apply_func(lambda a: a).skb.mark_as_X()

        try:
            with st.config(stats=True):
                st._api.grid_search(end, scoring="neg_mean_squared_error", return_predictions=True)
            self.fail("Expected RuntimeError")
        except RuntimeError as e:
            self.assertEqual("X and y nodes not found in the DAG",str(e))


    def test_search_choice_not_at_the_end1(self):
        data = st.as_data_op(self.df)
        X = data[["x"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        X = X + st.choose_from([0,1]).as_data_op()
        pred = X.skb.apply(DummyRegressor(), y=y)
        st._api.grid_search(pred, scoring="neg_mean_squared_error")

    def test_search_choice_not_at_the_end2(self):
        data = st.as_data_op(self.df)
        X = data[["x"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        X1 = X.assign(x_a= X["x"] + 1)
        X2 = X.assign(x_b = X["x"] - 1)
        X = 4 + st.choose_from([X1,X2]).as_data_op()
        pred = X.skb.apply(DummyRegressor(), y=y)
        with config(scheduler=True):
            pred.skb.make_grid_search(scoring="neg_mean_squared_error")

    def test_search_choice_not_at_the_end3(self):
        data = st.as_data_op(self.df)
        X = data[["x"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        X1 = X.assign(x_a= X["x"] + 1)
        X2 = X.assign(x_b = X["x"] - 1)
        X = 4 + st.choose_from([X1,X2]).as_data_op()
        pred = X.skb.apply(InputCheckEstimator(), y=y)
        # InputCheckEstimator has no `score`, so (as under skrub) the metric has to be
        # named: `scoring=None` would have nothing to fall back on.
        st._api.grid_search(pred, scoring="neg_mean_squared_error")

    def test_search_error_during_dataop_processing(self):
        data = st.as_data_op(self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        y = y.skb.apply_func(lambda a, m: (a, print(m))[0] if m != 'predict' else int("grr"), st.eval_mode())
        pred = X.skb.apply(DummyRegressor(), y=y)
        try:
            st._api.grid_search(pred, scoring="neg_mean_squared_error")
            self.fail("Expected RunTimeError")
        except RuntimeError as e:
            self.assertTrue(e.args[0].startswith("[predict] Error processing 'CallExec(<lambda>)': invalid literal for int() with base 10: 'grr'"))



    def test_search_with_stats(self):
        data = st.as_data_op(self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()

        X2 = X.skb.apply_func(lambda a: (a, time.sleep(0.01))[0])
        pred = X2.skb.apply(DummyRegressor(), y=y)
        # capture stdout
        with redirect_stdout(StringIO()) as stdout, st.config(stats=True, stats_top_k=1):
            st._api.grid_search(pred, scoring="neg_mean_squared_error", return_predictions=False)
        out = stdout.getvalue()
        def seconds(label):
            match = re.search(rf"^\s*{label}:\s+([\d.]+)$", out, re.MULTILINE)
            self.assertIsNotNone(match, label)
            return float(match.group(1))

        self.assertAlmostEqual(
            seconds("Total"), seconds("Optimization") + seconds("Execution"),
            delta=0.0002,
        )
        self.assertGreater(seconds("Optimization"), 0)
        self.assertGreater(seconds("Unshown operators"), 0)
        self.assertIn("Execution Statistics (seconds)", out)
        self.assertIn("share of all operator processing time", out)
        self.assertIn("serialize/deserialize times are included in Execution", out)
        self.assertIn("%", out.split("Heavy hitters")[1])
        row = next(line for line in out.splitlines() if "CallExec(<lambda>)" in line)
        fields = row.split()
        self.assertEqual(fields[1], "10")          # invocation count
        self.assertTrue(fields[-1].endswith("%"))


    def test_fused_attr(self):
        data = st.as_data_op(self.df)
        X = data[["x", "datetime"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        date = X["datetime"].skb.apply_func(pd.to_datetime, format="%Y-%m-%d %H:%M:%S")
        X = X.assign(year=date.dt.year)
        X = X.drop(columns=["datetime"])
        pred = X.skb.apply(DummyRegressor(), y=y)
        st._api.grid_search(pred, scoring="neg_mean_squared_error")


class RefusesRefitParams(ClassifierMixin, BaseEstimator):
    """Rejects any `set_params` once fitted, as CatBoost does."""

    def __init__(self, C=1.0):
        self.C = C

    def set_params(self, **params):
        if hasattr(self, "model_"):
            raise RuntimeError("You can't change params of fitted model.")
        return super().set_params(**params)

    def fit(self, X, y):
        self.model_ = LogisticRegression(C=self.C).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)


class ChoiceSearchTest(unittest.TestCase):
    """Grid search over estimator and hyperparameter choices scores what skrub scores."""

    def setUp(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
        df["t"] = (df.a > 0).astype(int)
        self.df = df

    def _plan(self, estimator, preprocessor=None):
        data = st.var("data", self.df)
        y = data["t"].skb.mark_as_y()
        X = data.drop(columns="t").skb.mark_as_X(cv=KFold(3), split_kwargs={})
        if preprocessor is not None:
            X = X.skb.apply(preprocessor)
        return X.skb.apply(estimator, y=y)

    def _assert_matches_skrub(self, pred, n_candidates):
        expected = pred.skb.make_grid_search(fitted=True, refit=False, scoring="accuracy").results_
        with config(scheduler=True):
            results = st._api.grid_search(dag=pred, cv=None, scoring="accuracy").results_
        self.assertEqual(len(results), n_candidates)
        self.assertEqual(len(expected), n_candidates)
        # Both are sorted by score, but ties may be ordered differently.
        np.testing.assert_allclose(sorted(results["scores"].to_list()),
                                   sorted(expected["mean_test_score"]))
        return results

    def test_param_choice(self):
        # #223: the choice nested in the estimator used to reach `fit` unresolved.
        pred = self._plan(LogisticRegression(C=st.choose_from([0.01, 1.0], name="C")))
        results = self._assert_matches_skrub(pred, 2)
        self.assertEqual(set(results["id"].to_list()), {"C:0.01", "C:1.0"})

    def test_param_choices_in_two_applies(self):
        pred = self._plan(LogisticRegression(C=st.choose_from([0.01, 1.0], name="C")),
                          preprocessor=st.choose_from([StandardScaler(), MinMaxScaler()],
                                                      name="scaler"))
        self._assert_matches_skrub(pred, 4)

    def _assert_ids_match_skrub(self, pred, labels):
        """Each candidate's score is the one skrub reports for the same grid point.

        ``labels`` maps each choice name to stratum's label of every skrub outcome.
        """
        results = self._assert_matches_skrub(pred, 4)
        expected = pred.skb.make_grid_search(fitted=True, refit=False,
                                             scoring="accuracy").results_
        expected_by_id = {
            ", ".join(f"{name}:{labels[name].get(row[name], row[name])}" for name in labels):
                row["mean_test_score"]
            for _, row in expected.iterrows()}
        ours_by_id = dict(zip(results["id"].to_list(), results["scores"].to_list()))
        self.assertEqual(ours_by_id.keys(), expected_by_id.keys())
        for candidate, score in expected_by_id.items():
            self.assertAlmostEqual(ours_by_id[candidate], score, msg=candidate)

    def test_independent_choices_join_after_the_predictor(self):
        # Unrolling stopped after the `C` choice, so the `wcol` choice on the other
        # branch reached the runtime as the GetItem key.
        data = st.var("data", self.df)
        y = data["t"].skb.mark_as_y()
        X = data.drop(columns="t").skb.mark_as_X(cv=KFold(3), split_kwargs={})
        pred = X.skb.apply(LogisticRegression(C=st.choose_from([0.01, 1.0], name="C")), y=y)
        key = st.as_data_op(st.choose_from(["a", "b"], name="wcol"))
        joined = pred.skb.apply_func(lambda p, w: p, X[key].abs())
        self._assert_ids_match_skrub(
            joined, {"C": {}, "wcol": {"a": "Opt0", "b": "Opt1"}})

    def test_independent_choices_join_before_the_predictor(self):
        # Every grid point scores differently, so a candidate named after another shows.
        data = st.var("data", self.df)
        y = data["t"].skb.mark_as_y()
        X = data.drop(columns="t").skb.mark_as_X(cv=KFold(3), split_kwargs={})
        first = st.as_data_op(st.choose_from(["a", "b"], name="first"))
        second = st.as_data_op(st.choose_from(["c", "a"], name="second"))
        feature = X[first].skb.apply_func(lambda u, v: (u + 0.5 * v).to_frame("f"),
                                          X[second])
        self._assert_ids_match_skrub(
            feature.skb.apply(LogisticRegression(), y=y),
            {"first": {"a": "Opt0", "b": "Opt1"}, "second": {"c": "Opt0", "a": "Opt1"}})

    def test_estimator_choice_refusing_set_params_once_fitted(self):
        # #224: the second grid point must not reconfigure an already fitted model.
        model = st.choose_from({"weak": RefusesRefitParams(C=0.01),
                                "strong": RefusesRefitParams(C=1.0)}, name="model")
        self._assert_matches_skrub(self._plan(model), 2)

    def test_estimator_choice_catboost(self):
        # #224, as reported.
        catboost = pytest.importorskip("catboost")

        class CB(catboost.CatBoostClassifier):
            def __sklearn_clone__(self):
                return CB(**self.get_params(deep=False))

        model = st.choose_from({"d2": CB(depth=2, iterations=20, verbose=0),
                                "d4": CB(depth=4, iterations=20, verbose=0)}, name="model")
        self._assert_matches_skrub(self._plan(model), 2)


class NestedChoiceSearchTest(unittest.TestCase):
    """A bare choice in an op's arguments is searched as skrub searches it.

    These choices used to reach the runtime unresolved unless wrapped in
    ``st.as_data_op``.
    """

    setUp = ChoiceSearchTest.setUp
    _assert_matches_skrub = ChoiceSearchTest._assert_matches_skrub
    _assert_ids_match_skrub = ChoiceSearchTest._assert_ids_match_skrub

    def _X_y(self):
        data = st.var("data", self.df)
        y = data["t"].skb.mark_as_y()
        return data.drop(columns="t").skb.mark_as_X(cv=KFold(3), split_kwargs={}), y

    def test_getitem_key(self):
        X, y = self._X_y()
        pred = X[[st.choose_from(["a", "b"], name="col")]].skb.apply(LogisticRegression(), y=y)
        results = self._assert_matches_skrub(pred, 2)
        self.assertEqual(set(results["id"].to_list()), {"col:Opt0", "col:Opt1"})

    def test_method_kwarg(self):
        X, y = self._X_y()
        pred = X.drop(columns=st.choose_from(["a", "b"], name="col"))
        self._assert_matches_skrub(pred.skb.apply(LogisticRegression(), y=y), 2)

    def test_binop_operand(self):
        X, y = self._X_y()
        pred = (X * st.choose_from([1.0, -1.0, 0.0], name="m")).skb.apply(LogisticRegression(), y=y)
        self._assert_matches_skrub(pred, 3)

    def test_call_arg(self):
        X, y = self._X_y()
        pred = X.skb.apply_func(lambda df, col: df[[col]], st.choose_from(["a", "c"], name="col"))
        self._assert_matches_skrub(pred.skb.apply(LogisticRegression(), y=y), 2)

    def test_apply_cols(self):
        X, y = self._X_y()
        scaled = X.skb.apply(StandardScaler(), cols=st.choose_from([["a"], ["b"]], name="cols"))
        self._assert_matches_skrub(scaled.skb.apply(LogisticRegression(), y=y), 2)

    def test_apply_cols_choice_nested_in_a_list(self):
        for fixed_cols in ([], ["c"]):
            with self.subTest(fixed_cols=fixed_cols):
                X, y = self._X_y()
                cols = [*fixed_cols, st.choose_from(["a", "b"], name="col")]
                scaled = X.skb.apply(StandardScaler(), cols=cols)
                results = self._assert_matches_skrub(
                    scaled.skb.apply(LogisticRegression(), y=y), 2)
                self.assertEqual(set(results["id"].to_list()), {"col:Opt0", "col:Opt1"})

    def test_discretized_numeric_choice_in_a_slice(self):
        X, y = self._X_y()
        pred = X.iloc[:, :st.choose_int(1, 3, n_steps=3, name="n")]
        results = self._assert_matches_skrub(pred.skb.apply(LogisticRegression(), y=y), 3)
        self.assertEqual(set(results["id"].to_list()), {"n:1", "n:2", "n:3"})

    def test_choice_nested_in_an_outcome(self):
        # skrub's grid is conditional: the inner choice only varies under its outcome.
        X, y = self._X_y()
        inner = st.choose_from(["b", "c"], name="inner")
        for key in ([st.choose_from([inner, "d"], name="outer")],
                    st.choose_from([["a", inner], ["d"]], name="outer")):
            with self.subTest(key=key):
                pred = X[key].skb.apply(LogisticRegression(), y=y)
                self._assert_matches_skrub(pred, 3)

    def test_dataop_outcome(self):
        X, y = self._X_y()
        cols = st.as_data_op(["a", "b"])
        pred = X[st.choose_from([cols, ["c"]], name="cols")].skb.apply(LogisticRegression(), y=y)
        self._assert_matches_skrub(pred, 2)

    def test_choice_used_twice_is_one_dimension(self):
        # skrub keys a choice by identity: every use takes the same outcome.
        X, y = self._X_y()
        col = st.choose_from(["a", "b"], name="col")
        for other in (X[[col]], X[[st.as_data_op(col)]]):
            with self.subTest(other=other):
                feature = X[[col]].skb.concat(
                    [other.rename(columns=lambda c: c + "_2")], axis=1)
                self._assert_matches_skrub(feature.skb.apply(LogisticRegression(), y=y), 2)

    def test_with_estimator_choice(self):
        X, y = self._X_y()
        pred = X[[st.choose_from(["a", "b"], name="col")]].skb.apply(
            LogisticRegression(C=st.choose_from([0.001, 1.0], name="C")), y=y)
        self._assert_ids_match_skrub(pred, {"col": {"a": "Opt0", "b": "Opt1"}, "C": {}})


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


_LOADS = {"n": 0}


def _load(seed):
    """Stands in for an expensive read upstream of both X and the groups."""
    _LOADS["n"] += 1
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    return df.assign(g=rng.integers(0, 3, 300), t=(df.a > 0).astype(int))


class PlanComputedSplitterTest(unittest.TestCase):
    """Regression tests for issue #225: a splitter declared with DataOps is computed by
    the plan, so what it reads is computed once, by the scheduler."""

    def _plan(self, cv):
        rows = st.var("seed", 0).skb.apply_func(_load)
        y = rows["t"].skb.mark_as_y()
        X = rows.drop(columns=["t"]).skb.mark_as_X(cv=cv, split_kwargs={"groups": rows["g"]})
        pred = X.skb.apply(LogisticRegression(), y=y)
        # Building the plan computes skrub's previews.
        _LOADS["n"] = 0
        return pred

    def _assert_matches_skrub(self, pred, cv=None):
        with config(scheduler=True):
            sched = st._api.grid_search(pred, cv=cv, scoring="accuracy")
        self.assertEqual(_LOADS["n"], 1)
        expected = pred.skb.make_grid_search(cv=cv, fitted=True, refit=False,
                                             scoring="accuracy")
        np.testing.assert_allclose(sched.results_["scores"].to_list(),
                                   expected.results_["mean_test_score"])

    def test_groups_computed_once(self):
        self._assert_matches_skrub(self._plan(GroupKFold(n_splits=3)))

    def test_x_mark_is_removed_without_splitting_other_source_consumers(self):
        from stratum.optimizer._optimize import convert_to_ops
        from stratum.optimizer.logical._ops import SplitXOp
        from stratum.optimizer.logical._split_ops import BuildSplitterOp, SplitOp, add_splitting_op
        from stratum.optimizer._op_utils import topological_iterator, validate_dag

        rows = st.var("seed", 0).skb.apply_func(_load)
        tmp = rows.drop(columns=["t"])
        y = rows["t"].skb.mark_as_y()
        X = tmp.skb.mark_as_X(cv=GroupKFold(n_splits=3),
                              split_kwargs={"groups": tmp["g"]})
        pred = X.skb.apply(LogisticRegression(), y=y)

        root = convert_to_ops(pred, env={"seed": 0})
        mark = next(op for op in topological_iterator(root) if isinstance(op, SplitXOp))
        with self.assertRaisesRegex(RuntimeError, "replaced by the splitting rewrite"):
            mark.process("fit_transform", [])
        root = add_splitting_op(root)
        validate_dag(root)
        ops = list(topological_iterator(root))
        self.assertFalse(any(isinstance(op, SplitXOp) for op in ops))
        split = next(op for op in ops if isinstance(op, SplitOp))
        splitter = next(op for op in ops if isinstance(op, BuildSplitterOp))
        source = split.inputs[0]
        self.assertIn(split, source.outputs)
        self.assertNotIn(splitter, source.outputs)
        # The groups path still reads the full source, independently of the fold X.
        self.assertTrue(any(source in op.inputs for op in ops if op in splitter.inputs))
        _LOADS["n"] = 0
        self._assert_matches_skrub(pred)

    def test_splitter_from_a_variable(self):
        # `cv` may itself be a DataOp; the plan resolves it like any other operand.
        self._assert_matches_skrub(self._plan(st.var("cv", GroupKFold(n_splits=3))))

    def test_explicit_cv_overrides_declared_splitter(self):
        self._assert_matches_skrub(self._plan(GroupKFold(n_splits=3)), cv=KFold(n_splits=2))


if __name__ == "__main__":
    unittest.main()
