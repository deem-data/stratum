"""Train/test unrolling: one run of the plan after the split is one fold."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import stratum as st
from stratum.frontend._api import grid_search
from stratum.frontend._conversion import convert_to_ops
from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.logical._candidate_ops import CollectCandidatesOp
from stratum.optimizer.logical._ops import BaseEstimatorOp, ImplOp, PredictorOp
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp
from stratum.optimizer.logical._split_ops import SplitOutput, add_splitting_op
from stratum.optimizer.physical.rewrites._dead_code_elimination import eliminate_dead_code
from stratum.optimizer.physical.rewrites._train_test_unrolling import unroll_train_test
from stratum.optimizer.physical._conversion import ConversionOp
from stratum.optimizer.physical._impl_selection import (
    DefaultImplementationSelector, GreedyImplementationSelector, select_implementations)
from stratum.optimizer.physical._lowering import lower_to_physical
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.physical._predictor_execs import (
    SklearnRandomForest, StratumHistogramRandomForestClassifier)
from stratum.optimizer.physical._representation import check_frame_representations
from stratum.optimizer.physical._selection_execs import PandasQuerySelectionOp
from stratum.runtime._scheduler import SequentialScheduler
from skrub._data_ops._data_ops import EvalMode


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    df["y"] = 2 * df["a"] - df["b"] + rng.normal(scale=.1, size=60)
    df["c"] = (df["y"] > 0).astype(int)
    return df


def xy(frame, target="y"):
    data = st.as_data_op(frame)
    return data[["a", "b"]].skb.mark_as_X(), data[target].skb.mark_as_y()


def modes(plan, cls=BaseEstimatorOp):
    return [(op.train, op.mode) for op in plan if isinstance(op, cls)]


def test_estimators_fit_on_train_rows_and_predict_on_test_rows(frame):
    X, y = xy(frame)
    plan, split_pos = optimize(X.skb.apply(StandardScaler()).skb.apply(Ridge(), y=y))
    assert sorted(modes(plan)) == [(False, "predict"), (False, "transform"),
                                  (True, "fit"), (True, "fit_transform")]
    for op in plan:
        if op.state_ref is not None:
            fit = op.inputs[op.state_ref.k]
            assert fit.produces_state and type(fit) is type(op)
    # Everything after the split runs once per fold, so all of it follows the split.
    assert all(op.train is not None for op in plan[split_pos + 1:])
    assert all(op.train is None for op in plan[:split_pos + 1])
    sides = {(op.is_x, op.train) for op in plan if isinstance(op, SplitOutput)}
    # A predict stage does not read y, and no scorer does here.
    assert sides == {(True, True), (False, True), (True, False)}


@pytest.mark.parametrize("weighted", [False, True])
def test_fit_and_predict_keep_the_same_selected_forest(frame, weighted):
    X, y = xy(frame, "c")
    fit_kwargs = {"sample_weight": X["a"].abs()} if weighted else None
    pred = X.skb.apply(RandomForestClassifier(n_estimators=3, max_depth=2, random_state=0),
                       y=y, fit_kwargs=fit_kwargs)

    plan, _ = optimize(pred, OptConfig(selector=GreedyImplementationSelector()))

    fit, = [op for op in plan if isinstance(op, PredictorOp) and op.produces_state]
    predict, = [op for op in plan if isinstance(op, PredictorOp) and op.state_ref is not None]
    # Native forests reject fit kwargs. Pruning prediction inputs must preserve
    # the implementation bound before stage construction.
    expected = SklearnRandomForest if weighted else StratumHistogramRandomForestClassifier
    assert type(fit) is type(predict) is expected
    assert predict.inputs[predict.state_ref.k] is fit
    assert fit.original_estimator is predict.original_estimator


def test_stateless_copies_preserve_selection_time_initialization(frame, monkeypatch):
    initialized = []
    initialize = PandasQuerySelectionOp.on_impl_selected

    def record(op, ctx):
        initialized.append(op)
        initialize(op, ctx)

    class BeforeUnrollingSelector(DefaultImplementationSelector):
        def choose(self, op, candidates, ctx):
            assert op.train is None or isinstance(op, SplitOutput)
            assert op.mode == op.fitting_mode
            return super().choose(op, candidates, ctx)

    monkeypatch.setattr(PandasQuerySelectionOp, "on_impl_selected", record)
    X, y = xy(frame)
    pred = X[X["a"] > 0].skb.apply(Ridge(), y=y[X["a"] > 0])
    plan, _ = optimize(pred, OptConfig(pandas_query=True, selector=BeforeUnrollingSelector()))

    copies = [op for op in plan if isinstance(op, PandasQuerySelectionOp)]
    assert len(initialized) == 1 and len(copies) == 2
    assert {op.mode for op in copies} == {"fit_transform", "predict"}
    for op in copies:
        assert op.query == initialized[0].query
        assert op.query_params == initialized[0].query_params
        pd.testing.assert_frame_equal(op.process([frame]), frame[frame["a"] > 0])


def test_selected_frame_conversions_are_copied_for_both_row_sets(frame):
    class PolarsProjection(DefaultImplementationSelector):
        def choose(self, op, candidates, ctx):
            if isinstance(op, ColumnProjectionOp) and isinstance(op.inputs[0], SplitOutput):
                return next(c for c in candidates if c.backend_name == "polars")
            return super().choose(op, candidates, ctx)

    X, y = xy(frame)
    pred = X[["a"]].skb.apply(Ridge(), y=y)
    plan, split_pos = optimize(pred, OptConfig(selector=PolarsProjection()), search=_search("r2"))

    conversions = [op for op in plan if isinstance(op, ConversionOp)]
    assert len(conversions) == 2
    assert {op.inputs[0].train for op in conversions} == {False, True}
    assert len({(op.source, op.target) for op in conversions}) == 1
    check_frame_representations(plan)
    sched = SequentialScheduler(plan, split_pos)
    sched.grid_search(KFold(3))
    reference = pred.skb.make_grid_search(fitted=True, cv=KFold(3), scoring="r2")
    np.testing.assert_allclose(sched.results_["scores"], reference.results_["mean_test_score"])


def test_a_predictor_only_fits_unless_the_training_rows_read_it(frame):
    X, y = xy(frame)
    plan, _ = optimize(X.skb.apply(Ridge(), y=y))
    assert sorted(modes(plan, PredictorOp)) == [(False, "predict"), (True, "fit")]

    stacked = X.skb.apply(Ridge(), y=y).skb.apply_func(
        lambda p: pd.DataFrame({"p": p})).skb.apply(LinearRegression(), y=y)
    plan, _ = optimize(stacked)
    # The first model's predictions on the training rows fit the second one.
    assert sorted(modes(plan, PredictorOp)) == [
        (False, "predict"), (False, "predict"),
        (True, "fit"), (True, "fit"), (True, "predict")]


def test_unrolling_leaves_unused_training_predictions_for_dce(frame):
    X, y = xy(frame)
    predict = add_splitting_op(convert_to_ops(X.skb.apply(Ridge(), y=y)))
    root = CollectCandidatesOp(["default"], inputs=[predict])
    predict.add_output(root)
    ctx = PlanContext.from_flags(OptConfig())
    root = select_implementations(lower_to_physical(root, ctx), ctx)
    predict, = root.inputs

    assert unroll_train_test(root) is root
    fit = predict.inputs[predict.state_ref.k]
    on_train, = [out for out in fit.outputs if out is not predict]
    train_tail, = on_train.outputs
    assert on_train.mode == "predict" and on_train.train is True
    assert isinstance(train_tail, CollectCandidatesOp)

    eliminate_dead_code(root)

    assert fit.outputs == [predict]
    assert not on_train.inputs and not on_train.outputs
    assert not train_tail.inputs and not train_tail.outputs


def test_postprocessing_runs_only_on_test_rows(frame):
    calls = []

    def tail(predictions):
        calls.append(len(predictions))
        return predictions

    with st.config_context(eager_data_ops=False):
        X, y = xy(frame)
        pred = X.skb.apply(Ridge(), y=y).skb.apply_func(tail)
        grid_search(pred, cv=KFold(3), scoring="r2")

    assert calls == [20, 20, 20]


class CenterPredictions(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.mean_ = np.mean(X)
        return self

    def transform(self, X):
        return X - self.mean_


def test_transformer_after_predictor_keeps_its_fit_and_training_inputs(frame):
    X, y = xy(frame)
    pred = X.skb.apply(Ridge(), y=y).skb.apply(CenterPredictions(), no_wrap=True)

    ours = grid_search(pred, cv=KFold(3), scoring="r2").results_
    features, target = frame[["a", "b"]], frame["y"]
    expected = []
    for train, test in KFold(3).split(frame):
        model = Ridge().fit(features.iloc[train], target.iloc[train])
        center = model.predict(features.iloc[train]).mean()
        predictions = model.predict(features.iloc[test]) - center
        expected.append(r2_score(target.iloc[test], predictions))

    np.testing.assert_allclose(ours["scores"], [np.mean(expected)])


def test_stacking_scores_like_skrub(frame):
    X, y = xy(frame)

    def stacked():
        X, y = xy(frame)
        first = X.skb.apply(Ridge(alpha=st.choose_from([0.1, 10.0], name="alpha")), y=y)
        features = first.skb.apply_func(lambda p: pd.DataFrame({"p": p}))
        return features.skb.apply(LinearRegression(), y=y)

    ours = grid_search(stacked(), cv=KFold(3), scoring="r2").results_
    theirs = stacked().skb.make_grid_search(fitted=True, cv=KFold(3), scoring="r2").results_
    np.testing.assert_allclose(sorted(ours["scores"]), sorted(theirs["mean_test_score"]))


def test_the_test_rows_answer_with_the_scorers_response(frame):
    X, y = xy(frame, "c")
    first = X.skb.apply(LogisticRegression(), y=y)
    pred = first.skb.apply_func(lambda p: pd.DataFrame({"p": p})).skb.apply(
        LogisticRegression(), y=y)
    plan, _ = optimize(pred.skb.with_scoring("roc_auc"), search=_search("roc_auc"))
    modes = sorted(op.mode for op in plan if isinstance(op, PredictorOp) and op.state_ref is not None)
    # The terminal model uses the scorer's response method. The upstream model
    # produces predictions as features for both train and test copies.
    assert modes == ["decision_function", "predict", "predict"]


def _search(scoring):
    from stratum.optimizer._optimize import SearchConfig
    from stratum.optimizer.logical._scoring import resolve_scoring
    return SearchConfig(metric=resolve_scoring(scoring))


def test_the_test_rows_read_eval_mode_in_their_mode(frame):
    X, y = xy(frame)
    pred = X.skb.apply(Ridge(), y=y).skb.apply_func(
        lambda p, mode: p if mode == "fit_transform" else -p, st.eval_mode())
    plan, _ = optimize(pred)
    sources = [op for op in plan if isinstance(op, ImplOp) and isinstance(op.skrub_impl, EvalMode)]
    # Only the test copy consumes eval_mode; the unused training branch is eliminated.
    assert [op.mode for op in sources] == ["predict"]


def test_what_only_fitting_reads_is_not_computed_on_the_test_rows(frame):
    X, y = xy(frame)
    calls = []

    def weights(y):
        calls.append(len(y))
        return np.ones(len(y))

    pred = X.skb.apply(Ridge(), y=y, fit_kwargs={"sample_weight": y.skb.apply_func(weights)})
    calls.clear()  # skrub's preview, when the pipeline is built
    grid_search(pred, cv=KFold(3), scoring="r2")
    # One call per fold, on its training rows (40 of 60), as in skrub.
    assert calls == [40, 40, 40]


def test_a_root_estimator_stays_the_root(frame):
    """Without a candidate set, the plan ends in the model's predict stage."""
    X, y = xy(frame)
    plan, _ = optimize(X.skb.apply(Ridge(), y=y), OptConfig(unroll_choices=False))
    assert isinstance(plan[-1], PredictorOp) and plan[-1].mode == "predict" and plan[-1].train is False
    assert plan[-1].inputs[plan[-1].state_ref.k].mode == "fit"


def test_a_plan_without_a_split_is_not_unrolled(frame):
    plan, split_pos = optimize(st.as_data_op(frame)[["a"]].skb.apply(StandardScaler()))
    assert split_pos is None
    assert all(op.train is None and op.mode == op.fitting_mode for op in plan)


def test_execution_uses_bound_modes(frame, monkeypatch):
    X, y = xy(frame)
    plan, split_pos = optimize(X.skb.apply(StandardScaler()).skb.apply(Ridge(), y=y))
    bound = [(op.mode, op.train) for op in plan]

    def resolve_at_runtime(*args):
        raise AssertionError("execution must use the mode bound during planning")

    monkeypatch.setattr(BaseEstimatorOp, "effective_mode", resolve_at_runtime)
    assert len(SequentialScheduler(plan, split_pos).evaluate()) == 12
    assert [(op.mode, op.train) for op in plan] == bound


@pytest.mark.parametrize("unroll_choices", [False, True])
def test_full_data_predictor_has_separate_fit_and_predict(frame, unroll_choices):
    features, target = frame[["a", "b"]], frame["y"]
    dag = st.as_data_op(features).skb.apply(Ridge(), y=target, no_wrap=True)
    plan, split_pos = optimize(dag, OptConfig(unroll_choices=unroll_choices))
    assert split_pos is None
    fit, predict = [op for op in plan if isinstance(op, PredictorOp)]
    assert (fit.mode, predict.mode) == ("fit", "predict")
    assert predict.inputs[predict.state_ref.k] is fit
    assert all(op.train is None for op in plan)

    scheduler = SequentialScheduler(plan, split_pos)
    actual = scheduler.evaluate()
    expected = Ridge().fit(features, target).predict(features)
    np.testing.assert_allclose(actual, expected)
    assert scheduler.memory_usage == 0
    assert not hasattr(fit.original_estimator, "coef_")


def test_predictor_before_split_runs_once_on_full_data(frame):
    data = st.as_data_op(frame)
    prefix = data[["a", "b"]].skb.apply(Ridge(), y=data["y"], no_wrap=True)
    X = prefix.skb.apply_func(lambda p: pd.DataFrame({"p": p})).skb.mark_as_X()
    y = data["y"].skb.mark_as_y()
    plan, split_pos = optimize(X.skb.apply(Ridge(), y=y), search=_search("r2"))
    fit, predict = [op for op in plan[:split_pos] if isinstance(op, PredictorOp)]
    assert (fit.mode, predict.mode) == ("fit", "predict")
    assert predict.inputs[predict.state_ref.k] is fit
    assert fit.train is predict.train is None

    scheduler = SequentialScheduler(plan, split_pos)
    scheduler.grid_search(KFold(3))
    features = Ridge().fit(frame[["a", "b"]], frame["y"]).predict(frame[["a", "b"]])[:, None]
    expected = []
    for train, test in KFold(3).split(frame):
        model = Ridge().fit(features[train], frame["y"].iloc[train])
        expected.append(r2_score(frame["y"].iloc[test], model.predict(features[test])))
    np.testing.assert_allclose(scheduler.results_["scores"], [np.mean(expected)])
    assert scheduler.memory_usage == 0
