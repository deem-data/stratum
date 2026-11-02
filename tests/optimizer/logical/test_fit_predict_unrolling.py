"""Fit/predict unrolling: one run of the plan after the split is one fold."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import stratum as st
from stratum._api import grid_search
from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.logical._ops import BaseEstimatorOp, ImplOp, PredictorOp
from stratum.optimizer.logical._split_ops import SplitOutput
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


def stages(plan, cls=BaseEstimatorOp):
    return [(op.stage, op.mode) for op in plan if isinstance(op, cls)]


def test_estimators_fit_on_train_rows_and_predict_on_test_rows(frame):
    X, y = xy(frame)
    plan, split_pos = optimize(X.skb.apply(StandardScaler()).skb.apply(Ridge(), y=y))
    assert sorted(stages(plan)) == [("fit", "fit_transform"), ("fit", "fit_transform"),
                                    ("predict", "predict"), ("predict", "transform")]
    for op in plan:
        if op.stage == "predict":
            fit = op.inputs[op.state_ref.k]
            assert fit.stage == "fit" and type(fit) is type(op)
    # Everything after the split runs once per fold, so all of it follows the split.
    assert all(isinstance(op, SplitOutput) or op.stage is not None or op is plan[-1]
               for op in plan[split_pos + 1:])
    sides = {(op.is_x, op.train) for op in plan if isinstance(op, SplitOutput)}
    # A predict stage does not read y, and no scorer does here.
    assert sides == {(True, True), (False, True), (True, False)}


def test_a_predictor_only_fits_unless_the_training_rows_read_it(frame):
    X, y = xy(frame)
    plan, _ = optimize(X.skb.apply(Ridge(), y=y))
    assert sorted(stages(plan, PredictorOp)) == [("fit", "fit_transform"), ("predict", "predict")]

    stacked = X.skb.apply(Ridge(), y=y).skb.apply_func(
        lambda p: pd.DataFrame({"p": p})).skb.apply(LinearRegression(), y=y)
    plan, _ = optimize(stacked)
    # The first model's predictions on the training rows fit the second one.
    assert sorted(stages(plan, PredictorOp)) == [
        ("fit", "fit_transform"), ("fit", "fit_transform"),
        ("predict", "predict"), ("predict", "predict"), ("predict", "predict")]


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
    modes = sorted(op.mode for op in plan if isinstance(op, PredictorOp) and op.stage == "predict")
    # The last model serves the scorer; the first one feeds a model, so it predicts,
    # on the training rows as on the test rows.
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
    # Only the test copy reads it here: the train copy has no tail after the model.
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
    assert isinstance(plan[-1], PredictorOp) and plan[-1].stage == "predict"
    assert plan[-1].inputs[plan[-1].state_ref.k].stage == "fit"


def test_a_plan_without_a_split_is_not_unrolled(frame):
    plan, split_pos = optimize(st.as_data_op(frame)[["a"]].skb.apply(StandardScaler()))
    assert split_pos is None
    assert all(op.stage is None and op.pass_mode is None for op in plan)
