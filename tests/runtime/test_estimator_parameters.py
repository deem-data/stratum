"""Graph-fed estimator parameters execute with and without literal inlining."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import stratum as st
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import OptConfig, SearchConfig, optimize
from stratum.optimizer.logical._scoring import resolve_scoring
from stratum.runtime._scheduler import SequentialScheduler


@pytest.fixture(autouse=True)
def configuration():
    with st.config_context(eager_data_ops=False), st.config(num_threads=2):
        yield


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("mean", [False, True])
@pytest.mark.parametrize("inline", [False, True])
def test_scalar_parameter_reaches_transformer(nested, mean, inline):
    frame = pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 8.0, 3.0]})
    parameter = st.as_data_op(mean)
    scaler = StandardScaler(with_mean=parameter)
    estimator = Pipeline([("scale", scaler)]) if nested else scaler
    dag = st.as_data_op(frame).skb.apply(estimator, no_wrap=True)

    scheduler = SequentialScheduler(*optimize(dag, OptConfig(literal_inlining=inline)))
    result = scheduler.evaluate()

    np.testing.assert_allclose(result, StandardScaler(with_mean=mean).fit_transform(frame))
    assert scaler.with_mean is parameter
    assert not hasattr(scaler, "mean_")


def variance(series):
    return float(np.var(series.to_numpy()))


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("inline", [False, True])
def test_computed_parameter_is_bound_per_training_fold(nested, inline):
    frame = pd.DataFrame({"a": np.arange(1.0, 19.0) ** 2, "b": np.arange(18.0) % 3})
    target = frame["a"] * 0.5 + frame["b"]
    X = st.var("X", frame).skb.mark_as_X()
    y = st.var("y", target).skb.mark_as_y()
    alpha = X["a"].skb.apply_func(variance)
    intercept = st.as_data_op(False)
    ridge = Ridge(alpha=alpha, fit_intercept=intercept)
    estimator = Pipeline([("ridge", ridge)]) if nested else ridge
    dag = X.skb.apply(estimator, y=y)
    cv = KFold(3)
    search = SearchConfig(resolve_scoring("r2"), return_predictions=True)
    scheduler = SequentialScheduler(*optimize(
        dag, OptConfig(literal_inlining=inline), env=get_data(dag), search=search))

    predictions = scheduler.grid_search(cv)

    scores, alphas = [], []
    for actual, (train, test) in zip(predictions, cv.split(frame), strict=True):
        alphas.append(variance(frame.iloc[train]["a"]))
        model = Ridge(alpha=alphas[-1], fit_intercept=False).fit(frame.iloc[train], target.iloc[train])
        expected = model.predict(frame.iloc[test])
        np.testing.assert_allclose(actual["vals"][0], expected)
        scores.append(r2_score(target.iloc[test], expected))
    assert len(set(alphas)) == 3
    np.testing.assert_allclose(scheduler.results_["scores"][0], np.mean(scores))
    assert ridge.alpha is alpha and not hasattr(ridge, "coef_")
    assert ridge.fit_intercept is intercept
