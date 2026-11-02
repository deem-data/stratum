"""Estimators are keyed by class and parameters, when stratum knows they are exact."""
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from skrub import StringEncoder, TableVectorizer

import stratum as st
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._estimator_lineage import estimator_lineage, selector_lineage
from stratum.optimizer._lineage import LineageEncoder, Uncacheable, lineage_key
from stratum.optimizer._optimize import optimize
from stratum.optimizer.logical._ops import BaseEstimatorOp


def key(estimator) -> str:
    enc = LineageEncoder()
    enc.encode(estimator_lineage(estimator))
    return enc.hexdigest()


def test_parameters_name_the_fit():
    assert key(Ridge()) == key(Ridge())
    assert key(Ridge(alpha=1.0)) != key(Ridge(alpha=2.0))
    assert key(Ridge()) != key(LogisticRegression())
    # How it runs is not what it computes.
    assert key(RandomForestClassifier(random_state=0, n_jobs=1)) == key(
        RandomForestClassifier(random_state=0, n_jobs=8))


def test_nested_estimators_are_part_of_the_key():
    seeded = StringEncoder(random_state=0)
    assert key(TableVectorizer(high_cardinality=seeded)) != key(
        TableVectorizer(high_cardinality=StringEncoder(n_components=5, random_state=0)))


def test_unseeded_randomness_is_uncacheable():
    with pytest.raises(Uncacheable, match="random_state=None"):
        key(RandomForestClassifier())
    with pytest.raises(Uncacheable, match="StringEncoder has random_state=None"):
        key(TableVectorizer())  # its default high-cardinality encoder is unseeded
    with pytest.raises(Uncacheable):
        key(LogisticRegression(solver="saga"))
    # A seed parameter that only matters for some configurations.
    assert key(LogisticRegression()) and key(Ridge(solver="cholesky"))


def test_default_seeded_libraries():
    lightgbm = pytest.importorskip("lightgbm")
    assert key(lightgbm.LGBMClassifier()) != key(lightgbm.LGBMClassifier(num_leaves=7))


def test_unknown_estimators_are_uncacheable():
    class Shift(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X + 1

    with pytest.raises(Uncacheable, match="not an estimator stratum knows"):
        key(Shift())
    with pytest.raises(Uncacheable):
        key(FunctionTransformer(lambda x: x))  # a known class running a UDF
    assert key(StandardScaler())


def test_selectors():
    import skrub.selectors as s
    def sel(x):
        enc = LineageEncoder()
        enc.encode(selector_lineage(x))
        return enc.hexdigest()
    assert sel(s.all()) == sel(s.all())
    assert sel(s.cols("a")) != sel(s.cols("b"))
    assert sel(s.numeric() | s.cols("a")) != sel(s.numeric() & s.cols("a"))
    assert sel(s.glob("a*")) != sel(s.glob("b*"))
    with pytest.raises(Uncacheable, match="user predicate"):
        selector_lineage(s.filter(lambda c: True))


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": range(20), "b": range(20, 40), "y": [0, 1] * 10}).to_csv(
        path, index=False)
    return path


def model(csv_path, estimator):
    data = st.as_data_op(str(csv_path)).skb.apply_func(pd.read_csv)
    X = data[["a", "b"]].skb.mark_as_X()
    y = data["y"].skb.mark_as_y()
    return X.skb.apply(StandardScaler()).skb.apply(estimator, y=y)


def estimator_keys(dag):
    """Each estimator op's key, over stand-in keys for its inputs (what precedes
    the estimators is keyed by the plan, see test_reuse_planning)."""
    plan, _ = optimize(dag, env=get_data(dag))
    keys = []
    for op in plan:
        if isinstance(op, BaseEstimatorOp) and op.stage == "fit":
            try:
                keys.append(lineage_key(op, [f"input{k}" for k in range(len(op.inputs))]))
            except Uncacheable:
                keys.append(None)
    return keys


def test_estimator_ops_are_keyed_in_plans(csv_path):
    scaler, forest = estimator_keys(model(csv_path, RandomForestClassifier(random_state=0)))
    assert scaler is not None and forest is not None
    scaler2, forest2 = estimator_keys(model(csv_path, RandomForestClassifier(random_state=1)))
    assert scaler2 == scaler and forest2 != forest
    _, unseeded = estimator_keys(model(csv_path, RandomForestClassifier()))
    assert unseeded is None
