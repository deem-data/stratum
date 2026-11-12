"""Estimator and selector lineage preserves configuration and rejects unknown code."""
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
import skrub.selectors as s
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from skrub import StringEncoder, TableVectorizer

import stratum as st
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._estimator_lineage import estimator_lineage, selector_lineage
from stratum.optimizer._lineage import (
    LineageEncoder, Uncacheable, compute_lineage_keys, lineage_key)
from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.logical._base import OperandRef
from stratum.optimizer.logical._ops import BaseEstimatorOp, TransformerOp
from stratum.optimizer.logical._projection_ops import ColumnSelectorOp
from tests.optimizer.test_operator_lineage import plan_keys, read_csv


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
    plan, _, _ = optimize(dag, env=get_data(dag))
    keys = []
    for op in plan:
        if isinstance(op, BaseEstimatorOp):
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


@pytest.mark.parametrize("estimator", [
    Ridge(solver="sag"), LogisticRegression(solver="liblinear"),
    Lasso(selection="random"), ElasticNet(selection="random"),
    RandomForestClassifier(random_state=np.random.RandomState(0)),
    RandomForestClassifier(random_state=np.random.default_rng(0)),
    Pipeline([("forest", RandomForestClassifier())]),
    FrozenEstimator(StandardScaler().fit([[0.0], [1.0]])),
])
def test_unsupported_estimator_state_is_rejected(estimator):
    with pytest.raises(Uncacheable):
        key(estimator)


@pytest.mark.parametrize("estimator", [
    Lasso(), ElasticNet(), Ridge(solver="sag", random_state=0),
    LogisticRegression(solver="saga", random_state=0),
    Pipeline([("scale", StandardScaler()), ("ridge", Ridge())]),
])
def test_deterministic_configurations_are_supported(estimator):
    assert key(estimator) == key(estimator)


def test_nested_parameter_values_keep_their_meaning():
    assert key(CountVectorizer(dtype=np.int32)) != key(CountVectorizer(dtype=np.int64))
    assert key(TableVectorizer(high_cardinality="passthrough", specific_transformers=[
        (StandardScaler(), s.numeric())])) != key(TableVectorizer(
            high_cardinality="passthrough", specific_transformers=[
                (StandardScaler(), s.cols("a"))]))
    # Converting a dict subclass to a dict would erase its default behavior.
    from collections import defaultdict
    with pytest.raises(Uncacheable):
        key(RandomForestClassifier(random_state=0, class_weight=defaultdict(float)))


def test_lineage_names_the_prototype_after_fitting():
    op = TransformerOp(StandardScaler())
    before = lineage_key(op, ["source"])
    op.estimator.fit([[1.0, 2.0], [3.0, 4.0]])
    op.estimator.set_params(with_mean=False)
    assert lineage_key(op, ["source"]) == before
    assert lineage_key(TransformerOp(StandardScaler(with_mean=False)), ["source"]) != before


@pytest.mark.parametrize("selector", [
    s.all(), s.cols("a", "b"), ~s.cols("a"), s.numeric() | s.cols("a"),
    s.numeric() & s.cols("a"), s.all() - s.cols("b"), s.numeric() ^ s.cols("b"),
    s.glob("a*"), s.regex("^a"), s.has_dtype(np.dtype("float64")),
])
def test_deferred_column_selectors_are_keyed(selector):
    op = ColumnSelectorOp(selector)
    assert lineage_key(op, ["source"]) == lineage_key(ColumnSelectorOp(selector), ["source"])
    assert lineage_key(op, ["source"]) != lineage_key(op, ["changed-source"])


def test_unknown_selector_behavior_is_rejected():
    class CustomAll(type(s.all())):
        def _matches(self, column):
            return random.choice([True, False])

    for selector in (CustomAll(), s.filter(lambda col: True), s.filter(random.choice)):
        with pytest.raises(Uncacheable):
            lineage_key(ColumnSelectorOp(selector), ["source"])


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_selector_and_estimator_plans_are_keyed(csv_path, backend):
    def build(cols):
        return read_csv(csv_path).skb.select(s.numeric()).skb.apply(
            StandardScaler(), cols=cols, exclude_cols=s.cols("y"))

    before = plan_keys(build(s.cols("a")), backend)
    assert all(value is not None for _, value in before)
    assert plan_keys(build(s.cols("a")), backend) == before
    changed = plan_keys(build(s.cols("b")), backend)
    assert before[0] == changed[0] and before[-1] != changed[-1]


def test_graph_fed_parameters_and_selectors_use_operand_keys(csv_path):
    def build(cols):
        return read_csv(csv_path).skb.apply(
            StandardScaler(),
            cols=st.as_data_op(cols), exclude_cols=st.as_data_op(["y"]))

    before = plan_keys(build(["a", "b"]))
    assert all(value is not None for _, value in before)
    assert plan_keys(build(["a", "b"])) == before
    assert plan_keys(build(["a"]))[-1] != before[-1]
    assert selector_lineage(OperandRef(1)) == OperandRef(1)

    # Hyperparameters with actual operand bindings use those inputs' keys.
    op = TransformerOp(StandardScaler(with_mean=st.as_data_op(True)),
                       param_refs={"with_mean": OperandRef(1)})
    assert lineage_key(op, ["source", "true"]) != lineage_key(op, ["source", "false"])


def test_unbound_graph_parameters_are_uncacheable():
    with pytest.raises(Uncacheable, match="no graph input binding"):
        key(StandardScaler(with_mean=st.as_data_op(True)))


def test_converted_graph_parameters_use_operand_lineage(csv_path):
    keys = []
    for mean in (True, False):
        dag = read_csv(csv_path).skb.apply(
            StandardScaler(with_mean=st.as_data_op(mean)))
        plan, *_ = optimize(dag, OptConfig(literal_inlining=False))
        keys.append(compute_lineage_keys(plan)[plan[-1]])
    assert all(value is not None for value in keys)
    assert keys[0] != keys[1]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("mean", [False, True])
def test_inlined_parameters_match_literal_lineage(csv_path, nested, mean):
    def build(parameter):
        scaler = StandardScaler(with_mean=parameter)
        estimator = Pipeline([("scale", scaler)]) if nested else scaler
        return read_csv(csv_path).skb.apply(estimator)

    assert plan_keys(build(st.as_data_op(mean))) == plan_keys(build(mean))


def test_a_bound_but_unresolved_seed_is_uncacheable():
    op = TransformerOp(RandomForestClassifier(random_state=st.as_data_op(None)),
                       param_refs={"random_state": OperandRef(1)})
    with pytest.raises(Uncacheable, match="random seed cannot be checked"):
        lineage_key(op, ["source", "none"])


def test_an_unkeyed_input_blocks_estimator_and_descendants(csv_path):
    dag = read_csv(csv_path).skb.apply_func(lambda frame: frame).skb.apply(
        StandardScaler()).skb.select(s.cols("a"))
    plan, *_ = optimize(dag, env=get_data(dag))
    reasons = {}
    keys = compute_lineage_keys(plan, reasons=reasons)
    assert keys[plan[0]] is not None
    assert all(keys[op] is None for op in plan[1:])
    assert "an input has no key" in reasons.values()


def test_estimator_and_selector_keys_are_stable_across_processes(csv_path):
    script = textwrap.dedent("""
        import json
        import sys
        from sklearn.preprocessing import StandardScaler
        import skrub.selectors as s
        from tests.optimizer.test_operator_lineage import plan_keys, read_csv
        dag = read_csv(sys.argv[1]).skb.select(s.numeric()).skb.apply(
            StandardScaler(), cols=s.all() - s.cols("y"))
        print(json.dumps([key for _, key in plan_keys(dag)]))
    """)
    expected = [value for _, value in plan_keys(read_csv(csv_path).skb.select(
        s.numeric()).skb.apply(StandardScaler(), cols=s.all() - s.cols("y")))]
    assert all(value is not None for value in expected)
    for seed in ("1", "2"):
        result = subprocess.run([sys.executable, "-c", script, str(csv_path)],
                                env={**os.environ, "PYTHONHASHSEED": seed},
                                cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, check=True, timeout=60)
        assert json.loads(result.stdout.strip().splitlines()[-1]) == expected
