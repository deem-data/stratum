"""SimpleImputer implementation selection and worker-budget propagation."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.impute import SimpleImputer

import stratum as st
from stratum import _rust_backend as rb
from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.logical._ops import TransformerOp
from stratum.optimizer.physical._impl_selection import bind_op
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.physical._transform_execs import (
    SimpleImputerOp, RustSimpleImputer, SklearnSimpleImputer, lower_transformer,
)


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent", "constant"])
def test_selection(selector, strategy):
    ctx = replace(PlanContext.from_flags(), implementation_selector=selector, parallelism=2)
    op = lower_transformer(TransformerOp(estimator=SimpleImputer(strategy=strategy)), ctx)
    assert type(op) is SimpleImputerOp
    bind_op(op, ctx)
    assert type(op) is (RustSimpleImputer if selector == "greedy" else SklearnSimpleImputer)
    if selector == "greedy":
        assert clone(op.estimator)._stratum_worker_budget == 2
        assert clone(op.estimator)._stratum_force_rust


@pytest.mark.parametrize("params", [
    dict(copy=False), dict(missing_values=-1), dict(strategy=np.nanmax),
    dict(strategy="constant", fill_value="missing"),
])
def test_unsupported_parameters(params):
    op = SimpleImputerOp(estimator=SimpleImputer(**params))
    bind_op(op, replace(PlanContext.from_flags(), implementation_selector="greedy"))
    assert type(op) is SklearnSimpleImputer


def test_unavailable_runtime(monkeypatch):
    monkeypatch.setattr(rb, "simple_imputer_fit", None)
    op = SimpleImputerOp(estimator=SimpleImputer())
    bind_op(op, replace(PlanContext.from_flags(), implementation_selector="greedy"))
    assert type(op) is SklearnSimpleImputer


@pytest.mark.parametrize("inline", [False, True])
def test_deferred_parameters(inline):
    with st.config(implementation_selector="greedy"):
        frame = st.as_data_op(pd.DataFrame({"a": [1., np.nan]}))
        plan = frame.skb.apply(SimpleImputer(strategy=st.as_data_op("median")))
        ops, *_ = optimize(plan, OptConfig(literal_inlining=inline))
    imputer, = [op for op in ops if isinstance(op, SimpleImputerOp)]
    assert type(imputer) is (RustSimpleImputer if inline else SklearnSimpleImputer)
