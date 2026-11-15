"""One-hot encoding lowering, reference selection, and conservative fallbacks."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder

from stratum import _rust_backend as rb
import stratum as st
from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.logical._ops import TransformerOp
from stratum.optimizer.physical._impl_selection import bind_op
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.physical._transform_execs import (
    OneHotEncoderOp, RustOneHotEncoder, SklearnOneHotEncoder, lower_transformer,
)


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("drop", [None, "if_binary"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse_output", [True, False])
def test_selection(monkeypatch, selector, drop, dtype, sparse_output):
    # Selection is independent of legacy runtime flags; no kernel is run here.
    monkeypatch.setattr(rb, "HAVE_RUST", True)
    ctx = replace(PlanContext.from_flags(), implementation_selector=selector,
                  rust_backend=False, allow_patch=False)
    op = lower_transformer(TransformerOp(estimator=OneHotEncoder(
        drop=drop, dtype=dtype, sparse_output=sparse_output, handle_unknown="ignore",
    )), ctx)
    assert type(op) is OneHotEncoderOp
    bind_op(op, ctx)
    assert type(op) is (SklearnOneHotEncoder if selector == "default" else RustOneHotEncoder)
    assert op.estimator.drop == drop
    assert op.estimator.dtype == dtype
    if selector == "greedy":
        assert clone(op.estimator)._stratum_force_rust


@pytest.mark.parametrize("params", [
    {"drop": "first"}, {"drop": ["x"]}, {"dtype": np.int32},
    {"handle_unknown": "error"}, {"min_frequency": 2}, {"max_categories": 2},
    {"categories": [["x", "y"]]},
])
def test_unsupported_options_keep_sklearn(params):
    estimator = OneHotEncoder(handle_unknown="ignore").set_params(**params)
    op = OneHotEncoderOp(estimator=estimator)
    bind_op(op, replace(PlanContext.from_flags(), implementation_selector="greedy"))
    assert type(op) is SklearnOneHotEncoder
    assert op.estimator is estimator


def test_missing_native_runtime_keeps_sklearn(monkeypatch):
    monkeypatch.setattr(rb, "ohe_transform", None)
    op = OneHotEncoderOp(estimator=OneHotEncoder(handle_unknown="ignore"))
    bind_op(op, replace(PlanContext.from_flags(), implementation_selector="greedy"))
    assert type(op) is SklearnOneHotEncoder


@pytest.mark.parametrize("inline", [False, True])
def test_plan_valued_drop_requires_literal_inlining(inline):
    with st.config(implementation_selector="greedy"):
        frame = st.as_data_op(pd.DataFrame({"a": ["x", "y"]}))
        plan = frame.skb.apply(OneHotEncoder(
            drop=st.as_data_op(None), handle_unknown="ignore", sparse_output=False,
        ))
        ops, *_ = optimize(plan, OptConfig(literal_inlining=inline))
    op, = [op for op in ops if isinstance(op, OneHotEncoderOp)]
    expected = RustOneHotEncoder if inline and rb.HAVE_RUST else SklearnOneHotEncoder
    assert type(op) is expected
