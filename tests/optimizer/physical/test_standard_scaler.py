"""Dedicated StandardScaler lowering and implementation selection."""
from dataclasses import replace

import pytest
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from stratum import _rust_backend as rb
import stratum as st
from stratum.adapters.standard_scaler import supports_rust_standard_scaler
from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.logical._ops import TransformerOp
from stratum.optimizer.physical._impl_selection import bind_op
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.physical._transform_execs import (
    NumpyStandardScalerExec, RustStandardScaler, SklearnStandardScaler,
    StandardScalerOp, lower_transformer,
)


@pytest.mark.parametrize("selector", ["default", "greedy"])
def test_lower_and_select(selector):
    ctx = PlanContext.from_flags()
    ctx = replace(ctx, implementation_selector=selector, parallelism=2)
    op = lower_transformer(TransformerOp(estimator=StandardScaler()), ctx)
    assert type(op) is StandardScalerOp
    bind_op(op, ctx)
    expected = SklearnStandardScaler if selector == "default" else (
        RustStandardScaler if rb.standard_scale_fit else NumpyStandardScalerExec
    )
    assert type(op) is expected
    if expected is RustStandardScaler:
        assert clone(op.estimator)._stratum_force_rust
        assert op.original_estimator._stratum_worker_budget == 2
        assert clone(op.estimator)._stratum_worker_budget == 2


def test_missing_native_runtime_uses_numpy(monkeypatch):
    monkeypatch.setattr(rb, "standard_scale_fit", None)
    op = StandardScalerOp(estimator=StandardScaler())
    bind_op(op, replace(PlanContext.from_flags(), implementation_selector="greedy"))
    assert type(op) is NumpyStandardScalerExec


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("parameter", ["with_mean", "with_std", "copy"])
@pytest.mark.parametrize("numpy_only", [False, True])
def test_literal_parameters_enable_specialized_implementations(monkeypatch, inline, parameter, numpy_only):
    # Exercise selection with and without native kernels on every installation.
    # No kernel is executed by this test.
    monkeypatch.setattr(rb, "HAVE_RUST", True)
    monkeypatch.setattr(rb, "standard_scale_fit", None if numpy_only else object())
    monkeypatch.setattr(rb, "standard_scale_transform", object())
    frame = st.as_data_op(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    estimator = StandardScaler(**{parameter: st.as_data_op(True)})
    with st.config(implementation_selector="greedy", num_threads=2):
        plan, *_ = optimize(frame.skb.apply(estimator), OptConfig(literal_inlining=inline))
    scaler, = [op for op in plan if isinstance(op, StandardScalerOp)]

    expected = SklearnStandardScaler if not inline else (
        NumpyStandardScalerExec if numpy_only else RustStandardScaler)
    assert type(scaler) is expected
    assert set(scaler.param_refs) == (set() if inline else {parameter})
    if inline:
        assert scaler.original_estimator.get_params()[parameter] is True


@pytest.mark.parametrize("parameter", ["with_mean", "with_std", "copy"])
@pytest.mark.parametrize("value", [True, False])
def test_deferred_parameters_are_not_evaluated_by_rust_supports(monkeypatch, parameter, value):
    # Exercise the parameter guard even on installations without native kernels.
    monkeypatch.setattr(rb, "HAVE_RUST", True)
    monkeypatch.setattr(rb, "standard_scale_fit", object())
    monkeypatch.setattr(rb, "standard_scale_transform", object())
    estimator = StandardScaler(**{parameter: st.as_data_op(value)})
    supported, reason = supports_rust_standard_scaler(estimator)
    assert not supported
    assert "requires" in reason


def test_numpy_boolean_parameters_remain_supported(monkeypatch):
    monkeypatch.setattr(rb, "HAVE_RUST", True)
    monkeypatch.setattr(rb, "standard_scale_fit", object())
    monkeypatch.setattr(rb, "standard_scale_transform", object())
    assert supports_rust_standard_scaler(StandardScaler(
        with_mean=np.bool_(True), with_std=np.bool_(True), copy=np.bool_(True),
    ))[0]


def test_subclass_keeps_custom_behavior():
    class CustomScaler(StandardScaler):
        pass
    op = lower_transformer(TransformerOp(estimator=CustomScaler()), PlanContext.from_flags())
    assert not isinstance(op, StandardScalerOp)
