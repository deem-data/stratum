"""Dedicated StandardScaler lowering and implementation selection."""
from dataclasses import replace

import pytest
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from stratum import _rust_backend as rb
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


def test_subclass_keeps_custom_behavior():
    class CustomScaler(StandardScaler):
        pass
    op = lower_transformer(TransformerOp(estimator=CustomScaler()), PlanContext.from_flags())
    assert not isinstance(op, StandardScalerOp)
