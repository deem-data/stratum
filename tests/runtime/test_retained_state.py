"""The runtime owns fitted objects until their last state consumer finishes."""
import gc
import weakref
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import stratum as st
from stratum.optimizer.physical.rewrites._lifetime_planning import plan_input_removals
from stratum.optimizer._optimize import optimize
from stratum.optimizer.logical._base import All, OperandRef
from stratum.optimizer.physical._generic_execs import ValueExec
from stratum.optimizer.physical._predictor_execs import SklearnKNeighbors
from stratum.optimizer.physical._transform_execs import PassthroughTransformer
from stratum.optimizer.physical._projection_execs import PandasColumnSelectorOp
from stratum.runtime._reuse_cache import ReuseCache
from stratum.runtime._scheduler import SequentialScheduler


def estimator_plan(*, transformer=False, readers=1):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = rng.integers(0, 2, len(X))
    source = ValueExec(X)
    if transformer:
        fit = PassthroughTransformer(estimator=StandardScaler(), cols=All(), no_wrap=True)
    else:
        fit = SklearnKNeighbors(estimator=KNeighborsClassifier(), y=y, cols=All(), no_wrap=True)
    fit.inputs = [source]
    predictions = []
    for _ in range(readers):
        predict = fit.clone()
        predict.mode = "transform" if transformer else "predict"
        predict.inputs = [source, fit]
        predict.state_ref = OperandRef(1)
        predictions.append(predict)
    fit.outputs = predictions
    source.outputs = [fit, *predictions]
    ops = [source, fit, *predictions]
    plan_input_removals(ops, set())
    return ops


@pytest.mark.parametrize("transformer", [False, True])
def test_state_is_measured_once_and_freed_after_last_reader(transformer):
    source, fit, first, last = estimator_plan(transformer=transformer, readers=2)
    prototype = fit.estimator
    # A fit-transform output has a data consumer, independently of state readers.
    if transformer:
        first.inputs.append(fit)
        fit.outputs.append(first)
        plan_input_removals([source, fit, first, last], set())
    cache = ReuseCache()
    last.cache_key = "predictions"
    sched = SequentialScheduler([source, fit, first, last], None, reuse_cache=cache)
    sched.process_op(source)
    sched.process_op(fit)
    state = sched.state_store.get(fit)
    reference = weakref.ref(state)
    size = fit.retained_state_size()
    assert size > 1
    assert sched.retained_state_memory_usage == size
    assert fit.estimator is prototype
    assert fit.estimator is not state
    assert (fit in sched.pool.live_variable_map) == transformer
    del state

    fit._measure_retained_state = Mock(side_effect=AssertionError("must not size on reads"))
    sched.process_op(first)
    assert reference() is not None
    assert sched.retained_state_memory_usage == size
    sched.process_op(last)
    gc.collect()
    assert reference() is None, "the plan must not retain the fitted model"
    assert sched.retained_state_memory_usage == 0
    assert sched.memory_usage == sched.pool.memory_usage
    assert cache.memory_usage > 0
    fit._measure_retained_state.assert_not_called()
    sched._finish()
    assert sched.memory_usage == 0


def test_state_store_failure_cleans_up_execution(monkeypatch):
    ops = estimator_plan()
    sched = SequentialScheduler(ops, None)
    fitted = []

    def fail(key, estimator, *, size):
        fitted.append(weakref.ref(estimator))
        raise MemoryError("state insertion failed")

    monkeypatch.setattr(sched.state_store, "put", fail)
    with pytest.raises(RuntimeError, match="state insertion failed") as error:
        sched.evaluate()
    assert isinstance(error.value.__context__, MemoryError)
    del error  # exception tracebacks can retain execution locals
    gc.collect()
    assert fitted and fitted[0]() is None
    assert sched.state_store.memory_usage == sched.pool.memory_usage == 0


def test_prediction_failure_releases_state_and_pins(monkeypatch):
    ops = estimator_plan()
    sched = SequentialScheduler(ops, None)
    fitted = []

    def fail(inputs):
        fitted.append(weakref.ref(inputs[1]))
        raise ValueError("prediction failed")

    monkeypatch.setattr(ops[-1], "process", fail)
    with pytest.raises(RuntimeError, match="prediction failed"):
        sched.evaluate()
    gc.collect()
    assert fitted[0]() is None
    assert not sched.state_store
    assert sched.pool.active_count == 0


def test_no_fold_execution_releases_one_shot_fitted_state():
    ops, split_pos = optimize(st.as_data_op(pd.DataFrame({"a": [1., 2., 3.]}))
                             .skb.apply(StandardScaler()))
    sched = SequentialScheduler(ops, split_pos)
    result = sched.evaluate()
    assert len(result) == 3
    assert sched.memory_usage == 0
    for op in ops:
        if hasattr(op, "original_estimator"):
            assert not hasattr(op.estimator, "mean_")
            assert not hasattr(op.original_estimator, "mean_")


def test_column_selector_state_is_owned_by_the_store():
    train = pd.DataFrame({"a": [1., 2.], "b": [3., 4.]})
    test = pd.DataFrame({"a": [5.], "b": [6.]})
    source, test_source = ValueExec(train), ValueExec(test)
    fit = PandasColumnSelectorOp(selector=["a"])
    predict = PandasColumnSelectorOp(selector=["a"])
    predict.mode = "transform"
    fit.inputs, fit.outputs = [source], [predict]
    predict.inputs = [test_source, fit]
    predict.state_ref = OperandRef(1)
    ops = [source, test_source, fit, predict]
    plan_input_removals(ops, set())
    sched = SequentialScheduler(ops, None)
    for op in ops[:-1]:
        sched.process_op(op)
    state = sched.state_store.get(fit)
    assert state == ["a"]
    assert not hasattr(fit, "selected_columns")
    del state
    sched.process_op(predict)
    result = sched.pool.pin(predict)
    sched.pool.unpin(predict)
    pd.testing.assert_frame_equal(result, test[["a"]])
    assert not sched.state_store
    sched._finish()


def test_failed_input_lookup_balances_already_pinned_buffers():
    source, fit, predict = estimator_plan()
    sched = SequentialScheduler([source, fit, predict], None)
    sched.process_op(source)
    with pytest.raises(RuntimeError):
        sched.process_op(predict)  # fit has not produced its state yet
    assert sched.pool.live_variable_map[source].pin_count == 0
    sched._finish()
