"""Recompile selected plans at ordinary operator boundaries."""
import gc
import weakref
from unittest.mock import patch

import numpy as np
import pytest

from stratum.optimizer._plan import Plan
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical.rewrites._lifetime_planning import plan_input_removals
from stratum.runtime._reuse_cache import ReuseCache
from stratum.runtime._scheduler import SequentialScheduler
from tests.runtime.test_retained_state import estimator_plan
from tests.runtime.test_reuse_execution import csv_path, model, scheduler_for  # noqa: F401
from tests.optimizer.test_reuse_planning import features


class Add(PhysicalOp):
    fields = ("name", "amount")
    cacheable = True

    def __init__(self, name, calls, *inputs, amount=1):
        super().__init__(name=name, inputs=list(inputs))
        self.amount, self.calls = amount, calls
        for in_ in inputs:
            in_.outputs.append(self)

    def process(self, inputs):
        self.calls.append(self.name)
        return (sum(inputs) if inputs else np.arange(8)) + self.amount


def scheduler(ops, cache=None):
    plan_input_removals(ops, set())
    return SequentialScheduler(ops, None, reuse_cache=cache)


def graph(calls):
    source = Add("source", calls)
    left = Add("left", calls, source)
    right = Add("right", calls, source, amount=2)
    sink = Add("sink", calls, left, right)
    return [source, left, right, sink]


@pytest.mark.parametrize("position", range(5))
def test_recompile_at_every_position_preserves_completed_producers(position):
    calls = []
    ops = graph(calls)
    sched = scheduler(ops)
    try:
        for _ in range(position):
            sched.step()
        completed = list(sched.executed)
        edges = [(op, list(op.inputs), list(op.outputs), list(op.remove_after))
                 for op in ops]
        remaining = sched.recompile()
        assert not set(remaining.ops) & set(ops)
        assert sched.executed == completed and sched.position == 0
        members = set(remaining.ops)
        assert all(in_ in members or in_ in completed
                   for op in remaining.ops for in_ in op.inputs)
        for op, inputs, outputs, removals in edges:
            assert (op.inputs, op.outputs, op.remove_after) == (inputs, outputs, removals)
        np.testing.assert_array_equal(sched.compute(), 2 * np.arange(8) + 6)
        assert calls == ["source", "left", "right", "sink"]
        assert sched.pool.active_count == 0
    finally:
        sched._finish()


def test_repeated_recompilation_reuses_new_hits_and_keeps_materialized_hits():
    calls, cache = [], ReuseCache()
    source = Add("source", calls)
    middle = Add("middle", calls, source)
    sink = Add("sink", calls, middle)
    sched = scheduler([source, middle, sink], cache)
    try:
        sched.recompile()
        executed_source = sched.step()
        pending_middle = sched.plan.ops[sched.position]
        cache.put(pending_middle.lineage, np.arange(8) + 2)
        sched.recompile()
        hit = next(op for op in sched.plan.ops if op.cache_hit is not None)
        assert hit.instance_of is middle
        assert executed_source not in sched.pool.live_variable_map
        cache.clear()
        sched.recompile()
        assert len(sched.plan.ops) == 1
        assert sched.plan.ops[0].inputs == [hit]
        np.testing.assert_array_equal(sched.compute(), np.arange(8) + 3)
        assert calls == ["source", "sink"]
        assert sched.pool.active_count == 0
    finally:
        sched._finish()


def test_recompile_an_already_materialized_result():
    calls, cache = [], ReuseCache()
    sched = scheduler(graph(calls), cache)
    try:
        sched.recompile()
        result = np.arange(8)
        cache.put(sched.plan.result.lineage, result)
        sched.recompile()
        assert len(sched.plan.ops) == 1 and sched.plan.result.cache_hit is not None
        cache.clear()
        sched.recompile()
        assert sched.plan.ops == []
        assert sched.compute() is result
        assert calls == []
    finally:
        sched._finish()


@pytest.mark.parametrize("transformer", [False, True])
def test_recompile_after_fitting_keeps_the_existing_state(transformer):
    source, fit, predict = estimator_plan(transformer=transformer)
    sched = scheduler([source, fit, predict])
    try:
        sched.step()
        sched.step()
        state = sched.state_store.get(fit)
        reference = weakref.ref(state)
        state_bytes = sched.retained_state_memory_usage
        edges = list(fit.outputs)
        sched.recompile()
        new_predict = sched.plan.ops[0]
        assert new_predict.inputs[new_predict.state_ref.k] is fit
        assert new_predict.mode == predict.mode and type(new_predict) is type(predict)
        assert sched.state_store.get(fit) is state
        assert fit.outputs == edges and sched.retained_state_memory_usage == state_bytes
        del state
        assert len(sched.compute()) == 100
        gc.collect()
        assert reference() is None and not sched.state_store
    finally:
        sched._finish()


def test_a_prediction_hit_releases_the_state_it_no_longer_reads(csv_path):
    from sklearn.linear_model import Ridge

    data = features(csv_path)
    dag = data[["a", "b"]].skb.apply(Ridge(), y=data["y"], no_wrap=True)
    cache = ReuseCache()
    sched = scheduler_for(dag, cache)
    try:
        while not (fit := sched.step()).produces_state:
            pass
        assert fit in sched.state_store
        predict = next(op for op in sched.plan.ops[sched.position:]
                       if op.state_ref is not None)
        expected = np.ones(60)
        cache.put(predict.lineage, expected)
        sched.recompile()
        assert not sched.state_store
        np.testing.assert_array_equal(sched.compute(), expected)
        assert sched.pool.active_count == 0
    finally:
        sched._finish()


def test_recompilation_does_not_load_spilled_boundary_values(tmp_path):
    sched = scheduler(graph([]))
    sched.pool._spill_root = tmp_path
    sched.pool.memory_budget = 1
    try:
        source = sched.step()
        assert sched.pool.live_variable_map[source].is_spilled
        with patch("stratum.runtime._buffer_pool.deserialize_object",
                   side_effect=AssertionError("recompilation loaded data")):
            sched.recompile()
        np.testing.assert_array_equal(sched.compute(), 2 * np.arange(8) + 6)
        assert not list(tmp_path.iterdir())
    finally:
        sched._finish()


@pytest.mark.parametrize("state", [False, True])
def test_missing_boundary_values_reject_recompilation_without_replacing_the_plan(state):
    ops = estimator_plan()
    sched = scheduler(ops)
    try:
        sched.step()
        if state:
            sched.step()
            sched.state_store.remove(ops[1])
        else:
            sched.pool.remove(ops[0])
        previous, position = sched.plan, sched.position
        with pytest.raises(ValueError, match="missing fitted state" if state else "missing data"):
            sched.recompile()
        assert sched.plan is previous and sched.position == position
    finally:
        sched._finish()


def test_recompilation_is_rejected_inside_operator_execution():
    op = Add("source", [])
    sched = scheduler([op])
    op.process = lambda inputs: sched.recompile()
    try:
        with pytest.raises(RuntimeError, match="requires an operator boundary"):
            sched.step()
        assert sched.position == 0 and sched.executed == []
    finally:
        sched._finish()


def test_profiles_and_operator_attribution_survive_recompilation():
    sched = scheduler(graph([]))
    original = sched.plan.ops
    from stratum._profiler._observer import observe
    timings = observe(sched).node_timings
    try:
        sched.step()
        sched.recompile(profile=timings)
        sched.step()
        sched.recompile()
        assert sched.plan.profile is timings
        sched.compute()
        assert set(timings) == set(original)
    finally:
        sched._finish()


def test_copy_preserves_shared_inputs_and_edge_multiplicity():
    source = Add("source", [])
    sink = Add("sink", [], source, source)
    copied = Plan([source, sink]).copy()
    new_source, new_sink = copied.ops
    assert new_sink.inputs == [new_source, new_source]
    assert new_source.outputs == [new_sink, new_sink]
    assert source.outputs == [sink, sink]


@pytest.mark.parametrize("boundary", ["preprocessing", "split", "fit", "every"])
@pytest.mark.parametrize("reuse", [False, True])
def test_recompile_during_preprocessing_and_inside_every_fold(csv_path, boundary, reuse):
    baseline = scheduler_for(model(csv_path), search=True)
    expected = baseline.grid_search(cv=3)
    sched = scheduler_for(model(csv_path), ReuseCache() if reuse else None, search=True)
    step = sched.step
    positions = []

    def recompile_after_step():
        op = step()
        if op is not None and (
            boundary == "every"
            or boundary == "preprocessing" and sched.cv_id == -1
            or boundary == "split" and op.is_split_op
            or boundary == "fit" and op.produces_state
        ):
            positions.append(sched.cv_id)
            edges = [(old, list(old.inputs), list(old.outputs)) for old in sched.executed]
            sched.recompile()
            for old, inputs, outputs in edges:
                assert old.inputs == inputs and old.outputs == outputs
        return op

    sched.step = recompile_after_step
    actual = sched.grid_search(cv=3)
    assert positions
    if boundary != "preprocessing":
        assert {0, 1, 2} <= set(positions)
    assert sched.results_.equals(baseline.results_)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(left["vals"][0], right["vals"][0])
    assert sched.pool.active_count == 0 and not sched.state_store


def test_a_materialized_none_is_a_valid_boundary_value():
    source = Add("source", [])
    sink = Add("sink", [], source)
    source.process = lambda inputs: None
    sink.process = lambda inputs: inputs == [None]
    sched = scheduler([source, sink])
    try:
        sched.step()
        sched.recompile()
        assert sched.compute() is True
    finally:
        sched._finish()
