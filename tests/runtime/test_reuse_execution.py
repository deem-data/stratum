"""Scheduler reuse preserves results, cache ownership, and buffer lifetimes."""
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import Ridge

import stratum as st
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import OptConfig, SearchConfig, optimize
from stratum.optimizer.physical.rewrites._reuse_planning import CacheHit
from stratum.optimizer.logical._scoring import resolve_scoring
from stratum.optimizer.logical._ops import PredictorOp
from stratum.optimizer.physical._impl_selection import FlagBasedSelector
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.runtime import _buffer_pool, _scheduler
from stratum.runtime._reuse_cache import ReuseCache
from stratum.runtime._scheduler import SequentialScheduler
from tests.optimizer.test_reuse_planning import features, hits, model


@pytest.fixture(autouse=True)
def spill_directory(tmp_path, monkeypatch):
    root = tmp_path / "buffers"
    monkeypatch.setattr(_scheduler, "BufferPool", lambda: _buffer_pool.BufferPool(spill_root=root))
    return root


@pytest.fixture
def csv_path(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": rng.random(60), "b": rng.random(60), "y": rng.random(60)}).to_csv(
        path, index=False)
    return path


def plan_for(dag, cache=None, backend="pandas", search=False):
    search_config = SearchConfig(resolve_scoring("r2"), return_predictions=True) if search else None
    return optimize(dag, OptConfig(selector=FlagBasedSelector(backend=backend)),
                    env=get_data(dag), reuse_cache=cache, search=search_config)


def scheduler_for(dag, cache=None, backend="pandas", search=False):
    return SequentialScheduler(*plan_for(dag, cache, backend, search),
                               reuse_cache=cache, print_heavy_hitters=True)


def assert_equal(actual, expected):
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected)
    elif isinstance(expected, pl.DataFrame):
        assert actual.equals(expected)
    else:
        np.testing.assert_array_equal(actual, expected)


def assert_released(scheduler, directory):
    assert scheduler.pool.active_count == 0 and scheduler.pool.memory_usage == 0
    assert not scheduler.state_store
    assert not directory.exists() or not list(directory.iterdir())


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_a_whole_plan_hit_skips_execution_and_caller_mutations_are_isolated(
        csv_path, backend, spill_directory):
    cache = ReuseCache()
    baseline = scheduler_for(features(csv_path), backend=backend).evaluate()
    cold = scheduler_for(features(csv_path), cache, backend)
    eligible = {op.cache_key for op in cold.linearized_dag if op.cache_key is not None}
    first = cold.evaluate()
    assert_equal(first, baseline)
    assert cache.stats.puts == len(eligible) > 0
    assert_released(cold, spill_directory)

    # Both a newly computed output and a reused one belong to their caller.
    for result in (first, scheduler_for(features(csv_path), cache, backend).evaluate()):
        if backend == "pandas":
            result.loc[0, "a"] = -100
        else:
            result[0, "a"] = -100
        warm = scheduler_for(features(csv_path), cache, backend)
        assert len(warm.linearized_dag) == 1 and hits(warm.linearized_dag)
        warm.linearized_dag[0].process = Mock(side_effect=AssertionError("cache hit executed"))
        assert_equal(warm.evaluate(), baseline)
        assert warm.timings == []
        assert_released(warm, spill_directory)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_evaluation_reuses_only_before_the_split(csv_path, backend, spill_directory):
    cache = ReuseCache()
    baseline = scheduler_for(model(csv_path), backend=backend).evaluate()
    cold = scheduler_for(model(csv_path), cache, backend)
    assert_equal(cold.evaluate(), baseline)
    puts = cache.stats.puts
    warm = scheduler_for(model(csv_path), cache, backend)
    assert hits(warm.linearized_dag)
    assert all(op.cache_key is None and op.cache_hit is None
               for op in warm.linearized_dag[warm.pos_split_op:])
    for op in hits(warm.linearized_dag):
        op.process = Mock(side_effect=AssertionError("cache hit executed"))
    assert_equal(warm.evaluate(), baseline)
    assert cache.stats.puts == puts
    assert_released(cold, spill_directory)
    assert_released(warm, spill_directory)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_an_evaluated_prefix_serves_cross_validation(csv_path, backend, spill_directory):
    cache = ReuseCache()
    scheduler_for(features(csv_path), cache, backend).evaluate()
    baseline = scheduler_for(model(csv_path), backend=backend, search=True)
    expected_predictions = baseline.grid_search(cv=3)
    for _ in range(2):
        reused = scheduler_for(model(csv_path), cache, backend, search=True)
        assert hits(reused.linearized_dag)
        predictions = reused.grid_search(cv=3)
        assert_equal(reused.results_, baseline.results_)
        for actual, expected in zip(predictions, expected_predictions, strict=True):
            assert_equal(actual, expected)
        assert_released(reused, spill_directory)


def test_model_edits_reuse_features_and_fit_the_new_model(csv_path):
    cache = ReuseCache()
    scheduler_for(model(csv_path), cache).evaluate()
    expected = scheduler_for(model(csv_path, alpha=2.0)).evaluate()
    changed = scheduler_for(model(csv_path, alpha=2.0), cache)
    assert hits(changed.linearized_dag)
    assert_equal(changed.evaluate(), expected)


def test_full_data_predictions_are_cached_without_caching_fitted_state(csv_path):
    def pipeline():
        data = features(csv_path)
        return data[["a", "b"]].skb.apply(Ridge(), y=data["y"], no_wrap=True)

    cache = ReuseCache()
    cold = scheduler_for(pipeline(), cache)
    assert cold.pos_split_op is None
    fit, predict = [op for op in cold.linearized_dag if isinstance(op, PredictorOp)]
    assert fit.mode == "fit" and fit.cache_key is None
    assert predict.mode == "predict" and predict.cache_key is not None
    expected = cold.evaluate()

    warm = scheduler_for(pipeline(), cache)
    assert not any(op.produces_state for op in warm.linearized_dag)
    assert hits(warm.linearized_dag)
    assert_equal(warm.evaluate(), expected)


def test_a_cached_fit_output_does_not_replace_required_state(csv_path):
    from sklearn.preprocessing import StandardScaler
    from stratum.optimizer._lineage import compute_lineage_keys
    from stratum.optimizer.physical.rewrites._reuse_planning import apply_reuse_cache, cacheable_region
    from stratum.optimizer.physical.rewrites._lifetime_planning import plan_input_removals
    from stratum.optimizer.physical.rewrites._linearization import linearize_dag

    data = features(csv_path)
    dag = data[["a", "b"]].skb.apply(StandardScaler()).skb.apply(Ridge(), y=data["y"])
    plan, split_pos = plan_for(dag)
    fit = next(op for op in plan if isinstance(op, PredictorOp) and op.produces_state)
    cache = ReuseCache()
    keys = compute_lineage_keys(plan, cacheable_region(plan))
    cache.put(keys[fit], None)
    apply_reuse_cache(plan[-1], cache)
    assert fit.cache_hit is None and fit.cache_key is None
    plan, split_pos = linearize_dag(plan[-1])
    plan_input_removals(plan, set())
    actual = SequentialScheduler(plan, split_pos, reuse_cache=cache).evaluate()
    assert_equal(actual, scheduler_for(dag).evaluate())


@pytest.mark.parametrize("after_loading", [False, True])
def test_cache_eviction_does_not_invalidate_a_planned_hit(csv_path, after_loading):
    cache = ReuseCache()
    expected = scheduler_for(model(csv_path), cache).evaluate()
    plan = plan_for(model(csv_path), cache)
    assert hits(plan[0])
    if not after_loading:
        cache.clear()
    # Hits are owned by the plan, so they can run without a cache to write to.
    warm = SequentialScheduler(*plan)
    if after_loading:
        cache.clear()
    assert_equal(warm.evaluate(), expected)


def test_a_plan_hands_its_hits_to_only_one_scheduler(csv_path):
    cache = ReuseCache()
    scheduler_for(features(csv_path), cache).evaluate()
    plan = plan_for(features(csv_path), cache)
    first = SequentialScheduler(*plan, reuse_cache=cache)
    with pytest.raises(RuntimeError, match="optimize the plan again"):
        SequentialScheduler(*plan, reuse_cache=cache)
    assert_equal(first.evaluate(), scheduler_for(features(csv_path)).evaluate())


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_cold_and_warm_reuse_work_with_pool_spilling(csv_path, backend, spill_directory):
    cache = ReuseCache()
    baseline = scheduler_for(model(csv_path), backend=backend, search=True)
    baseline.grid_search(cv=3)
    for _ in range(2):
        sched = scheduler_for(model(csv_path), cache, backend, search=True)
        sched.pool.memory_budget = 1
        sched.pool.check_for_eviction()  # also spill any hits loaded at construction
        sched.grid_search(cv=3)
        assert_equal(sched.results_, baseline.results_)
        assert sched.pool.stats.evictions > 0
        assert cache.stats.puts > 0 and cache.memory_usage > 0
        assert_released(sched, spill_directory)


def test_rejected_cache_admission_does_not_affect_execution(csv_path, spill_directory):
    cache = ReuseCache(memory_budget=1)
    expected = scheduler_for(features(csv_path)).evaluate()
    for _ in range(2):
        sched = scheduler_for(features(csv_path), cache)
        assert not hits(sched.linearized_dag)
        assert_equal(sched.evaluate(), expected)
        assert_released(sched, spill_directory)
    assert cache.stats.rejected > 0 and cache.stats.puts == 0 and len(cache) == 0


def _increment_frame(frame):
    # Operators own any copy needed by an in-place implementation.
    frame = frame.copy()
    frame["a"] += 1
    return frame


def test_a_udf_that_copies_before_writing_preserves_the_cached_source(csv_path):
    cache = ReuseCache()
    expected = pd.read_csv(csv_path)
    for _ in range(2):
        data = st.deferred(pd.read_csv)(str(csv_path))
        sched = scheduler_for(data.skb.apply_func(_increment_frame), cache)
        actual = sched.evaluate()
        pd.testing.assert_frame_equal(actual, expected.assign(a=expected["a"] + 1))
        source = scheduler_for(st.deferred(pd.read_csv)(str(csv_path)), cache)
        assert hits(source.linearized_dag)
        pd.testing.assert_frame_equal(source.evaluate(), expected)


def test_numpy_cache_results_are_isolated_from_callers(tmp_path):
    path = tmp_path / "values.npy"
    expected = np.arange(10)
    np.save(path, expected)
    cache = ReuseCache()
    for _ in range(3):
        sched = scheduler_for(st.deferred(np.load)(str(path)), cache)
        actual = sched.evaluate()
        assert_equal(actual, expected)
        actual[:] = -1


class ResultOp(PhysicalOp):
    """A controlled result for runtime ownership tests, independent of planning."""

    def __init__(self, value, key=None):
        super().__init__()
        self.value = value
        self.cache_key = key

    def process(self, inputs):
        return self.value


@pytest.mark.parametrize("value", [
    np.arange(3),
    pd.DataFrame({"x": [1, 2]}),
    pl.DataFrame({"x": [1, 2]}),
    {"arrays": [np.arange(3)], "options": {"x": [1, 2]}},
])
def test_cache_and_pool_share_values_with_read_only_consumers(value):
    cache = ReuseCache()
    for warm in (False, True):
        source = ResultOp(value, "shared")
        if warm:
            source.cache_hit = CacheHit("shared", cache.get("shared"))
            source.process = Mock(side_effect=AssertionError("cache hit executed"))
        sink = ResultOp(None)
        sink.inputs = [source]
        seen = []

        def read(inputs):
            seen.append(inputs[0])
            return len(inputs[0])

        sink.process = read
        # A hit can also run without a cache to write to.
        sched = SequentialScheduler([source, sink], None, reuse_cache=None if warm else cache)
        try:
            with patch.object(_scheduler, "_copy_result", side_effect=AssertionError("copied")):
                sched.process_op(source)
                assert cache.get("shared") is value
                assert sched.pool.pin(source) is value
                sched.pool.unpin(source)
                sched.process_op(sink)
            assert seen[0] is value
        finally:
            sched._finish()


def test_execution_without_reuse_returns_its_original_result():
    value = np.arange(3)
    assert SequentialScheduler([ResultOp(value)], None).evaluate() is value


def test_nested_results_are_isolated_when_returned_to_callers():
    value = {"arrays": [np.arange(3)], "options": {"x": [1, 2]}}
    expected = {"arrays": [np.arange(3)], "options": {"x": [1, 2]}}
    cache = ReuseCache()
    cold = SequentialScheduler([ResultOp(value, "nested")], None, reuse_cache=cache)
    result = cold.evaluate()
    result["arrays"][0][:] = -1
    result["options"]["x"].append(3)
    for _ in range(2):
        op = ResultOp(None)
        op.cache_hit = CacheHit("nested", cache.get("nested"))
        actual = SequentialScheduler([op], None).evaluate()
        assert_equal(actual["arrays"][0], expected["arrays"][0])
        assert actual["options"] == expected["options"]
        actual["arrays"][0][:] = -1
        actual["options"]["x"].append(3)


def test_a_cached_none_is_consumed_without_execution():
    cache = ReuseCache()
    assert SequentialScheduler([ResultOp(None, "none")], None, reuse_cache=cache).evaluate() is None
    op = ResultOp("must not execute")
    op.cache_hit = CacheHit("none", cache.get("none"))
    assert SequentialScheduler([op], None).evaluate() is None


@pytest.mark.parametrize("kind", ["frame", "series", "array", "nested-frame",
                                  "polars-frame", "polars-series"])
def test_mutable_objects_inside_values_cannot_change_the_cache(kind):
    if kind in ("frame", "nested-frame"):
        value = pd.DataFrame([[[1], [3]], [[2], [4]]], columns=["x", "x"], index=[5, 9])
        if kind == "nested-frame":
            value = {"frames": [value]}
            cell = lambda result: result["frames"][0].iloc[0, 0]
        else:
            cell = lambda result: result.iloc[0, 0]
    elif kind == "series":
        value = pd.Series([[1], [2]], index=[5, 9], name="items")
        cell = lambda result: result.iloc[0]
    elif kind in ("polars-frame", "polars-series"):
        value = pl.Series("items", [[1], [2]], dtype=pl.Object)
        if kind == "polars-frame":
            value = value.to_frame()
            cell = lambda result: result["items"][0]
        else:
            cell = lambda result: result[0]
    else:
        value = np.empty(2, dtype=object)
        value[:] = [[1], [2]]
        cell = lambda result: result[0]
    cache = ReuseCache()
    cold = SequentialScheduler([ResultOp(value, "objects")], None, reuse_cache=cache)
    cell(cold.evaluate()).append(99)
    assert cell(cache.get("objects")) == [1]
    for _ in range(2):
        op = ResultOp(None)
        op.cache_hit = CacheHit("objects", cache.get("objects"))
        warm = SequentialScheduler([op], None).evaluate()
        assert cell(warm) == [1]
        cell(warm).append(99)
    assert cell(cache.get("objects")) == [1]


def test_grid_search_detaches_predictions_stored_in_object_cells():
    value = np.arange(3)
    table = pl.DataFrame({"id": ["candidate"],
                          "vals": pl.Series("vals", [value], dtype=pl.Object)})
    sched = SequentialScheduler([], None, reuse_cache=ReuseCache())
    sched._grid_search = Mock(return_value=[table])
    result = sched.grid_search()[0]
    np.testing.assert_array_equal(result["vals"][0], value)
    result["vals"][0][:] = -1
    np.testing.assert_array_equal(value, np.arange(3))


def test_admission_never_reloads_a_spilled_output(spill_directory):
    cache = ReuseCache()
    op = ResultOp(np.arange(10), "array")
    sched = SequentialScheduler([op], None, reuse_cache=cache)
    sched.pool.memory_budget = 1
    with patch.object(_buffer_pool, "deserialize_object", side_effect=AssertionError("reloaded")):
        sched.process_op(op)
        assert sched.pool.stats.evictions == 1
        assert_equal(cache.get("array"), op.value)
        sched._finish()
    assert_released(sched, spill_directory)


def _fail(frame):
    raise ValueError("deliberate execution failure")


@pytest.mark.parametrize("method", ["evaluate", "grid_search"])
def test_execution_failure_releases_buffers_and_keeps_successful_admissions(
        csv_path, method, spill_directory):
    cache = ReuseCache()
    data = st.deferred(pd.read_csv)(str(csv_path))
    with st.config_context(eager_data_ops=False):
        sched = scheduler_for(data.skb.apply_func(_fail), cache)
    sched.pool.memory_budget = 1
    with pytest.raises(RuntimeError, match="deliberate execution failure"):
        getattr(sched, method)()
    assert cache.stats.puts > 0
    assert_released(sched, spill_directory)
    pd.testing.assert_frame_equal(scheduler_for(data, cache).evaluate(), pd.read_csv(csv_path))


def test_cross_validation_failure_releases_buffers(csv_path, spill_directory):
    cache = ReuseCache()
    sched = scheduler_for(model(csv_path), cache, search=True)
    sched.pool.memory_budget = 1
    with pytest.raises(ValueError, match="n_splits"):
        sched.grid_search(cv=100)
    assert cache.stats.puts > 0
    assert_released(sched, spill_directory)


def test_constructor_failure_cleans_up_hits_already_loaded(spill_directory, monkeypatch):
    first, consumed = ResultOp(None), ResultOp(None)
    first.cache_hit = CacheHit("first", np.arange(10))
    consumed.cache_hit = CacheHit("consumed", None)
    consumed.cache_hit.take()
    monkeypatch.setattr(_buffer_pool, "_resolve_memory_budget", lambda fraction: 1)
    with pytest.raises(RuntimeError, match="already consumed"):
        SequentialScheduler([first, consumed], None)
    assert not list(spill_directory.iterdir())
