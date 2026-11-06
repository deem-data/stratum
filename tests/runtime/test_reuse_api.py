"""Configured reuse shares one memory cache across public API runs."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pandas as pd
import pytest

import stratum as st
from stratum import _api
from stratum.runtime import _buffer_pool, _reuse_cache, _scheduler
from tests.optimizer.test_reuse_planning import features, hits, model
from tests.runtime.test_reuse_execution import assert_equal


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(_reuse_cache, "_active", None)
    monkeypatch.setattr(_buffer_pool, "_host_memory_bytes", lambda: 10_000_000)
    monkeypatch.setattr(_buffer_pool, "_DEFAULT_SPILL_ROOT", tmp_path / "buffers")
    with st.config(reuse_cache=False, reuse_cache_memory_fraction=0.1):
        yield


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "data.csv"
    rng = np.random.default_rng(0)
    pd.DataFrame({"a": rng.random(60), "b": rng.random(60), "y": rng.random(60)}).to_csv(
        path, index=False)
    return path


@pytest.fixture
def schedulers(monkeypatch):
    created = []

    def create(*args, **kwargs):
        scheduler = _scheduler.SequentialScheduler(*args, **kwargs)
        created.append(scheduler)
        return scheduler

    monkeypatch.setattr(_api, "SequentialScheduler", create)
    return created


def test_disabled_runs_do_not_allocate_a_cache(csv_path, monkeypatch, schedulers):
    def forbidden(*args, **kwargs):
        pytest.fail("Disabled reuse must not resolve a budget or allocate a cache")

    monkeypatch.setattr(_reuse_cache, "_resolve_memory_budget", forbidden)
    assert _reuse_cache.active_reuse_cache() is None
    _api.evaluate(features(csv_path))
    assert _reuse_cache._active is None
    assert schedulers[-1].reuse_cache is None
    assert all(op.cache_key is None and op.cache_hit is None
               for op in schedulers[-1].linearized_dag)


def test_budget_changes_preserve_instance_and_evict_lru_entries(monkeypatch):
    monkeypatch.setattr(_buffer_pool, "_host_memory_bytes", lambda: 1000)
    with st.config(reuse_cache=True, reuse_cache_memory_fraction=0.24):
        cache = _reuse_cache.active_reuse_cache()
        assert cache.memory_budget == 240
        for key in ("a", "b", "c"):
            assert cache.put(key, np.arange(10, dtype=np.float64))
        cache.get("a")
        with st.config(reuse_cache_memory_fraction=0.08):
            assert _reuse_cache.active_reuse_cache() is cache
            assert cache.memory_budget == 80 and cache.memory_usage == 80
            assert "a" in cache and len(cache) == 1 and cache.stats.evictions == 2
        assert _reuse_cache.active_reuse_cache() is cache
        assert cache.memory_budget == 240 and len(cache) == 1


def test_unknown_host_memory_uses_a_finite_fallback(monkeypatch, caplog):
    monkeypatch.setattr(_buffer_pool, "_host_memory_bytes", lambda: None)
    with st.config(reuse_cache=True):
        assert _reuse_cache.active_reuse_cache().memory_budget == 2 * 1024**3
    assert "reuse cache budget" in caplog.text


def test_tiny_fraction_does_not_turn_off_eviction():
    with st.config(reuse_cache=True, reuse_cache_memory_fraction=1e-20):
        cache = _reuse_cache.active_reuse_cache()
        assert cache.memory_budget == 1
        assert not cache.put("oversized", np.arange(2))


def test_concurrent_first_callers_share_one_cache(monkeypatch):
    constructor = _reuse_cache.ReuseCache
    created = []
    ready = Barrier(8)

    def create(**kwargs):
        cache = constructor(**kwargs)
        created.append(cache)
        return cache

    def get_cache(_):
        ready.wait(timeout=10)
        return _reuse_cache.active_reuse_cache()

    monkeypatch.setattr(_reuse_cache, "ReuseCache", create)
    with st.config(reuse_cache=True), ThreadPoolExecutor(max_workers=8) as executor:
        caches = list(executor.map(get_cache, range(8)))
    assert len(created) == 1
    assert all(cache is created[0] for cache in caches)


@pytest.mark.parametrize("selector", ["default", "greedy"])
def test_evaluation_reuses_pruned_plans_and_can_be_temporarily_disabled(
        csv_path, schedulers, selector):
    with st.config(implementation_selector=selector):
        expected = _api.evaluate(features(csv_path))
    with st.config(reuse_cache=True, implementation_selector=selector, stats=True):
        assert_equal(_api.evaluate(features(csv_path)), expected)
        cache = schedulers[-1].reuse_cache
        assert cache is _reuse_cache._active and cache.stats.puts > 0
        assert_equal(_api.evaluate(features(csv_path)), expected)
        warm = schedulers[-1]
        assert warm.reuse_cache is cache
        assert len(warm.linearized_dag) == 1 and hits(warm.linearized_dag)
        assert warm.timings == [] and warm.pool.active_count == 0
        before = cache.stats
        with st.config(reuse_cache=False, implementation_selector=selector):
            assert_equal(_api.evaluate(features(csv_path)), expected)
            assert schedulers[-1].reuse_cache is None
            assert not hits(schedulers[-1].linearized_dag)
            assert _reuse_cache.active_reuse_cache() is None
        assert cache.stats == before
        assert_equal(_api.evaluate(features(csv_path)), expected)
        assert schedulers[-1].reuse_cache is cache
        assert hits(schedulers[-1].linearized_dag)
    assert _reuse_cache.active_reuse_cache() is None
    assert _reuse_cache._active is cache and len(cache) > 0


@pytest.mark.parametrize("selector", ["default", "greedy"])
def test_evaluation_and_grid_search_share_cached_features(csv_path, selector):
    with st.config(scheduler=True, implementation_selector=selector):
        baseline, predictions = model(csv_path).skb.make_grid_search(
            cv=3, scoring="r2", return_predictions=True)
    with st.config(scheduler=True, reuse_cache=True, implementation_selector=selector):
        _api.evaluate(features(csv_path))
        cache = _reuse_cache._active
        for _ in range(2):
            sched, actual = model(csv_path).skb.make_grid_search(
                cv=3, scoring="r2", return_predictions=True)
            assert sched.reuse_cache is cache
            assert hits(sched.linearized_dag)
            assert all(op.cache_key is None and op.cache_hit is None
                       for op in sched.linearized_dag[sched.pos_split_op:])
            assert_equal(sched.results_, baseline.results_)
            for result, expected in zip(actual, predictions, strict=True):
                assert_equal(result, expected)
            assert sched.pool.active_count == 0
