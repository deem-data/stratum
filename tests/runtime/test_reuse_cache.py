"""Memory reuse retains references, respects its byte budget, and is thread-safe."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from threading import Barrier
import weakref

import numpy as np
import pandas as pd
import polars as pl
import pytest

from stratum.optimizer._lineage import lineage_key
from stratum.optimizer.logical._ops import ValueOp
from stratum.runtime._object_size import get_size
from stratum.runtime._reuse_cache import ReuseCache


def _arr(n):
    return np.arange(n, dtype=np.float64)


def test_put_and_get():
    cache = ReuseCache()
    value = _arr(10)
    assert cache.put("key", value)
    assert "key" in cache and "missing" not in cache and len(cache) == 1
    assert cache.get("key") is value
    assert not cache.put("key", _arr(3)), "the existing entry is kept"
    assert cache.get("key") is value
    assert cache.memory_usage == value.nbytes
    assert cache.get("missing") is None
    miss = object()
    assert cache.get("missing", miss) is miss
    assert asdict(cache.stats) == dict(puts=1, rejected=1, hits=2, misses=2, evictions=0)


def test_a_cached_none_is_a_hit():
    cache = ReuseCache()
    assert cache.put("none", None)
    miss = object()
    assert cache.get("none", miss) is None
    assert cache.stats.hits == 1 and cache.stats.misses == 0


@pytest.mark.parametrize("value", [
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
    pd.Series([1, 2]), pl.DataFrame({"a": [1, 2]}), pl.Series([1, 2]),
    _arr(10), (pd.DataFrame({"a": [1]}), {"b": _arr(3)}),
    object(),  # Admission has no serialization or cost filter.
])
def test_supported_objects_are_kept_by_reference(value):
    cache = ReuseCache()
    assert cache.put("value", value)
    assert cache.get("value") is value
    assert cache.memory_usage == get_size(value)


def test_lru_access_changes_eviction_order():
    cache = ReuseCache(memory_budget=160)
    for key in ("a", "b"):
        assert cache.put(key, _arr(10))
    cache.get("a")
    assert cache.put("c", _arr(10))
    assert "a" in cache and "c" in cache and "b" not in cache
    assert cache.memory_usage == 160
    assert cache.stats.evictions == 1
    # Membership checks and rejected duplicates do not refresh LRU order.
    assert "a" in cache
    assert not cache.put("a", _arr(10))
    assert cache.put("d", _arr(10))
    assert "a" not in cache and "c" in cache and "d" in cache


def test_oversized_entries_do_not_displace_existing_values():
    cache = ReuseCache(memory_budget=80)
    value = _arr(10)
    assert cache.put("fits", value)
    assert not cache.put("too-large", _arr(11))
    assert cache.get("fits") is value and len(cache) == 1
    assert cache.memory_usage == 80 and cache.stats.evictions == 0
    assert cache.stats.rejected == 1


def test_large_insert_can_evict_multiple_entries():
    cache = ReuseCache(memory_budget=240)
    for key in ("a", "b", "c"):
        cache.put(key, _arr(10))
    assert cache.put("large", _arr(25))
    assert len(cache) == 1 and cache.memory_usage == 200
    assert cache.stats.evictions == 3


def test_reader_keeps_its_object_after_eviction_and_clear():
    cache = ReuseCache(memory_budget=80)
    cache.put("old", _arr(10))
    held = cache.get("old")
    cache.put("new", _arr(10))
    assert "old" not in cache
    cache.clear()
    np.testing.assert_array_equal(held, _arr(10))
    assert len(cache) == 0 and cache.memory_usage == 0


def test_eviction_and_clear_release_cache_references():
    cache = ReuseCache(memory_budget=80)
    value = _arr(10)
    reference = weakref.ref(value)
    cache.put("old", value)
    del value
    assert reference() is not None
    cache.put("new", _arr(10))
    assert reference() is None
    reference = weakref.ref(cache.get("new"))
    cache.clear()
    assert reference() is None


def test_explicit_eviction_returns_bytes_released():
    cache = ReuseCache()
    for key in ("a", "b", "c"):
        cache.put(key, _arr(10))
    assert cache.evict(0) == 0 and len(cache) == 3
    assert cache.evict(81) == 160
    assert cache.memory_usage == 80 and "c" in cache
    assert cache.evict(1000) == 80
    assert cache.evict(1) == 0 and cache.memory_usage == 0


def test_budget_changes_evict_immediately_and_zero_is_unlimited():
    cache = ReuseCache(memory_budget=0)
    for key in ("a", "b", "c"):
        cache.put(key, _arr(10))
    assert cache.memory_usage == 240
    cache.get("a")
    cache.memory_budget = 80
    assert len(cache) == 1 and "a" in cache and cache.memory_usage == 80
    cache.memory_budget = 0
    assert cache.put("large", _arr(100))
    assert cache.memory_usage == 880


@pytest.mark.parametrize("budget, error", [(-1, ValueError), (1.5, TypeError)])
def test_invalid_budget_does_not_change_cache(budget, error):
    with pytest.raises(error):
        ReuseCache(memory_budget=budget)
    cache = ReuseCache(memory_budget=80)
    cache.put("a", _arr(10))
    with pytest.raises(error):
        cache.memory_budget = budget
    assert cache.memory_budget == 80 and cache.memory_usage == 80 and "a" in cache


def test_clear_also_removes_zero_size_entries_and_allows_reuse():
    cache = ReuseCache(memory_budget=80)
    cache.put("empty", _arr(0))
    cache.put("a", _arr(10))
    before = cache.stats
    cache.clear()
    assert len(cache) == 0 and cache.memory_usage == 0
    assert cache.stats == before
    assert cache.put("a", _arr(10))
    cache.put("b", _arr(10))
    assert "a" not in cache and "b" in cache and cache.stats.evictions == 1


def test_stats_are_independent_snapshots():
    cache = ReuseCache()
    before = cache.stats
    cache.put("a", _arr(10))
    assert before.puts == 0 and cache.stats.puts == 1
    before.hits = 100
    assert cache.stats.hits == 0


def test_instances_share_neither_entries_nor_lru_state():
    first, second = ReuseCache(), ReuseCache()
    first.put("key", _arr(10))
    assert second.get("key") is None and second.memory_usage == 0
    assert first.stats.misses == 0


def test_merkle_keys_reuse_equivalent_configurations():
    cache = ReuseCache()
    key = lineage_key(ValueOp("source-a"), [])
    value = _arr(10)
    cache.put(key, value)
    assert cache.get(lineage_key(ValueOp("source-a"), [])) is value
    assert cache.get(lineage_key(ValueOp("source-b"), [])) is None


def test_concurrent_duplicate_puts_admit_exactly_one_value():
    cache = ReuseCache()
    workers = 8
    start = Barrier(workers)

    def insert(i):
        value = np.full(10, i, dtype=np.int64)
        start.wait(timeout=10)
        return cache.put("shared", value), value

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(insert, range(workers)))
    winners = [value for admitted, value in results if admitted]
    assert len(winners) == 1 and cache.get("shared") is winners[0]
    assert cache.memory_usage == 80 and len(cache) == 1
    assert cache.stats.puts == 1 and cache.stats.rejected == workers - 1


def test_concurrent_reads_writes_evictions_and_clear():
    cache = ReuseCache(memory_budget=320)
    workers = 8
    start = Barrier(workers)
    missing = object()

    def run(worker):
        start.wait(timeout=10)
        for i in range(100):
            key = f"{worker}:{i}"
            cache.put(key, np.full(10, worker * 100 + i, dtype=np.int64))
            held = cache.get(key, missing)
            if i % 7 == 0:
                cache.evict(80)
            if i % 23 == 0:
                cache.clear()
            if held is not missing:
                np.testing.assert_array_equal(held, np.full(10, worker * 100 + i))
            assert 0 <= cache.memory_usage <= 320

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, range(workers)))
    stats = cache.stats
    assert stats.puts == workers * 100 and stats.rejected == 0
    assert stats.hits + stats.misses == workers * 100
    assert cache.memory_usage == len(cache) * 80
