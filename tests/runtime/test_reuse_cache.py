"""ReuseCache: a lineage-keyed memory/disk cache that survives the process."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from stratum.runtime._eviction_policy import CostSizePolicy, make_eviction_policy
from stratum.runtime._object_serialization import serialize_object
from stratum.runtime._reuse_cache import ReuseCache


def _arr(n):
    return np.arange(n, dtype=np.float64)  # 8 bytes per element


class Opaque:
    """An object the serializer has no format for."""


@pytest.fixture
def root(tmp_path):
    return tmp_path / "cache"


def files(root):
    return sorted(p.name for p in root.iterdir()
                  if p.name not in ("manifest.json", "history.json"))


def test_put_and_get(root):
    cache = ReuseCache(root)
    assert cache.put("k", _arr(10))
    assert "k" in cache and "j" not in cache
    np.testing.assert_array_equal(cache.get("k"), _arr(10))
    assert cache.stats.memory_hits == 1
    assert not cache.put("k", _arr(3)), "entries are immutable"
    np.testing.assert_array_equal(cache.get("k"), _arr(10))
    assert cache.get("missing") is None
    miss = object()
    assert cache.get("missing", miss) is miss
    assert cache.stats.misses == 2


def test_a_cached_none_is_a_hit(root):
    cache = ReuseCache(root)
    cache.put("none", None)
    miss = object()
    assert cache.get("none", miss) is None


def test_a_reader_keeps_its_object_after_eviction(root):
    cache = ReuseCache(root, memory_budget=10_000)
    cache.put("unserializable", Opaque())
    held = cache.get("unserializable")
    cache.put("a", _arr(700))
    cache.put("b", _arr(700))  # over budget: the opaque entry cannot spill and is lost
    assert "unserializable" not in cache
    assert isinstance(held, Opaque), "nothing waits for the reader, nor breaks it"


def test_memory_eviction_spills_and_reload_keeps_the_file(root):
    cache = ReuseCache(root, memory_budget=10_000)
    cache.put("a", _arr(500))
    cache.put("b", _arr(500))
    cache.put("c", _arr(500))  # 12_000 B: a (LRU) spills
    assert not cache.entries["a"].in_memory and cache.entries["a"].handle is not None
    assert cache.memory_usage == 8_000 and len(files(root)) == 1
    np.testing.assert_array_equal(cache.get("a"), _arr(500))  # b spills to make room
    assert cache.stats.disk_hits == 1
    assert cache.entries["a"].handle is not None, "a loaded entry keeps its disk copy"
    cache.put("d", _arr(500))  # c, the LRU now, spills
    cache.put("e", _arr(500))  # a leaves memory again, without being rewritten
    assert not cache.entries["a"].in_memory
    assert cache.stats.spills == 3  # a, b, c -- never a twice


def test_unserializable_entries_are_dropped(root):
    cache = ReuseCache(root, memory_budget=10_000)
    cache.put("lost", Opaque())
    cache.put("a", _arr(700))
    cache.put("b", _arr(700))  # over budget: the opaque entry is the oldest
    assert "lost" not in cache and cache.stats.drops == 1
    assert cache.entries["a"].handle is not None


def test_disk_eviction(root, tmp_path):
    one_copy = serialize_object(_arr(10), tmp_path / "probe").size_on_disk
    cache = ReuseCache(root, disk_budget=one_copy)
    cache.put("old", _arr(10))
    cache.put("new", _arr(10))
    cache.flush()  # writes both; the older one is evicted from disk at once
    assert "old" in cache and cache.entries["old"].handle is None
    assert cache.entries["old"].in_memory, "an entry in memory survives losing its file"
    assert cache.entries["new"].handle is not None
    cache.evict(cache.memory_usage)  # "old" cannot reach disk and is dropped
    cache.disk_budget = 1
    cache.check_for_eviction()  # "new" loses its file, its only copy
    assert "old" not in cache and "new" not in cache
    assert files(root) == []


def test_flush_persists_across_instances(root):
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}, index=[10, 20, 30])
    cache = ReuseCache(root)
    cache.put("frame", frame, cost=1.5, label="Assign(c)")
    cache.put("polars", pl.DataFrame({"a": [1, 2]}))
    cache.put("tuple", (frame, 3, "s"))
    cache.put("opaque", Opaque())
    cache.flush()

    reopened = ReuseCache(root)
    assert set(reopened.entries) == {"frame", "polars", "tuple"}
    assert reopened.cost("frame") == 1.5
    assert reopened.entries["frame"].label == "Assign(c)"
    assert reopened.tier("frame") == "disk" and reopened.tier("missing") is None
    pd.testing.assert_frame_equal(reopened.get("frame"), frame)
    assert reopened.get("polars").equals(pl.DataFrame({"a": [1, 2]}))
    t = reopened.get("tuple")
    pd.testing.assert_frame_equal(t[0], frame)
    assert t[1:] == (3, "s")


def test_restart_restores_lru_order_and_removes_orphans(root):
    cache = ReuseCache(root)
    for key in "abc":
        cache.put(key, _arr(10))
    cache.flush()
    cache.get("a")  # a is now the most recent
    cache.save_manifest()
    orphan = "0" * 32 + ".parquet"
    (root / orphan).write_bytes(b"left by a crash")
    (root / "notes.txt").write_text("not the cache's")

    reopened = ReuseCache(root)
    assert list(reopened.disk_policy.victims()) == ["b", "c", "a"]
    assert orphan not in files(root)
    assert "notes.txt" in files(root), "only files the cache writes are orphans"


def test_spilled_entries_survive_a_crash(root):
    cache = ReuseCache(root, memory_budget=10_000)
    cache.put("a", _arr(500))
    cache.put("b", _arr(500))
    cache.put("c", _arr(500))  # over budget: a is spilled
    assert cache.entries["a"].handle is not None
    # No flush: the manifest was rewritten when a reached disk.
    reopened = ReuseCache(root)
    assert set(reopened.entries) == {"a"}
    np.testing.assert_array_equal(reopened.get("a"), _arr(500))


def test_unreadable_manifest_is_ignored(root):
    root.mkdir()
    (root / "manifest.json").write_text("{not json")
    assert len(ReuseCache(root)) == 0


def test_put_spilled_takes_the_file(root, tmp_path):
    spill_dir = tmp_path / "pool"
    spill_dir.mkdir()
    handle = serialize_object(_arr(10), spill_dir / "x")
    cache = ReuseCache(root)
    assert cache.put_spilled("k", handle, size=80, cost=2.0)
    assert list(spill_dir.iterdir()) == []
    assert not cache.entries["k"].in_memory
    np.testing.assert_array_equal(cache.get("k"), _arr(10))
    skeleton = serialize_object([_arr(3)], spill_dir / "y")
    assert not cache.put_spilled("j", skeleton, size=24), "a skeleton spill cannot move"


def test_oversized_entries_are_rejected(root):
    cache = ReuseCache(root, memory_budget=100)
    assert not cache.put("big", _arr(100))
    assert "big" not in cache and cache.stats.rejected == 1


def test_entries_larger_than_the_disk_budget_never_reach_disk(root, tmp_path):
    cache = ReuseCache(root, disk_budget=5_000)
    cache.put("small", _arr(100))
    cache.put("big", _arr(1000))  # 8 KB: alone over the disk budget
    cache.evict(cache.memory_usage)  # both leave memory; only "small" reaches disk
    assert "big" not in cache and cache.stats.drops == 1
    assert "small" in cache and cache.entries["small"].handle is not None
    handle = serialize_object(_arr(1000), tmp_path / "x")
    assert not cache.put_spilled("big", handle, size=8_000)
    assert handle.path.exists(), "a rejected spill stays with its owner"


def test_an_unreadable_entry_is_discarded(root):
    cache = ReuseCache(root)
    cache.put("a", _arr(10))
    cache.flush()
    cache.evict(cache.memory_usage)
    cache.entries["a"].handle.path.write_bytes(b"damaged")
    assert cache.get("a") is None, "an unreadable entry reads as a miss"
    assert cache.stats.discards == 1
    assert "a" not in cache and files(root) == []
    assert len(ReuseCache(root)) == 0, "the manifest forgot it too"


def test_clear(root):
    cache = ReuseCache(root)
    cache.put("a", _arr(10))
    cache.put("b", _arr(10))
    cache.flush()
    cache.clear()
    assert len(cache) == 0 and files(root) == []
    assert cache.memory_usage == 0 and cache.disk_usage == 0
    assert len(ReuseCache(root)) == 0


def test_cost_size_policy():
    policy = CostSizePolicy()
    policy.on_insert("cheap", size=100, cost=1.0)  # 0.01 s/B
    policy.on_insert("dear", size=100, cost=10.0)  # 0.1 s/B
    policy.on_insert("big_dear", size=10_000, cost=10.0)  # 0.001 s/B
    policy.on_insert("unknown_old", size=100)
    policy.on_insert("unknown_new", size=100)
    policy.on_access("unknown_old")
    assert list(policy.victims()) == [
        "unknown_new", "unknown_old", "big_dear", "cheap", "dear"]
    policy.on_remove("cheap")
    assert "cheap" not in list(policy.victims())


def test_cache_evicts_in_policy_order(root):
    cache = ReuseCache(root, memory_budget=10_000, memory_policy=CostSizePolicy())
    cache.put("dear", _arr(500), cost=10.0)
    cache.put("cheap", _arr(500), cost=0.1)
    cache.put("new", _arr(500), cost=5.0)
    assert not cache.entries["cheap"].in_memory
    assert cache.entries["dear"].in_memory and cache.entries["new"].in_memory


def test_make_eviction_policy():
    assert isinstance(make_eviction_policy("cost_size"), CostSizePolicy)
    with pytest.raises(ValueError, match="valid policies"):
        make_eviction_policy("random")


# --- history --------------------------------------------------------------------

def test_history_outlives_the_entry(root):
    cache = ReuseCache(root, memory_budget=10_000)
    cache.observe("a", label="Assign(c)")
    cache.put("a", Opaque(), cost=2.0, label="Assign(c)")
    cache.get("a")
    cache.put("b", _arr(700))
    cache.put("c", _arr(700))  # "a" cannot spill: it leaves the cache
    assert "a" not in cache
    record = cache.history["a"]
    assert (record.seen, record.hits, record.admissions, record.evictions) == (1, 1, 1, 1)
    assert record.label == "Assign(c)" and record.cost == 2.0 and record.size > 0


def test_history_counts_admit_evict_cycles(root):
    cache = ReuseCache(root)
    for _ in range(3):  # each plan needs "a", admits it, and loses it again
        cache.observe("a")
        cache.put("a", _arr(10))
        cache.evict(cache.memory_usage)  # spilled to disk ...
        cache.evict_disk(cache.disk_usage)  # ... and deleted
        assert "a" not in cache
    record = cache.history["a"]
    assert (record.seen, record.admissions, record.spills, record.evictions) == (3, 3, 3, 3)
    assert record.hits == 0


def test_history_survives_a_restart(root):
    cache = ReuseCache(root)
    cache.observe("a", label="Join")
    cache.observe("a")
    cache.put("a", _arr(10), cost=1.0)
    cache.save_manifest()
    reopened = ReuseCache(root)
    assert reopened.history["a"].seen == 2
    assert reopened.history["a"].label == "Join"
    assert reopened.history["a"].admissions == 1


def test_clear_forgets_the_history(root):
    cache = ReuseCache(root)
    cache.observe("a")
    cache.clear()
    assert cache.history == {} and ReuseCache(root).history == {}
