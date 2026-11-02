"""Reuse planning and execution: plans take what earlier plans left in the cache."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

import stratum as st
from stratum._api import evaluate, grid_search
from stratum._config import config
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer._reuse_planning import (
    AdmissionPolicy, CacheNothing, cacheable_region, make_admission_policy)
from stratum.runtime import _buffer_pool as _bp
from stratum.runtime import _reuse_cache
from stratum.runtime._reuse_cache import ReuseCache, active_reuse_cache
from stratum.runtime._scheduler import SequentialScheduler


@pytest.fixture
def csv_path(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": rng.random(60), "b": rng.random(60),
                  "y": rng.random(60)}).to_csv(path, index=False)
    return path


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Reuse cache on, in a fresh directory, with a fresh process-level cache."""
    monkeypatch.setattr(_reuse_cache, "_active", None)
    with config(reuse_cache=True, reuse_cache_dir=str(tmp_path / "cache")):
        yield tmp_path / "cache"
    monkeypatch.setattr(_reuse_cache, "_active", None)


def restart():
    """What a new process sees: the disk tier only."""
    active_reuse_cache().flush()
    _reuse_cache._active = None


def features(csv_path):
    data = st.as_data_op(str(csv_path)).skb.apply_func(pd.read_csv)
    return data.assign(c=data["a"] * 2 + data["b"])


def model(csv_path, alpha=1.0):
    data = features(csv_path)
    X = data[["a", "b", "c"]].skb.mark_as_X()
    y = data["y"].skb.mark_as_y()
    return X.skb.apply(Ridge(alpha=alpha), y=y)


def plan_for(dag, cache, conf=None):
    return optimize(dag, conf, env=get_data(dag), reuse_cache=cache)


def hits(plan):
    return [op for op in plan if op.cache_hit is not None]


# --- the region -----------------------------------------------------------------

def test_region_is_what_runs_once(csv_path):
    dag = model(csv_path)
    plan, split_pos = optimize(dag, env=get_data(dag))
    region = cacheable_region(plan)
    assert region == set(plan[:split_pos])

    dag = features(csv_path)
    plan, split_pos = optimize(dag, env=get_data(dag))
    assert split_pos is None
    assert cacheable_region(plan) == set(plan)


# --- planning ----------------------------------------------------------------------

def test_cold_cache_flags_the_region_and_takes_nothing(csv_path, tmp_path):
    cache = ReuseCache(tmp_path / "c")
    plan, split_pos = plan_for(model(csv_path), cache)
    assert not hits(plan)
    flagged = {op for op in plan if op.cache_key is not None}
    assert flagged and flagged <= set(plan[:split_pos])
    assert cache.stats.planned_admissions == len(flagged)
    keyed = len(cache.history)
    plan_for(model(csv_path), cache)
    assert len(cache.history) == keyed
    assert all(record.seen == 2 for record in cache.history.values()), \
        "every keyed op is seen once per plan, cached or not"


def test_a_hit_keeps_its_place_and_loses_its_inputs(csv_path, tmp_path):
    cache = ReuseCache(tmp_path / "c")
    plan, split_pos = plan_for(model(csv_path), cache)
    x_op, y_op = plan[split_pos].inputs[:2]  # the split's inputs are X and y
    cache.put(x_op.cache_key, pd.DataFrame({"a": [1.0]}))
    plan, split_pos = plan_for(model(csv_path), cache)
    [hit] = hits(plan)
    assert plan[split_pos].inputs[0] is hit, "the split still reads X from it"
    assert type(hit) is type(x_op) and str(hit) == str(x_op), "the op itself"
    assert hit.inputs == [] and hit.cache_key is None
    assert hit.cache_hit.tier == "memory" and hit.cache_hit.key == x_op.cache_key
    # y still needs the source, so only what X alone needed is pruned.
    assert plan[split_pos].inputs[1].cache_hit is None
    assert hit.outputs == [plan[split_pos]]


def test_the_whole_result_can_be_a_single_hit(csv_path, tmp_path):
    cache = ReuseCache(tmp_path / "c")
    plan, _ = plan_for(features(csv_path), cache)
    sink = plan[-1]
    cache.put(sink.cache_key, pd.DataFrame({"x": [1]}))
    plan, split_pos = plan_for(features(csv_path), cache)
    assert len(plan) == 1 and type(plan[0]) is type(sink)
    assert plan[0].cache_hit is not None
    assert split_pos is None


def test_a_hit_is_handed_to_one_scheduler(csv_path, tmp_path):
    cache = ReuseCache(tmp_path / "c")
    plan, _ = plan_for(features(csv_path), cache)
    cache.put(plan[-1].cache_key, pd.DataFrame({"x": [1]}))
    plan, split_pos = plan_for(features(csv_path), cache)
    SequentialScheduler(plan, split_pos, reuse_cache=cache)
    with pytest.raises(RuntimeError, match="optimize the plan again"):
        SequentialScheduler(plan, split_pos, reuse_cache=cache)


def test_admission_policy_is_pluggable(csv_path, tmp_path):
    seen = {}

    class FirstOnly(AdmissionPolicy):
        def select(self, candidates, keys, cache):
            seen["candidates"] = candidates
            return set(candidates[:1])

    cache = ReuseCache(tmp_path / "c")
    plan, _ = plan_for(model(csv_path), cache, OptConfig(reuse_admission=FirstOnly()))
    assert len(seen["candidates"]) > 1
    assert [op for op in plan if op.cache_key is not None] == seen["candidates"][:1]

    plan, _ = plan_for(model(csv_path), cache, OptConfig(reuse_admission=CacheNothing()))
    assert all(op.cache_key is None for op in plan)
    with pytest.raises(ValueError, match="valid policies"):
        make_admission_policy("most")


# --- execution ------------------------------------------------------------------

def test_evaluate_reuses_the_pre_split_part(csv_path, cache_dir):
    first = evaluate(model(csv_path))
    cache = active_reuse_cache()
    assert cache.stats.puts > 0 and cache.stats.planned_hits == 0
    second = evaluate(model(csv_path))
    assert cache.stats.planned_hits > 0
    assert cache.stats.memory_hits == cache.stats.planned_hits
    np.testing.assert_array_equal(first, second)


def test_a_plan_without_split_is_cached_whole(csv_path, cache_dir):
    first = evaluate(features(csv_path))
    second = evaluate(features(csv_path))
    pd.testing.assert_frame_equal(first, second)
    cache = active_reuse_cache()
    assert cache.stats.planned_hits == 1, "the second plan is one hit: the sink"
    # The caller gets its own object: mutating it leaves the cached entry intact.
    second["a"] = -1.0
    pd.testing.assert_frame_equal(evaluate(features(csv_path)), first)


def test_a_read_is_keyed_by_the_categories_it_reads_with(tmp_path, cache_dir):
    path = tmp_path / "c.csv"
    path.write_text("c\na\nb\na\n")

    def read(categories):
        return evaluate(st.as_data_op(str(path)).skb.apply_func(
            pd.read_csv, dtype={"c": pd.CategoricalDtype(categories)}))["c"]

    assert read(["a", "b"]).tolist() == ["a", "b", "a"]
    assert read(["a"]).isna().tolist() == [False, True, False]


def test_the_cache_may_evict_a_hit_before_the_plan_runs(csv_path, tmp_path):
    cache = ReuseCache(tmp_path / "c")
    dag = model(csv_path)
    plan, split_pos = plan_for(dag, cache)
    expected = SequentialScheduler(plan, split_pos, reuse_cache=cache).evaluate()
    plan, split_pos = plan_for(dag, cache)
    assert hits(plan)
    cache.clear()  # nothing is pinned or leased: every entry may go
    result = SequentialScheduler(plan, split_pos, reuse_cache=cache).evaluate()
    np.testing.assert_array_equal(result, expected)


def test_an_unreadable_entry_is_recomputed(csv_path, cache_dir):
    first = evaluate(model(csv_path))
    cache = active_reuse_cache()
    cache.flush()
    cache.evict(cache.memory_usage)  # every entry is on disk only
    for entry in cache.entries.values():
        entry.handle.path.write_bytes(b"damaged")  # e.g. unreadable after an upgrade
    np.testing.assert_array_equal(evaluate(model(csv_path)), first)
    assert cache.stats.discards > 0 and cache.stats.planned_hits == 0


def test_runs_report_cache_activity_per_node(csv_path, cache_dir, tmp_path):
    runs = tmp_path / "runs"
    with st.dev_config(diagnostics_dir=runs):
        evaluate(model(csv_path))
        evaluate(model(csv_path))
    first, second = sorted((json.loads(f.read_text()) for f in runs.glob("*.json")),
                           key=lambda run: run["started_at"])

    report = first["reuse_cache"]
    nodes = report["nodes"]
    assert report["stats"]["planned_hits"] == 0
    assert report["stats"]["puts"] == report["entries"] > 0
    admitted = [n for n in nodes.values() if n["status"] == "admitted"]
    assert admitted and all(n["stored"] and n["size"] > 0 for n in admitted)
    assert all(n["label"] for n in admitted)
    # After the split, summed over the folds: one fold here.
    per_fold = [n for n in nodes.values() if n["status"] == "fold"]
    assert per_fold and all(n["folds"] == 1 for n in per_fold)
    assert sum(n["admitted"] for n in per_fold) > 0
    executed = {n["id"] for n in first["plans"][-1]["nodes"]}
    assert {int(i) for i in nodes} <= executed

    report = second["reuse_cache"]
    taken = [n for n in report["nodes"].values() if n["status"] == "hit"]
    fold_hits = sum(n["hits"] for n in report["nodes"].values() if n["status"] == "fold")
    assert taken and fold_hits
    assert report["stats"]["planned_hits"] == len(taken) + fold_hits
    assert report["stats"]["puts"] == 0, "counts are per run"
    assert all(n["tier"] == "memory" and n["load_seconds"] >= 0 for n in taken)
    assert all(n["label"] for n in taken), "a hit names the op that computed it"


def _double(df):
    return df * 2


def test_runs_report_why_an_op_has_no_key(csv_path, cache_dir, tmp_path):
    runs = tmp_path / "runs"
    data = features(csv_path).skb.apply_func(_double)
    X = data[["a", "b", "c"]].skb.mark_as_X()
    y = data["y"].skb.mark_as_y()
    with st.dev_config(diagnostics_dir=runs):
        evaluate(X.skb.apply(Ridge(), y=y))
    run = json.loads(next(runs.glob("*.json")).read_text())
    reasons = {n.get("reason") for n in run["reuse_cache"]["nodes"].values()
               if n["status"] == "unkeyed"}
    assert "an input has no key" in reasons
    assert any("cacheable" in str(r) for r in reasons), reasons


def test_runs_without_a_cache_report_none(csv_path, tmp_path):
    with st.dev_config(diagnostics_dir=tmp_path / "runs"):
        evaluate(model(csv_path))
    run = json.loads(next((tmp_path / "runs").glob("*.json")).read_text())
    assert run["reuse_cache"] is None


def test_entries_survive_a_restart(csv_path, cache_dir):
    first = evaluate(model(csv_path))
    restart()
    second = evaluate(model(csv_path))
    cache = active_reuse_cache()
    assert cache.stats.planned_hits > 0
    assert cache.stats.disk_hits == cache.stats.planned_hits
    np.testing.assert_array_equal(first, second)


def test_an_edited_model_reuses_the_features(csv_path, cache_dir):
    evaluate(model(csv_path, alpha=1.0))
    cache = active_reuse_cache()
    entries = set(cache.entries)
    evaluate(model(csv_path, alpha=2.0))
    assert cache.stats.planned_hits > 0
    # Everything before the split is a hit; only the new model's fit and its
    # predictions are added.
    added = [cache.entries[key].label for key in set(cache.entries) - entries]
    assert sorted(added) == ["SklearnRidge_fit state", "SklearnRidge_predict"]


def test_an_evaluate_prefix_serves_a_grid_search(csv_path, cache_dir):
    evaluate(features(csv_path))
    cache = active_reuse_cache()
    hits_before = cache.stats.planned_hits
    search = grid_search(model(csv_path), cv=3, scoring="r2")
    assert cache.stats.planned_hits > hits_before
    with config(reuse_cache=False):
        baseline = grid_search(model(csv_path), cv=3, scoring="r2")
    assert search.results_["scores"].to_list() == baseline.results_["scores"].to_list()


def test_spilled_buffers_are_handed_over(csv_path, cache_dir):
    # A one-byte pool spills every buffer nobody holds pinned. The pre-split
    # buffers the folds share are released only at the end, by then spilled, and
    # move into the cache as files rather than being loaded back.
    with patch.object(_bp, "_resolve_memory_budget", return_value=1):
        search = grid_search(model(csv_path), cv=3, scoring="r2")
    cache = active_reuse_cache()
    taken = [e for e in cache.entries.values() if e.handle is not None and not e.in_memory]
    assert taken, "spilled buffers keep living on disk, in the cache"
    with config(reuse_cache=False):
        baseline = grid_search(model(csv_path), cv=3, scoring="r2")
    assert search.results_["scores"].to_list() == baseline.results_["scores"].to_list()
    with patch.object(_bp, "_resolve_memory_budget", return_value=1):
        again = grid_search(model(csv_path), cv=3, scoring="r2")
    assert cache.stats.disk_hits > 0
    assert again.results_["scores"].to_list() == baseline.results_["scores"].to_list()


def test_admission_none_only_reads(csv_path, cache_dir):
    evaluate(model(csv_path))
    puts = active_reuse_cache().stats.puts
    with config(reuse_cache_admission="none"):
        evaluate(features(csv_path).assign(d=1))
    assert active_reuse_cache().stats.puts == puts


def test_config_validation():
    with pytest.raises(ValueError, match="reuse_cache_admission"):
        st.set_config(reuse_cache_admission="some")
    with pytest.raises(ValueError, match="reuse_cache_eviction"):
        st.set_config(reuse_cache_eviction="random")
    with pytest.raises(ValueError, match="reuse_cache_disk_budget"):
        st.set_config(reuse_cache_disk_budget=-1)
    with pytest.raises(ValueError, match="reuse_cache_memory_fraction"):
        st.set_config(reuse_cache_memory_fraction=2)
