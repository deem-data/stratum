"""Reuse decisions describe the executed plan; artifact counters cover its interval."""
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

import stratum as st
from stratum import _api
from stratum.optimizer.physical.rewrites import _reuse_planning
from stratum.runtime import _buffer_pool, _reuse_cache
from tests.optimizer.test_reuse_planning import features, model
from tests.profiler.test_diagnostics import artifact, fail
from tests.runtime.test_reuse_execution import assert_equal


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    cache = _reuse_cache.ReuseCache()
    monkeypatch.setattr(_reuse_cache, "_active", cache)
    monkeypatch.setattr(_buffer_pool, "_host_memory_bytes", lambda: 10_000_000)
    monkeypatch.setattr(_buffer_pool, "_DEFAULT_SPILL_ROOT", tmp_path / "buffers")
    with st.config(reuse_cache=True, reuse_cache_memory_fraction=0.1, stats=False):
        yield cache


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "data.csv"
    rng = np.random.default_rng(0)
    pd.DataFrame({"a": rng.random(30), "b": rng.random(30), "y": rng.random(30)}).to_csv(
        path, index=False)
    return path


def physical_plan(document):
    return next(plan for plan in document["plans"] if plan["name"] == "physical_impl")


@pytest.mark.parametrize("selector", ["default", "greedy"])
def test_cold_and_warm_reports_follow_pruned_physical_node_ids(
        tmp_path, csv_path, cache, selector, capsys):
    with st.config(implementation_selector=selector):
        with st.dev_config(diagnostics_dir=tmp_path / "cold"):
            expected = _api.evaluate(features(csv_path))
        cold = artifact(tmp_path / "cold")
        report = cold["reuse_cache"]
        assert report["stats_scope"] == "run_interval"
        assert report["stats"] == asdict(cache.stats)
        assert report["stats"]["puts"] > 0 and report["stats"]["hits"] == 0
        assert report["entries"] == len(cache)
        assert report["memory_usage"] == cache.memory_usage > 0
        assert report["memory_budget"] == cache.memory_budget
        plan = physical_plan(cold)
        assert set(report["nodes"]) == {str(node["id"]) for node in plan["nodes"]}
        assert {node["status"] for node in report["nodes"].values()} <= {"eligible", "keyed"}
        root_key = report["nodes"][str(plan["nodes"][-1]["id"])]["key"]
        assert all(node["stored"] and node["size"] > 0
                   for node in report["nodes"].values() if node["status"] == "eligible")

        with st.dev_config(diagnostics_dir=tmp_path / "warm"):
            assert_equal(_api.evaluate(features(csv_path)), expected)
    warm = artifact(tmp_path / "warm")
    assert len(physical_plan(warm)["nodes"]) == 1 < len(plan["nodes"])
    assert warm["reuse_cache"]["nodes"] == {"0": {
        "status": "hit", "key": root_key, "label": physical_plan(warm)["nodes"][0]["label"]}}
    assert warm["node_timings"] == {}  # The hit was loaded, not computed.
    assert warm["reuse_cache"]["stats"]["puts"] == 0
    assert cache.stats.puts == report["stats"]["puts"] > 0
    assert warm["reuse_cache"]["stats"]["hits"] == 1
    assert "Reuse cache" not in capsys.readouterr().out


def test_split_dependent_nodes_are_reported_as_outside(tmp_path, csv_path):
    _api.evaluate(features(csv_path))
    with st.dev_config(diagnostics_dir=tmp_path / "search"):
        sched = _api.grid_search(model(csv_path), cv=2, scoring="r2")
    run = artifact(tmp_path / "search")
    nodes = run["reuse_cache"]["nodes"]
    assert len(nodes) == len(sched.linearized_dag)
    assert any(node["status"] == "hit" for node in nodes.values())
    assert all(nodes[str(i)] == {"status": "outside"}
               for i in range(sched.pos_split_op, len(sched.linearized_dag)))


def test_unkeyed_inputs_explain_why_descendants_cannot_be_reused(tmp_path):
    data = st.as_data_op(pd.DataFrame({"a": [1, 2, 3]}))
    with st.dev_config(diagnostics_dir=tmp_path / "unkeyed"):
        _api.evaluate(data.assign(b=data["a"] * 2))
    nodes = artifact(tmp_path / "unkeyed")["reuse_cache"]["nodes"].values()
    reasons = {node["reason"] for node in nodes if node["status"] == "unkeyed"}
    assert "in-memory input frame" in reasons
    assert "an input has no key" in reasons


def test_constants_are_keyed_but_not_eligible_for_storage(tmp_path):
    with st.dev_config(diagnostics_dir=tmp_path / "constant"):
        assert _api.evaluate(st.as_data_op(42)) == 42
    report = artifact(tmp_path / "constant")["reuse_cache"]
    assert report["nodes"]["0"]["status"] == "keyed"
    assert report["nodes"]["0"]["key"]
    assert report["stats"]["puts"] == 0 and report["entries"] == 0


def test_eligible_does_not_claim_successful_admission(tmp_path, csv_path):
    with st.config(reuse_cache_memory_fraction=1e-20), st.dev_config(
            diagnostics_dir=tmp_path / "rejected"):
        _api.evaluate(features(csv_path))
    report = artifact(tmp_path / "rejected")["reuse_cache"]
    assert any(node["status"] == "eligible" for node in report["nodes"].values())
    assert all(not node["stored"] for node in report["nodes"].values()
               if node["status"] == "eligible")
    assert report["stats"]["rejected"] > 0 and report["stats"]["puts"] == 0
    assert report["entries"] == 0 and report["memory_usage"] == 0


def test_stats_print_cache_lifetime_counters_and_memory(csv_path, cache, capsys):
    with st.config(stats=True):
        _api.evaluate(features(csv_path))
        capsys.readouterr()
        _api.evaluate(features(csv_path))
    output = capsys.readouterr().out
    assert "Reuse cache (cache lifetime counters):" in output
    assert "Hits: 1" in output
    assert f"misses: {cache.stats.misses}" in output
    assert f"insertions: {cache.stats.puts}" in output
    assert "rejections: 0, evictions: 0" in output
    assert f"Entries: {len(cache)}, memory:" in output


def test_disabled_reuse_has_no_cache_report(tmp_path, csv_path, capsys):
    with st.config(reuse_cache=False, stats=True), st.dev_config(
            diagnostics_dir=tmp_path / "disabled"):
        _api.evaluate(features(csv_path))
    assert artifact(tmp_path / "disabled")["reuse_cache"] is None
    assert "Reuse cache" not in capsys.readouterr().out


def test_disabled_diagnostics_do_not_collect_reasons_or_snapshot_cache(
        csv_path, cache, monkeypatch):
    original = _reuse_planning.compute_lineage_keys
    calls = []

    def compute(*args, **kwargs):
        calls.append(kwargs["reasons"])
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Disabled diagnostics must not collect cache reports")

    monkeypatch.setattr(_reuse_planning, "compute_lineage_keys", compute)
    monkeypatch.setattr(_reuse_planning, "record_reuse_plan", forbidden)
    monkeypatch.setattr(cache, "snapshot", forbidden)
    _api.evaluate(features(csv_path))
    assert calls == [None] and len(cache) > 0


def test_failed_execution_exports_decisions_and_partial_cache_state(tmp_path, csv_path):
    dag = features(csv_path).skb.apply_func(fail, st.eval_mode())
    with st.dev_config(diagnostics_dir=tmp_path / "failed"):
        with pytest.raises(RuntimeError, match="deliberate failure"):
            _api.evaluate(dag)
    run = artifact(tmp_path / "failed")
    assert run["status"] == "failed" and run["error"]["stage"] == "execution"
    assert run["reuse_cache"]["stats"]["puts"] > 0
    assert run["reuse_cache"]["nodes"]
    assert sum(t["failures"] for t in run["node_timings"].values()) == 1


def test_failed_optimization_exports_cache_state_without_node_ids(tmp_path, monkeypatch):
    def fail_optimize(*args, **kwargs):
        raise ValueError("invalid plan")

    monkeypatch.setattr(_api, "optimize", fail_optimize)
    with st.dev_config(diagnostics_dir=tmp_path / "failed"):
        with pytest.raises(ValueError, match="invalid plan"):
            _api.evaluate(st.as_data_op(1))
    run = artifact(tmp_path / "failed")
    assert run["error"]["stage"] == "optimization"
    assert run["reuse_cache"]["nodes"] == {} and run["reuse_cache"]["entries"] == 0


def test_cache_report_failure_does_not_fail_execution(tmp_path, cache, monkeypatch, caplog):
    def fail_snapshot(**kwargs):
        raise ValueError("cannot report cache")

    monkeypatch.setattr(cache, "snapshot", fail_snapshot)
    with st.dev_config(diagnostics_dir=tmp_path / "failed"):
        assert _api.evaluate(st.as_data_op(42)) == 42
    assert "Could not export Stratum diagnostics" in caplog.text
