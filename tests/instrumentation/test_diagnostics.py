"""Pin identity, CV accounting, and the filesystem boundary consumed by the GUI."""
import json

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

import stratum as st
from stratum._api import evaluate, grid_search
from stratum.optimizer.logical._ops import Op
from stratum.runtime._scheduler import SequentialScheduler
from stratum._instrumentation._observer import observe


class SameLabelOp(Op):
    def process(self, mode, inputs):
        return 1


def pipeline():
    data = st.as_data_op(pd.DataFrame({"x": range(20), "y": range(20)}))
    x = data[["x"]].skb.mark_as_X()
    y = data["y"].skb.mark_as_y()
    return x.skb.apply(DummyRegressor(), y=y)


def artifact(directory):
    files = list(directory.glob("*.json"))
    assert len(files) == 1
    assert not list(directory.glob("*.tmp"))
    document = json.loads(files[0].read_text())
    assert document["run_id"] == files[0].stem
    assert document["schema_version"] == 1
    return document


def test_same_label_nodes_have_distinct_runtime_and_disabled_collection():
    a, b = SameLabelOp(), SameLabelOp()
    assert str(a) == str(b)
    sched = SequentialScheduler([a, b], None, [], print_heavy_hitters=True)
    node_timings = observe(sched).node_timings
    sched.process_op(a)
    sched.process_op(b)
    sched.mode = "predict"
    sched.cv_id = 2
    sched.process_op(a)
    assert node_timings[a].count == 2
    assert node_timings[b].count == 1
    assert node_timings[a].phases[("predict", 2)]["count"] == 1
    assert len(sched.timings) == 3
    sched.pool.remove_all()
    disabled = SequentialScheduler([a], None, [])
    disabled.process_op(a)
    assert disabled.observer is None
    assert disabled.timings is None
    disabled.pool.remove_all()


def test_export_without_printed_stats_has_plan_node_cv_attribution(tmp_path, capsys):
    with st.config(stats=False), st.dev_config(diagnostics_dir=tmp_path):
        sched = grid_search(pipeline(), cv=2, scoring="neg_mean_squared_error")
    assert "Heavy hitters" not in capsys.readouterr().out
    run = artifact(tmp_path)
    assert run["status"] == "completed"
    assert run["timings"]["total"] == pytest.approx(
        run["timings"]["optimization"] + run["timings"]["execution"])
    plan = next(p for p in run["plans"] if p["name"] == "physical_impl")
    assert [n["id"] for n in plan["nodes"]] == list(range(len(sched.linearized_dag)))
    assert {"logical", "lowered", "physical_impl", "relational_rewrite"} <= {p["name"] for p in run["plans"]}
    for snapshot in run["plans"]:
        for node in snapshot["nodes"]:
            assert all(i < node["id"] for i in node["inputs"])
    assert sum(t["seconds"] for t in run["node_timings"].values()) == pytest.approx(
        sum(h["seconds"] for h in run["heavy_hitters"]))
    fitted = [t for t in run["node_timings"].values() if {p["fold"] for p in t["phases"]} == {0, 1}]
    assert fitted
    assert any({p["mode"] for p in t["phases"]} >= {"fit_transform", "predict"} for t in fitted)
    assert run["buffer_pool"]["hits"] > 0


def test_evaluate_export_and_config_restore(tmp_path):
    original = st.get_config()
    with st.dev_config(diagnostics_dir=tmp_path):
        result = evaluate(pipeline())
    assert len(result) == 4
    assert artifact(tmp_path)["kind"] == "evaluate"
    assert st.get_config() == original


def test_dev_config_restores_and_rejects_unknown_flags(tmp_path):
    from stratum._config import DEV_FLAGS
    with st.dev_config(diagnostics_dir=tmp_path, hardware_sampling_interval=.5):
        assert DEV_FLAGS.diagnostics_dir == tmp_path
    assert DEV_FLAGS.diagnostics_dir is None
    assert DEV_FLAGS.hardware_sampling_interval == .1
    with pytest.raises(TypeError, match="diagnostic_dir"):
        with st.dev_config(diagnostic_dir=tmp_path):
            pass
    assert "diagnostics_dir" not in st.get_config()


def fail(value, mode):
    if mode != "preview":
        raise ValueError("deliberate failure")
    return value


def test_failed_run_exports_partial_measurements_and_preserves_exception(tmp_path):
    dag = st.as_data_op(1).skb.apply_func(fail, st.eval_mode())
    with st.config(stats=False), st.dev_config(diagnostics_dir=tmp_path):
        with pytest.raises(RuntimeError, match="deliberate failure"):
            evaluate(dag)
    run = artifact(tmp_path)
    assert run["status"] == "failed"
    assert run["error"]["stage"] == "execution"
    assert "deliberate failure" in run["error"]["message"]
    assert sum(t["failures"] for t in run["node_timings"].values()) == 1


def test_optimization_failure_exports_stage(tmp_path, monkeypatch):
    def fail_optimize(*args, **kwargs):
        raise ValueError("invalid plan")
    monkeypatch.setattr(st._api, "optimize", fail_optimize)
    with st.dev_config(diagnostics_dir=tmp_path):
        with pytest.raises(ValueError, match="invalid plan"):
            evaluate(st.as_data_op(1))
    run = artifact(tmp_path)
    assert run["error"]["stage"] == "optimization"
    assert run["plans"] == []
    assert run["timings"]["execution"] is None


def test_export_failure_does_not_fail_computation(tmp_path, caplog):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    with st.dev_config(diagnostics_dir=blocked):
        result = evaluate(pipeline())
    assert len(result) == 4
    assert "Could not export Stratum diagnostics" in caplog.text
