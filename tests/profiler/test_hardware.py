import json
from time import perf_counter, sleep
from types import SimpleNamespace

import pytest

import stratum as st
from stratum._api import evaluate, grid_search
from stratum._profiler import _hardware
from stratum._profiler._hardware import HardwareTracker, hardware_run
from stratum._profiler._hardware_sampler import sample_tree
from stratum.runtime._scheduler import SequentialScheduler
from stratum._profiler._observer import observe
from tests.profiler.test_diagnostics import SameLabelOp, pipeline


def test_disabled_tracking_does_not_spawn_or_read_cpu_counters(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Disabled hardware profiling performed work")
    monkeypatch.setattr(_hardware.subprocess, "Popen", forbidden)
    monkeypatch.setattr(_hardware, "process_time", forbidden)
    with st.dev_config(hardware_utilization=False):
        result = grid_search(pipeline(), cv=2, scoring="neg_mean_squared_error")
    assert not hasattr(result, "hardware_utilization")


def test_sampler_start_failure_preserves_result_and_exports_error(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("sampler deliberately unavailable")
    monkeypatch.setattr(_hardware.subprocess, "Popen", unavailable)
    with st.dev_config(hardware_utilization=True, diagnostics_dir=tmp_path):
        result = evaluate(pipeline())
    assert len(result) == 4
    run = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert run["status"] == "completed"
    assert run["hardware"]["status"] == "unavailable"
    assert "deliberately unavailable" in run["hardware"]["error"]


def test_sampler_report_failure_preserves_result_and_reaps_sampler(tmp_path, monkeypatch):
    class BrokenSampler:
        def __init__(self, *args, **kwargs):
            from io import BytesIO
            self.stdout = BytesIO(b'{"ready":true}\n')
            self.returncode = None
        def communicate(self, input=None, timeout=None):
            self.returncode = 0
            return b"invalid report", b""
        def poll(self):
            return self.returncode
        def kill(self):
            self.returncode = -9
    monkeypatch.setattr(_hardware.subprocess, "Popen", BrokenSampler)
    with st.dev_config(hardware_utilization=True, diagnostics_dir=tmp_path):
        result = grid_search(pipeline(), cv=2, scoring="neg_mean_squared_error")
    assert len(result.results_) == 1
    run = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert run["status"] == "completed"
    assert run["hardware"]["status"] == "unavailable"
    assert run["hardware"]["nodes"]  # Engine CPU measurements survive a sampler failure.


def test_hardware_starts_before_input_preparation(tmp_path, monkeypatch):
    from stratum import _api
    original = _api.get_data
    def get_data(dag):
        assert _hardware.get_hardware_tracker() is not None
        sleep(.03)
        return original(dag)
    monkeypatch.setattr(_api, "get_data", get_data)
    with st.dev_config(hardware_utilization=True, diagnostics_dir=tmp_path):
        evaluate(pipeline())
    run = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert run["hardware"]["wall_seconds"] >= run["timings"]["total"] + .03


def test_memory_join_uses_half_open_windows_and_reports_missing_samples(monkeypatch):
    tracker = HardwareTracker()
    a, b, c = object(), object(), object()
    tracker.bind_scheduler(SimpleNamespace(linearized_dag=[a, b, c]))
    cpu = iter([0.2, 0.1, 0.001])
    monkeypatch.setattr(_hardware, "process_time", lambda: next(cpu))
    for op, start, end in [(a, 0, 1), (b, 1, 2), (c, 2, 2.05)]:
        phase = tracker.end(op, "predict", 1, 0, start, end)
        phase.tracking_seconds += .00001
        tracker.boundary_seconds += .00001
        tracker.boundary_calls += 2
    tracker.started, tracker.finished, tracker.cpu_seconds = 0, 2.05, .301
    report = tracker._assemble({"samples": [
        {"timestamp": .25, "tree_rss_bytes": 100},
        {"timestamp": .75, "tree_rss_bytes": 300},
        {"timestamp": 1.0, "tree_rss_bytes": 500},
        {"timestamp": 1.5, "tree_rss_bytes": 700},
        {"timestamp": 2.05, "tree_rss_bytes": 999},
        {"timestamp": 1.25, "collection_start": .99, "tree_rss_bytes": 999},
    ]})
    nodes = report["nodes"]
    assert nodes["0"]["rss_mean_bytes"] == 200
    assert nodes["0"]["calls_with_memory_samples"] == 1
    assert nodes["1"]["rss_peak_bytes"] == 700
    assert nodes["2"]["rss_mean_bytes"] is None
    assert nodes["2"]["rss_peak_bytes"] is None
    assert nodes["2"]["memory_sample_count"] == 0
    assert nodes["0"]["process_cpu_percent"] == pytest.approx(20)
    assert sum(n["tracking_seconds"] for n in nodes.values()) == pytest.approx(report["overhead"]["boundary_seconds"])


def test_window_retention_is_bounded_but_cpu_totals_remain_exact(monkeypatch):
    monkeypatch.setattr(_hardware, "MAX_WINDOWS", 1)
    monkeypatch.setattr(_hardware, "process_time", lambda: .2)
    tracker = HardwareTracker()
    op = object()
    tracker.bind_scheduler(SimpleNamespace(linearized_dag=[op]))
    tracker.end(op, "predict", 0, 0, 1, 2)
    tracker.end(op, "predict", 1, 0, 2, 3)
    assert len(tracker.windows) == 1
    assert tracker.dropped_windows == 1
    assert sum(p.process_cpu_seconds for p in tracker.nodes[0].values()) == pytest.approx(.4)
    assert sum(p.calls for p in tracker.nodes[0].values()) == 2


def test_sampler_excludes_its_own_process_and_handles_pid_reuse():
    class Proc:
        def __init__(self, pid, born, cpu, rss):
            self.pid, self.born, self.cpu, self.rss = pid, born, cpu, rss
        def oneshot(self):
            from contextlib import nullcontext
            return nullcontext()
        def memory_info(self):
            return SimpleNamespace(rss=self.rss)
        def cpu_times(self):
            return SimpleNamespace(user=self.cpu, system=0)
        def create_time(self):
            return self.born
    root = Proc(1, 0, 5, 100)
    worker = Proc(2, 1, 2, 200)
    sampler = Proc(3, 1, 50, 1000)
    root.children = lambda **kwargs: [worker, sampler]
    errors = SimpleNamespace(NoSuchProcess=ProcessLookupError, AccessDenied=PermissionError)
    previous = {}
    assert sample_tree(root, errors, previous, 3) == (100, 300, 7)
    root.cpu += 1
    worker.born, worker.cpu = 2, .5  # Same PID, new process.
    assert sample_tree(root, errors, previous, 3) == (100, 300, 1.5)
    assert len(previous) == 2


# Polls straddling an operator boundary are discarded, and a poll can take tens
# of milliseconds on loaded CI runners, so each window spans many intervals.
OP_SECONDS = .5


class BusyOp(SameLabelOp):
    def process(self, mode, inputs):
        end = perf_counter() + OP_SECONDS
        while perf_counter() < end:
            sum(range(1000))
        return 1


class IdleOp(SameLabelOp):
    def process(self, mode, inputs):
        sleep(OP_SECONDS)
        return 1


def test_real_service_tracks_cpu_memory_and_reaps_process():
    busy, idle = BusyOp(), IdleOp()
    with hardware_run(True, .02) as tracker:
        assert tracker.error is None, tracker.error
        sched = SequentialScheduler([busy, idle], None, [])
        observe(sched, tracker, collect_node_timings=False)
        sched.process_op(busy)
        sched.process_op(idle)
        sched.pool.remove_all()
    assert tracker.process.poll() == 0
    report = sched.hardware_utilization
    assert report["status"] == "completed"
    assert report["sample_count"] >= 3
    assert report["overhead"]["boundary_calls"] == 4
    assert report["overhead"]["service_cpu_seconds"] > 0
    assert report["overhead"]["startup_seconds"] > 0
    nodes = report["nodes"]
    assert nodes["0"]["process_cpu_seconds"] > nodes["1"]["process_cpu_seconds"]
    assert nodes["0"]["process_cpu_percent"] > nodes["1"]["process_cpu_percent"]
    assert nodes["0"]["memory_sample_count"] > 0
    assert nodes["1"]["rss_peak_bytes"] > 0


def memory_workload(value, mode):
    if mode != "preview":
        data = bytearray(16 * 2**20)
        data[::4096] = b"x" * (len(data) // 4096)
        sleep(.08)
    return value


def test_api_exports_hardware_and_overhead_without_stats(tmp_path, capsys):
    dag = st.as_data_op(1).skb.apply_func(memory_workload, st.eval_mode())
    with st.config(stats=False), st.dev_config(
            hardware_utilization=True, hardware_sampling_interval=.02, diagnostics_dir=tmp_path):
        evaluate(dag)
    run = json.loads(next(tmp_path.glob("*.json")).read_text())
    hw = run["hardware"]
    assert hw["status"] == "completed", hw["error"]
    assert hw["nodes"]
    assert any(n["rss_peak_bytes"] is not None for n in hw["nodes"].values())
    assert hw["wall_seconds"] >= run["timings"]["total"]
    assert "Tracking overhead" not in capsys.readouterr().out


def test_failed_operation_still_records_hardware_and_stops_service(tmp_path, capsys):
    from tests.profiler.test_diagnostics import fail
    dag = st.as_data_op(1).skb.apply_func(fail, st.eval_mode())
    with st.config(stats=True), st.dev_config(
            hardware_utilization=True, hardware_sampling_interval=.02, diagnostics_dir=tmp_path):
        with pytest.raises(RuntimeError, match="deliberate failure"):
            evaluate(dag)
    run = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert run["status"] == "failed"
    assert run["hardware"]["status"] == "completed"
    assert run["hardware"]["nodes"]
    assert "Tracking overhead" in capsys.readouterr().out
