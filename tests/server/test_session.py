"""Memory cache inspection during a server session."""
import json
from types import SimpleNamespace

import numpy as np

from stratum.runtime._reuse_cache import ReuseCache
from stratum.server import _session


def test_session_tracks_live_entries_and_throttles_cache_samples(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(_session, "time", SimpleNamespace(time=lambda: now[0]))
    cache = ReuseCache(memory_budget=160)
    recorder = _session.SessionRecorder(tmp_path, cache, tmp_path / "server")
    try:
        first = recorder.job_started(1, "first.py", [], str(tmp_path))
        cache.put("a", np.arange(10), label="source")
        now[0] += 1.1
        cache.get("a")
        cache.put("b", np.arange(10), label="transform")
        recorder.job_finished(0, 1.1, ["run-1"])

        second = recorder.job_started(2, "second.py", [], str(tmp_path))
        now[0] += 1.1
        cache.put("c", np.arange(10), label="prediction")
        recorder.job_finished(0, 1.1, ["run-2"])
        report = json.loads(recorder.path.read_text())
        entries = {entry["key"]: entry for entry in report["entries"]}
        assert set(entries) == {"b", "c"}
        assert entries["b"]["job_id"] == first
        assert entries["c"]["job_id"] == second
        assert report["jobs"][0]["stats"]["puts"] == 2
        assert report["jobs"][1]["stats"]["puts"] == 1
        assert report["jobs"][1]["stats"]["evictions"] == 1
        assert "a" not in recorder.entry_jobs

        cache.clear()
        recorder.write()
        assert json.loads(recorder.path.read_text())["entries"] == []
        assert recorder.entry_jobs == {}
    finally:
        recorder.close()
    samples = [json.loads(line) for line in recorder.timeline_path.read_text().splitlines()]
    assert [sample["event"] for sample in samples] == [
        "start", "job_start", "read", "job_end", "job_start", "admit", "job_end", "clear", "stop"]
    assert samples[2]["job_id"] == first and samples[2]["memory_usage"] == 80
    assert samples[-1]["memory_usage"] == 0
    assert not cache.listeners


def test_reinserted_entries_belong_to_the_job_that_recomputed_them(tmp_path):
    cache = ReuseCache(memory_budget=80)
    recorder = _session.SessionRecorder(tmp_path, cache, tmp_path / "server")
    try:
        recorder.job_started(1, "first.py", [], str(tmp_path))
        cache.put("shared", np.arange(10), label="source")
        recorder.job_finished(0, 0, [])
        second = recorder.job_started(2, "second.py", [], str(tmp_path))
        cache.put("temporary", np.arange(10))  # Evicts the first job's entry.
        cache.put("shared", np.arange(10), label="source")
        recorder.job_finished(0, 0, [])
        report = json.loads(recorder.path.read_text())
        [entry] = report["entries"]
        assert entry["key"] == "shared" and entry["job_id"] == second
        assert report["jobs"][1]["added_entries"] == 1
    finally:
        recorder.close()
