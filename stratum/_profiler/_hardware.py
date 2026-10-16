"""Local profiling API with an external sampler and no per-node IPC.

CPU-counter boundaries remain in the engine; sampled RSS is joined to those
windows once, after execution. Both buffers are bounded, with losses reported.
"""
from __future__ import annotations

from bisect import bisect_right
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, asdict
import json
import logging
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from time import perf_counter, process_time

logger = logging.getLogger(__name__)
_active_tracker = ContextVar("stratum_hardware_tracker", default=None)
MAX_SAMPLES = 20_000
MAX_WINDOWS = 100_000


def get_hardware_tracker():
    return _active_tracker.get()


@dataclass(slots=True)
class _Metrics:
    calls: int = 0
    wall_seconds: float = 0.0
    process_cpu_seconds: float = 0.0
    tracking_seconds: float = 0.0
    memory_sample_count: int = 0
    calls_with_memory_samples: int = 0
    rss_sum_bytes: int = 0
    rss_peak_bytes: int | None = None


def _public_metrics(metrics):
    result = asdict(metrics)
    total = result.pop("rss_sum_bytes")
    count = result["memory_sample_count"]
    result["rss_mean_bytes"] = total / count if count else None
    wall = result["wall_seconds"]
    result["process_cpu_percent"] = 100 * result["process_cpu_seconds"] / wall if wall else 0.0
    return result


class HardwareTracker:
    """One sampler per API run. begin/end update local counters, never the service."""

    def __init__(self, interval=0.1):
        self.interval = interval
        self.process = None
        self.scheduler = None
        self.ids = {}
        self.nodes = {}
        self.windows = []
        self.dropped_windows = 0
        self.boundary_seconds = 0.0
        self.boundary_calls = 0
        self.startup_seconds = self.shutdown_seconds = self.aggregation_seconds = 0.0
        self.report = None
        self.error = None

    def start(self):
        start = perf_counter()
        self.started = start
        self.cpu_started = process_time()
        try:
            sampler = Path(__file__).with_name("_hardware_sampler.py")
            self.process = subprocess.Popen(
                [sys.executable, str(sampler), str(os.getpid()),
                 str(self.interval), str(MAX_SAMPLES)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=0,
            )
            ready = queue.Queue(maxsize=1)
            # A short-lived startup reader makes pipe readiness portable to Windows.
            threading.Thread(target=lambda: ready.put(self.process.stdout.readline()), daemon=True).start()
            line = ready.get(timeout=5)
            if not line or not json.loads(line).get("ready"):
                raise RuntimeError("Sampler did not become ready; install stratum-ai[hardware]")
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._kill()
            logger.warning("Hardware tracking unavailable: %s", self.error)
        finally:
            self.startup_seconds = perf_counter() - start

    def bind_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.ids = {op: i for i, op in enumerate(scheduler.linearized_dag)}

    def begin(self):
        return process_time()

    def end(self, op, mode, fold, token, start, end):
        cpu = max(0.0, process_time() - token)
        node_id = self.ids[op]
        node = self.nodes.get(node_id)
        if node is None:
            node = self.nodes[node_id] = {}
        key = (mode, fold)
        phase = node.get(key)
        if phase is None:
            phase = node[key] = _Metrics()
        phase.calls += 1
        phase.wall_seconds += end - start
        phase.process_cpu_seconds += cpu
        if len(self.windows) < MAX_WINDOWS:
            self.windows.append((start, end, node_id, key))
        else:
            self.dropped_windows += 1
        return phase

    def _kill(self):
        if self.process is not None:
            try:
                if self.process.poll() is None:
                    self.process.kill()
                self.process.communicate(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                logger.warning("Could not reap hardware sampler", exc_info=True)

    def stop(self):
        if self.report is not None:
            return
        self.finished = perf_counter()
        self.cpu_seconds = max(0.0, process_time() - self.cpu_started)
        start = perf_counter()
        sampled = None
        try:
            if self.process is not None and self.error is None:
                stdout, stderr = self.process.communicate(input=b"stop\n", timeout=5)
                if self.process.returncode:
                    raise RuntimeError(stderr.decode("utf-8", errors="replace")[-1000:])
                sampled = json.loads(stdout)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            logger.warning("Could not finish hardware tracking: %s", self.error)
            self._kill()
        finally:
            self.shutdown_seconds = perf_counter() - start
        start = perf_counter()
        self.report = self._assemble(sampled)
        self.aggregation_seconds = perf_counter() - start
        self.report["overhead"]["aggregation_seconds"] = self.aggregation_seconds
        if self.scheduler is not None:
            self.scheduler.hardware_utilization = self.report

    def _assemble(self, sampled):
        sampled = sampled or {}
        samples = sampled.get("samples", [])
        windows = sorted(self.windows, key=lambda w: w[0])
        starts = [window[0] for window in windows]
        seen = set()
        for sample in samples:
            index = bisect_right(starts, sample["timestamp"]) - 1
            if index < 0:
                continue
            start, end, node_id, key = windows[index]
            # A poll that straddles an operator boundary cannot be assigned safely.
            if not start <= sample.get("collection_start", sample["timestamp"]) <= sample["timestamp"] < end:
                continue
            metrics = self.nodes[node_id][key]
            metrics.memory_sample_count += 1
            metrics.rss_sum_bytes += sample["tree_rss_bytes"]
            metrics.rss_peak_bytes = max(metrics.rss_peak_bytes or 0, sample["tree_rss_bytes"])
            if index not in seen:
                metrics.calls_with_memory_samples += 1
            seen.add(index)
        wall = self.finished - self.started
        nodes = {}
        for node_id, phases in self.nodes.items():
            total = _Metrics()
            for phase in phases.values():
                total.calls += phase.calls
                total.wall_seconds += phase.wall_seconds
                total.process_cpu_seconds += phase.process_cpu_seconds
                total.tracking_seconds += phase.tracking_seconds
                total.memory_sample_count += phase.memory_sample_count
                total.calls_with_memory_samples += phase.calls_with_memory_samples
                total.rss_sum_bytes += phase.rss_sum_bytes
                if phase.rss_peak_bytes is not None:
                    total.rss_peak_bytes = max(total.rss_peak_bytes or 0, phase.rss_peak_bytes)
            nodes[str(node_id)] = {
                **_public_metrics(total),
                "phases": [{"mode": mode, "fold": fold, **_public_metrics(phase)}
                           for (mode, fold), phase in phases.items()],
            }
        return {
            "status": "unavailable" if self.error else "completed", "error": self.error,
            "interval_seconds": self.interval, "wall_seconds": wall,
            "process_cpu_seconds": self.cpu_seconds,
            "process_cpu_percent": 100 * self.cpu_seconds / wall if wall else 0.0,
            "sample_count": sampled.get("sample_count", 0), "summary": sampled.get("summary", {}),
            "timeline": [{"time_seconds": sample["timestamp"] - self.started,
                          **{k: v for k, v in sample.items() if k not in ("timestamp", "collection_start")}} for sample in samples],
            "nodes": nodes,
            "overhead": {
                "startup_seconds": self.startup_seconds, "shutdown_seconds": self.shutdown_seconds,
                "aggregation_seconds": 0.0, "boundary_seconds": self.boundary_seconds,
                "boundary_calls": self.boundary_calls,
                "service_cpu_seconds": sampled.get("service_cpu_seconds", 0.0),
                "service_peak_rss_bytes": sampled.get("service_peak_rss_bytes", 0),
                "sampling_seconds": sampled.get("sampling_seconds", 0.0),
                "sampling_calls": sampled.get("sample_count", 0),
                "failed_samples": sampled.get("failed_samples", 0),
                "dropped_samples": sampled.get("dropped_samples", 0), "dropped_windows": self.dropped_windows,
            },
        }


@contextmanager
def hardware_run(enabled, interval):
    if not enabled:
        yield None
        return
    tracker = HardwareTracker(interval)
    tracker.start()
    token = _active_tracker.set(tracker if tracker.error is None else None)
    try:
        yield tracker
    finally:
        _active_tracker.reset(token)
        try:
            tracker.stop()
        except Exception:
            logger.warning("Hardware tracking finalization failed", exc_info=True)
