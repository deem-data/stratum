"""Optional, versioned run artifacts. No web server or GUI dependency lives here."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import json
from time import perf_counter
from uuid import uuid4

logger = logging.getLogger(__name__)
_active_run = ContextVar("stratum_diagnostics", default=None)
_plan_capture = ContextVar("stratum_plan_capture", default=None)
_capture_svg = ContextVar("stratum_capture_svg", default=False)
_job = ContextVar("stratum_job", default=None)


@dataclass
class NodeTiming:
    """Operator processing only; input/output buffer work is measured separately."""

    count: int = 0
    seconds: float = 0.0
    buffer_seconds: float = 0.0
    failures: int = 0
    phases: dict = field(default_factory=dict)

    def record(self, mode, fold, seconds, buffer_seconds, failed=False):
        self.count += 1
        self.seconds += seconds
        self.buffer_seconds += buffer_seconds
        self.failures += int(failed)
        key = (mode, fold)
        phase = self.phases.get(key)
        if phase is None:
            phase = self.phases[key] = {
                "mode": mode, "fold": fold, "count": 0, "seconds": 0.0,
                "buffer_seconds": 0.0, "failures": 0,
            }
        phase["count"] += 1
        phase["seconds"] += seconds
        phase["buffer_seconds"] += buffer_seconds
        phase["failures"] += int(failed)

    def to_dict(self):
        return {
            "count": self.count, "seconds": self.seconds,
            "buffer_seconds": self.buffer_seconds, "failures": self.failures,
            "phases": list(self.phases.values()),
        }


def plan_snapshot(name, ops, split_pos=None):
    """Use plan-local positions as IDs; freeze labels before further IR mutations."""
    ids = {op: i for i, op in enumerate(ops)}
    return {
        "name": name, "split_pos": split_pos,
        "nodes": [{
            "id": i, "label": str(op), "implementation": type(op).__name__,
            "inputs": [ids[inp] for inp in op.inputs],
            "output_type": op.output_type.name,
            "train": op.train, "mode": op.mode,
        } for i, op in enumerate(ops)],
    }


def capture_plan(name, ops, split_pos=None):
    """Only traverse when runtime diagnostics or in-memory capture is enabled."""
    plans = _plan_capture.get()
    if plans is not None:
        ops = list(ops)
        snapshot = plan_snapshot(name, ops, split_pos)
        plans.append(snapshot)
        if _capture_svg.get():
            from stratum._profiler._graph import graph_svg, op_graph
            snapshot["svg"] = graph_svg(op_graph(ops, name))
        return
    run = _active_run.get()
    if run is not None:
        try:
            run.plans.append(plan_snapshot(name, list(ops), split_pos))
        except Exception:
            logger.warning("Could not capture diagnostics plan %s", name, exc_info=True)


def diagnostics_enabled():
    return _active_run.get() is not None or _plan_capture.get() is not None


@contextmanager
def capture_plans(*, include_svg=False):
    """Capture optimizer checkpoints in memory without a scheduler or disk export.

    Unlike best-effort runtime export, capture errors propagate to the caller.
    Nested captures restore the previous sink when they finish.
    """
    plans = []
    token = _plan_capture.set(plans)
    svg_token = _capture_svg.set(include_svg)
    try:
        yield plans
    finally:
        _capture_svg.reset(svg_token)
        _plan_capture.reset(token)


def current_diagnostics():
    return _active_run.get()


def record_reuse_plan(ops, region, keys, reasons):
    """Record planning decisions for surviving nodes, before execution."""
    run = current_diagnostics()
    if run is None:
        return
    try:
        decisions = {}
        for op in ops:
            if op.cache_hit is not None:
                decision = {"status": "hit", "key": op.cache_hit.key}
            elif op.cache_key is not None:
                decision = {"status": "eligible", "key": op.cache_key}
            elif op not in region:
                decision = {"status": "outside"}
            elif keys.get(op) is None:
                decision = {"status": "unkeyed", "reason": reasons.get(op)}
            else:
                decision = {"status": "keyed", "key": keys[op]}
            decisions[op] = decision
        run.reuse_plan = decisions
    except Exception:
        logger.warning("Could not capture reuse diagnostics", exc_info=True)


@contextmanager
def job_context(job_id: str):
    """Attribute the runs exported inside to ``job_id`` (a server job).

    Yields the job's record; ``run_ids`` collects the runs exported for it.
    """
    job = {"job_id": job_id, "run_ids": []}
    token = _job.set(job)
    try:
        yield job
    finally:
        _job.reset(token)


class RunDiagnostics:
    def __init__(self, directory, kind, config):
        self.directory = Path(directory).expanduser()
        self.run_id = uuid4().hex
        self.job = _job.get()
        self.kind = kind
        self.config = config
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.plans = []
        self.scheduler = None
        self.timings = None
        self.stage = "preparation"
        self.optimization_start = None
        self.execution_start = None
        self.hardware = None
        self.reuse_cache = None
        self.reuse_stats_before = None
        self.reuse_plan = {}

    def attach_reuse_cache(self, cache):
        """Start the counter interval before this run plans its cache reuse."""
        self.reuse_cache = cache
        self.reuse_stats_before = asdict(cache.stats)

    def reuse_cache_report(self, sched):
        """Cache-wide state plus this run's planning decisions by physical node ID.

        Counters cover the run's interval, as in server job reports. Concurrent
        users of the same cache can contribute to that interval too.
        """
        report = self.reuse_cache.snapshot(include_entries=True)
        entries = {entry["key"]: entry for entry in report.pop("entry_details")}
        report["stats"] = {name: value - self.reuse_stats_before[name]
                           for name, value in report["stats"].items()}
        report["stats_scope"] = "run_interval"
        nodes = report["nodes"] = {}
        for i, op in enumerate(sched.linearized_dag if sched is not None else ()):
            decision = self.reuse_plan.get(op)
            if decision is None:
                continue
            node = dict(decision)
            if node["status"] in ("hit", "eligible"):
                node["label"] = str(op)
            if node["status"] == "eligible":
                entry = entries.get(node["key"])
                node["stored"] = entry is not None
                if entry is not None:
                    node["size"] = entry["size"]
            nodes[str(i)] = node
        return report

    def partial_timings(self, start):
        """Timings of a run that did not reach ``record_timings``, split by stage marks."""
        end = perf_counter()
        if self.optimization_start is None:
            return {"total": end - start, "optimization": None, "execution": None}
        execution_start = self.execution_start or end
        return {
            "total": end - self.optimization_start,
            "optimization": execution_start - self.optimization_start,
            "execution": end - self.execution_start if self.execution_start else None,
        }

    def publish(self, error=None):
        sched = self.scheduler
        document = {
            "schema_version": 1, "run_id": self.run_id, "kind": self.kind,
            "job_id": self.job["job_id"] if self.job is not None else None,
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed" if error is not None else "completed",
            "error": {"type": type(error).__name__, "message": str(error),
                      "stage": self.stage} if error is not None else None,
            "config": self.config, "plans": self.plans,
            "timings": self.timings,
            "node_timings": {}, "heavy_hitters": [], "buffer_pool": None,
            "hardware": self.hardware,
            "reuse_cache": None,
        }
        if self.reuse_cache is not None:
            document["reuse_cache"] = self.reuse_cache_report(sched)
        if sched is not None:
            node_timings = (sched.observer.node_timings if sched.observer is not None else None) or {}
            document["node_timings"] = {
                str(i): node_timings[op].to_dict()
                for i, op in enumerate(sched.linearized_dag)
                if op in node_timings
            }
            groups = {}
            for op, timing in node_timings.items():
                label = str(op)
                group = groups.setdefault(label, {"label": label, "count": 0, "seconds": 0.0})
                group["count"] += timing.count
                group["seconds"] += timing.seconds
            document["heavy_hitters"] = sorted(groups.values(), key=lambda g: g["seconds"], reverse=True)
            document["buffer_pool"] = asdict(sched.pool.stats)

        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{self.run_id}.json"
        temporary = self.directory / f".{self.run_id}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(document, stream, separators=(",", ":"), allow_nan=False)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        if self.job is not None:
            self.job["run_ids"].append(self.run_id)


@contextmanager
def diagnostic_run(directory, kind, config):
    """Export failures too; diagnostics must never change the computation outcome."""
    if directory is None:
        yield None
        return
    run = RunDiagnostics(directory, kind, config)
    token = _active_run.set(run)
    start = perf_counter()
    error = None
    try:
        yield run
    except BaseException as exc:
        error = exc
        raise
    finally:
        _active_run.reset(token)
        if run.timings is None:
            run.timings = run.partial_timings(start)
        try:
            run.publish(error)
        except Exception:
            logger.warning("Could not export Stratum diagnostics to %s", directory, exc_info=True)
