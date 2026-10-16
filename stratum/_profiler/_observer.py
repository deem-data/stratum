"""Per-operator hooks the scheduler calls when an observer is attached.

The scheduler only checks ``self.observer is not None``; all timing and
hardware bookkeeping lives here.
"""
from __future__ import annotations

from time import perf_counter

from stratum._profiler._diagnostics import NodeTiming


def observe(sched, tracker=None, collect_node_timings=True):
    """Attach a RunObserver to ``sched`` and return it."""
    sched.observer = RunObserver(sched, tracker, collect_node_timings)
    return sched.observer


class RunObserver:
    def __init__(self, sched, tracker=None, collect_node_timings=True):
        self.node_timings = {} if collect_node_timings else None
        self.tracker = tracker
        if tracker is not None:
            tracker.bind_scheduler(sched)

    def before(self, op):
        """Called right before ``op.process``; returns a token for after/failed."""
        if self.tracker is None:
            return None
        mark = perf_counter()
        cpu = self.tracker.begin()
        return cpu, perf_counter() - mark

    def after(self, op, mode, fold, token, t1, t2, buffer_seconds):
        """Called once the output is in the buffer pool."""
        if token is not None:
            # The begin() call ran inside the scheduler's buffer window.
            buffer_seconds = max(0.0, buffer_seconds - token[1])
            self._end_hardware(op, mode, fold, token, t1, t2)
        if self.node_timings is not None:
            self._record(op, mode, fold, t2 - t1, buffer_seconds)

    def failed(self, op, mode, fold, token, t0, t1, t2):
        """Called on any processing error; t1/t2 are None if never reached."""
        if t1 is None:
            return  # Failed while pinning inputs, before the operator ran.
        if token is not None:
            self._end_hardware(op, mode, fold, token, t1, t2 if t2 is not None else perf_counter())
        if self.node_timings is not None and t2 is None:
            self._record(op, mode, fold, perf_counter() - t1, t1 - t0, failed=True)

    def _end_hardware(self, op, mode, fold, token, t1, t2):
        cpu, overhead = token
        tracker = self.tracker
        mark = perf_counter()
        metrics = tracker.end(op, mode, fold, cpu, t1, t2)
        overhead += perf_counter() - mark
        metrics.tracking_seconds += overhead
        tracker.boundary_seconds += overhead
        tracker.boundary_calls += 2

    def _record(self, op, mode, fold, seconds, buffer_seconds, failed=False):
        timing = self.node_timings.get(op)
        if timing is None:
            timing = self.node_timings[op] = NodeTiming()
        timing.record(mode, fold, seconds, buffer_seconds, failed=failed)
