"""Hooks the public API calls into. Each is a no-op unless instrumentation is configured."""
from __future__ import annotations

from functools import wraps
from time import perf_counter

from stratum._config import DEV_FLAGS, FLAGS, get_config
from stratum._instrumentation._diagnostics import current_diagnostics, diagnostic_run
from stratum._instrumentation._hardware import get_hardware_tracker, hardware_run
from stratum._instrumentation._observer import observe


def enabled():
    return DEV_FLAGS.diagnostics_dir is not None or DEV_FLAGS.hardware_utilization


def record_run(kind):
    """Own both profilers at API entry, before data preparation or optimization."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if not enabled():
                return function(*args, **kwargs)
            with diagnostic_run(DEV_FLAGS.diagnostics_dir, kind, get_config()) as diagnostics:
                tracker = None
                try:
                    with hardware_run(DEV_FLAGS.hardware_utilization, DEV_FLAGS.hardware_sampling_interval) as tracker:
                        return function(*args, **kwargs)
                finally:
                    if tracker is not None and tracker.report is not None:
                        if diagnostics is not None:
                            diagnostics.hardware = tracker.report
                        if FLAGS.stats:
                            hardware_stats_printer(tracker.report)
        return wrapped
    return decorate


def optimization_started():
    diagnostics = current_diagnostics()
    if diagnostics is not None:
        diagnostics.stage = "optimization"
        diagnostics.optimization_start = perf_counter()


def execution_started(sched):
    """Attach the per-operator observer to the scheduler about to run."""
    diagnostics, tracker = current_diagnostics(), get_hardware_tracker()
    if diagnostics is None and tracker is None:
        return
    observe(sched, tracker, collect_node_timings=diagnostics is not None)
    if diagnostics is not None:
        diagnostics.scheduler = sched
        diagnostics.stage = "execution"
        diagnostics.execution_start = perf_counter()


def record_timings(run):
    diagnostics = current_diagnostics()
    if diagnostics is not None:
        diagnostics.timings = vars(run)


def hardware_stats_printer(report):
    overhead = report["overhead"]
    print("Hardware utilization (complete API run; 100% CPU = one core):")
    print(f"  Status:       {report['status']}")
    print(f"  Wall:         {report['wall_seconds']:.4f} s")
    print(f"  Engine CPU:   {report['process_cpu_percent']:.1f}% ({report['process_cpu_seconds']:.4f} core-s)")
    peak = report["summary"].get("tree_rss_peak_bytes")
    print(f"  Tree peak RSS: {peak / 2**20:.2f} MiB" if peak is not None else "  Tree peak RSS: unavailable")
    print(f"  Samples:      {report['sample_count']}")
    print("Tracking overhead (local API calls; no per-node IPC):")
    print(f"  Boundaries:   {overhead['boundary_seconds']:.6f} s / {overhead['boundary_calls']} calls")
    print(f"  Startup:      {overhead['startup_seconds']:.6f} s")
    print(f"  Shutdown:     {overhead['shutdown_seconds']:.6f} s")
    print(f"  Aggregation:  {overhead['aggregation_seconds']:.6f} s")
    print(f"  Sampler CPU:  {overhead['service_cpu_seconds']:.6f} core-s (separate process)")
    print(f"  Dropped:      {overhead['dropped_samples']} samples, {overhead['dropped_windows']} node windows")
    print()
