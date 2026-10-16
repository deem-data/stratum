"""Inspection-only profiler

Not part of the production runtime. Everything here is off by default and only
activates when ``DEV_FLAGS.diagnostics_dir`` or ``DEV_FLAGS.hardware_utilization`` is set
(directly or via ``stratum.dev_config``)
(or a caller opens ``capture_plans``). Call sites check ``enabled()`` before
calling a hook, so a disabled profiler is never started.
"""
from stratum._profiler._run import (
    enabled, execution_started, optimization_started, record_run, record_timings,
)
