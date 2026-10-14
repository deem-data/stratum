"""Inspection-only instrumentation consumed by Stratum Inspector (stratum-gui).

Not part of the production runtime. Everything here is off by default and only
activates when ``DEV_FLAGS.diagnostics_dir`` or ``DEV_FLAGS.hardware_utilization`` is set
(directly or via ``stratum.dev_config``)
(or a caller opens ``capture_plans``); otherwise every hook is a no-op.
"""
from stratum._instrumentation._run import (
    execution_started, optimization_started, record_run, record_timings,
)
