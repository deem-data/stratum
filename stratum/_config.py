from __future__ import annotations
import os
from contextlib import contextmanager
from dataclasses import dataclass
import logging

def _env_bool(name, default=False):
    val = os.getenv(name)
    if val is None:
        return bool(default)
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return bool(default)

def _env_int(name, default=0):
    v = os.getenv(name)
    return int(v) if v is not None else int(default)


@dataclass
class _Flags:
    rust_backend: bool = _env_bool("SKRUB_RUST", False)
    num_threads: int = _env_int("SKRUB_RUST_THREADS", 0)      # 0 => backend decides
    debug_timing: bool = _env_bool("SKRUB_RUST_DEBUG_TIMING", False)
    allow_patch: bool = _env_bool("SKRUB_RUST_ALLOW_PATCH", True)
    scheduler: bool =  False
    stats: bool = False # TODO if we want to use that flag on other runtimes we need to set envirenment variable as well
    stats_top_k: int = 20
    debug_graph: bool = False
    open_graph: bool = False
    explain_linear_plan: bool = False
    cse: bool = True
    DEBUG: bool = False
    force_polars: bool = _env_bool("STRATUM_FORCE_POLARS", False)
    pandas_query: bool = _env_bool("STRATUM_PANDAS_QUERY", False)
    fast_dataops_convert: bool = True
    validate_dag: bool = True
    make_selection_op: bool = True
    rechunk: bool = True
    buffer_pool_memory_budget: int = 0

FLAGS = _Flags()

def set_config(rust_backend: bool | None = None,
    num_threads: int | None = None,
    debug_timing: bool | None = None,
    allow_patch: bool | None = None,
    stats: bool | None = None,
    stats_top_k: int | None = None,
    scheduler: bool = False,
    debug_graph: bool = False,
    open_graph: bool = False,
    explain_linear_plan: bool = False,
    DEBUG: bool | None = None,
    force_polars: bool = False,
    pandas_query: bool = False,
    cse: bool = True,
    fast_dataops_convert: bool = True,
    validate_dag: bool = True,
    make_selection_op: bool = True,
    rechunk: bool = True,
buffer_pool_memory_budget: int = 0
               ) -> None:
    """Runtime toggles (synced env for Rust to read).

    Parameter:
    -----------

        rust_backend: bool, default false
            Enable/disable rust backend. It is a feature flag for the Rust backend.

        num_threads: int >= 0 (0 lets backend decide), default 0
            Set the number of threads for the multithreaded rust operations.

        debug_timing: bool, default false
            Print the timing in standard output.

        allow_patch: bool, default true
            Allows disabling runtime backend swapping in sensitive contexts. This is a soft
            kill-switch for disabling all non-sklearn backends, even if their flags are set.

        scheduler: bool, default false
            Enable/disable stratum's scheduler instead of skrub's make_grid_search.

        stratum_stats: bool, default false
            Enable/disable stratum statistics. This will print the heavy hitters of a DataOp DAG execution.

        stats_top_k: int >= 0, default 20
            Set the number of heavy hitters to print when stats is enabled.

        open_graph: bool, default true
            Open the graph after optimization.

        explain_linear_plan: bool, default false
            Print a text-based linear execution plan after optimization.

        DEBUG: bool, default false
            Enable/disable debug mode.

        force_polars: bool, default false
            Force use of Polars instead of Pandas for dataframe operations.

        pandas_query: bool, default false
            Evaluate MASK selections on the pandas backend via ``DataFrame.query()``
            when the predicate is expressible as a query string (no OperandLeaf / str
            accessor); otherwise fall back to boolean-mask indexing.
    """
    if rust_backend is not None:
        FLAGS.rust_backend = bool(rust_backend)
        os.environ["SKRUB_RUST"] = "1" if FLAGS.rust_backend else "0"
    if num_threads is not None:
        if not (isinstance(num_threads, int) and num_threads >= 0):
            raise ValueError("num_threads must be an int >= 0")
        FLAGS.num_threads = int(num_threads)
        os.environ["SKRUB_RUST_THREADS"] = str(FLAGS.num_threads)
    if debug_timing is not None:
        FLAGS.debug_timing = bool(debug_timing)
        os.environ["SKRUB_RUST_DEBUG_TIMING"] = "1" if FLAGS.debug_timing else "0"
    if allow_patch is not None:
        FLAGS.allow_patch = bool(allow_patch)
        os.environ["SKRUB_RUST_ALLOW_MONKEYPATCH"] = "1" if FLAGS.allow_patch else "0"
    if stats is not None:
        FLAGS.stats = bool(stats)
    if stats_top_k is not None:
        if not (isinstance(stats_top_k, int) and stats_top_k >= 0):
            raise ValueError("stats_top_k must be an int >= 0")
        FLAGS.stats_top_k = int(stats_top_k)
    if DEBUG is not None:
        FLAGS.DEBUG = bool(DEBUG)
        os.environ["STRATUM_DEBUG"] = "1" if FLAGS.DEBUG else "0"
    #FIXME: This is a temporary flag. Remove once we have the operator selector.
    if force_polars is not None:
        FLAGS.force_polars = bool(force_polars)
        os.environ["STRATUM_FORCE_POLARS"] = "1" if FLAGS.force_polars else "0"
    FLAGS.pandas_query = bool(pandas_query)
    os.environ["STRATUM_PANDAS_QUERY"] = "1" if FLAGS.pandas_query else "0"
    # TODO: Select between multiple schedulers in the future.
    FLAGS.scheduler = bool(scheduler)
    FLAGS.cse = bool(cse)
    FLAGS.debug_graph = bool(debug_graph)
    FLAGS.open_graph = bool(open_graph)
    FLAGS.buffer_pool_memory_budget = int(buffer_pool_memory_budget)
    FLAGS.explain_linear_plan = bool(explain_linear_plan)
    FLAGS.make_selection_op = bool(make_selection_op)
    FLAGS.rechunk = bool(rechunk)

    #FIXME: This should be the default. No need to set it. Remove.
    FLAGS.fast_dataops_convert = bool(fast_dataops_convert)
    FLAGS.validate_dag = bool(validate_dag)


def get_config() -> dict:
    # Shallow copy for safety
    return vars(FLAGS).copy() # asdict if we want a deep copy

@contextmanager
def config(**kwargs):
    """Temporarily override runtime config inside a context."""
    original = get_config()
    set_config(**kwargs)
    stratum_logger = logging.getLogger("stratum")
    prev_level = stratum_logger.level
    if kwargs.get("DEBUG", False):
        # set for this module stratum only
        print("DEBUG MODE ENABLED")
        logging.basicConfig(level=logging.INFO)
        stratum_logger.setLevel(logging.DEBUG)
    try:
        yield
    finally:
        set_config(**original)
        stratum_logger.setLevel(prev_level)
