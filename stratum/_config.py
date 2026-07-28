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


#: IR levels the optimizer can print a linear plan for:
#: ``logical`` (after logical rewrites), ``physical`` (after lowering) and
#: ``physical_impl`` (after implementation selection, i.e.,  the executable plan).
EXPLAIN_LEVELS = ("logical", "physical", "physical_impl")
IMPLEMENTATION_SELECTOR_MODES = ("default", "greedy")


def _read_implementation_selector(value: str) -> str:
    """Read and validate the configured implementation-selector mode."""
    if not isinstance(value, str):
        raise ValueError(
            "implementation_selector must be one of "
            f"{list(IMPLEMENTATION_SELECTOR_MODES)}, got {value!r}.")
    mode = value.strip().lower()
    if mode not in IMPLEMENTATION_SELECTOR_MODES:
        raise ValueError(
            "Invalid implementation_selector "
            f"{value!r}; valid modes are {list(IMPLEMENTATION_SELECTOR_MODES)}.")
    return mode


def _read_explain_levels(value) -> tuple[str, ...]:
    """Coerce the ``explain`` config into a tuple of IR levels to print.

    ``None``/``False`` -> off; ``True`` -> the executable plan only
    (``("physical_impl",)``); a level (or list of levels) -> those levels.
    """
    if value is None or value is False:
        return ()
    if value is True:
        return ("physical_impl",)
    levels = (value,) if isinstance(value, str) else tuple(value)
    invalid = [lvl for lvl in levels if lvl not in EXPLAIN_LEVELS]
    if invalid:
        raise ValueError(
            f"Invalid explain level(s) {invalid}; valid levels are {list(EXPLAIN_LEVELS)}.")
    return levels


# FIXME: Not all flags need environment variables, only the ones that are shared across backends
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
    explain: tuple[str, ...] = ()
    cse: bool = True
    DEBUG: bool = False
    force_polars: bool = _env_bool("STRATUM_FORCE_POLARS", False)
    implementation_selector: str = _read_implementation_selector(
        os.getenv("STRATUM_IMPLEMENTATION_SELECTOR", "default"))
    pandas_query: bool = _env_bool("STRATUM_PANDAS_QUERY", False)
    fast_dataops_convert: bool = True
    validate_dag: bool = True
    make_selection_op: bool = True
    make_map_op: bool = True
    make_column_projection: bool = True
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
    explain: bool | str | list[str] | None = None,
    DEBUG: bool | None = None,
    force_polars: bool = False,
    pandas_query: bool = False,
    cse: bool = True,
    fast_dataops_convert: bool = True,
    validate_dag: bool = True,
    make_selection_op: bool = True,
    make_map_op: bool = True,
    make_column_projection: bool = True,
    rechunk: bool = True,
    buffer_pool_memory_budget: int = 0,
    implementation_selector: str = "default",
               ) -> None:
    """Runtime toggles (synced env for Rust to read).

    Parameter:
    -----------

        rust_backend: bool, default false
            Legacy feature flag for Rust execution. The physical
            operator selector should choose Rust through the registry instead.

        num_threads: int >= 0 (0 lets backend decide), default 0
            Set the number of threads for the multithreaded rust operations.

        debug_timing: bool, default false
            Print the timing in standard output.

        allow_patch: bool, default true
            Legacy kill-switch for direct adapter Rust execution. It does not
            control physical operator registration.

        scheduler: bool, default false
            Enable/disable stratum's scheduler instead of skrub's make_grid_search.

        stratum_stats: bool, default false
            Enable/disable stratum statistics. This will print the heavy hitters of a DataOp DAG execution.

        stats_top_k: int >= 0, default 20
            Set the number of heavy hitters to print when stats is enabled.

        open_graph: bool, default true
            Open the graph after optimization.

        explain: bool | str | list[str], default None
            Print text-based linear execution plans during optimization. ``True``
            prints the executable plan (equivalent to ``["physical_impl"]``); a
            list selects which IR levels to print, any of ``"logical"`` (after
            logical rewrites), ``"physical"`` (after lowering) and
            ``"physical_impl"`` (after implementation selection).

        DEBUG: bool, default false
            Enable/disable debug mode.

        force_polars: bool, default false
            Legacy frame-backend flag. It does not override the configured
            implementation selector.

        implementation_selector: str, default "default"
            Implementation-selection policy. ``"default"`` prefers pandas/
            sklearn-skrub; ``"greedy"`` prefers efficient backends
            (rust/polars) first.

        pandas_query: bool, default false
            Evaluate MASK selections on the pandas backend via ``DataFrame.query()``
            when the predicate is expressible as a query string (no OperandLeaf / str
            accessor); otherwise fall back to boolean-mask indexing.
    """
    implementation_selector = _read_implementation_selector(implementation_selector)

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
        os.environ["SKRUB_RUST_ALLOW_PATCH"] = "1" if FLAGS.allow_patch else "0"
    if stats is not None:
        FLAGS.stats = bool(stats)
    if stats_top_k is not None:
        if not (isinstance(stats_top_k, int) and stats_top_k >= 0):
            raise ValueError("stats_top_k must be an int >= 0")
        FLAGS.stats_top_k = int(stats_top_k)
    if DEBUG is not None:
        FLAGS.DEBUG = bool(DEBUG)
        os.environ["STRATUM_DEBUG"] = "1" if FLAGS.DEBUG else "0"
    if force_polars is not None:
        FLAGS.force_polars = bool(force_polars)
        os.environ["STRATUM_FORCE_POLARS"] = "1" if FLAGS.force_polars else "0"
    FLAGS.implementation_selector = implementation_selector
    os.environ["STRATUM_IMPLEMENTATION_SELECTOR"] = implementation_selector
    FLAGS.pandas_query = bool(pandas_query)
    os.environ["STRATUM_PANDAS_QUERY"] = "1" if FLAGS.pandas_query else "0"
    # TODO: Select between multiple schedulers in the future.
    FLAGS.scheduler = bool(scheduler)
    FLAGS.cse = bool(cse)
    FLAGS.debug_graph = bool(debug_graph)
    FLAGS.open_graph = bool(open_graph)
    FLAGS.buffer_pool_memory_budget = int(buffer_pool_memory_budget)
    FLAGS.explain = _read_explain_levels(explain)
    FLAGS.make_selection_op = bool(make_selection_op)
    FLAGS.make_map_op = bool(make_map_op)
    FLAGS.make_column_projection = bool(make_column_projection)
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
