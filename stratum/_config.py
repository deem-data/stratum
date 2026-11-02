from __future__ import annotations
import os
from contextlib import contextmanager
from dataclasses import dataclass
import logging

from stratum.runtime._eviction_policy import EVICTION_POLICIES

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
#: Reuse-cache admission policies (see stratum/optimizer/_reuse_planning.py) and
#: eviction policies (see stratum/runtime/_eviction_policy.py), by config name.
REUSE_CACHE_ADMISSION_POLICIES = ("all", "none")
REUSE_CACHE_EVICTION_POLICIES = tuple(EVICTION_POLICIES)


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


# Memory fraction of host memory to use for the buffer pool.
_DEFAULT_BUFFER_POOL_MEMORY_FRACTION = 0.7 # default 70%
# Memory fraction of host memory to use for the reuse cache's memory tier.
_DEFAULT_REUSE_CACHE_MEMORY_FRACTION = 0.1
_DEFAULT_REUSE_CACHE_DISK_BUDGET = 10 * 1024**3  # 10 GiB

def _read_memory_fraction(value, name="buffer_pool_memory_fraction") -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a float in (0, 1], got {value!r}.")
    if not 0 < fraction <= 1:
        raise ValueError(
            f"{name} must be in (0, 1], got {value!r}.")
    return fraction


def _read_choice(value, name, choices) -> str:
    if value not in choices:
        raise ValueError(f"Invalid {name} {value!r}; valid values are {list(choices)}.")
    return value


# FIXME: Not all flags need environment variables, only the ones that are shared across backends
@dataclass
class _Flags:
    rust_backend: bool = _env_bool("SKRUB_RUST", False)
    num_threads: int = _env_int("SKRUB_RUST_THREADS", 0)      # 0 => backend decides
    debug_timing: bool = _env_bool("SKRUB_RUST_DEBUG_TIMING", False)
    allow_patch: bool = _env_bool("SKRUB_RUST_ALLOW_PATCH", True)
    scheduler: bool =  False
    stats: bool = False # TODO: sync this flag if stats are supported by other runtimes
    stats_top_k: int = 20
    debug_graph: bool = False
    open_graph: bool = False
    graph_format: str = "svg"
    explain: tuple[str, ...] = ()
    cse: bool = True
    DEBUG: bool = False
    implementation_selector: str = _read_implementation_selector(
        os.getenv("STRATUM_IMPLEMENTATION_SELECTOR", "default"))
    validate_dag: bool = True
    make_selection_op: bool = True
    make_map_op: bool = True
    make_column_projection: bool = True
    rechunk: bool = True
    buffer_pool_memory_fraction: float = _DEFAULT_BUFFER_POOL_MEMORY_FRACTION
    reuse_cache: bool = _env_bool("STRATUM_REUSE_CACHE", False)
    reuse_cache_dir: str = os.getenv("STRATUM_REUSE_CACHE_DIR", ".stratum/reuse_cache")
    reuse_cache_memory_fraction: float = _DEFAULT_REUSE_CACHE_MEMORY_FRACTION
    reuse_cache_disk_budget: int = _DEFAULT_REUSE_CACHE_DISK_BUDGET
    reuse_cache_admission: str = "all"
    reuse_cache_eviction: str = "lru"

FLAGS = _Flags()


@dataclass
class _DevFlags:
    """Developer-only switches, not part of set_config/get_config.

    Set them directly (``DEV_FLAGS.diagnostics_dir = ...``) or scoped with
    ``with stratum.dev_config(diagnostics_dir=...):``.
    """
    # Export plan snapshots and per-node timings for profiling
    diagnostics_dir: str | os.PathLike | None = None
    # Sample CPU/RSS in a separate process (needs the hardware extra).
    hardware_utilization: bool = False
    # Seconds between hardware samples.
    hardware_sampling_interval: float = 0.1

DEV_FLAGS = _DevFlags()


@contextmanager
def dev_config(**kwargs):
    """Temporarily override DEV_FLAGS inside a context."""
    unknown = set(kwargs) - set(vars(DEV_FLAGS))
    if unknown:
        raise TypeError(f"Unknown dev flag(s): {sorted(unknown)}")
    original = vars(DEV_FLAGS).copy()
    vars(DEV_FLAGS).update(kwargs)
    try:
        yield DEV_FLAGS
    finally:
        vars(DEV_FLAGS).update(original)

def set_config(rust_backend: bool | None = None,
    num_threads: int | None = None,
    debug_timing: bool | None = None,
    allow_patch: bool | None = None,
    stats: bool | None = None,
    stats_top_k: int | None = None,
    scheduler: bool | None = None,
    debug_graph: bool | None = None,
    open_graph: bool | None = None,
    graph_format: str | None = None,
    explain: bool | str | list[str] | None = None,
    DEBUG: bool | None = None,
    cse: bool | None = None,
    validate_dag: bool | None = None,
    make_selection_op: bool | None = None,
    make_map_op: bool | None = None,
    make_column_projection: bool | None = None,
    rechunk: bool | None = None,
    buffer_pool_memory_fraction: float | None = None,
    implementation_selector: str | None = None,
    reuse_cache: bool | None = None,
    reuse_cache_dir: str | os.PathLike | None = None,
    reuse_cache_memory_fraction: float | None = None,
    reuse_cache_disk_budget: int | None = None,
    reuse_cache_admission: str | None = None,
    reuse_cache_eviction: str | None = None,
               ) -> None:
    """Runtime toggles (synced env for Rust to read).

    Only the arguments given change; ``None`` keeps a setting as it is. The
    defaults below are those of a fresh process.

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

        stats: bool, default false
            Print optimization and scheduler execution timing,
            DataOp heavy hitters, and BufferPool statistics.

        stats_top_k: int >= 0, default 20
            Set the number of heavy hitters to print when stats is enabled.
            Their percentages use all operator processing time as the denominator.

        open_graph: bool, default false
            Open the graph after optimization.

        graph_format: str, default "svg"
            Output format of the debug graphs. Keep "svg" for anything but a
            small plan: the bitmap formats ("png", "jpg") are capped at 32767
            pixels per side by the cairo renderer, and graphviz silently scales
            a larger drawing down to fit, which is what turns the labels of a
            big plan into unreadable smudge. SVG has no such cap, renders about
            10x faster, and leaves the labels as real text the browser can
            search.

        explain: bool | str | list[str], default off
            Print text-based linear execution plans during optimization. ``False``
            turns it off; ``True``
            prints the executable plan (equivalent to ``["physical_impl"]``); a
            list selects which IR levels to print, any of ``"logical"`` (after
            logical rewrites), ``"physical"`` (after lowering) and
            ``"physical_impl"`` (after implementation selection).

        DEBUG: bool, default false
            Enable/disable debug mode.

        implementation_selector: str, default "default"
            Implementation-selection policy, and with it the dataframe backend:
            ``"default"`` prefers pandas/sklearn-skrub; ``"greedy"`` prefers
            efficient backends (rust/polars) first. This is the only user-facing
            way to put a pipeline on polars -- the backend is a property of the
            selector, not a flag beside it.

        buffer_pool_memory_fraction: float in (0, 1], default 0.7
            Share of the detected system memory used as the buffer pool's memory
            budget. The byte budget is resolved when a BufferPool is constructed
            (fixed 2 GiB fallback when host memory cannot be detected).

        reuse_cache: bool, default false
            Keep intermediates of the pre-split part of a plan in a persistent
            reuse cache, and read them back instead of recomputing them in later
            plans, in this process or in later ones using the same directory.

        reuse_cache_dir: path, default ".stratum/reuse_cache"
            Where the reuse cache keeps its disk tier and manifest.

        reuse_cache_memory_fraction: float in (0, 1], default 0.1
            Share of the detected system memory used as the reuse cache's memory
            budget. Entries beyond it are spilled to its disk tier.

        reuse_cache_disk_budget: int >= 0, default 10 GiB
            Bytes the reuse cache may keep on disk; 0 means unlimited.

        reuse_cache_admission: str, default "all"
            Which intermediates to add to the reuse cache: ``"all"`` (every
            keyed one but constants) or ``"none"`` (read the cache only).

        reuse_cache_eviction: str, default "lru"
            What the reuse cache evicts first, in both tiers: ``"lru"`` or
            ``"cost_size"`` (cheapest to recompute per byte). Fixed when the
            process's cache is created.
    """
    if implementation_selector is not None:
        implementation_selector = _read_implementation_selector(implementation_selector)
    if explain is not None:
        explain = _read_explain_levels(explain)
    if buffer_pool_memory_fraction is not None:
        buffer_pool_memory_fraction = _read_memory_fraction(buffer_pool_memory_fraction)
    if reuse_cache_memory_fraction is not None:
        reuse_cache_memory_fraction = _read_memory_fraction(
            reuse_cache_memory_fraction, "reuse_cache_memory_fraction")
    if reuse_cache_disk_budget is not None and not (
            isinstance(reuse_cache_disk_budget, int) and reuse_cache_disk_budget >= 0):
        raise ValueError("reuse_cache_disk_budget must be an int >= 0")
    if reuse_cache_admission is not None:
        _read_choice(reuse_cache_admission, "reuse_cache_admission",
                     REUSE_CACHE_ADMISSION_POLICIES)
    if reuse_cache_eviction is not None:
        _read_choice(reuse_cache_eviction, "reuse_cache_eviction",
                     REUSE_CACHE_EVICTION_POLICIES)
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
    if implementation_selector is not None:
        FLAGS.implementation_selector = implementation_selector
        os.environ["STRATUM_IMPLEMENTATION_SELECTOR"] = implementation_selector
    # TODO: Select between multiple schedulers in the future.
    toggles = {"scheduler": scheduler, "cse": cse, "debug_graph": debug_graph,
               "open_graph": open_graph, "make_selection_op": make_selection_op,
               "make_map_op": make_map_op, "make_column_projection": make_column_projection,
               "rechunk": rechunk, "validate_dag": validate_dag}
    for name, value in toggles.items():
        if value is not None:
            setattr(FLAGS, name, bool(value))
    if graph_format is not None:
        FLAGS.graph_format = str(graph_format)
    if buffer_pool_memory_fraction is not None:
        FLAGS.buffer_pool_memory_fraction = buffer_pool_memory_fraction
    if explain is not None:
        FLAGS.explain = explain
    if reuse_cache is not None:
        FLAGS.reuse_cache = bool(reuse_cache)
    if reuse_cache_dir is not None:
        FLAGS.reuse_cache_dir = os.fspath(reuse_cache_dir)
    if reuse_cache_memory_fraction is not None:
        FLAGS.reuse_cache_memory_fraction = reuse_cache_memory_fraction
    if reuse_cache_disk_budget is not None:
        FLAGS.reuse_cache_disk_budget = reuse_cache_disk_budget
    if reuse_cache_admission is not None:
        FLAGS.reuse_cache_admission = reuse_cache_admission
    if reuse_cache_eviction is not None:
        FLAGS.reuse_cache_eviction = reuse_cache_eviction


def get_config() -> dict:
    # Shallow copy for safety
    return vars(FLAGS).copy() # asdict if we want a deep copy

_CONFIG_ENV_VARS = (
    "SKRUB_RUST", "SKRUB_RUST_THREADS", "SKRUB_RUST_DEBUG_TIMING",
    "SKRUB_RUST_ALLOW_PATCH", "STRATUM_DEBUG",
    "STRATUM_IMPLEMENTATION_SELECTOR",
)

@contextmanager
def config(**kwargs):
    """Temporarily override runtime config inside a context."""
    original = get_config()
    original_env = {name: os.environ.get(name) for name in _CONFIG_ENV_VARS}
    stratum_logger = logging.getLogger("stratum")
    prev_level = stratum_logger.level
    try:
        set_config(**kwargs)
        if kwargs.get("DEBUG", False):
            # set for this module stratum only
            print("DEBUG MODE ENABLED")
            logging.basicConfig(level=logging.INFO)
            stratum_logger.setLevel(logging.DEBUG)
        yield
    finally:
        # Restore by writing the snapshot back rather than replaying it through
        # `set_config`: `get_config` dumps every FLAGS field, so a replay only
        # works while the two stay in exact lockstep, and silently breaks every
        # `with config(...)` block the moment a field is not a set_config
        # parameter.
        for name, value in original.items():
            setattr(FLAGS, name, value)
        for name, value in original_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        stratum_logger.setLevel(prev_level)
