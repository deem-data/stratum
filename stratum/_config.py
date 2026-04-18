from __future__ import annotations
import os
from contextlib import contextmanager
from dataclasses import dataclass

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

def _env_str(name, default=None):
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("", "none", "null"):
        return None
    return s

@dataclass
class _Flags:
    rust_backend: bool = _env_bool("SKRUB_RUST", False)
    num_threads: int = _env_int("SKRUB_RUST_THREADS", 0)      # 0 => backend decides
    debug_timing: bool = _env_bool("SKRUB_RUST_DEBUG_TIMING", False)
    allow_patch: bool = _env_bool("SKRUB_RUST_ALLOW_PATCH", True)
    scheduler: bool =  False
    stats: bool = False # TODO if we want to use that flag on other runtimes we need to set envirenment variable as well
    stats_top_k: int = 20
    open_graph: bool = False,
    cse: bool = True,
    DEBUG: bool = False
    force_polars: bool = _env_bool("STRATUM_FORCE_POLARS", False)
    fast_dataops_convert: bool = True

FLAGS = _Flags()

def set_config(rust_backend: bool | None = None,
           num_threads: int | None = None,
           debug_timing: bool | None = None,
           allow_patch: bool | None = None,
           stats: bool | None = None,
           stats_top_k: int | None = None,
           scheduler: bool | None = None,
           open_graph: bool | None = None,
           DEBUG: bool | None = None,
           force_polars: bool | None = None,
           cse: bool = True,
           fast_dataops_convert: bool = True) -> None:
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

        stratum_stats: bool, default false
            Enable/disable stratum statistics. This will print the heavy hitters of a DataOp DAG execution.

        stats_top_k: int >= 0, default 20
            Set the number of heavy hitters to print when stats is enabled.

        open_graph: bool, default true
            Open the graph after optimization.

        DEBUG: bool, default false
            Enable/disable debug mode.

        force_polars: bool, default false
            Force use of Polars instead of Pandas for dataframe operations.
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
    if scheduler is not None:
        FLAGS.scheduler = bool(scheduler)
    if stats is not None:
        FLAGS.stats = bool(stats)
    if stats_top_k is not None:
        if not (isinstance(stats_top_k, int) and stats_top_k >= 0):
            raise ValueError("stats_top_k must be an int >= 0")
        FLAGS.stats_top_k = int(stats_top_k)
    if open_graph is not None:
        FLAGS.open_graph = bool(open_graph)
    if DEBUG is not None:
        FLAGS.DEBUG = bool(DEBUG)
        os.environ["STRATUM_DEBUG"] = "1" if FLAGS.DEBUG else "0"
    if force_polars is not None:
        FLAGS.force_polars = bool(force_polars)
        os.environ["STRATUM_FORCE_POLARS"] = "1" if FLAGS.force_polars else "0"
    if cse is not None:
        FLAGS.cse = bool(cse)
    if fast_dataops_convert is not None:
        FLAGS.fast_dataops_convert = bool(fast_dataops_convert)


def get_config() -> dict:
    # Shallow copy for safety
    return {
        "rust_backend": FLAGS.rust_backend,
        "num_threads": FLAGS.num_threads,
        "debug_timing": FLAGS.debug_timing,
        "allow_patch": FLAGS.allow_patch,
        "scheduler": FLAGS.scheduler,
        "stats": FLAGS.stats,
        "stats_top_k": FLAGS.stats_top_k,
        "open_graph": FLAGS.open_graph,
        "DEBUG" : FLAGS.DEBUG,
        "force_polars": FLAGS.force_polars,
        "cse": FLAGS.cse,
        "fast_dataops_convert": FLAGS.fast_dataops_convert,
    }

@contextmanager
def config(**kwargs):
    """Temporarily override runtime config inside a context."""
    original = get_config()
    set_config(**kwargs)
    try:
        yield
    finally:
        set_config(**original)
