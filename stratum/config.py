from __future__ import annotations
import os
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
    return s == "true"

def _env_int(name, default=0):
    v = os.getenv(name)
    return int(v) if v is not None else int(default)

@dataclass
class _Flags:
    rust_backend: bool = _env_bool("SKRUB_RUST", False)
    num_threads: int = _env_int("SKRUB_RUST_THREADS", 0)      # 0 => backend decides
    debug_timing: bool = _env_bool("SKRUB_RUST_DEBUG_TIMING", False)
    allow_patch: bool = _env_bool("SKRUB_RUST_ALLOW_PATCH", True)
    stratum_stats: bool = _env_bool("STRATUM_STATS", False)

FLAGS = _Flags()

def set_config(rust_backend: bool | None = None,
           num_threads: int | None = None,
           debug_timing: bool | None = None,
           allow_patch: bool | None = None,
           stratum_stats: bool | None = None) -> None:
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
    if stratum_stats is not None:
        FLAGS.stratum_stats = bool(stratum_stats)
        os.environ["STRATUM_STATS"] = "1" if FLAGS.stratum_stats else "0"

def get_config() -> dict:
    # Shallow copy for safety
    return {
        "rust_backend": FLAGS.rust_backend,
        "num_threads": FLAGS.num_threads,
        "debug_timing": FLAGS.debug_timing,
        "allow_patch": FLAGS.allow_patch,
    }