from __future__ import annotations
import os
import time
from . _config import get_config

T0 = time.perf_counter()

# Set the rust backend related config knobs
def __getattr__(name):
    rc = get_config()
    if name == "USE_RUST":      #feature flag for rust backend
        use_rust = os.getenv("SKRUB_RUST", "0") == "1" or bool(rc.get("rust_backend", False))

        # Sync env flags to python config for the rust backend to read dynamically
        if use_rust:
            # Set SKRUB_RUST_DEBUG_TIMING to 1 if debug_timing is true
            os.environ["SKRUB_RUST_DEBUG_TIMING"] = "1" if bool(rc.get("debug_timing", False)) else "0"
            # Set SKRUB_RUST_THREADS from num_threads
            os.environ["SKRUB_RUST_THREADS"] = str(int(rc.get("num_threads", 0)))
        return use_rust

    if name == "NUM_THREADS":   #number of threads for all rust OPs. 0 -> global threadpool.
        nt = os.getenv("SKRUB_RUST_THREADS")
        return int(nt) if nt is not None else int(rc.get("num_threads", 0))

    if name == "DEBUG_TIMING":  #print debug timing
        dt = os.getenv("SKRUB_RUST_DEBUG_TIMING")
        return (dt == "1") if dt is not None else bool(rc.get("debug_timing", False))

    if name == "ALLOW_PATCH":   #kill-switch for all non-sklearn backends. This ignores feature flags.
        ap = os.getenv("SKRUB_RUST_ALLOW_PATCH")
        return (ap == "1") if ap is not None else bool(rc.get("allow_patch", False))
    raise AttributeError(name)

try:
    from . import _rust_backend_native as native
    HAVE_RUST = True
except Exception as e:
    native = False
    HAVE_RUST = False
    _import_error = e

# Utility methods for timing
def start_timing():
    if __getattr__("DEBUG_TIMING"):
        return time.perf_counter()
    return None

def print_timing(msg, start_time):
    if start_time is not None and __getattr__("DEBUG_TIMING"):
        end_time = time.perf_counter()
        print(f"[python] [{(end_time- T0) * 1000:.1f}] {msg}: {(end_time - start_time) * 1000:.1f}ms")


# pandas or polars series -> list (best-effort, minimal overhead)
def _to_list(col):
    try:
        return col.tolist()
    except Exception:
        pass
    try:
        return col.to_list()
    except Exception:
        pass
    return list(col)

#---------------------------------------------

# Re-export compiled rust functions
hashing_tfidf_fit = getattr(native, "hashing_tfidf_csr", None) if native else None
hashing_tfidf_transform = getattr(native, "hashing_tfidf_csr_with_idf", None) if native else None
tfidf_fit = getattr(native, "tfidf_fit_csr", None) if native else None
tfidf_transform = getattr(native, "tfidf_transform_csr", None) if native else None
fd_fit = getattr(native, "fd_fit_from_csr", None) if native else None
fd_transform = getattr(native, "fd_transform_from_csr", None) if native else None
truncated_svd_fit = getattr(native, "truncated_svd_fit_from_csr", None) if native else None
truncated_svd_transform = getattr(native, "truncated_svd_transform_from_csr", None) if native else None
ohe_transform = getattr(native, "ohe_transform_csr", None) if native else None
csr_to_dense = getattr(native, "csr_to_dense", None) if native else None
standard_scale_fit = getattr(native, "standard_scale_fit", None) if native else None
standard_scale_transform = getattr(native, "standard_scale_transform", None) if native else None
