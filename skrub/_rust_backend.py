import os
import time

from skrub._config import get_rust_config

# Set the rust backend related config knobs
def __getattr__(name):
    rc = get_rust_config()
    if name == "USE_RUST":      #feature flag for rust backend
        use_rust = os.getenv("SKRUB_RUST", "0") == "1" or bool(rc.get("enable_rust", False))

        # Sync env flags to python config for the rust backend to read dynamically
        if use_rust:
            # Set SKRUB_RUST_DEBUG_TIMING to 1 if debug_timing is true
            os.environ["SKRUB_RUST_DEBUG_TIMING"] = "1" if bool(rc.get("debug_timing", False)) else "0"
            # Set SKRUB_RUST_THREADS from num_threads
            os.environ["SKRUB_RUST_THREADS"] = str(int(rc.get("num_threads", 0)))
        return use_rust

    if name == "NUM_THREADS":   #number of threads for all rust OPs. 0 -> global threadpool.
        return os.getenv("SKRUB_RUST_THREADS", "0") == "1" or int(rc.get("num_threads", 0))

    if name == "DEBUG_TIMING":  #print debug timing
        return os.getenv("SKRUB_RUST_DEBUG_TIMING", "0") == "1" or bool(rc.get("debug_timing", False))

    if name == "ALLOW_PATCH":   #kill-switch for all non-sklearn backends. This ignores feature flags.
        return os.getenv("SKRUB_RUST_ALLOW_MONKEYPATCH", "0") == "1" or bool(rc.get("allow_monkeypatch", False))
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
    rc = get_rust_config()
    if os.getenv("SKRUB_RUST_DEBUG_TIMING", "0") == "1" or bool(rc.get("debug_timing", True)):
        return time.perf_counter()
    else:
        return None

def print_timing(msg, start_time):
    rc = get_rust_config()
    if os.getenv("SKRUB_RUST_DEBUG_TIMING", "0") == "1" or bool(rc.get("debug_timing", True)):
        end_time = time.perf_counter()
        print(f"[python] {msg}: {(end_time - start_time):8.3f}s")


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
hashing_tfidf_csr = getattr(native, "hashing_tfidf_csr", None)
fd_embedding = getattr(native, "fd_embed_from_csr", None)