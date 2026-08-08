"""Shared helpers for the MinMaxScaler macrobenchmark."""
import gc

import numpy as np
from sklearn.preprocessing import MinMaxScaler as SKMinMaxScaler
from stratum import MinMaxScaler as RustyMinMaxScaler, set_config

SEED = 67


def get_synthetic_data(n_rows, n_cols, seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(n_rows, n_cols), dtype=np.float32)


def get_synthetic_data_64(n_rows, n_cols, seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(n_rows, n_cols), dtype=np.float64)


def warmup(n_jobs: int) -> None:
    """Touch both backends once so import/JIT overhead doesn't skew timings.

    Also locks in the Rust thread pool at `n_jobs` threads: it is built once
    per process (see _rust/src/threads.rs) from whatever SKRUB_RUST_THREADS
    is set to on first use, so this must run before any timed rust call.
    """
    data = get_synthetic_data(2048, 4)

    set_config(rust_backend=False)
    SKMinMaxScaler().fit_transform(data)

    set_config(rust_backend=True, num_threads=n_jobs)
    RustyMinMaxScaler(n_jobs=n_jobs).fit_transform(data)

    gc.collect()
