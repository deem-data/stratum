"""Measure retained native memory across repeated Rust StringEncoder fits.

The old integer-registry implementation retained one TF-IDF and FD model per
cycle, so resident memory grew approximately linearly after estimator deletion.
With Python-owned handles, post-GC resident memory should stabilize after
allocator warm-up.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import random
import string
import sys
import threading
import time
from ctypes import wintypes

import pandas as pd

from stratum.adapters.string_encoder import RustyStringEncoder


def make_strings(n_rows: int, length: int = 24) -> pd.Series:
    rng = random.Random(0)
    values = [
        "".join(rng.choices(string.ascii_lowercase, k=length))
        for _ in range(n_rows)
    ]
    return pd.Series(values, name="text")


if sys.platform == "win32":
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]


    _get_process_memory_info = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    _get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    _get_process_memory_info.restype = wintypes.BOOL
    _get_current_process = ctypes.WinDLL("kernel32").GetCurrentProcess
    _get_current_process.restype = wintypes.HANDLE
    _current_process = _get_current_process()
else:
    import psutil

    _process = psutil.Process()


def rss_mib() -> float:
    if sys.platform == "win32":
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not _get_process_memory_info(
            _current_process, ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError()
        return counters.WorkingSetSize / (1024 * 1024)

    return _process.memory_info().rss / (1024 * 1024)


def sample_peak(stop: threading.Event, samples: list[float]) -> None:
    while not stop.wait(0.005):
        samples.append(rss_mib())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--rows", type=int, default=3_000)
    args = parser.parse_args()

    values = make_strings(args.rows)
    gc.collect()
    baseline = rss_mib()
    post_gc = []

    print(f"baseline_rss_mib={baseline:.1f}")
    for cycle in range(args.cycles):
        samples = [rss_mib()]
        stop = threading.Event()
        sampler = threading.Thread(
            target=sample_peak, args=(stop, samples), daemon=True
        )
        sampler.start()
        encoder = RustyStringEncoder(
            n_components=10,
            random_state=0,
        )
        encoder._stratum_force_rust = True
        output = encoder.fit_transform(values)
        fitted_rss = rss_mib()
        stop.set()
        sampler.join()
        sampled_peak = max(samples)
        del output, encoder
        gc.collect()
        # Give the native allocator a chance to return released pages before
        # recording retained memory.
        time.sleep(0.05)
        collected_rss = rss_mib()
        post_gc.append(collected_rss)
        print(
            f"cycle={cycle + 1} "
            f"sampled_peak_rss_mib={sampled_peak:.1f} "
            f"fitted_rss_mib={fitted_rss:.1f} "
            f"post_gc_rss_mib={collected_rss:.1f}"
        )

    retained_growth = post_gc[-1] - post_gc[0] if post_gc else 0.0
    peak_observed = max(post_gc, default=baseline)
    print(f"post_warmup_retained_growth_mib={retained_growth:.1f}")
    print(f"peak_post_gc_rss_mib={peak_observed:.1f}")


if __name__ == "__main__":
    main()
