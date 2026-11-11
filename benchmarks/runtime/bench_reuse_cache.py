"""Reuse-cache overhead benchmark (no extra benchmark dependencies).

Run from the repository root::

    PYTHONPATH=. .venv/bin/python benchmarks/runtime/bench_reuse_cache.py --output /tmp/reuse.json
    PYTHONPATH=. .venv/bin/python benchmarks/runtime/bench_reuse_cache.py --baseline /tmp/reuse.json

- JSON retains all samples and environment details. ``--max-regression-percent 20``
  fails comparisons above that budget.
- CSV pipeline: cold clears the reuse cache and recomputes; warm verifies a cached
  result hit. Both include planning, scheduler labels and result copying.
"""
from __future__ import annotations

import argparse
import gc
from itertools import cycle
import json
import logging
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
from tempfile import TemporaryDirectory
from timeit import Timer
import tracemalloc

import numpy as np
import pandas as pd

import stratum as st
from stratum.frontend import _api
from stratum.runtime import _reuse_cache


def measure(call, repeats, seconds):
    # Warm up and calibrate outside the retained samples. Keep cyclic GC enabled
    # because repeated pipeline evaluation creates graphs with reference cycles.
    timer = Timer(call, setup=gc.enable)
    number = 1
    while timer.timeit(number) < seconds:
        number *= 2
    samples = []
    for _ in range(repeats):
        gc.collect()
        samples.append(timer.timeit(number) * 1e9 / number)
    return {"value": median(samples), "unit": "ns/call", "samples": samples,
            "iterations": number}


def cache_cases(payload):
    for count in (128, 10_000):
        cache = _reuse_cache.ReuseCache()
        keys = [f"{i:032x}" for i in range(count)]
        for key in keys:
            cache.put(key, payload)
        rotating = cycle(keys)
        yield f"hit/{count}", lambda: cache.get(next(rotating))
        if hasattr(cache, "snapshot"):
            yield f"snapshot/summary/{count}", cache.snapshot
            yield f"snapshot/entries/{count}", lambda: cache.snapshot(include_entries=True)
        if hasattr(cache, "listeners"):
            cache.listeners.append(lambda event: None)
            yield f"hit/noop_listener/{count}", lambda: cache.get(next(rotating))

    cache = _reuse_cache.ReuseCache(memory_budget=128 * payload.nbytes)
    keys = [f"{i:032x}" for i in range(129)]
    for key in keys[:128]:
        cache.put(key, payload)
    yield "miss", lambda: cache.get("missing")
    yield "put/duplicate", lambda: cache.put(keys[0], payload)
    rotating = cycle(keys[128:] + keys[:128])
    yield "put/evict", lambda: cache.put(next(rotating), payload)
    cache = _reuse_cache.ReuseCache(memory_budget=1)
    yield "put/oversized", lambda: cache.put("oversized", payload)


def bookkeeping_bytes(payload, count=10_000):
    # Keys and cached data are allocated before tracing, so neither is counted.
    keys = [f"{i:032x}" for i in range(count)]
    gc.collect()
    tracemalloc.start()
    try:
        cache = _reuse_cache.ReuseCache()
        for key in keys:
            cache.put(key, payload)
        retained = tracemalloc.get_traced_memory()[0]
        assert len(cache) == count
        return {"value": retained / count, "unit": "bytes/entry"}
    finally:
        tracemalloc.stop()


def pipeline_cases(directory):
    path = directory / "input.csv"
    pd.DataFrame({"a": np.arange(128), "b": np.arange(128)}).to_csv(path, index=False)
    data = st.deferred(pd.read_csv)(str(path))
    dag = data.assign(c=data["a"] * 2 + data["b"])
    previous = _reuse_cache._active
    cache = _reuse_cache._active = _reuse_cache.ReuseCache()
    try:
        with st.config(reuse_cache=True, stats=False, rust_backend=False,
                       implementation_selector="default", explain=(), DEBUG=False), st.dev_config(
                           diagnostics_dir=None, hardware_utilization=False):
            def cold():
                cache.clear()
                return _api.evaluate(dag)

            np.testing.assert_array_equal(cold()["c"], np.arange(128) * 3)
            assert cache.stats.puts > 0
            yield "pipeline/cold", cold
            before = cache.stats.hits
            _api.evaluate(dag)
            assert cache.stats.hits == before + 1, "warm pipeline must reuse its root"
            yield "pipeline/warm", lambda: _api.evaluate(dag)
            with st.dev_config(diagnostics_dir=directory / "reports"):
                yield "pipeline/warm_export", lambda: _api.evaluate(dag)
                # A warm plan has one surviving node, even when the shared cache
                # contains thousands of unrelated results from earlier runs.
                payload = np.zeros(128, dtype=np.float64)
                for i in range(10_000):
                    cache.put(f"unrelated-{i:032x}", payload)
                yield "pipeline/warm_export/10000", lambda: _api.evaluate(dag)
    finally:
        _reuse_cache._active = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-regression-percent", type=float)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seconds", type=float, default=0.03, help="minimum seconds per timing sample")
    args = parser.parse_args()
    if args.repeats < 1 or not 0 < args.seconds < float("inf"):
        parser.error("repeats and seconds must be positive and finite")
    if args.max_regression_percent is not None and (
            args.baseline is None or not 0 <= args.max_regression_percent < float("inf")):
        parser.error("a regression budget requires --baseline and a finite nonnegative percentage")
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    logging.getLogger("stratum").setLevel(logging.ERROR)
    source = Path(st.__file__).resolve().parent.parent
    revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
    result = {
        "schema_version": 1,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "source": str(source), "revision": revision.stdout.strip(),
                        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                        "numpy": np.__version__, "pandas": pd.__version__, "skrub": st.__skrub_version__},
        "repeats": args.repeats, "seconds": args.seconds, "metrics": {},
    }
    print(f"Source: {source}; Python {platform.python_version()}", flush=True)
    failures = []

    def record(name, metric):
        result["metrics"][name] = metric
        comparison = ""
        old = baseline["metrics"].get(name) if baseline else None
        if old is not None:
            if old["unit"] != metric["unit"]:
                raise ValueError(f"incompatible baseline unit for {name}")
            percent = (metric["value"] / old["value"] - 1) * 100
            comparison = f"  ({percent:+.1f}%)"
            if args.max_regression_percent is not None and percent > args.max_regression_percent:
                failures.append(name)
        print(f"{name:34s} {metric['value']:12.1f} {metric['unit']}{comparison}", flush=True)

    payload = np.zeros(128, dtype=np.float64)
    # Consume each generator immediately: its cache/configuration changes on resume.
    for name, call in cache_cases(payload):
        record(name, measure(call, args.repeats, args.seconds))
    record("bookkeeping/10000", bookkeeping_bytes(payload))
    with TemporaryDirectory(prefix="stratum-reuse-bench-") as directory:
        for name, call in pipeline_cases(Path(directory)):
            record(name, measure(call, args.repeats, args.seconds))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if baseline:
        missing = baseline["metrics"].keys() - result["metrics"].keys()
        if missing:
            parser.error(f"baseline metrics missing from this run: {sorted(missing)}")
    if failures:
        print("Regression budget exceeded: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
