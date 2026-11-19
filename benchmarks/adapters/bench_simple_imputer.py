"""SimpleImputer end-to-end adapter latency with output/state parity checks.

Run:
  python benchmarks/adapters/bench_simple_imputer.py --rows 100000 --cols 64 --workers 4

Includes validation and Python dispatch, not just kernel time. Fused transform
uses one output allocation, avoiding sklearn's full missing mask, repeated fill
values, coordinate arrays, and indicator concatenation. Median uses selection
(expected O(rows) per column); mode sorts (O(rows log rows)); mean/constant are
linear. Fit workspace is O(rows * active workers), not a full masked matrix.
"""
import argparse
import gc
from statistics import median
from time import perf_counter
import warnings

import numpy as np
from sklearn.impute import SimpleImputer

from stratum import _rust_backend as rb
from stratum.adapters.simple_imputer import RustySimpleImputer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--strategies", nargs="+", default=["mean", "median", "most_frequent", "constant"])
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--distribution", choices=["normal", "counts"], default="normal")
    parser.add_argument("--missing-rate", type=float, default=0.2)
    parser.add_argument("--keep-empty", action="store_true")
    args = parser.parse_args()
    if not rb.simple_imputer_fit:
        parser.error("build the Rust extension before benchmarking")
    if min(args.rows, args.cols, args.workers, args.repeats) < 1 or not 0 <= args.missing_rate <= 1:
        parser.error("positive sizes/workers/repeats and missing rate in [0, 1] required")
    warnings.filterwarnings("ignore", message="Skipping features without any observed values")
    rng = np.random.default_rng(42)
    X = (rng.integers(-20, 21, size=(args.rows, args.cols)) if args.distribution == "counts"
         else rng.normal(size=(args.rows, args.cols))).astype(args.dtype)
    X[rng.random(X.shape) < args.missing_rate] = np.nan
    X[:, -1] = np.nan
    test = X.copy()
    test[::3, 0] = np.nan
    print(f"rows={args.rows} cols={args.cols} dtype={args.dtype} workers={args.workers} "
          f"distribution={args.distribution} missing={args.missing_rate} "
          f"keep_empty={args.keep_empty} repeats={args.repeats}")
    for strategy in args.strategies:
        for indicator in (False, True):
            params = dict(strategy=strategy, add_indicator=indicator, keep_empty_features=args.keep_empty)
            if strategy == "constant":
                params["fill_value"] = -999
            reference = SimpleImputer(**params).fit(X)
            expected = reference.transform(test)
            times = {}
            for backend, factory in (("sklearn", SimpleImputer), ("rust", RustySimpleImputer)):
                stage = {"fit_transform": [], "transform": []}
                for repeat in range(args.repeats + 1):
                    gc.collect()
                    estimator = factory(**params)
                    if backend == "rust":
                        estimator._bind_native_worker_budget(args.workers)
                        estimator._stratum_force_rust = True
                    start = perf_counter()
                    output = estimator.fit_transform(X)
                    fit_transform_s = perf_counter() - start
                    del output
                    start = perf_counter()
                    output = estimator.transform(test)
                    transform_s = perf_counter() - start
                    assert output.dtype == expected.dtype
                    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-7)
                    np.testing.assert_allclose(
                        np.asarray(estimator.statistics_, float),
                        np.asarray(reference.statistics_, float), rtol=1e-6, equal_nan=True,
                    )
                    del output
                    if repeat:  # First full run warms pools/allocations.
                        stage["fit_transform"].append(fit_transform_s)
                        stage["transform"].append(transform_s)
                times[backend] = {key: median(value) for key, value in stage.items()}
            print(f"{strategy} indicator={indicator}", " ".join(
                f"{key}: sklearn={times['sklearn'][key]:.4f}s rust={times['rust'][key]:.4f}s "
                f"speedup={times['sklearn'][key] / times['rust'][key]:.2f}x"
                for key in ("fit_transform", "transform")
            ))


if __name__ == "__main__":
    main()
