"""OneHotEncoder fit/transform latency, including sklearn category lookup.

Run: python benchmarks/adapters/bench_onehot_encoder.py --rows 100000

Covers realistic dense categoricals and high-cardinality sparse output,
both drop options and both precisions, with exact parity checks outside timing.
Emission remains O(rows * features + nnz); dense output needs O(rows * width)
memory. Float64 is allocated directly, avoiding a temporary float32 dense matrix.
"""
import argparse
import gc
import warnings
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

import stratum as st
from stratum import _rust_backend as rb
from stratum.adapters.one_hot_encoder import RustyOneHotEncoder


def check(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    if sparse.issparse(expected):
        for name in ("data", "indices", "indptr"):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    else:
        np.testing.assert_array_equal(actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not rb.HAVE_RUST:
        parser.error("build the Rust extension before benchmarking")
    if min(args.rows, args.workers, args.repeats) < 1:
        parser.error("rows, workers and repeats must be positive")
    # Unknown-category behavior is checked below; do not time warning output.
    warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)
    rng = np.random.default_rng(42)
    for sparse_output, cardinality in ((False, 12), (True, 2000)):
        X = pd.DataFrame({
            "deck": rng.choice(["A", "B"], args.rows),
            "planet": rng.choice(["Earth", "Europa", "Mars"], args.rows),
            "category": np.array([f"c{i}" for i in range(cardinality)])[rng.integers(cardinality, size=args.rows)],
        })
        test = X.iloc[:min(10000, args.rows)].copy()
        test.iloc[::10, 2] = "unseen"
        for drop in (None, "if_binary"):
            for dtype in (np.float32, np.float64):
                params = dict(drop=drop, dtype=dtype, handle_unknown="ignore", sparse_output=sparse_output)
                reference = OneHotEncoder(**params)
                expected = reference.fit_transform(X)
                expected_test = reference.transform(test)
                for name, factory in (("sklearn", OneHotEncoder), ("rust", RustyOneHotEncoder)):
                    times = {"fit_transform": [], "transform": []}
                    with st.config(rust_backend=False, allow_patch=False, num_threads=args.workers):
                        # A forced binding measures the same adapter used by greedy.
                        warm = factory(**params)
                        if name == "rust":
                            warm._stratum_force_rust = True
                        warm.fit_transform(X.iloc[:min(2048, args.rows)])
                        for _ in range(args.repeats):
                            gc.collect()
                            estimator = factory(**params)
                            if name == "rust":
                                estimator._stratum_force_rust = True
                            start = perf_counter()
                            actual = estimator.fit_transform(X)
                            times["fit_transform"].append(perf_counter() - start)
                            check(actual, expected)
                            del actual
                            start = perf_counter()
                            actual = estimator.transform(test)
                            times["transform"].append(perf_counter() - start)
                            check(actual, expected_test)
                            del actual
                    print(f"{name} sparse={sparse_output} drop={drop} dtype={np.dtype(dtype).name}",
                          " ".join(f"{stage}={min(values):.4f}s" for stage, values in times.items()))
                del expected, expected_test


if __name__ == "__main__":
    main()
