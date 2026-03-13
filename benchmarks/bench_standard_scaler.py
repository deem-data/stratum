import argparse
import time
from contextlib import nullcontext
from itertools import product

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler as SKStandardScaler

from stratum.adapters.standard_scaler import NumpyStandardScaler, RustyStandardScaler
from stratum._config import config


def _run_single_case(n_rows: int, n_cols: int, seed: int = 0, rep: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(n_rows, n_cols), dtype=np.float32)

    cases = (
        ("sklearn", SKStandardScaler(copy=True), nullcontext()),
        ("rust", RustyStandardScaler(copy=True), config(rust_backend=True, allow_patch=True, debug_timing=False)),
        ("numpy_copy_true", NumpyStandardScaler(copy=True), nullcontext()),
        ("numpy_copy_false", NumpyStandardScaler(copy=False), nullcontext()),
    )

    rows: list[dict] = []
    for backend, scaler, ctx in cases:
        with ctx:
            t0 = time.perf_counter()
            scaler.fit(X)
            t1 = time.perf_counter()
            fit_time = t1 - t0

            t0 = time.perf_counter()
            scaler.transform(X)
            t1 = time.perf_counter()
            transform_time = t1 - t0

        rows.append(
            {
                "backend": backend,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "fit_time_s": fit_time,
                "transform_time_s": transform_time,
                "rep": rep,
            }
        )
    return rows


def run_benchmark(
    n_rows_list: list[int],
    n_cols_list: list[int],
    *,
    reps: int = 3,
    seed: int = 0,
) -> pl.DataFrame:
    """Run the benchmark for all (n_rows, n_cols) combinations and return an averaged Polars DataFrame.

    Each (backend, n_rows, n_cols) combination is repeated `reps` times and
    timings are averaged over repetitions.
    """

    # init thread pool for rust
    _run_single_case(100, 10, seed=0, rep=1)

    all_rows: list[dict] = []
    for n_rows, n_cols in product(n_rows_list, n_cols_list):
        for rep in range(reps):
            # Change seed per repetition to avoid reusing the exact same data
            all_rows.extend(_run_single_case(n_rows* 1000, n_cols, seed=seed + rep, rep=rep))

    df = pl.DataFrame(all_rows)
    return (
        df.group_by(["backend", "n_rows", "n_cols"])
        .agg(
            pl.col("fit_time_s").mean().alias("fit_time_s"),
            pl.col("transform_time_s").mean().alias("transform_time_s"),
        )
        .sort(["backend", "n_rows", "n_cols"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark StandardScaler implementations on random numpy arrays.\n"
            "Fit and transform times are measured separately for each combination\n"
            "of n_rows and n_cols."
        )
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        nargs="+",
        required=True,
        help="List of row counts to benchmark, in thousands, e.g. --n-rows 10 50",
    )
    parser.add_argument(
        "--n_cols",
        type=int,
        nargs="+",
        required=True,
        help="List of column counts to benchmark, e.g. --n-cols 10 100",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Number of repetitions per (n_rows, n_cols) combination to average over (default: 3)",
    )

    args = parser.parse_args()

    df = run_benchmark(args.n_rows, args.n_cols, reps=args.reps, seed=0)
    output_path = "standard_scaler_benchmark.csv"
    df.write_csv(output_path)
    print(f"Wrote benchmark results to {output_path}")
    print(df.show(limit=df.shape[0]))


if __name__ == "__main__":
    main()

