"""
Benchmarks stratum's Rust-backed MinMaxScaler against scikit-learn's, for a
single thread count, and appends the results to a CSV.

The Rust thread pool is built once per process (see _rust/src/threads.rs),
so sweeping across thread counts requires re-invoking this script once per
--n-jobs value in a fresh process -- see run_macrobenchmark_minmax.sh.
"""

import argparse
import time
from pathlib import Path
from statistics import median

from sklearn.preprocessing import MinMaxScaler as SKMinMaxScaler
from stratum import MinMaxScaler as RustyMinMaxScaler, set_config

from minmax_benchmarkutils import warmup, get_synthetic_data

N_REPS = 5
N_COLS_LIST = [10, 50, 100]
N_ROWS_LIST = [100_000, 1_000_000, 10_000_000]
DEFAULT_OUT = Path(__file__).parent / "results" / "macrobenchmark_minmax.csv"


def benchmark_rust(data, n_jobs):
    set_config(rust_backend=True, num_threads=n_jobs)
    scaler = RustyMinMaxScaler(n_jobs=n_jobs)

    fit_times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        scaler.fit(data)
        fit_times.append((time.perf_counter() - t0) * 1000)
    fit_ms = median(fit_times)

    transform_times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        scaler.transform(data)
        transform_times.append((time.perf_counter() - t0) * 1000)
    transform_ms = median(transform_times)

    return fit_ms, transform_ms


def benchmark_sklearn(data):
    set_config(rust_backend=False)
    scaler = SKMinMaxScaler()

    fit_times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        scaler.fit(data)
        fit_times.append((time.perf_counter() - t0) * 1000)
    fit_ms = median(fit_times)

    transform_times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        scaler.transform(data)
        transform_times.append((time.perf_counter() - t0) * 1000)
    transform_ms = median(transform_times)

    return fit_ms, transform_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-jobs",
        type=int,
        required=True,
        help="Thread count for this run. The Rust thread pool is fixed for "
        "the process lifetime, so run this script once per n_jobs value.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    warmup(args.n_jobs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out.exists()

    with open(args.out, "a") as f:
        if write_header:
            f.write("n_jobs,n_cols,n_rows,version,fit_ms,transform_ms\n")

        for n_cols in N_COLS_LIST:
            for n_rows in N_ROWS_LIST:
                data = get_synthetic_data(n_rows=n_rows, n_cols=n_cols)

                # sklearn's timing doesn't depend on n_jobs; only run it once.
                if args.n_jobs == 1:
                    fit_ms, transform_ms = benchmark_sklearn(data)
                    f.write(
                        f"{args.n_jobs},{n_cols},{n_rows},sklearn,{fit_ms:.9f},{transform_ms:.9f}\n"
                    )
                    f.flush()

                fit_ms, transform_ms = benchmark_rust(data, args.n_jobs)
                f.write(
                    f"{args.n_jobs},{n_cols},{n_rows},rust,{fit_ms:.9f},{transform_ms:.9f}\n"
                )
                f.flush()


if __name__ == "__main__":
    main()
