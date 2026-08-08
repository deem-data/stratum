"""
Benchmarks stratum's Rust-backed CountVectorizer against scikit-learn's, for a
single thread count, and appends the results to a CSV.

The Rust thread pool is built once per process (see _rust/src/threads.rs),
so sweeping across thread counts requires re-invoking this script once per
--n-jobs value in a fresh process -- see run_macrobenchmark_countvectorizer.sh.
"""

import argparse
import time
from pathlib import Path
from statistics import median

from sklearn.feature_extraction.text import CountVectorizer as SKCountVectorizer
from stratum import set_config
from stratum.adapters.count_vectorizer import RustyCountVectorizer

from countvectorizer_benchmarkutils import warmup, get_synthetic_data

N_REPS = 5
N_UNIQUE_WORDS_LIST = [1_000, 10_000, 50_000]
DATASET_LENGTH_LIST = [10_000, 100_000, 1_000_000]
DEFAULT_OUT = Path(__file__).parent / "results" / "macrobenchmark_countvectorizer.csv"


def benchmark_rust(data, n_jobs):
    set_config(rust_backend=True, num_threads=n_jobs)
    cv = RustyCountVectorizer(n_jobs=n_jobs)

    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        cv.fit_transform(data)
        times.append((time.perf_counter() - t0) * 1000)

    return median(times)


def benchmark_sklearn(data):
    set_config(rust_backend=False)
    cv = SKCountVectorizer()

    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        cv.fit_transform(data)
        times.append((time.perf_counter() - t0) * 1000)

    return median(times)


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
            f.write("n_jobs,n_unique_words,dataset_length,version,time_ms\n")

        for n_unique_words in N_UNIQUE_WORDS_LIST:
            for dataset_length in DATASET_LENGTH_LIST:
                data = get_synthetic_data(
                    n_unique_words=n_unique_words,
                    max_length=10,
                    n_sentences=dataset_length,
                    max_sentence_len=100,
                )

                # sklearn's timing doesn't depend on n_jobs; only run it once.
                if args.n_jobs == 1:
                    ms = benchmark_sklearn(data)
                    f.write(
                        f"{args.n_jobs},{n_unique_words},{dataset_length},sklearn,{ms}\n"
                    )
                    f.flush()

                ms = benchmark_rust(data, args.n_jobs)
                f.write(
                    f"{args.n_jobs},{n_unique_words},{dataset_length},rust,{ms:.9f}\n"
                )
                f.flush()


if __name__ == "__main__":
    main()
