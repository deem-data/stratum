"""Compare CountVectorizer fit/transform latency with exact output checks.

Run from the repository root:
    .venv/bin/python benchmarks/adapters/bench_count_vectorizer.py --rows 25000

The native hot path remains linear in corpus size plus vocabulary sorting.
Counts use int64 (sklearn's default); parallel chunks retain temporary CSR buffers.
Unicode/unsupported inputs deliberately retain sklearn execution cost.
"""
import argparse
from time import perf_counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import stratum as st
from stratum import _rust_backend as rb
from stratum.adapters.count_vectorizer import RustyCountVectorizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=25000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--transform-rows", type=int, default=None)
    args = parser.parse_args()
    if rb.count_vectorize_fit_transform is None:
        parser.error("build the Rust extension before benchmarking")

    rng = np.random.default_rng(42)
    words = np.array([f"token_{index}" for index in range(2000)])
    documents = [" ".join(words[row]) for row in rng.integers(0, len(words), (args.rows, 40))]
    reference = CountVectorizer()
    expected = reference.fit_transform(documents)
    transform_documents = documents[:args.transform_rows]
    expected_transform = reference.transform(transform_documents)

    with st.config(allow_patch=True, rust_backend=True, num_threads=args.workers):
        for name, factory in (("sklearn", CountVectorizer), ("rust", RustyCountVectorizer)):
            timings = {"fit_transform": [], "transform": []}
            # Warm pools and allocations before reporting steady-state latency.
            factory().fit_transform(documents)
            for _ in range(args.repeats):
                estimator = factory()
                start = perf_counter()
                actual = estimator.fit_transform(documents)
                timings["fit_transform"].append(perf_counter() - start)
                assert actual.dtype == expected.dtype
                assert (actual != expected).nnz == 0
                start = perf_counter()
                transformed = estimator.transform(transform_documents)
                timings["transform"].append(perf_counter() - start)
                assert (transformed != expected_transform).nnz == 0
                # Keep explicit sklearn dictionary export outside timed regions.
                assert estimator.vocabulary_ == reference.vocabulary_
            print(name, " ".join(f"{stage}={min(times):.4f}s" for stage, times in timings.items()))


if __name__ == "__main__":
    main()
