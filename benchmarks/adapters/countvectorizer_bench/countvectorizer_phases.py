import argparse
import os
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
import pandas as pd
from stratum import set_config
from stratum.adapters.count_vectorizer import RustyCountVectorizer

from countvectorizer_benchmarkutils import warmup, get_synthetic_data

N_REPS = 5
N_UNIQUE_WORDS = 10_000
DATASET_LENGTH_LIST = [10_000, 1_000_000]
DEFAULT_OUT = Path(__file__).parent / "results" / "macrobenchmark_countvectorizer_phases.csv"


@contextmanager
def capture_rust_stderr():
    """Redirect the OS-level stderr fd so Rust's `eprintln!` output (which
    bypasses Python's sys.stderr) can be captured and parsed."""
    saved_fd = os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), 2)
    box = {"text": ""}
    try:
        yield box
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        tmp.seek(0)
        box["text"] = tmp.read().decode("utf-8", errors="replace")
        tmp.close()


def parse_ms(text, label):
    pattern = re.escape(f"[rust] {label}: ") + r"(\d+)ms"
    return [int(m) for m in re.findall(pattern, text)]


def median(values):
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 0:
        return (values[mid - 1] + values[mid]) / 2
    return values[mid]


def benchmark_phases(data, n_jobs):
    set_config(rust_backend=True, allow_patch=True, debug_timing=True, num_threads=n_jobs)
    cv = RustyCountVectorizer(n_jobs=n_jobs)

    with capture_rust_stderr() as box:
        t0 = time.perf_counter()
        for _ in range(N_REPS):
            cv.fit_transform(data)
        outer_ms = (time.perf_counter() - t0) * 1000 / N_REPS
    text = box["text"]

    total_samples = parse_ms(text, "count_vectorize_fit_transform total")
    if not total_samples:
        raise RuntimeError(
            "No '[rust] count_vectorize_fit_transform total' timing found -- "
            "the Rust fastpath did not run (check supports_rust_count_vectorizer "
            "eligibility and rust_backend/allow_patch config)."
        )

    map_samples = parse_ms(text, "count_vectorize_fit map")
    reduce_samples = parse_ms(text, "count_vectorize_fit reduce")
    sorting_samples = parse_ms(text, "count_vectorize_fit sorting")

    map_ms = median(map_samples)
    reduce_ms = median(reduce_samples) + median(sorting_samples)
    rust_total_ms = median(total_samples)
    ffi_ms = outer_ms - rust_total_ms

    return ffi_ms, map_ms, reduce_ms


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
            f.write("n_jobs,dataset_length,ffi_ms,map_ms,reduce_ms\n")

        for dataset_length in DATASET_LENGTH_LIST:
            data = get_synthetic_data_f64(
                n_unique_words=N_UNIQUE_WORDS,
                max_length=10,
                n_sentences=dataset_length,
                max_sentence_len=100,
            )
            data = pd.Series(data)

            ffi_ms, map_ms, reduce_ms = benchmark_phases(data, args.n_jobs)
            f.write(
                f"{args.n_jobs},{dataset_length},{ffi_ms:.9f},{map_ms:.9f},{reduce_ms:.9f}\n"
            )
            f.flush()


if __name__ == "__main__":
    main()
