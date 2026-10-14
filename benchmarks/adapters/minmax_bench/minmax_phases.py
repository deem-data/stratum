import argparse
import os
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from stratum import MinMaxScaler as RustyMinMaxScaler, set_config

from minmax_benchmarkutils import warmup, get_synthetic_data

N_REPS = 5
N_COLS = 50
N_ROWS_LIST = [100_0000, 10_000_000]
DEFAULT_OUT = Path(__file__).parent / "results" / "minmax_phases.csv"


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
    scaler = RustyMinMaxScaler(n_jobs=n_jobs)

    with capture_rust_stderr() as box:
        t0 = time.perf_counter()
        for _ in range(N_REPS):
            scaler.fit(data)
        fit_outer_ms = (time.perf_counter() - t0) * 1000 / N_REPS

        t0 = time.perf_counter()
        for _ in range(N_REPS):
            scaler.transform(data)
        transform_outer_ms = (time.perf_counter() - t0) * 1000 / N_REPS
    text = box["text"]

    fit_samples = parse_ms(text, "minmax scale fit")
    transform_samples = parse_ms(text, "minmax scale transform")
    if len(fit_samples) != N_REPS or len(transform_samples) != N_REPS:
        raise RuntimeError(
            f"Expected {N_REPS} '[rust] minmax scale fit/transform' timing "
            f"samples each, got {len(fit_samples)}/{len(transform_samples)} -- "
            "either the Rust fastpath didn't run at all (check "
            "_supported_params eligibility and rust_backend/allow_patch "
            "config), or it silently fell back to sklearn on a subset of "
            "reps (see RustyMinMaxScaler.transform's try/except)."
        )

    fit_ms = median(fit_samples)
    transform_ms = median(transform_samples)
    outer_ms = fit_outer_ms + transform_outer_ms
    ffi_ms = outer_ms - fit_ms - transform_ms

    return ffi_ms, fit_ms, transform_ms


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
            f.write("n_jobs,n_rows,n_cols,ffi_ms,fit_ms,transform_ms\n")

        for n_rows in N_ROWS_LIST:
            data = get_synthetic_data(n_rows=n_rows, n_cols=N_COLS)

            ffi_ms, fit_ms, transform_ms = benchmark_phases(data, args.n_jobs)
            f.write(
                f"{args.n_jobs},{n_rows},{N_COLS},{ffi_ms:.9f},{fit_ms:.9f},{transform_ms:.9f}\n"
            )
            f.flush()


if __name__ == "__main__":
    main()
