"""
Benchmark: Rust normalization kernels vs. sklearn equivalents.

Metrics: sklearn time, rust time, speedup, and a correctness check comparing the
Rust output against sklearn's element-wise.

Kernels compared
----------------
  Normalizer(norm='l2')    sklearn.preprocessing.Normalizer(norm='l2')
  Normalizer(norm='l1')    sklearn.preprocessing.Normalizer(norm='l1')
  Normalizer(norm='max')   sklearn.preprocessing.Normalizer(norm='max')
  (each also has an in-place variant, selected with copy=False)

Both sides run the same stratum.Normalizer estimator. The backend is switched
with skrub.set_config(rust_backend=...), so the sklearn column measures the
fallback path the adapter itself would take. The Rust thread count is frozen at
the first kernel call, so --threads re-runs this script once per thread count
with SKRUB_RUST_THREADS set, the same way bench_normalization_charts.py does.

Size tiers
----------
  --small    10k-1M rows        (the default when no tier is given)
  --large    1M rows, wider
  --xlarge   2M-5M rows
  --xxlarge  10M rows (~2 GB f32)

Tiers combine, and naming any tier selects exactly the tiers named, so
--xxlarge runs the 10M rows alone. With no tier flag at all you get --small.
Note this differs from the older meaning of --large, which used to append the
large tier to small rather than replace it; pass --small --large for that.

All inputs are float32. Results are printed as a table.

Correctness
-----------
Each kernel's Rust output is compared to sklearn's with
np.allclose(rtol=1e-3, atol=1e-4); the table shows PASS/FAIL and the max
absolute error. Correctness is size independent, so inputs above
VERIFY_MAX_MB are reported as skipped rather than verified twice over.
Pass --no-verify to skip the check entirely.

Usage
-----
    cd <repo_root>
    uv run python benchmarks/adapters/bench_normalization.py

    # Larger sizes:
    uv run python benchmarks/adapters/bench_normalization.py --large
    uv run python benchmarks/adapters/bench_normalization.py --xlarge --xxlarge

    # One norm only, copy variants only:
    uv run python benchmarks/adapters/bench_normalization.py --l2 --no-inplace

    # Thread sweep:
    uv run python benchmarks/adapters/bench_normalization.py --threads 1,4,8

    # Skip the correctness check:
    uv run python benchmarks/adapters/bench_normalization.py --no-verify

    # Show the per-call kernel timings from the backend:
    uv run python benchmarks/adapters/bench_normalization.py --debug-timing
"""
import os
os.environ["SKRUB_RUST"] = "1"

import argparse
import gc
import subprocess
import sys
import time

import numpy as np

import stratum as skrub
from stratum import Normalizer
from stratum import _rust_backend as rb

# ── guard ────────────────────────────────────────────────────────────────────
if not rb.HAVE_RUST:
    sys.exit(
        "Rust backend not built. Run:\n"
        "  cd _rust && maturin develop --release"
    )

# ── size tiers (mirrors bench_normalization_charts.py) ────────────────────────
SIZE_SMALL   = [(10_000, 100), (100_000, 100), (100_000, 1_000), (1_000_000, 50)]
SIZE_LARGE   = [(1_000_000, 100)]
SIZE_XLARGE  = [(2_000_000, 100), (5_000_000, 50)]
# 1e6x500 is 2 GB, the same footprint as 1e7x50, so it belongs here rather than
# in --large: the tiers are named for row count but what costs time is bytes.
SIZE_XXLARGE = [(1_000_000, 500), (10_000_000, 50)]

_n_cpu = os.cpu_count() or 1
ALL_THREADS = sorted(set(t for t in [1, 2, 4, 8, _n_cpu] if t <= _n_cpu))

# ── correctness tolerances ────────────────────────────────────────────────────
VERIFY_RTOL = 1e-3
VERIFY_ATOL = 1e-4
# Verify only on inputs at or below this size (MB) — correctness is size
# independent, so there is no need to pay for the huge tiers.
VERIFY_MAX_MB = 200

# ── backend selection ─────────────────────────────────────────────────────────

DEBUG_TIMING = False

def use_backend(rust: bool):
    """Point stratum at the Rust kernels or back at sklearn.

    stratum.Normalizer reads the config on every call, so the same fitted
    estimator serves both columns of the table.

    Note that the thread count is deliberately NOT set here. The Rust side
    caches it in a Lazy/OnceLock (util.rs NUM_THREADS, threads.rs POOL), so it
    is frozen at the first kernel call and set_config(num_threads=...) after
    that is a no-op. --threads therefore re-runs this script in a subprocess
    per thread count, the same way bench_normalization_charts.py does.
    """
    if rust:
        skrub.set_config(rust_backend=True, debug_timing=DEBUG_TIMING)
    else:
        skrub.set_config(rust_backend=False, debug_timing=DEBUG_TIMING)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_data(n_rows: int, n_cols: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Mix of positive and negative values; some near-zero rows
    data = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
    # Sprinkle all-zero rows (~1%) to stress edge-case handling
    zero_idx = rng.integers(0, n_rows, size=max(1, n_rows // 100))
    data[zero_idx] = 0.0
    return data


def timeit(fn, n_reps: int = 5) -> float:
    """Return median wall-clock time over n_reps repetitions (seconds)."""
    times = []
    for _ in range(n_reps):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def warmup(fn, n: int = 3):
    for _ in range(n):
        fn()
    gc.collect()


def timeit_inplace(make_buf, fn, n_reps: int = 5) -> float:
    """Like timeit, but rebuilds the (mutated) buffer before each rep so the
    copy cost stays outside the timed region."""
    for _ in range(3):
        fn(make_buf())
    gc.collect()
    times = []
    for _ in range(n_reps):
        buf = make_buf()
        gc.collect()
        t0 = time.perf_counter()
        fn(buf)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def verify_arrays(rust_out, sklearn_out) -> dict:
    """Compare a Rust kernel's output against sklearn's element-wise."""
    ru = np.asarray(rust_out, dtype=np.float64)
    sk = np.asarray(sklearn_out, dtype=np.float64)
    diff = np.abs(ru - sk)
    return {
        "passed":  bool(np.allclose(ru, sk, rtol=VERIFY_RTOL, atol=VERIFY_ATOL)),
        "max_abs": float(diff.max()),
        "max_rel": float((diff / (np.abs(sk) + 1e-6)).max()),
    }


# ── per-kernel benchmark ──────────────────────────────────────────────────────

def bench_normalize(data, norms, n_reps: int, verify: bool) -> dict:
    results = {}

    for norm in norms:
        enc = Normalizer(norm=norm, copy=True).fit(data)

        use_backend(False)
        warmup(lambda: enc.transform(data))
        sklearn_t = timeit(lambda: enc.transform(data), n_reps)

        use_backend(True)
        warmup(lambda: enc.transform(data))
        rust_t = timeit(lambda: enc.transform(data), n_reps)

        v = None
        if verify:
            use_backend(True)
            rust_out = enc.transform(data)
            use_backend(False)
            v = verify_arrays(rust_out, enc.transform(data))
        results[f"normalize_{norm}"] = (sklearn_t, rust_t, v)

    return results


def bench_normalize_inplace(data, norms, n_reps: int, verify: bool) -> dict:
    results = {}

    for norm in norms:
        # copy=False makes transform write through the buffer it is handed
        enc = Normalizer(norm=norm, copy=False).fit(data)

        use_backend(False)
        sklearn_t = timeit_inplace(lambda: data.copy(), enc.transform, n_reps)

        use_backend(True)
        rust_t = timeit_inplace(lambda: data.copy(), enc.transform, n_reps)

        v = None
        if verify:
            use_backend(True)
            buf = data.copy()
            enc.transform(buf)
            use_backend(False)
            ref = Normalizer(norm=norm, copy=True).fit(data).transform(data)
            v = verify_arrays(buf, ref)
        results[f"normalize_{norm}_inplace"] = (sklearn_t, rust_t, v)

    return results


# ── printing ──────────────────────────────────────────────────────────────────

COL_W = 22

def _ms(t: float) -> str:
    return f"{t * 1000:.1f} ms"


def print_header():
    header = (
        f"{'kernel':<{COL_W}}"
        f"{'sklearn':>{COL_W}}"
        f"{'rust':>{COL_W}}"
        f"{'speedup':>{COL_W}}"
        f"{'correctness':>{COL_W}}"
    )
    print(header)
    print("-" * len(header))


def print_row(name: str, sklearn_t: float, rust_t: float, verify_info: dict | None) -> bool:
    """Print a result row; returns True if a correctness check FAILED."""
    speedup = sklearn_t / rust_t if rust_t > 0 else float("inf")
    marker = " ✓" if speedup >= 1.0 else " ✗"

    failed = False
    if verify_info is None:
        corr = "— skipped"
    else:
        failed = not verify_info["passed"]
        tag = "PASS" if verify_info["passed"] else "FAIL"
        corr = f"{tag}  (max_abs={verify_info['max_abs']:.1e})"

    print(
        f"{name:<{COL_W}}"
        f"{_ms(sklearn_t):>{COL_W}}"
        f"{_ms(rust_t):>{COL_W}}"
        f"{speedup:>{COL_W-4}.2f}x{marker:>2}"
        f"{corr:>{COL_W}}"
    )
    return failed


# ── main ──────────────────────────────────────────────────────────────────────

def run_suite(sizes, norms, n_reps: int, verify: bool, inplace: bool) -> bool:
    """Run every kernel across every size.

    Returns True if any correctness check failed."""
    any_fail = False
    for n_rows, n_cols in sizes:
        mb = n_rows * n_cols * 4 / 1e6
        # Correctness does not depend on input size, so skip it on the huge tiers
        size_verify = verify and mb <= VERIFY_MAX_MB

        data = make_data(n_rows, n_cols)

        print(f"\n{'='*110}")
        print(f"  Shape: ({n_rows:,} × {n_cols})   ~{mb:.0f} MB float32   reps={n_reps}"
              f"   threads={os.environ.get('SKRUB_RUST_THREADS', '0') or '0'}"
              f"   verify={'on' if size_verify else 'off'}")
        print(f"{'='*110}")
        print_header()

        groups = [bench_normalize]
        if inplace:
            groups.append(bench_normalize_inplace)
        for group in groups:
            for name, (sk, ru, v) in group(data, norms, n_reps, size_verify).items():
                any_fail |= print_row(name, sk, ru, v)

        del data
        gc.collect()
    return any_fail


def run_thread_sweep(thread_counts, args) -> int:
    """Re-run this script once per thread count, in a fresh process each time.

    util.rs caches SKRUB_RUST_THREADS in a Lazy and threads.rs builds its pool
    in a OnceLock, so a thread count can only be chosen before the first kernel
    call. Returns the exit code to propagate."""
    child_args = []
    for flag in ("l1", "l2", "max", "small", "large", "xlarge", "xxlarge"):
        if getattr(args, flag):
            child_args.append(f"--{flag}")
    if args.no_inplace:
        child_args.append("--no-inplace")
    if args.no_verify:
        child_args.append("--no-verify")
    if args.debug_timing:
        child_args.append("--debug-timing")
    child_args += ["--reps", str(args.reps)]

    failed = False
    for n_threads in thread_counts:
        print(f"\n{'#'*110}")
        print(f"#  THREADS = {n_threads}")
        print(f"{'#'*110}")
        env = dict(os.environ, SKRUB_RUST_THREADS=str(n_threads))
        rc = subprocess.run([sys.executable, os.path.abspath(__file__)] + child_args,
                            env=env).returncode
        failed |= rc != 0
    return 1 if failed else 0


def main():
    global DEBUG_TIMING

    parser = argparse.ArgumentParser(description="Bench Rust normalization vs sklearn")
    # kernels
    parser.add_argument("--l1",  action="store_true", help="include L1 normalize")
    parser.add_argument("--l2",  action="store_true", help="include L2 normalize")
    parser.add_argument("--max", action="store_true", help="include Max normalize")
    parser.add_argument("--no-inplace", action="store_true",
                        help="drop in-place kernel variants")
    # sizes
    parser.add_argument("--small",   action="store_true", help="10k-1M rows (default)")
    parser.add_argument("--large",   action="store_true", help="1M rows, wider")
    parser.add_argument("--xlarge",  action="store_true", help="2M-5M rows")
    parser.add_argument("--xxlarge", action="store_true", help="10M rows (~2 GB f32)")
    # run control
    parser.add_argument("--threads", default=None, metavar="LIST",
                        help=f"comma-separated thread counts (max {_n_cpu}, "
                             f"default: backend default)")
    parser.add_argument("--reps", type=int, default=5, help="Repetitions per kernel (default 5)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the rust-vs-sklearn correctness check")
    parser.add_argument("--debug-timing", action="store_true",
                        help="Print the backend's per-call kernel timings")
    args = parser.parse_args()

    DEBUG_TIMING = args.debug_timing

    # Kernels: no flag means all three
    norms = [n for n, on in (("l2", args.l2), ("l1", args.l1), ("max", args.max)) if on]
    if not norms:
        norms = ["l2", "l1", "max"]

    # Sizes: tiers are additive and --small is the default
    sizes = []
    if args.small or not (args.large or args.xlarge or args.xxlarge):
        sizes += SIZE_SMALL
    if args.large:
        sizes += SIZE_LARGE
    if args.xlarge:
        sizes += SIZE_XLARGE
    if args.xxlarge:
        sizes += SIZE_XXLARGE

    verify = not args.no_verify

    # A thread sweep needs one process per thread count, because the Rust pool
    # is built once from SKRUB_RUST_THREADS and never rebuilt.
    if args.threads:
        thread_counts = [int(t) for t in args.threads.split(",") if t.strip()]
        over = [t for t in thread_counts if t > _n_cpu]
        if over:
            print(f"warning: requested more threads than cores ({over} > {_n_cpu})")
        sys.exit(run_thread_sweep(thread_counts, args))

    print(f"sklearn {__import__('sklearn').__version__}  |  numpy {np.__version__}  "
          f"|  reps={args.reps}  |  verify={'on' if verify else 'off'}  "
          f"|  norms={','.join(norms)}  "
          f"|  threads={os.environ.get('SKRUB_RUST_THREADS', '0') or '0'}")
    any_fail = run_suite(sizes, norms, args.reps, verify,
                         inplace=not args.no_inplace)

    if verify:
        print()
        if any_fail:
            print("RESULT: one or more kernels FAILED the correctness check ✗")
            sys.exit(1)
        print("RESULT: all kernels passed the correctness check ✓")


if __name__ == "__main__":
    main()
