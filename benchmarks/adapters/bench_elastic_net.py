"""
Benchmark: Rust ElasticNet vs sklearn ElasticNet.

Both sides run the same stratum.ElasticNet estimator, with
skrub.set_config(rust_backend=...) choosing the backend, so the sklearn column
measures the fallback path the adapter itself would take and both sides receive
the identical float32 input. Timing covers the whole fit / predict call,
including the input validation both backends perform.

Metrics: fit time, predict time, and coefficient agreement between the two
backends.

Usage
-----
    cd <repo_root>
    uv run python benchmarks/adapters/bench_elastic_net.py            # default small sizes
    uv run python benchmarks/adapters/bench_elastic_net.py --reps 10
    uv run python benchmarks/adapters/bench_elastic_net.py --large    # only large   (~100 MB)
    uv run python benchmarks/adapters/bench_elastic_net.py --xlarge   # only v. large (~1 GB)
    uv run python benchmarks/adapters/bench_elastic_net.py --xxlarge  # only vv large (~2 GB)
    uv run python benchmarks/adapters/bench_elastic_net.py --debug-timing

Size flags are mutually exclusive and each runs ONLY its own tier.
"""
import os
os.environ["SKRUB_RUST"] = "1"

import argparse
import gc
import sys
import time

import numpy as np

import stratum as skrub
from stratum import ElasticNet
from stratum import _rust_backend as rb

if not rb.HAVE_RUST:
    sys.exit("Rust backend not built. Run: cd _rust && maturin develop --release")

MAX_ITER = 1000
TOL = 1e-4

# ── backend selection ─────────────────────────────────────────────────────────

DEBUG_TIMING = False
NUM_THREADS = 0        # 0 = let the backend decide (Rayon global pool = one per core)

def use_backend(rust: bool):
    """Point stratum at the Rust solver or back at sklearn.

    stratum.ElasticNet reads the config on every call, so one estimator class
    serves both columns of the table.

    The thread count must be set before the first kernel call: the Rust side
    caches it in a Lazy (util.rs NUM_THREADS) and builds its pool in a OnceLock
    (threads.rs POOL), so later changes are silently ignored. One process
    therefore measures exactly one thread count, which is why --threads takes a
    single value rather than a sweep.
    """
    if rust:
        skrub.set_config(rust_backend=True, debug_timing=DEBUG_TIMING,
                         num_threads=NUM_THREADS)
    else:
        skrub.set_config(rust_backend=False, debug_timing=DEBUG_TIMING)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_data(n_rows, n_cols, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_cols)).astype(np.float32)
    true_coef = rng.standard_normal(n_cols).astype(np.float32)
    y = (X @ true_coef + 0.1 * rng.standard_normal(n_rows)).astype(np.float32)
    return X, y


def timeit(fn, n_reps=5):
    times = []
    for _ in range(n_reps):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def warmup(fn, n=2):
    for _ in range(n):
        fn()
    gc.collect()


# ── benchmark ─────────────────────────────────────────────────────────────────

COL_W = 22

def _ms(t): return f"{t * 1000:.1f} ms"

def print_header():
    h = (f"{'kernel':<{COL_W}}"
         f"{'sklearn':>{COL_W}}"
         f"{'rust':>{COL_W}}"
         f"{'speedup':>{COL_W}}")
    print(h)
    print("-" * len(h))

def print_row(name, sk_t, ru_t, extra=""):
    if sk_t is None:
        print(f"{name:<{COL_W}}{'— skipped':>{COL_W}}{_ms(ru_t):>{COL_W}}"
              f"{'—':>{COL_W}}{extra}")
        return
    sp = sk_t / ru_t if ru_t > 0 else float("inf")
    mark = " ✓" if sp >= 1.0 else " ✗"
    print(f"{name:<{COL_W}}{_ms(sk_t):>{COL_W}}{_ms(ru_t):>{COL_W}}"
          f"{sp:>{COL_W - 2}.2f}x{mark}{extra}")


def _fresh(alpha, l1_ratio):
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=MAX_ITER,
                      tol=TOL, fit_intercept=True)


def bench_one(X, y, alpha, l1_ratio, n_reps, compare_sklearn=True):
    """Time fit and predict on each backend and compare the fitted coefficients."""
    results = {}
    for rust in ((False, True) if compare_sklearn else (True,)):
        use_backend(rust)
        # A fresh estimator per repetition: fit must not reuse prior state.
        warmup(lambda: _fresh(alpha, l1_ratio).fit(X, y))
        fit_t = timeit(lambda: _fresh(alpha, l1_ratio).fit(X, y), n_reps)

        model = _fresh(alpha, l1_ratio).fit(X, y)
        warmup(lambda: model.predict(X))
        pred_t = timeit(lambda: model.predict(X), n_reps)
        results[rust] = (fit_t, pred_t, np.asarray(model.coef_, dtype=np.float32),
                         int(getattr(model, "n_iter_", -1)))
        del model
        gc.collect()

    ru_fit, ru_pred, ru_coef, ru_iter = results[True]
    if not compare_sklearn:
        return None, ru_fit, None, ru_pred, None, ru_iter, None
    sk_fit, sk_pred, sk_coef, sk_iter = results[False]
    coef_mse = float(np.mean((ru_coef - sk_coef) ** 2))
    return sk_fit, ru_fit, sk_pred, ru_pred, coef_mse, ru_iter, sk_iter


def run_suite(sizes, alpha, l1_ratio, n_reps, compare_sklearn=True):
    for n_rows, n_cols in sizes:
        gb = n_rows * n_cols * 4 / 1e9
        size_str = f"~{gb*1000:.0f} MB" if gb < 1 else f"~{gb:.1f} GB"
        print(f"\n{'='*88}")
        print(f"  ({n_rows:,} × {n_cols})  {size_str} f32  alpha={alpha}  l1_ratio={l1_ratio}  "
              f"max_iter={MAX_ITER}  reps={n_reps}")
        print(f"{'='*88}")

        X, y = make_data(n_rows, n_cols)
        sk_fit_t, ru_fit_t, sk_pred_t, ru_pred_t, coef_mse, n_iter, sk_iter = \
            bench_one(X, y, alpha, l1_ratio, n_reps, compare_sklearn)

        print_header()
        # Both iteration counts, because the solvers stop on different criteria
        # (max relative |dw_j| here, duality gap in sklearn). Without sklearn's
        # count there is no way to tell a faster iteration from a shorter run.
        iters = f"  (n_iter rust={n_iter}"
        iters += f", sklearn={sk_iter})" if sk_iter is not None else ")"
        print_row("elastic_net fit", sk_fit_t, ru_fit_t, iters)
        mse_extra = f"  coef_mse={coef_mse:.2e}" if coef_mse is not None else ""
        print_row("elastic_net predict", sk_pred_t, ru_pred_t, mse_extra)
        del X, y
        gc.collect()


# Size tiers — mutually exclusive; each flag runs ONLY its own set.
DEFAULT_SIZES = [(10_000, 50), (50_000, 50), (100_000, 50), (100_000, 100)]
LARGE_SIZES   = [(500_000, 50), (1_000_000, 20)]                       # ~100 MB
XLARGE_SIZES  = [(2_000_000, 100), (5_000_000, 50), (10_000_000, 20)]  # ~1 GB
XXLARGE_SIZES = [(10_000_000, 50)]                                    # ~2 GB


def main():
    global DEBUG_TIMING, NUM_THREADS

    p = argparse.ArgumentParser()
    p.add_argument("--reps",     type=int,   default=None,
                   help="timed repetitions (default: 5, or 3 for --xlarge/--xxlarge)")
    p.add_argument("--alpha",    type=float, default=0.1)
    p.add_argument("--l1-ratio", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=0, metavar="N",
                   help=f"Rust thread count (default 0 = one per core, {os.cpu_count()} here)")
    p.add_argument("--debug-timing", action="store_true",
                   help="print the backend's per-call kernel timings")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--large",   action="store_true",
                   help="run ONLY large matrices (~100 MB f32)")
    g.add_argument("--xlarge",  action="store_true",
                   help="run ONLY very large matrices (~1 GB f32)")
    g.add_argument("--xxlarge", action="store_true",
                   help="run ONLY very very large matrices (~2 GB f32)")
    args = p.parse_args()

    DEBUG_TIMING = args.debug_timing
    NUM_THREADS = args.threads

    if args.xxlarge:
        sizes = XXLARGE_SIZES
    elif args.xlarge:
        sizes = XLARGE_SIZES
    elif args.large:
        sizes = LARGE_SIZES
    else:
        sizes = DEFAULT_SIZES

    # Very/very-very large fits are seconds each; fewer reps keeps runtime sane.
    big = args.xlarge or args.xxlarge
    n_reps = args.reps if args.reps is not None else (3 if big else 5)

    print(f"sklearn {__import__('sklearn').__version__}  |  numpy {np.__version__}  "
          f"|  reps={n_reps}  |  threads={args.threads or 'default (%d)' % (os.cpu_count() or 1)}")
    run_suite(sizes, args.alpha, args.l1_ratio, n_reps, compare_sklearn=True)


if __name__ == "__main__":
    main()