"""Benchmark the stratum QuantileTransformer against scikit-learn.

For each (rows x cols) cell of a config grid, measures ``fit`` / ``transform`` /
``fit_transform`` wall-clock time for both implementations and how far apart
their results are. Rows are written and flushed as each cell finishes, so a
crash mid-run keeps whatever was already measured.

Two config flavours are supported, the same split as logisticRegression:

  * ``config.json``             - one ``qt_params`` dict, timing enabled.
  * ``config_correctness.json`` - a list of ``qt_param_sets`` with
    ``measure_time`` false. Every cell is run once per parameter set, so
    agreement is checked across the awkward cases (NaNs, ties, subsampling)
    rather than a single clean one.

Two further axes are crossed with the parameter sets:

  * ``n_jobs`` (top level) is stratum-only -- the ``ThreadPoolExecutor`` width in
    ``_transform``. It changes only how fast the transform is, never what it
    returns, so the plots draw it as curves within one figure. A config that
    omits it leaves each set's own value alone.
  * ``output_distribution`` may be a single string or a list, and a list is
    swept like ``rows`` and ``cols``. Unlike ``n_jobs`` it changes WHAT
    ``transform`` computes -- ``uniform`` runs scikit-learn's own
    ``_transform_col``, ``normal`` the ``ndtri``/``ndtr`` rewrite -- so each
    value becomes its own parameter set, and therefore its own figure.

Agreement is reported as raw distances (MAE and max |Δ|), not as pass/fail
against a tolerance: what is being measured is float64 round-off, and the number
itself is the result.

Both implementations receive the SAME float64 array, so any difference in the
CSV is a difference in arithmetic rather than in inputs. (The float32 fallback
path, where the kernel declines and the adapter degrades to scikit-learn's own
``_dense_fit``, is covered by
``stratum/tests/adapters/test_quantile_transformer.py``, not here.)

The column prefix is ``stratum_`` rather than ``rust_`` because only ``fit`` is
native. The three measured methods speed up for different reasons:

  * ``fit``       - ``_dense_fit`` replaces ``np.nanpercentile`` with a rayon
    kernel that sorts the columns in parallel. Bounded by ``subsample``: at
    scikit-learn's default of 10 000 the kernel never sees more than 10 000
    rows, so the fit cost stops growing with ``rows``.
  * ``transform`` - stays in NumPy. The wins are ``ndtri``/``ndtr`` instead of
    ``scipy.stats.norm.ppf``/``cdf`` (``normal`` output only), a skipped reverse
    interpolation pass when the quantiles are strictly increasing, an all-finite
    fast path, and the ``n_jobs`` thread fan-out over features.
  * ``fit_transform`` - both of the above, on the same data.

``fit`` depends on neither axis, so its timing is measured once per distinct fit
and reused (see ``_FIT_TIMING_CACHE``) instead of being re-measured per thread
count and distribution.

Sparse input is out of scope: the native kernel backs ``_dense_fit`` only, and
sparse matrices go through scikit-learn's inherited ``_sparse_fit``.

Run:  ./.venv/bin/python benchmark/quantileTransformer/benchmark.py [config.json]
      ./.venv/bin/python benchmark/quantileTransformer/benchmark.py config_correctness.json

A bare config file name is resolved next to this script. Output always lands in
``results/`` next to this script; the config's ``output_csv`` names the file,
not its directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer as SklearnQuantileTransformer

from stratum.adapters.quantile_transformer import QuantileTransformer

# CSV schema. Timing columns are written as NaN when `measure_time` is false.
CSV_COLUMNS = [
    "rows", "cols", "param_set", "repeats", "test_size",
    "n_fit_rows", "n_transform_rows", "data_kind",
    "n_quantiles", "output_distribution", "subsample", "n_jobs",
    # --- timing -------------------------------------------------------------
    "stratum_fit_time", "stratum_fit_std", "sklearn_fit_time", "sklearn_fit_std",
    "stratum_transform_time", "stratum_transform_std",
    "sklearn_transform_time", "sklearn_transform_std",
    "stratum_fit_transform_time", "stratum_fit_transform_std",
    "sklearn_fit_transform_time", "sklearn_fit_transform_std",
    "fit_speedup", "transform_speedup", "fit_transform_speedup",
    # --- agreement on the fitted state --------------------------------------
    "quantiles_mae", "quantiles_max_abs_diff", "quantiles_shape_match",
    "stratum_quantiles_monotonic", "sklearn_quantiles_monotonic",
    # --- agreement on the transformed data ----------------------------------
    "transform_mae", "transform_max_abs_diff", "transform_nan_agree",
    "fit_transform_mae", "fit_transform_max_abs_diff",
    # --- agreement on the inverse, and each implementation's round-trip -----
    "inverse_mae", "inverse_max_abs_diff",
    "roundtrip_max_abs_diff_stratum", "roundtrip_max_abs_diff_sklearn",
    # --- provenance ---------------------------------------------------------
    "stratum_fastpath", "qt_params",
]

# Fixed output directory next to this script; the config names the file only.
RESULTS_DIRNAME = "results"

NAN = float("nan")

# Fit timings, keyed by everything a fit actually depends on. Neither sweep axis
# is in the key, and that is deliberate: n_jobs is a transform-only thread count
# and output_distribution is read only by transform/inverse_transform, so
# including them would pay for the same measurement once per combination.
_FIT_TIMING_CACHE: dict = {}


def time_call(fn, repeats, warmup):
    """Return (mean, std) wall-clock seconds over `repeats` timed runs of fn()."""
    for _ in range(warmup):
        fn()
    samples = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples[i] = time.perf_counter() - t0
    return float(samples.mean()), float(samples.std())


def build_param_sets(cfg):
    """Return the list of per-cell parameter dicts the config asks for.

    ``qt_params`` (one dict) is the speed form, ``qt_param_sets`` (a list) the
    agreement form. A top-level ``n_jobs`` list is crossed with whichever of the
    two is present and overrides the value a set carries; without it, each set
    keeps its own.
    """
    base = cfg.get("qt_param_sets") or [cfg["qt_params"]]
    n_jobs = cfg.get("n_jobs")
    if n_jobs is None:
        return [dict(p) for p in base]
    if not isinstance(n_jobs, list):
        n_jobs = [n_jobs]
    return [dict(p, n_jobs=n) for p, n in product(base, n_jobs)]


def build_output_distributions(cfg):
    """Return the ``output_distribution`` values to sweep, as a grid axis.

    A top-level list (or single string) is the axis and overrides whatever the
    parameter sets carry. ``None`` means there is no axis, and each parameter set
    keeps its own value -- which is what config_correctness.json does, where the
    distribution is one of the things a set is defined by.
    """
    values = cfg.get("output_distribution")
    if values is None:
        return [None]
    return values if isinstance(values, list) else [values]


def make_data(n_rows, n_cols, spec, random_state):
    """Build the synthetic matrix a parameter set asks for, as float64.

    ``spec`` is the config's ``data`` block (or a parameter set's override), and
    may be absent -- the default is clean normal data. ``kind`` picks the column
    distribution; the remaining keys perturb it:

      * ``normal``    - the reference case. Distinct values, no ties, no NaNs.
      * ``lognormal`` - heavily skewed, so the order statistics bunch up at one
        end and `np.interp` works on a very unevenly spaced ``xp``.
      * ``ties``      - integer levels, i.e. repeated values per column. These
        produce repeated quantiles, which is the entire reason the forward
        transform interpolates in both directions and averages.
      * ``constant``  - one value per column. Degenerate quantiles, which leaves
        the bounds-clipping branches of transform as the only active code.

    ``nan_frac`` sprinkles NaNs uniformly (exercises the kernel's NaN-dropping
    gather and the masked branch of ``_transform_col_fast``); ``n_all_nan_cols``
    turns whole columns to NaN (the kernel's all-NaN early return, which must
    match ``np.nanpercentile``'s all-NaN slice).
    """
    spec = spec or {}
    kind = spec.get("kind", "normal")
    rng = np.random.default_rng(spec.get("random_state", random_state))

    if kind == "normal":
        X = rng.standard_normal((n_rows, n_cols))
    elif kind == "lognormal":
        X = np.exp(spec.get("sigma", 1.0) * rng.standard_normal((n_rows, n_cols)))
    elif kind == "ties":
        n_levels = spec.get("n_levels", 32)
        X = rng.integers(0, n_levels, size=(n_rows, n_cols)).astype(np.float64)
    elif kind == "constant":
        X = np.repeat(rng.standard_normal((1, n_cols)), n_rows, axis=0)
    else:
        raise ValueError(f"unsupported data kind: {kind!r}")

    nan_frac = spec.get("nan_frac", 0.0)
    if nan_frac:
        mask = rng.random(X.shape) < nan_frac
        X[mask] = np.nan

    n_all_nan = spec.get("n_all_nan_cols", 0)
    if n_all_nan:
        X[:, : min(n_all_nan, n_cols)] = np.nan

    return np.ascontiguousarray(X, dtype=np.float64)


def diff_stats(a, b):
    """(mae, max_abs_diff, nan_agree) for two arrays, ignoring NaN positions.

    NaNs are compared as a mask rather than as values: `nan_agree` is the
    fraction of entries where the two arrays agree on being NaN, and the numeric
    statistics run over the entries where neither is NaN. Mixing the two would
    either poison every statistic with NaN or hide a NaN that only one
    implementation produced.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return NAN, NAN, NAN

    nan_a, nan_b = np.isnan(a), np.isnan(b)
    nan_agree = float(np.mean(nan_a == nan_b)) if a.size else 1.0

    both_finite = ~(nan_a | nan_b)
    if not both_finite.any():
        return 0.0, 0.0, nan_agree

    d = np.abs(a[both_finite] - b[both_finite])
    return float(d.mean()), float(d.max()), nan_agree


def is_monotonic(quantiles):
    """True if every non-degenerate column of `quantiles_` is non-decreasing.

    ``quantiles_[:, i]`` is fed to ``np.interp`` as its ``xp``, which is only
    defined for a non-decreasing ``xp``, so this is a precondition of the whole
    transform rather than a cosmetic property. All-NaN columns are skipped
    (there is no order to check). ``np.nanpercentile`` does not guarantee it
    either, which is why the CSV records the flag for BOTH implementations
    instead of asserting on ours.
    """
    q = np.asarray(quantiles, dtype=np.float64)
    ok_cols = ~np.isnan(q).all(axis=0)
    if not ok_cols.any():
        return True
    d = np.diff(q[:, ok_cols], axis=0)
    return bool(np.all(np.isnan(d) | (d >= 0.0)))


def run_cell(n_rows, n_cols, output_distribution, cfg, p):
    """Benchmark one (rows, cols, distribution, param-set) cell.

    Returns a dict row for the CSV. ``output_distribution`` is the grid axis
    value; ``None`` falls back to the parameter set's own.
    """
    random_state = cfg["random_state"]
    data_spec = p.get("data", cfg.get("data"))
    X = make_data(n_rows, n_cols, data_spec, random_state)

    # Fit on the training split; the held-out split is what the agreement
    # columns are computed on. The TIMED transform runs on X_fit (see below), so
    # `n_transform_rows` describes the agreement split only, not the timings.
    X_fit, X_tf = train_test_split(
        X, test_size=cfg["test_size"], random_state=random_state
    )

    # Column-major on purpose: it is the layout scikit-learn's validation hands
    # over for a DataFrame, and the one where the kernel's per-column gather is
    # a memcpy rather than a strided walk. Both implementations get it.
    X_fit = np.asarray(X_fit, dtype=np.float64, order="F")
    X_tf = np.asarray(X_tf, dtype=np.float64, order="F")

    n_quantiles = p.get("n_quantiles", 1000)
    if output_distribution is None:
        output_distribution = p.get("output_distribution", "uniform")
    subsample = p.get("subsample")  # null in JSON -> None -> use every row
    n_jobs = p.get("n_jobs", 1)

    # `copy` stays at its default True for both implementations: with copy=False
    # `transform` rewrites its input in place, so the second timed repeat would
    # be transforming already-transformed data.
    common = {
        "n_quantiles": n_quantiles,
        "output_distribution": output_distribution,
        "subsample": subsample,
        "random_state": random_state,
        "copy": True
    }

    def new_stratum():
        return QuantileTransformer(n_jobs=n_jobs, **common)

    def new_sklearn():
        # Same random_state, so `resample` draws the same rows in both when
        # `subsample` is active -- otherwise the two would fit different samples
        # and every agreement column below would be meaningless.
        return SklearnQuantileTransformer(**common)

    measure_time = cfg.get("measure_time", True)
    repeats, warmup = cfg["repeats"], cfg["warmup"]

    # --- fitted models (also the fast-path probe) ---------------------------------
    # `simplefilter("always")` defeats the once-per-location dedup, which would
    # otherwise hide a fallback warning on every cell after the first.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        st_model = new_stratum().fit(X_fit)
    fastpath = not any("Falling back" in str(w.message) for w in caught)
    # scikit-learn's fit is silenced instead: on a set with all-NaN columns
    # np.nanpercentile raises "All-NaN slice encountered" once per cell, which is
    # the branch that set exists to exercise rather than something to report.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sk_model = new_sklearn().fit(X_fit)

    # --- timing -------------------------------------------------------------------
    # A fresh estimator per fit repeat, so what is measured is a cold fit rather
    # than a re-fit on warm caches. The n_quantiles > n_samples warning (raised
    # by scikit-learn's own `fit`) is silenced so it cannot dominate the
    # measurement or the console. Transform is timed on X_fit rather than X_tf:
    # the larger array gives the more stable measurement, and the two differ
    # only by a constant row factor.
    if measure_time:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_key = (
                n_rows, n_cols, n_quantiles, subsample,
                json.dumps(data_spec or {}, sort_keys=True),
            )
            if fit_key in _FIT_TIMING_CACHE:
                st_fit_t, st_fit_s, sk_fit_t, sk_fit_s = _FIT_TIMING_CACHE[fit_key]
            else:
                st_fit_t, st_fit_s = time_call(
                    lambda: new_stratum().fit(X_fit), repeats, warmup)
                sk_fit_t, sk_fit_s = time_call(
                    lambda: new_sklearn().fit(X_fit), repeats, warmup)
                _FIT_TIMING_CACHE[fit_key] = (st_fit_t, st_fit_s, sk_fit_t, sk_fit_s)

            st_tf_t, st_tf_s = time_call(lambda: st_model.transform(X_fit), repeats, warmup)
            sk_tf_t, sk_tf_s = time_call(lambda: sk_model.transform(X_fit), repeats, warmup)
            st_ft_t, st_ft_s = time_call(
                lambda: new_stratum().fit_transform(X_fit), repeats, warmup
            )
            sk_ft_t, sk_ft_s = time_call(
                lambda: new_sklearn().fit_transform(X_fit), repeats, warmup
            )
    else:
        st_fit_t = st_fit_s = sk_fit_t = sk_fit_s = NAN
        st_tf_t = st_tf_s = sk_tf_t = sk_tf_s = NAN
        st_ft_t = st_ft_s = sk_ft_t = sk_ft_s = NAN

    # --- agreement ----------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st_tf = st_model.transform(X_tf)
        sk_tf = sk_model.transform(X_tf)
        st_ft = new_stratum().fit_transform(X_fit)
        sk_ft = new_sklearn().fit_transform(X_fit)
        # Inverse of the SAME input for both, so this isolates the inverse path
        # instead of compounding whatever difference the forward pass produced.
        st_inv = st_model.inverse_transform(sk_tf.copy())
        sk_inv = sk_model.inverse_transform(sk_tf.copy())
        # Round-trip: each implementation against the original data. Clipping at
        # the tails makes this non-zero for both; what matters is that the two
        # numbers agree, not that either is small.
        st_round = st_model.inverse_transform(st_tf.copy())
        sk_round = sk_model.inverse_transform(sk_tf.copy())

    q_mae, q_max, _ = diff_stats(st_model.quantiles_, sk_model.quantiles_)
    tf_mae, tf_max, tf_nan_agree = diff_stats(st_tf, sk_tf)
    ft_mae, ft_max, _ = diff_stats(st_ft, sk_ft)
    inv_mae, inv_max, _ = diff_stats(st_inv, sk_inv)
    _, st_round_max, _ = diff_stats(st_round, X_tf)
    _, sk_round_max, _ = diff_stats(sk_round, X_tf)

    def speedup(sk_t, st_t):
        return sk_t / st_t if st_t > 0 else NAN

    return {
        "rows": n_rows, "cols": n_cols, "param_set": p.get("name", "default"),
        "repeats": repeats if measure_time else 0, "test_size": cfg["test_size"],
        "n_fit_rows": X_fit.shape[0], "n_transform_rows": X_tf.shape[0],
        "data_kind": (data_spec or {}).get("kind", "normal"),
        "n_quantiles": n_quantiles, "output_distribution": output_distribution,
        "subsample": subsample, "n_jobs": n_jobs,
        "stratum_fit_time": st_fit_t, "stratum_fit_std": st_fit_s,
        "sklearn_fit_time": sk_fit_t, "sklearn_fit_std": sk_fit_s,
        "stratum_transform_time": st_tf_t, "stratum_transform_std": st_tf_s,
        "sklearn_transform_time": sk_tf_t, "sklearn_transform_std": sk_tf_s,
        "stratum_fit_transform_time": st_ft_t, "stratum_fit_transform_std": st_ft_s,
        "sklearn_fit_transform_time": sk_ft_t, "sklearn_fit_transform_std": sk_ft_s,
        "fit_speedup": speedup(sk_fit_t, st_fit_t),
        "transform_speedup": speedup(sk_tf_t, st_tf_t),
        "fit_transform_speedup": speedup(sk_ft_t, st_ft_t),
        "quantiles_mae": q_mae, "quantiles_max_abs_diff": q_max,
        "quantiles_shape_match": st_model.quantiles_.shape == sk_model.quantiles_.shape,
        "stratum_quantiles_monotonic": is_monotonic(st_model.quantiles_),
        "sklearn_quantiles_monotonic": is_monotonic(sk_model.quantiles_),
        "transform_mae": tf_mae, "transform_max_abs_diff": tf_max,
        "transform_nan_agree": tf_nan_agree,
        "fit_transform_mae": ft_mae, "fit_transform_max_abs_diff": ft_max,
        "inverse_mae": inv_mae, "inverse_max_abs_diff": inv_max,
        "roundtrip_max_abs_diff_stratum": st_round_max,
        "roundtrip_max_abs_diff_sklearn": sk_round_max,
        "stratum_fastpath": fastpath,
        # The set as it was actually run: the distribution axis overrides what the
        # set carries, so it is merged in rather than dumped from `p`.
        "qt_params": json.dumps(
            {k: v for k, v in p.items() if not k.startswith("_")}
            | {"output_distribution": output_distribution},
            sort_keys=True,
        ),
    }


def main():
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", nargs="?", default=None,
                    help="config file, resolved relative to this script if not a path "
                         "(default: config.json)")
    ap.add_argument("--config", dest="config_opt", default=None,
                    help="same as the positional argument; wins if both are given")
    args = ap.parse_args()

    # A bare name like "config_correctness.json" is looked up next to this script,
    # so the benchmark runs the same from the repo root or from its own directory.
    chosen = args.config_opt or args.config or "config.json"
    cfg_path = Path(chosen)
    if not cfg_path.is_absolute() and not cfg_path.exists():
        cfg_path = script_dir / chosen

    cfg = json.loads(cfg_path.read_text())
    measure_time = cfg.get("measure_time", True)
    param_sets = build_param_sets(cfg)
    distributions = build_output_distributions(cfg)

    # `.name` strips any directory the config's output_csv may carry, so runs
    # cannot scatter CSVs across the repo.
    out_dir = script_dir / RESULTS_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / Path(cfg["output_csv"]).name
    n_cells = (len(cfg["rows"]) * len(cfg["cols"])
               * len(distributions) * len(param_sets))
    print(f"[bench] config {cfg_path} -> {out_csv}", flush=True)
    print(f"[bench] {len(param_sets)} parameter combination(s), {n_cells} cells",
          flush=True)

    # Fresh CSV with header; rows appended (and flushed) as each cell finishes.
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        f.flush()
        for n_rows in cfg["rows"]:
            for n_cols in cfg["cols"]:
                for output_distribution in distributions:
                    for p in param_sets:
                        label = f"{p.get('name', 'default')} n_jobs={p.get('n_jobs', 1)}"
                        dist = output_distribution or p.get("output_distribution")
                        print(f"[bench] rows={n_rows} cols={n_cols} {dist} {label} ...",
                              flush=True)
                        row = run_cell(n_rows, n_cols, output_distribution, cfg, p)
                        writer.writerow(row)
                        f.flush()

                        flag = "" if row["stratum_fastpath"] else "  (FELL BACK!)"
                        print("       ", end="")
                        if measure_time:
                            print(
                                f" fit x{row['fit_speedup']:.2f}  "
                                f"transform x{row['transform_speedup']:.2f}  "
                                f"fit_transform x{row['fit_transform_speedup']:.2f} ",
                                end="", flush=True,
                            )
                        print(
                            f" q_max={row['quantiles_max_abs_diff']:.2e}  "
                            f"tf_max={row['transform_max_abs_diff']:.2e}  "
                            f"inv_max={row['inverse_max_abs_diff']:.2e}{flag}",
                            flush=True,
                        )

    print(f"[bench] done -> {out_csv}")


if __name__ == "__main__":
    main()
