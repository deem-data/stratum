from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from stratum.adapters.logistic_regression import (
    LogisticRegression as RustLogisticRegression,
)

# CSV schema. Timing columns are written as NaN when `measure_time` is false.
CSV_COLUMNS = [
    "rows", "cols", "param_set", "repeats", "test_size",
    # --- timing -------------------------------------------------------------
    "rust_fit_time", "rust_fit_std", "sklearn_fit_time", "sklearn_fit_std",
    "rust_predict_time", "rust_predict_std", "sklearn_predict_time", "sklearn_predict_std",
    "rust_predict_proba_time", "rust_predict_proba_std",
    "sklearn_predict_proba_time", "sklearn_predict_proba_std",
    "fit_speedup", "predict_speedup", "predict_proba_speedup",
    # --- accuracy, train and test kept separate -----------------------------
    "rust_acc_train", "rust_acc_test", "sklearn_acc_train", "sklearn_acc_test",
    "pred_agree_train", "pred_agree_test",
    # --- KL between the two probability vectors, both directions plus the max.
    # KL is asymmetric, so the headline agreement number is the worse direction
    # rather than whichever one flatters the implementation ------------------
    "kl_sklearn_to_rust_train", "kl_sklearn_to_rust_test",
    "kl_rust_to_sklearn_train", "kl_rust_to_sklearn_test",
    "kl_max_train", "kl_max_test",
    # --- cross-entropy of each model against the true labels ----------------
    "logloss_rust_train", "logloss_rust_test",
    "logloss_sklearn_train", "logloss_sklearn_test",
    # --- raw probability / parameter distances ------------------------------
    "proba_mae_test", "proba_max_abs_diff_test",
    "param_mae", "param_max_abs_diff",
    "rust_fastpath", "sklearn_solver", "lr_params",
]

# Probabilities are clipped away from {0, 1} before any log; the Rust kernel
# returns float32, which saturates to exactly 0.0 / 1.0 for confident samples.
EPS = 1e-7

# Fixed output directory next to this script; the config names the file only.
RESULTS_DIRNAME = "results"


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


def params_vector(model):
    """Flatten [coef_, intercept_] into one float64 vector for MAE comparison."""
    coef = np.asarray(model.coef_, dtype=np.float64).ravel()
    intercept = np.asarray(model.intercept_, dtype=np.float64).ravel()
    return np.concatenate([coef, intercept])


def pos_proba(model, X):
    """P(y = 1) as float64, the second column of predict_proba."""
    return np.asarray(model.predict_proba(X), dtype=np.float64)[:, 1]


def binary_cross_entropy(target, pred):
    """Binary cross-entropy H(target, pred) = -mean(t*log(p) + (1-t)*log(1-p)).

    `target` may be hard labels (0/1) or soft ones. Against the true labels this
    is ordinary log-loss. Against another model's probabilities it is NOT a
    distance -- its floor is H(target), not zero; use `kl_divergence` for that.
    """
    t = np.clip(np.asarray(target, dtype=np.float64), EPS, 1.0 - EPS)
    p = np.clip(np.asarray(pred, dtype=np.float64), EPS, 1.0 - EPS)
    return float(-(t * np.log(p) + (1.0 - t) * np.log1p(-p)).mean())


def kl_divergence(target, pred):
    """KL(target || pred) = H(target, pred) - H(target), in nats per sample.

    Zero exactly when the two probability vectors agree and growing with the
    disagreement, unlike the raw cross-entropy whose floor moves with the
    target's own entropy. Both arguments go through the same clipping, so the
    entropy term cancels and the result cannot go negative beyond round-off.
    """
    return binary_cross_entropy(target, pred) - binary_cross_entropy(target, target)


def make_sample_weight(spec, n_samples):
    """Build the per-sample fit weights a parameter set asks for, or None.

    ``kind = "lognormal"`` draws w = exp(sigma * N(0, 1)) from a seeded
    generator: positive, reproducible, spread over roughly a decade at
    sigma = 0.5. Per-sample rather than per-class on purpose -- a two-valued
    weight vector is still matched by a kernel that only tracks a class-level
    scale factor, a continuous one is not.

    Returned float32 and C-contiguous, which is what the adapter's
    `_check_before_fit` requires; anything else drops the cell to the sklearn
    fallback and `rust_fastpath` goes false. scikit-learn accepts float32 fine.
    """
    if not spec:
        return None
    kind = spec.get("kind", "lognormal")
    if kind != "lognormal":
        raise ValueError(f"unsupported sample_weight kind: {kind!r}")
    rng = np.random.default_rng(spec.get("random_state", 0))
    w = np.exp(spec.get("sigma", 0.5) * rng.standard_normal(n_samples))
    return np.ascontiguousarray(w, dtype=np.float32)


def run_cell(n_rows, n_cols, cfg, p):
    """Benchmark one (rows, cols, param-set) cell; return a dict row for the CSV."""
    frac = cfg.get("n_informative_frac", 0.5)
    n_informative = max(2, int(frac * n_cols))
    X, y = make_classification(
        n_samples=n_rows,
        n_features=n_cols,
        n_informative=n_informative,
        n_redundant=0,
        n_classes=2,
        random_state=cfg["random_state"],
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=cfg["random_state"]
    )

    # Rust fast-path arrays: float32 + uint32 + C-contiguous. sklearn keeps float64.
    Xtr_rust = np.ascontiguousarray(X_tr, dtype=np.float32)
    Xte_rust = np.ascontiguousarray(X_te, dtype=np.float32)
    ytr_rust = np.ascontiguousarray(y_tr, dtype=np.uint32)
    Xtr_sk = np.ascontiguousarray(X_tr, dtype=np.float64)
    Xte_sk = np.ascontiguousarray(X_te, dtype=np.float64)

    # scikit-learn's lbfgs only supports l2; l1_ratio > 0 needs saga. The config
    # carries the solver per parameter set so both stay on the same objective.
    sk_solver = p.get("sklearn_solver", "lbfgs")

    # One weight vector per cell, handed to BOTH implementations. Weights affect
    # the FIT only -- accuracy, log-loss and KL below stay unweighted, so those
    # columns remain comparable across parameter sets.
    sample_weight = make_sample_weight(p.get("sample_weight"), len(y_tr))

    def new_rust():
        return RustLogisticRegression(
            C=p["C"], l1_ratio=p["l1_ratio"], fit_intercept=p["fit_intercept"],
            tol=p["tol"], max_iter=p["max_iter"],
        )

    def new_sklearn():
        # sklearn >=1.8 deprecated `penalty`; the L1/L2 mix is set via l1_ratio + C
        # (l1_ratio=0.0 is pure L2), matching how the Rust kernel derives l1/l2 reg.
        return SklearnLogisticRegression(
            C=p["C"], fit_intercept=p["fit_intercept"], tol=p["tol"],
            max_iter=p["max_iter"], solver=sk_solver, l1_ratio=p["l1_ratio"],
        )

    def fit_rust():
        return new_rust().fit(Xtr_rust, ytr_rust, sample_weight=sample_weight)

    def fit_sklearn():
        return new_sklearn().fit(Xtr_sk, y_tr, sample_weight=sample_weight)

    measure_time = cfg.get("measure_time", True)
    repeats, warmup = cfg["repeats"], cfg["warmup"]
    nan = float("nan")

    # --- fitted models (also the fast-path probe) ---------------------------------
    # `simplefilter("always")` defeats the once-per-location dedup, which would
    # otherwise hide a fallback warning on every cell after the first.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rust_model = fit_rust()
    fastpath = not any("Falling back" in str(w.message) for w in caught)
    sk_model = fit_sklearn()

    # --- timing (fresh estimator each fit repeat so we measure a cold fit) --------
    if measure_time:
        rust_fit_t, rust_fit_s = time_call(fit_rust, repeats, warmup)
        sk_fit_t, sk_fit_s = time_call(fit_sklearn, repeats, warmup)
        rust_pred_t, rust_pred_s = time_call(lambda: rust_model.predict(Xte_rust), repeats, warmup)
        sk_pred_t, sk_pred_s = time_call(lambda: sk_model.predict(Xte_sk), repeats, warmup)
        rust_pp_t, rust_pp_s = time_call(lambda: rust_model.predict_proba(Xte_rust), repeats, warmup)
        sk_pp_t, sk_pp_s = time_call(lambda: sk_model.predict_proba(Xte_sk), repeats, warmup)
    else:
        rust_fit_t = rust_fit_s = sk_fit_t = sk_fit_s = nan
        rust_pred_t = rust_pred_s = sk_pred_t = sk_pred_s = nan
        rust_pp_t = rust_pp_s = sk_pp_t = sk_pp_s = nan

    # --- accuracy, train and test separately --------------------------------------
    rust_pred_tr = np.asarray(rust_model.predict(Xtr_rust))
    rust_pred_te = np.asarray(rust_model.predict(Xte_rust))
    sk_pred_tr = sk_model.predict(Xtr_sk)
    sk_pred_te = sk_model.predict(Xte_sk)

    rust_acc_train = accuracy_score(y_tr, rust_pred_tr)
    rust_acc_test = accuracy_score(y_te, rust_pred_te)
    sk_acc_train = accuracy_score(y_tr, sk_pred_tr)
    sk_acc_test = accuracy_score(y_te, sk_pred_te)

    # Fraction of samples where the two implementations pick the same class.
    pred_agree_train = float(np.mean(rust_pred_tr == sk_pred_tr))
    pred_agree_test = float(np.mean(rust_pred_te == sk_pred_te))

    # --- KL divergence between the two probability vectors ------------------------
    rust_p_tr, rust_p_te = pos_proba(rust_model, Xtr_rust), pos_proba(rust_model, Xte_rust)
    sk_p_tr, sk_p_te = pos_proba(sk_model, Xtr_sk), pos_proba(sk_model, Xte_sk)

    kl_s2r_train = kl_divergence(sk_p_tr, rust_p_tr)
    kl_s2r_test = kl_divergence(sk_p_te, rust_p_te)
    kl_r2s_train = kl_divergence(rust_p_tr, sk_p_tr)
    kl_r2s_test = kl_divergence(rust_p_te, sk_p_te)
    kl_max_train = max(kl_s2r_train, kl_r2s_train)
    kl_max_test = max(kl_s2r_test, kl_r2s_test)

    # --- log-loss of each model against the ground truth --------------------------
    ll_rust_train = binary_cross_entropy(y_tr, rust_p_tr)
    ll_rust_test = binary_cross_entropy(y_te, rust_p_te)
    ll_sk_train = binary_cross_entropy(y_tr, sk_p_tr)
    ll_sk_test = binary_cross_entropy(y_te, sk_p_te)

    # --- raw distances ------------------------------------------------------------
    proba_diff = np.abs(rust_p_te - sk_p_te)
    param_diff = np.abs(params_vector(rust_model) - params_vector(sk_model))

    def speedup(sk_t, rust_t):
        return sk_t / rust_t if rust_t > 0 else nan

    return {
        "rows": n_rows, "cols": n_cols, "param_set": p.get("name", "default"),
        "repeats": repeats if measure_time else 0, "test_size": cfg["test_size"],
        "rust_fit_time": rust_fit_t, "rust_fit_std": rust_fit_s,
        "sklearn_fit_time": sk_fit_t, "sklearn_fit_std": sk_fit_s,
        "rust_predict_time": rust_pred_t, "rust_predict_std": rust_pred_s,
        "sklearn_predict_time": sk_pred_t, "sklearn_predict_std": sk_pred_s,
        "rust_predict_proba_time": rust_pp_t, "rust_predict_proba_std": rust_pp_s,
        "sklearn_predict_proba_time": sk_pp_t, "sklearn_predict_proba_std": sk_pp_s,
        "fit_speedup": speedup(sk_fit_t, rust_fit_t),
        "predict_speedup": speedup(sk_pred_t, rust_pred_t),
        "predict_proba_speedup": speedup(sk_pp_t, rust_pp_t),
        "rust_acc_train": rust_acc_train, "rust_acc_test": rust_acc_test,
        "sklearn_acc_train": sk_acc_train, "sklearn_acc_test": sk_acc_test,
        "pred_agree_train": pred_agree_train, "pred_agree_test": pred_agree_test,
        "kl_sklearn_to_rust_train": kl_s2r_train,
        "kl_sklearn_to_rust_test": kl_s2r_test,
        "kl_rust_to_sklearn_train": kl_r2s_train,
        "kl_rust_to_sklearn_test": kl_r2s_test,
        "kl_max_train": kl_max_train, "kl_max_test": kl_max_test,
        "logloss_rust_train": ll_rust_train, "logloss_rust_test": ll_rust_test,
        "logloss_sklearn_train": ll_sk_train, "logloss_sklearn_test": ll_sk_test,
        "proba_mae_test": float(proba_diff.mean()),
        "proba_max_abs_diff_test": float(proba_diff.max()),
        "param_mae": float(param_diff.mean()),
        "param_max_abs_diff": float(param_diff.max()),
        "rust_fastpath": fastpath, "sklearn_solver": sk_solver,
        "lr_params": json.dumps(p, sort_keys=True),
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

    # `lr_param_sets` (a list) sweeps several initialisations; `lr_params` (a single
    # dict) is the one-configuration speed-run form.
    param_sets = cfg.get("lr_param_sets") or [cfg["lr_params"]]

    # `.name` strips any directory the config's output_csv may carry, so runs
    # cannot scatter CSVs across the repo.
    out_dir = script_dir / RESULTS_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / Path(cfg["output_csv"]).name
    print(f"[bench] config {cfg_path} -> {out_csv}", flush=True)

    # Fresh CSV with header; rows appended (and flushed) as each cell finishes.
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        f.flush()
        for n_rows in cfg["rows"]:
            for n_cols in cfg["cols"]:
                for p in param_sets:
                    name = p.get("name", "default")
                    print(f"[bench] rows={n_rows} cols={n_cols} params={name} ...", flush=True)
                    row = run_cell(n_rows, n_cols, cfg, p)
                    writer.writerow(row)
                    f.flush()
                    flag = "" if row["rust_fastpath"] else "  (FELL BACK!)"
                    print("       ", end="")
                    if measure_time:
                        print(
                            f" fit x{row['fit_speedup']:.2f}  "
                            f"predict x{row['predict_speedup']:.2f}  "
                            f"proba x{row['predict_proba_speedup']:.2f}",
                            end="", flush=True,
                        )
                    print(
                        f" acc train rust={row['rust_acc_train']:.4f}/"
                        f"sk={row['sklearn_acc_train']:.4f}  "
                        f"test rust={row['rust_acc_test']:.4f}/"
                        f"sk={row['sklearn_acc_test']:.4f}  "
                        f"kl_max={row['kl_max_test']:.2e}  "
                        f"param_mae={row['param_mae']:.2e}{flag}",
                        flush=True,
                    )

    print(f"[bench] done -> {out_csv}")


if __name__ == "__main__":
    main()
