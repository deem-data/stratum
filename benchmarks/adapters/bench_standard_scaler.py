import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler as SKStandardScaler

from stratum._config import config
from stratum.adapters.standard_scaler import NumpyStandardScaler, RustyStandardScaler


data_cache: dict[tuple[int, int, int], np.ndarray] = {}
multi_data_cache: dict[tuple[int, int, int, int], list[np.ndarray]] = {}


def _format_compact_number(value: int) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        scaled = value / 1_000_000
        suffix = "M"
    elif abs_value >= 1_000:
        scaled = value / 1_000
        suffix = "K"
    else:
        return str(value)

    if float(scaled).is_integer():
        return f"{int(scaled)}{suffix}"
    return f"{scaled:.1f}".rstrip("0").rstrip(".") + suffix


def _get_data(*, n_rows: int, n_cols: int, seed: int) -> np.ndarray:
    key = (n_rows, n_cols, seed)
    if key not in data_cache:
        rng = np.random.default_rng(seed)
        data_cache[key] = rng.standard_normal(size=(n_rows, n_cols), dtype=np.float32)
    return data_cache[key]


def _get_multi_data(
    *,
    n_rows: int,
    n_cols: int,
    n_scalers: int,
    seed: int,
) -> list[np.ndarray]:
    key = (n_rows, n_cols, n_scalers, seed)
    if key not in multi_data_cache:
        rng = np.random.default_rng(seed)
        multi_data_cache[key] = [
            rng.standard_normal(size=(n_rows, n_cols), dtype=np.float32)
            for _ in range(n_scalers)
        ]
    return multi_data_cache[key]


def _time_fit_transform(scaler, X: np.ndarray, ctx) -> tuple[float, float]:
    with ctx:
        t0 = time.perf_counter()
        scaler.fit(X)
        t1 = time.perf_counter()
        fit_time = t1 - t0

        t0 = time.perf_counter()
        scaler.transform(X)
        t1 = time.perf_counter()
        transform_time = t1 - t0
    return fit_time, transform_time


def _aggregate(df: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    return (
        df.group_by(keys)
        .agg(
            pl.col("fit_time_s").mean().alias("fit_time_s"),
            pl.col("transform_time_s").mean().alias("transform_time_s"),
            pl.col("time_s").mean().alias("time_s"),
        )
        .sort(keys)
    )


def run_compare_backends(
    *,
    n_rows_list: list[int],
    n_cols_list: list[int],
    reps: int,
    seed: int,
) -> pl.DataFrame:
    _ = _time_fit_transform(
        RustyStandardScaler(copy=True),
        _get_data(n_rows=100, n_cols=10, seed=seed),
        config(rust_backend=True, allow_patch=True, debug_timing=False),
    )

    rows: list[dict] = []
    for n_rows_k, n_cols in product(n_rows_list, n_cols_list):
        n_rows = n_rows_k * 1000
        print(f"[compare] n_rows={n_rows_k}k n_cols={n_cols}")
        for rep in range(reps):
            X = _get_data(n_rows=n_rows, n_cols=n_cols, seed=seed + rep)
            cases = (
                ("sklearn", SKStandardScaler(copy=True), nullcontext()),
                ("rust", RustyStandardScaler(copy=True), config(rust_backend=True, allow_patch=True, debug_timing=False)),
                ("numpy_copy_true", NumpyStandardScaler(copy=True), nullcontext()),
                ("numpy_copy_false", NumpyStandardScaler(copy=False), nullcontext()),
            )
            for backend, scaler, ctx in cases:
                fit_time, transform_time = _time_fit_transform(scaler, X, ctx)
                rows.append(
                    {
                        "mode": "compare",
                        "backend": backend,
                        "n_rows": n_rows,
                        "n_cols": n_cols,
                        "rep": rep,
                        "fit_time_s": fit_time,
                        "transform_time_s": transform_time,
                        "time_s": None,
                    }
                )
    return _aggregate(pl.DataFrame(rows), ["mode", "backend", "n_rows", "n_cols"])


def run_n_jobs(
    *,
    n_rows_list: list[int],
    n_cols_list: list[int],
    n_jobs_list: list[int],
    reps: int,
    seed: int,
) -> pl.DataFrame:
    _ = _time_fit_transform(
        RustyStandardScaler(copy=True, n_jobs=max(n_jobs_list)),
        _get_data(n_rows=100, n_cols=10, seed=seed),
        config(rust_backend=True, allow_patch=True, debug_timing=False),
    )

    rows: list[dict] = []
    for n_rows_k, n_cols, n_jobs in product(n_rows_list, n_cols_list, n_jobs_list):
        n_rows = n_rows_k * 1000
        print(f"[n_jobs] n_rows={n_rows_k}k n_cols={n_cols} n_jobs={n_jobs}")
        for rep in range(reps):
            X = _get_data(n_rows=n_rows, n_cols=n_cols, seed=seed + rep)
            scaler = RustyStandardScaler(copy=True, n_jobs=n_jobs)
            fit_time, transform_time = _time_fit_transform(
                scaler,
                X,
                config(rust_backend=True, allow_patch=True, debug_timing=False),
            )
            rows.append(
                {
                    "mode": "n_jobs",
                    "backend": "rust",
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "n_jobs": n_jobs,
                    "rep": rep,
                    "fit_time_s": fit_time,
                    "transform_time_s": transform_time,
                    "time_s": None,
                }
            )
    return _aggregate(pl.DataFrame(rows), ["mode", "backend", "n_rows", "n_cols", "n_jobs"])


def run_parallel_scalers(
    *,
    n_rows_list: list[int],
    n_cols: int,
    n_scalers_list: list[int],
    reps: int,
    seed: int,
) -> pl.DataFrame:
    _ = _get_multi_data(n_rows=1000, n_cols=min(n_cols, 10), n_scalers=1, seed=seed)
    cores = os.cpu_count() or 1

    rows: list[dict] = []
    for n_rows_k, n_scalers in product(n_rows_list, n_scalers_list):
        n_rows = n_rows_k * 1000
        cores_per_scaler = max(1, cores // max(1, n_scalers))
        print(f"[parallel] n_rows={n_rows_k}k n_cols={n_cols} n_scalers={n_scalers}")

        for rep in range(reps):
            data = _get_multi_data(
                n_rows=n_rows,
                n_cols=n_cols,
                n_scalers=n_scalers,
                seed=seed + rep,
            )

            # --- sklearn baseline: sequential ---
            t0 = time.perf_counter()
            for X in data:
                SKStandardScaler(copy=True).fit_transform(X)
            t1 = time.perf_counter()
            rows.append(
                {
                    "mode": "parallel_scalers",
                    "backend": "sklearn",
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "n_scalers": n_scalers,
                    "n_jobs": 1,
                    "cores_per_scaler": 1,
                    "rep": rep,
                    "fit_time_s": None,
                    "transform_time_s": None,
                    "time_s": t1 - t0,
                    "parallel_mode": "sequential",
                }
            )

            # --- sklearn baseline: threaded ---
            def _sk_fit_transform(X: np.ndarray) -> np.ndarray:
                return SKStandardScaler(copy=True).fit_transform(X)

            pool_sk = ThreadPoolExecutor(max_workers=min(n_scalers, cores))
            try:
                t0 = time.perf_counter()
                futures = [pool_sk.submit(_sk_fit_transform, X) for X in data]
                _ = [f.result() for f in futures]
                t1 = time.perf_counter()
            finally:
                pool_sk.shutdown(wait=True, cancel_futures=False)
            rows.append(
                {
                    "mode": "parallel_scalers",
                    "backend": "sklearn",
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "n_scalers": n_scalers,
                    "n_jobs": 1,
                    "cores_per_scaler": 1,
                    "rep": rep,
                    "fit_time_s": None,
                    "transform_time_s": None,
                    "time_s": t1 - t0,
                    "parallel_mode": "parallel",
                }
            )

            # --- numpy (copy=True) baseline: sequential ---
            t0 = time.perf_counter()
            for X in data:
                NumpyStandardScaler(copy=True).fit_transform(X)
            t1 = time.perf_counter()
            rows.append(
                {
                    "mode": "parallel_scalers",
                    "backend": "numpy",
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "n_scalers": n_scalers,
                    "n_jobs": 1,
                    "cores_per_scaler": 1,
                    "rep": rep,
                    "fit_time_s": None,
                    "transform_time_s": None,
                    "time_s": t1 - t0,
                    "parallel_mode": "sequential",
                }
            )

            # --- numpy (copy=True) baseline: threaded ---
            def _np_fit_transform(X: np.ndarray) -> np.ndarray:
                return NumpyStandardScaler(copy=True).fit_transform(X)

            pool_np = ThreadPoolExecutor(max_workers=min(n_scalers, cores))
            try:
                t0 = time.perf_counter()
                futures = [pool_np.submit(_np_fit_transform, X) for X in data]
                _ = [f.result() for f in futures]
                t1 = time.perf_counter()
            finally:
                pool_np.shutdown(wait=True, cancel_futures=False)
            rows.append(
                {
                    "mode": "parallel_scalers",
                    "backend": "numpy",
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "n_scalers": n_scalers,
                    "n_jobs": 1,
                    "cores_per_scaler": 1,
                    "rep": rep,
                    "fit_time_s": None,
                    "transform_time_s": None,
                    "time_s": t1 - t0,
                    "parallel_mode": "parallel",
                }
            )

            # --- rust: sequential ---
            with config(rust_backend=True, allow_patch=True, debug_timing=False):
                t0 = time.perf_counter()
                for X in data:
                    RustyStandardScaler(copy=True, n_jobs=cores).fit_transform(X)
                t1 = time.perf_counter()
                rows.append(
                    {
                        "mode": "parallel_scalers",
                        "backend": "rust",
                        "n_rows": n_rows,
                        "n_cols": n_cols,
                        "n_scalers": n_scalers,
                        "n_jobs": cores,
                        "cores_per_scaler": cores,
                        "rep": rep,
                        "fit_time_s": None,
                        "transform_time_s": None,
                        "time_s": t1 - t0,
                        "parallel_mode": "sequential",
                    }
                )

                def _fit_one(X: np.ndarray) -> RustyStandardScaler:
                    return RustyStandardScaler(copy=True, n_jobs=cores_per_scaler).fit(X)

                def _transform_one(X: np.ndarray, scaler: RustyStandardScaler) -> np.ndarray:
                    return scaler.transform(X)

                pool = ThreadPoolExecutor(max_workers=min(n_scalers, cores))
                try:
                    t0 = time.perf_counter()
                    futures = [pool.submit(_fit_one, X) for X in data]
                    scalers = [future.result() for future in futures]
                    futures = [pool.submit(_transform_one, X, scalers[i]) for i, X in enumerate(data)]
                    _ = [future.result() for future in futures]
                    t1 = time.perf_counter()
                finally:
                    pool.shutdown(wait=True, cancel_futures=False)

                rows.append(
                    {
                        "mode": "parallel_scalers",
                        "backend": "rust",
                        "n_rows": n_rows,
                        "n_cols": n_cols,
                        "n_scalers": n_scalers,
                        "n_jobs": cores_per_scaler,
                        "cores_per_scaler": cores_per_scaler,
                        "rep": rep,
                        "fit_time_s": None,
                        "transform_time_s": None,
                        "time_s": t1 - t0,
                        "parallel_mode": "parallel",
                    }
                )

    return (
        pl.DataFrame(rows)
        .group_by(
            [
                "mode",
                "backend",
                "parallel_mode",
                "n_rows",
                "n_cols",
                "n_scalers",
                "n_jobs",
                "cores_per_scaler",
            ]
        )
        .agg(pl.col("time_s").mean().alias("time_s"))
        .sort(["mode", "n_rows", "n_scalers", "backend", "parallel_mode"])
    )


def plot_fit_transform(df: pl.DataFrame, *, title: str, pdf_path: str, png: bool) -> None:
    n_rows_values = sorted(df["n_rows"].unique().to_list())
    group_values = (
        sorted(df["backend"].unique().to_list())
        if "backend" in df.columns and "n_jobs" not in df.columns
        else sorted(df["n_jobs"].unique().to_list())
    )
    by_backend = "n_jobs" not in df.columns

    metric_colors = {"fit_time_s": "#1f77b4", "transform_time_s": "#2ca02c"}
    hatch_patterns = ["///", "\\\\\\", "xxx", "...", "+++", "|||", "ooo"]
    hatches = {v: hatch_patterns[i % len(hatch_patterns)] for i, v in enumerate(group_values)}

    fig, axes = plt.subplots(nrows=len(n_rows_values), ncols=1, figsize=(6, 2.5 * len(n_rows_values)), sharex=True)
    if len(n_rows_values) == 1:
        axes = [axes]

    bar_width = 0.24
    for ax, n_rows in zip(axes, n_rows_values):
        df_nr = df.filter(pl.col("n_rows") == n_rows)
        n_cols_values = sorted(df_nr["n_cols"].unique().to_list())
        x_base = np.arange(len(n_cols_values)) * (bar_width * (len(group_values) + 1))

        for i, g in enumerate(group_values):
            dfg = df_nr.filter(pl.col("backend") == g) if by_backend else df_nr.filter(pl.col("n_jobs") == g)
            dfg = (
                dfg.sort("n_cols")
                .join(pl.DataFrame({"n_cols": n_cols_values}), on="n_cols", how="right")
                .with_columns(
                    pl.col("fit_time_s").fill_null(0.0),
                    pl.col("transform_time_s").fill_null(0.0),
                )
            )
            offset = (i - (len(group_values) - 1) / 2) * bar_width
            fit = dfg["fit_time_s"]
            transform = dfg["transform_time_s"]
            ax.bar(x_base + offset, fit, width=bar_width, color=metric_colors["fit_time_s"], hatch=hatches[g], edgecolor="black")
            ax.bar(x_base + offset, transform, width=bar_width, bottom=fit, color=metric_colors["transform_time_s"], hatch=hatches[g], edgecolor="black")

        ax.set_title(f"n_rows = {_format_compact_number(n_rows)}")
        ax.set_ylabel("time [s]")
        ax.set_xticks(x_base, [_format_compact_number(v) for v in n_cols_values])

    axes[-1].set_xlabel("n_cols")
    metric_patches = [
        Patch(facecolor=metric_colors["fit_time_s"], edgecolor="black", label="fit"),
        Patch(facecolor=metric_colors["transform_time_s"], edgecolor="black", label="transform"),
    ]
    group_label = "backend" if by_backend else "n_jobs"
    group_patches = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=hatches[g],
            label=(
                f"{group_label}={g}"
                if by_backend
                else f"{group_label}={_format_compact_number(g)}"
            ),
        )
        for g in group_values
    ]
    fig.legend(handles=metric_patches + group_patches, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, title="metrics / grouping")
    fig.suptitle(title, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(pdf_path.replace(".pdf", ".png") if png else pdf_path)


def plot_parallel_scalers(df: pl.DataFrame, *, pdf_path: str, png: bool) -> None:
    n_rows_values = sorted(df["n_rows"].unique().to_list())
    n_scalers_values = sorted(df["n_scalers"].unique().to_list())
    backends = sorted(df["backend"].unique().to_list())
    parallel_modes = ["sequential", "parallel"]

    combo_labels = [(b, m) for b in backends for m in parallel_modes]
    n_combos = len(combo_labels)

    backend_colors = {"sklearn": "#d62728", "numpy": "#ff7f0e", "rust": "#1f77b4"}
    mode_hatches = {"sequential": "", "parallel": "///"}

    fig, axes = plt.subplots(
        nrows=len(n_rows_values),
        ncols=1,
        figsize=(0.4 * len(n_scalers_values) * n_combos, 3.0 * len(n_rows_values)),
        sharex=True,
    )
    if len(n_rows_values) == 1:
        axes = [axes]

    bar_width = 0.05
    group_width = bar_width * (n_combos + 1.0)

    for ax, n_rows in zip(axes, n_rows_values):
        df_nr = df.filter(pl.col("n_rows") == n_rows)
        x_base = np.arange(len(n_scalers_values)) * group_width

        for i, (backend, pmode) in enumerate(combo_labels):
            y_vals = []
            for n_scalers in n_scalers_values:
                row = df_nr.filter(
                    (pl.col("n_scalers") == n_scalers)
                    & (pl.col("backend") == backend)
                    & (pl.col("parallel_mode") == pmode)
                )
                y_vals.append(float(row["time_s"][0]) if row.shape[0] else 0.0)
            offset = (i - (n_combos - 1) / 2) * bar_width
            ax.bar(
                x_base + offset,
                y_vals,
                width=bar_width,
                color=backend_colors.get(backend, "#999999"),
                hatch=mode_hatches[pmode],
                edgecolor="black",
            )

        ax.set_title(f"n_rows = {_format_compact_number(n_rows)}")
        ax.set_ylabel("time [s]")
        ax.set_xticks(x_base, [_format_compact_number(v) for v in n_scalers_values])

    axes[-1].set_xlabel("n_scalers")
    legend_patches = [
        Patch(
            facecolor=backend_colors.get(b, "#999999"),
            edgecolor="black",
            hatch=mode_hatches[m],
            label=f"{b} ({m})",
        )
        for b, m in combo_labels
    ]
    fig.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        title="backend (mode)",
    )
    fig.suptitle("Parallel scalers benchmark", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.80])
    plt.savefig(pdf_path.replace(".pdf", ".png") if png else pdf_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified benchmark for StandardScaler variants (compare backends, n_jobs, and parallel scalers)."
    )
    parser.add_argument("--run_compare", action="store_true", help="Run backend comparison benchmark")
    parser.add_argument("--run_n_jobs", action="store_true", help="Run Rust n_jobs sweep benchmark")
    parser.add_argument("--run_parallel", action="store_true", help="Run parallel-scalers benchmark")
    parser.add_argument("--n_rows", type=int, nargs="+", required=True, help="Row counts in thousands, e.g. --n_rows 10 100")
    parser.add_argument("--n_cols", type=int, nargs="+", required=True, help="Column counts for compare/n_jobs, e.g. --n_cols 10 50")
    parser.add_argument("--n_jobs", type=int, nargs="+", default=[1], help="n_jobs list for --run_n_jobs")
    parser.add_argument("--n_scalers", type=int, nargs="+", default=[1, 2, 4], help="n_scalers list for --run_parallel")
    parser.add_argument("--parallel_n_cols", type=int, default=20, help="Single n_cols for --run_parallel")
    parser.add_argument("--reps", type=int, default=3, help="Repetitions per benchmark point")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed")
    parser.add_argument("--no_plot", action="store_false", dest="plot", help="Disable plotting")
    parser.add_argument("--png", action="store_true", help="Save plots as PNG (default: PDF)")
    parser.add_argument("--plot_only", action="store_true", help="Plot only (default: run benchmarks and plot)")

    args = parser.parse_args()

    if not (args.run_compare or args.run_n_jobs or args.run_parallel):
        parser.error("At least one mode must be selected: --run_compare, --run_n_jobs, --run_parallel")

    if args.run_compare:
        csv_path = "standard_scaler_benchmark.csv"
        if not args.plot_only:
            df_compare = run_compare_backends(
                n_rows_list=args.n_rows,
                n_cols_list=args.n_cols,
                reps=args.reps,
                seed=args.seed,
            )
            df_compare.write_csv(csv_path)
            print(f"Wrote compare benchmark results to {csv_path}")
        else:
            df_compare = pl.read_csv(csv_path)
        print(df_compare.show(limit=df_compare.shape[0]))
        if args.plot:
            plot_fit_transform(
                df_compare,
                title="StandardScaler fit + transform times vs n_cols",
                pdf_path="standard_scaler_benchmark_rows.pdf",
                png=args.png,
            )

    if args.run_n_jobs:
        csv_path = "standard_scaler_n_jobs_benchmark.csv"
        if not args.plot_only:
            df_n_jobs = run_n_jobs(
                n_rows_list=args.n_rows,
                n_cols_list=args.n_cols,
                n_jobs_list=args.n_jobs,
                reps=args.reps,
                seed=args.seed,
            )
            df_n_jobs.write_csv(csv_path)
            print(f"Wrote n_jobs benchmark results to {csv_path}")
        else:
            df_n_jobs = pl.read_csv(csv_path)
        print(df_n_jobs.show(limit=df_n_jobs.shape[0]))
        if args.plot:
            plot_fit_transform(
                df_n_jobs,
                title="Rust StandardScaler fit + transform times vs n_cols",
                pdf_path="standard_scaler_n_jobs_benchmark_rows.pdf",
                png=args.png,
            )

    if args.run_parallel:
        csv_path = "parallel_scalers_benchmark.csv"
        if not args.plot_only:
            df_parallel = run_parallel_scalers(
                n_rows_list=args.n_rows,
                n_cols=args.parallel_n_cols,
                n_scalers_list=args.n_scalers,
                reps=args.reps,
                seed=args.seed,
            )
            
            df_parallel.write_csv(csv_path)
        else:
            df_parallel = pl.read_csv(csv_path)
        print(f"Wrote parallel benchmark results to {csv_path}")
        print(df_parallel.show(limit=df_parallel.shape[0]))
        if args.plot:
            plot_parallel_scalers(
                df_parallel,
                pdf_path="parallel_scalers_benchmark_rows.pdf",
                png=args.png,
            )


if __name__ == "__main__":
    main()

