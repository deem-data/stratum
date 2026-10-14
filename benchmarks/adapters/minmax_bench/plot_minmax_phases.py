import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).parent

df = pd.read_csv(HERE / "results" / "minmax_phases.csv")

only_big = "--big" in sys.argv
args = [a for a in sys.argv[1:] if a != "--big"]
out_suffix = args[0] if args else "all"

UNIT = "ms"
n_jobs_vals = sorted(df["n_jobs"].unique())
row_vals = [10_000_000] if only_big else sorted(df["n_rows"].unique())

GRID_COLS = 1
GRID_ROWS = len(row_vals)

fig, axes = plt.subplots(
    GRID_ROWS,
    GRID_COLS,
    figsize=(6, 3.4 * GRID_ROWS),
    squeeze=False,
)

phase_colors = {
    "FFI/materialize": "#C4C4C4",
    "fit": "#f47e7e",
    "transform": "#c1272d",
}

x = np.arange(len(n_jobs_vals))
bar_width = 0.5

for i, n_rows in enumerate(row_vals):
    ax = axes[i][0]
    sub = df[df["n_rows"] == n_rows]

    ffi_times, fit_times, transform_times = [], [], []
    for n_jobs in n_jobs_vals:
        row = sub[sub.n_jobs == n_jobs]
        ffi_times.append(row.ffi_ms.iloc[0] if not row.empty else float("nan"))
        fit_times.append(row.fit_ms.iloc[0] if not row.empty else float("nan"))
        transform_times.append(row.transform_ms.iloc[0] if not row.empty else float("nan"))

    ax.bar(
        x,
        ffi_times,
        bar_width,
        color=phase_colors["FFI/materialize"],
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        x,
        fit_times,
        bar_width,
        bottom=ffi_times,
        color=phase_colors["fit"],
        edgecolor="black",
        linewidth=0.4,
    )
    transform_bottom = [f + m for f, m in zip(ffi_times, fit_times)]
    ax.bar(
        x,
        transform_times,
        bar_width,
        bottom=transform_bottom,
        color=phase_colors["transform"],
        edgecolor="black",
        linewidth=0.4,
    )

    totals = [f + m + r for f, m, r in zip(ffi_times, fit_times, transform_times)]
    for xi, tot in zip(x, totals):
        ax.annotate(
            f"{tot:.0f}ms",
            xy=(xi, tot),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(n_jobs_vals)
    if not only_big:
        ax.set_title(f"n_rows = {n_rows:,}")
    ax.set_xlabel("n_jobs")
    ax.set_ylabel(f"time ({UNIT})")
    ax.grid(True, axis="y", ls=":", alpha=0.5)

legend_handles = [Patch(color=phase_colors[p], label=p) for p in ("FFI/materialize", "fit", "transform")]
fig.legend(
    handles=legend_handles,
    title="phase",
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
)
fig.tight_layout()
(HERE / "plots").mkdir(parents=True, exist_ok=True)
fig.savefig(
    HERE / "plots" / f"minmax_phases_{out_suffix}.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()
