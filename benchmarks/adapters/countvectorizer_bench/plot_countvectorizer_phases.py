import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).parent

df = pd.read_csv(HERE / "results" / "countvectorizer_phases.csv")

only_1m = "--1m" in sys.argv
args = [a for a in sys.argv[1:] if a != "--1m"]
out_suffix = args[0] if args else "all"

UNIT = "ms"
n_jobs_vals = sorted(df["n_jobs"].unique())
row_vals = [1_000_000] if only_1m else sorted(df["dataset_length"].unique())

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
    "map": "#f47e7e",
    "reduce": "#c1272d",
}

x = np.arange(len(n_jobs_vals))
bar_width = 0.5

for i, dataset_length in enumerate(row_vals):
    ax = axes[i][0]
    sub = df[df["dataset_length"] == dataset_length]

    ffi_times, map_times, reduce_times = [], [], []
    for n_jobs in n_jobs_vals:
        row = sub[sub.n_jobs == n_jobs]
        ffi_times.append(row.ffi_ms.iloc[0] if not row.empty else float("nan"))
        map_times.append(row.map_ms.iloc[0] if not row.empty else float("nan"))
        reduce_times.append(row.reduce_ms.iloc[0] if not row.empty else float("nan"))

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
        map_times,
        bar_width,
        bottom=ffi_times,
        color=phase_colors["map"],
        edgecolor="black",
        linewidth=0.4,
    )
    reduce_bottom = [f + m for f, m in zip(ffi_times, map_times)]
    ax.bar(
        x,
        reduce_times,
        bar_width,
        bottom=reduce_bottom,
        color=phase_colors["reduce"],
        edgecolor="black",
        linewidth=0.4,
    )

    totals = [f + m + r for f, m, r in zip(ffi_times, map_times, reduce_times)]
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
    if not only_1m:
        ax.set_title(f"dataset_length = {dataset_length:,}")
    ax.set_xlabel("n_jobs")
    ax.set_ylabel(f"time ({UNIT})")
    ax.grid(True, axis="y", ls=":", alpha=0.5)

legend_handles = [Patch(color=phase_colors[p], label=p) for p in ("FFI/materialize", "map", "reduce")]
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
    HERE / "plots" / f"countvectorizer_phases_{out_suffix}.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()
