import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).parent

df_all = pd.read_csv(HERE / "results" / "macrobenchmark_minmax.csv")
df = df_all[df_all["version"] == "rust"]
df_sklearn = df_all[df_all["version"] == "sklearn"]

only_1m = "--1m" in sys.argv
args = [a for a in sys.argv[1:] if a != "--1m"]
out_suffix = args[0] if args else "all"

UNIT = "ms"
n_jobs_vals = sorted(df["n_jobs"].unique())
col_vals = sorted(df["n_cols"].unique())
row_vals = [1_000_000] if only_1m else [100_000, 1_000_000, 10_000_000]

GRID_COLS = 1
GRID_ROWS = len(row_vals)

fig, axes = plt.subplots(
    GRID_ROWS,
    GRID_COLS,
    figsize=(8, 3.4 * GRID_ROWS),
    squeeze=False,
)

njobs_colors = {
    1: "#C4C4C4",
    2: "#525252",
    4: "#f47e7e",
    8: "#620000",
    24: "#c1272d",
}
sklearn_color = "#444444"

FIT_HATCH = "///"

n_bars = len(n_jobs_vals) + 1
total_width = 0.7
bar_width = total_width / n_bars
offsets = np.linspace(
    -(total_width - bar_width) / 2, (total_width - bar_width) / 2, n_bars
)

for i, n_rows in enumerate(row_vals):
    ax = axes[i][0]
    sub = df[df["n_rows"] == n_rows]
    sub_sklearn = df_sklearn[df_sklearn["n_rows"] == n_rows]
    x = np.arange(len(col_vals))

    sklearn_fit, sklearn_transform = [], []
    for c in col_vals:
        row = sub_sklearn[sub_sklearn.n_cols == c]
        sklearn_fit.append(row.fit_ms.iloc[0] if not row.empty else float("nan"))
        sklearn_transform.append(
            row.transform_ms.iloc[0] if not row.empty else float("nan")
        )
    ax.bar(
        x + offsets[0],
        sklearn_fit,
        bar_width,
        color=sklearn_color,
        hatch=FIT_HATCH,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.bar(
        x + offsets[0],
        sklearn_transform,
        bar_width,
        bottom=sklearn_fit,
        color=sklearn_color,
    )
    sklearn_totals = [f + t for f, t in zip(sklearn_fit, sklearn_transform)]

    for offset, n_jobs in zip(offsets[1:], n_jobs_vals):
        fit_times, transform_times = [], []
        for c in col_vals:
            row = sub[(sub.n_cols == c) & (sub.n_jobs == n_jobs)]
            fit_times.append(row.fit_ms.iloc[0] if not row.empty else float("nan"))
            transform_times.append(
                row.transform_ms.iloc[0] if not row.empty else float("nan")
            )
        color = njobs_colors[n_jobs]
        ax.bar(
            x + offset,
            fit_times,
            bar_width,
            color=color,
            hatch=FIT_HATCH,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.bar(
            x + offset,
            transform_times,
            bar_width,
            bottom=fit_times,
            color=color,
        )
        totals = [f + t for f, t in zip(fit_times, transform_times)]

        if n_jobs in (1, n_jobs_vals[-1]):
            origin_nudge = (-1 if n_jobs == 1 else 0) * 0.015
            for xi_sklearn, xi_rust, tot, st in zip(
                x + offsets[0], x + offset, totals, sklearn_totals
            ):
                origin_y = st * (1 + origin_nudge)
                ax.annotate(
                    "",
                    xy=(xi_rust, tot),
                    xytext=(xi_sklearn, origin_y),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="black",
                        lw=0.8,
                        connectionstyle="angle,angleA=0,angleB=90",
                    ),
                )
                ax.annotate(
                    f"{st / tot:.1f}×",
                    xy=(xi_rust, (origin_y + tot) / 2),
                    xytext=(4, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(col_vals)
    if not only_1m:
        ax.set_title(f"n_rows = {n_rows:,}")
    ax.set_xlabel("n_cols")
    ax.set_ylabel(f"time ({UNIT})")
    ax.grid(True, axis="y", ls=":", alpha=0.5)

legend_handles = [Patch(color=sklearn_color, label="sklearn")]
legend_handles += [Patch(color=njobs_colors[j], label=str(j)) for j in n_jobs_vals]
legend_handles.append(
    Patch(
        facecolor="white",
        edgecolor="black",
        hatch=FIT_HATCH,
        label="fit (transform solid)",
    )
)
if not only_1m:
    addition = 0.05
else:
    addition = 0.12
fig.legend(
    handles=legend_handles,
    title="rust (n_jobs)",
    ncol=len(n_jobs_vals) + 2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1 + addition),
)
fig.tight_layout()
(HERE / "plots").mkdir(parents=True, exist_ok=True)
fig.savefig(
    HERE / "plots" / f"macrobenchmark_minmax_njobs_{out_suffix}.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()
