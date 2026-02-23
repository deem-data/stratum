"""Stacked vertical bar plot comparing per-Op time breakdown: Polars vs Pandas.

Usage:
    python plot_heavy_hitters.py --sizes "10K"
    python plot_heavy_hitters.py --sizes "10K 100K"
    python plot_heavy_hitters.py --sizes "10K 100K" -o heavy_hitters
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

DIR = Path(__file__).parent
MIN_FRAC = 0.01

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
]


def op_color_map(ops: list[str]) -> dict[str, str]:
    return {op: PALETTE[i % len(PALETTE)] for i, op in enumerate(ops)}


RENAME_MAP = {
    "read_csv": "ReadCSV",
}


def shorten_op(name: str) -> str:
    """Extract the inner name from wrapper ops like EstimatorOp(LGBMRegressor)."""
    m = re.match(r"\w+Op\(([^)]+)\)", name)
    if m:
        name = m.group(1)
    else:
        name = re.sub(r"\s*\[df\]$", "", name)
        m = re.match(r"(\w+Op)(?:\((.+)\))?", name)
        if m:
            name = m.group(2) if m.group(2) else m.group(1)

    if name in RENAME_MAP:
        return RENAME_MAP[name]
    m = re.match(r"<Concat:\s.*>", name)
    if m:
        return "Concat"
    m = re.match(r"Rusty(\w+)", name)
    if m:
        return m.group(1)
    return name


def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Op"] = df["Op"].apply(shorten_op)

    total = df["Time"].sum()
    threshold = total * MIN_FRAC
    major = df[df["Time"] >= threshold].copy()
    minor = df[df["Time"] < threshold]
    if not minor.empty:
        other = pd.DataFrame(
            {"Op": ["Other Ops"], "Time": [minor["Time"].sum()]}
        )
        major = pd.concat([major, other], ignore_index=True)

    return major.sort_values("Time", ascending=False).reset_index(drop=True)


def collect_ops_and_times(sizes: list[str], version=""):
    """Return (union of op names sorted by max time desc, {size: (polars_df, pandas_df)})."""
    data = {}
    all_ops_set: dict[str, float] = {}

    for size in sizes:
        polars_df = load_and_prepare(DIR / f"heavy_hitters{version}_polars_{size}.csv")
        pandas_df = load_and_prepare(DIR / f"heavy_hitters{version}_pandas_{size}.csv")
        data[size] = (polars_df, pandas_df)
        for op in set(polars_df["Op"]) | set(pandas_df["Op"]):
            t = max(
                polars_df.loc[polars_df["Op"] == op, "Time"].sum(),
                pandas_df.loc[pandas_df["Op"] == op, "Time"].sum(),
            )
            all_ops_set[op] = max(all_ops_set.get(op, 0.0), t)

    ops_sorted = sorted(all_ops_set, key=all_ops_set.get, reverse=True)
    return ops_sorted, data


parser = argparse.ArgumentParser()
parser.add_argument("--sizes", required=True, help='Space-separated sizes, e.g. "10K 100K"')
parser.add_argument("-v", required=False, help='Version, e.g. "_v2"', default="")
args = parser.parse_args()
sizes = args.sizes.split()
version = args.v
ops_sorted, data = collect_ops_and_times(sizes, version)
colors = op_color_map(ops_sorted)

n_sizes = len(sizes)
bar_width = 0.35

fig, axes = plt.subplots(1, n_sizes, figsize=(max(4, n_sizes * 3.2), 5.5),
                          sharey=False)
if n_sizes == 1:
    axes = [axes]

legend_handles = []

for gi, (size, ax) in enumerate(zip(sizes, axes)):
    polars_df, pandas_df = data[size]
    x_polars = 0
    x_pandas = bar_width

    bottom_polars = 0.0
    bottom_pandas = 0.0

    for oi, op in enumerate(ops_sorted):
        tp = polars_df.loc[polars_df["Op"] == op, "Time"].sum()
        td = pandas_df.loc[pandas_df["Op"] == op, "Time"].sum()

        c = colors[op]
        bp = ax.bar(x_polars, tp, bar_width, bottom=bottom_polars,
                    color=c, edgecolor="white", linewidth=0.4)
        bd = ax.bar(x_pandas, td, bar_width, bottom=bottom_pandas,
                    color=c, edgecolor="white", linewidth=0.4)

        if gi == 0:
            legend_handles.append((bp[0], op))

        bottom_polars += tp
        bottom_pandas += td

    for x_pos, total in [(x_polars, bottom_polars), (x_pandas, bottom_pandas)]:
        ax.text(x_pos, total + 0.05, f"{total:.2f}s",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks([x_polars, x_pandas])
    ax.set_xticklabels(["Polars", "Pandas"])
    ax.set_ylim(0, ax.get_ylim()[1] * (1.08 if size != "10K" else 1.3))
    ax.set_title(size)
    if gi == 0:
        ax.set_ylabel("Time (s)")


fig.legend(
    [h[0] for h in legend_handles],
    [h[1] for h in legend_handles],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=min(len(legend_handles), 4),
    fontsize=10,
    title="Op",
    title_fontsize=9,
    frameon=True,
)

fig.tight_layout(rect=[0, 0, 1, 0.85])

fig.savefig("heavy_hitters.pdf", bbox_inches="tight")
print(f"Saved heavy_hitters.pdf")
