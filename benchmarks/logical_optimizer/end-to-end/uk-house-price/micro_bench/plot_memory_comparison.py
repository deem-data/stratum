"""Plot memory usage comparison: stratum vs skrub vs base (2 processes).

Produces one figure per dataset size, with a rolling-average overlay to
smooth the fine-grained sampling noise.
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

DIR = "."

CONFIGS = [("1M",{"stratum": f"{DIR}/memory_usage_stratum_1M.csv",
        "skrub": f"{DIR}/memory_usage_skrub_1M.csv",
        "base": f"{DIR}/memory_usage_base2_1M.csv",
    }, 10, 105, "upper left",1.0),
    ("5M",{
        "stratum": f"{DIR}/memory_usage_stratum_5M.csv",
        "skrub": f"{DIR}/memory_usage_skrub_5M.csv",
        "base": f"{DIR}/memory_usage_base2_5M.csv",
    }, 70, 270, "upper right", 1.1),
]

COLORS = {
    "stratum": "#2196F3",
    "skrub": "#4CAF50",
    "base": "#FF9800",
}

OOM_FILES = {f"{DIR}/memory_usage_base2_5M.csv"}

ROLLING_WINDOW = 50


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time_sec"] = df["time_sec"] - df["time_sec"].iloc[0]
    return df


def gb(mb: float) -> float:
    return mb / 1024


parser = argparse.ArgumentParser()
parser.add_argument(
    "-o", "--output",
    help="Output prefix (e.g. 'mem' → mem_1M.pdf, mem_5M.pdf). "
         "Omit to show interactively.",
)
args = parser.parse_args()

for size, files, window, ylim, lloc, margin in CONFIGS:
    fig, ax = plt.subplots(figsize=(7, 2.5))
    t_max = 0
    for label, path in files.items():
        df = load(path)
        mem_gb = df["rss_mb"].apply(gb)
        t = df["time_sec"]
        t_max = max(t_max, t.max())
        #ax.plot(t, mem_gb, color=COLORS[label], alpha=0.20, linewidth=0.8)
        is_oom = path in OOM_FILES
        smoothed = mem_gb.rolling(window, center=True).mean()
        if is_oom:
            smoothed[0] = 0.0
            smoothed[len(smoothed)-1] = 250.0
        linestyle = "solid" #"--" if is_oom else "solid"
        ax.plot(t, smoothed, label=label, color=COLORS[label], linewidth=1.0, linestyle=linestyle)

        if is_oom:
            peak_idx = df["rss_mb"].idxmax()
            ax.plot(t.iloc[peak_idx], gb(df["rss_mb"].iloc[peak_idx]),
                    marker="x", color="red", markersize=5,
                    markeredgewidth=2, zorder=5)
            ax.annotate("OOM",
                        (t.iloc[peak_idx], gb(df["rss_mb"].iloc[peak_idx])),
                        textcoords="offset points", xytext=(-10, -15),
                        fontsize=10, fontweight="bold", color="red")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Memory (GB)")
    ax.legend(loc=lloc, fontsize=8)
    ax.set_ylim(0, ylim)
    #ax.margins(x=0.1)
    ax.set_xlim(left=0, right=t_max*margin)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    fig.tight_layout()

    if args.output:
        out = f"{args.output}_{size}.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved {out}")
    else:
        plt.show()
