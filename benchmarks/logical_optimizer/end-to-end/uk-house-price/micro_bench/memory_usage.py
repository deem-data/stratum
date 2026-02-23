"""Plot memory usage comparison: stratum vs skrub vs base (2 processes).

Produces one figure per dataset size, with a rolling-average overlay to
smooth the fine-grained sampling noise.
"""
import argparse
from pathlib import Path
import pandas as pd

DIR = "."

CONFIGS = [("1M",{"Stratum": f"{DIR}/memory_usage_stratum_1M.csv",
        "skrub": f"{DIR}/memory_usage_skrub_1M.csv",
        "Baseline (2 proc)": f"{DIR}/memory_usage_base2_1M.csv",
    }),
    ("5M",{
        "Stratum": f"{DIR}/memory_usage_stratum_5M.csv",
        "skrub": f"{DIR}/memory_usage_skrub_5M.csv",
        "Baseline (2 proc)": f"{DIR}/memory_usage_base2_5M.csv",
    }),
]

COLORS = {
    "Stratum": "#2196F3",
    "skrub": "#4CAF50",
    "Baseline (2 proc)": "#FF9800",
}

OOM_FILES = {f"{DIR}/memory_usage_base2_5M.csv"}


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

for size, files in CONFIGS:
    df_stratum = load(files["Stratum"])
    df_skrub = load(files["skrub"])
    df_base = load(files["Baseline (2 proc)"])

    # duration
    duration_stratum = df_stratum["time_sec"].iloc[-1] - df_stratum["time_sec"].iloc[0]
    duration_skrub = df_skrub["time_sec"].iloc[-1] - df_skrub["time_sec"].iloc[0]
    duration_base = df_base["time_sec"].iloc[-1] - df_base["time_sec"].iloc[0]

    print( "Max memory usage (stratum):", df_stratum["rss_mb"].max())
    print( "Max memory usage (skrub):", df_skrub["rss_mb"].max())
    print( "Max memory usage (base2):", df_base["rss_mb"].max())

    print(f"Duration (stratum): {duration_stratum:.2f}s")
    print(f"Duration (skrub): {duration_skrub:.2f}s")
    print(f"Duration (base2): {duration_base:.2f}s")