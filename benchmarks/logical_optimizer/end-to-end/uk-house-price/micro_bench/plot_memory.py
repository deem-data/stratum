"""Plot memory usage from memory_usage.csv (produced by gc_scheduler.py)."""
import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input", nargs="?", default="memory_usage.csv", help="CSV from gc_scheduler (default: memory_usage.csv)")
parser.add_argument("-o", "--output", help="Save figure to path")
args = parser.parse_args()

df = pd.read_csv(args.input)
df.plot(x="time_sec", y="rss_mb", legend=False)
plt.xlabel("Time (s)")
plt.ylabel("RSS (MB)")
plt.title("Process memory usage")
plt.tight_layout()
if args.output:
    plt.savefig(args.output)
else:
    plt.show()
