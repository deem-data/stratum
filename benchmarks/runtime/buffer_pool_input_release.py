import os
import sys

from sklearn.model_selection import ShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import stratum as skrub
import numpy as np
import pandas as pd
from time import sleep, perf_counter
from sklearn.dummy import DummyRegressor
from utils.memory_consumption_tracker import MemoryTracker
import matplotlib.pyplot as plt
import argparse
import logging
import polars as pl
logging.basicConfig(level=logging.INFO)
logging.getLogger("stratum").setLevel(logging.DEBUG)

def dummy_func(x, t: float=0.1):
    indices = np.arange(len(x))
    if isinstance(x, pl.DataFrame) or isinstance(x, pl.Series):
        out = x[indices]
    else:
        out = x.iloc[indices]
    sleep(t)
    return out

def main(use_skrub: bool, polars: bool):
    tracker = MemoryTracker(mode="process", interval_sec=0.02)
    tracker.start()
    with skrub.config_context(eager_data_ops=False):
        n = 30_000_000
        if not os.path.exists(f"input_{n}.csv"):
            cols = ["a", "b", "c", "d", "e", "f", "g", "h", "y"]
            df = pl.DataFrame({col: np.random.random(n) for col in cols})

            print(f"Memory usage: {df.estimated_size() / 1024**2} MB")

            df.write_csv(f"input_{n}.csv")
            del df
            print("CSV written")

        df = skrub.as_data_op(f"input_{n}.csv").skb.apply_func(pd.read_csv)
        X = df.drop("y", axis=1).skb.mark_as_X()
        y = df["y"]
        y = y.skb.apply_func(dummy_func, t=1.0).skb.mark_as_y()

        for _ in range(10):
            X = X.skb.apply_func(dummy_func, t=0.3)
        model = DummyRegressor()

        pred = X.skb.apply(model, y=y)
        cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        t0 = perf_counter()
        try:
            with skrub.config(scheduler=not use_skrub, stats=True, DEBUG=False, force_polars=polars):
                search = pred.skb.make_grid_search(cv=cv, n_jobs=1, scoring="r2", fitted=True, refit=False)
        finally:
            samples = tracker.stop()
        t1 = perf_counter()

    csv_path = os.path.join(os.path.dirname(__file__), f"memory_usage_{'skrub' if use_skrub else 'stratum'}_{'polars' if polars else 'pandas'}.csv")
    tracker.write_csv(csv_path)

    print(f"Time taken: {t1 - t0:.2f}s")
    print(search.results_)
    plot_memory(csv_path)


def plot_memory(csv_path: str):
    data = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data["time_sec"], data["rss_mb"], linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RSS (MB)")
    ax.set_title("Buffer Pool Benchmark - Memory Usage")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = csv_path.replace(".csv", ".pdf")
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skrub", action="store_true")
    parser.add_argument("--polars", action="store_true")
    args = parser.parse_args()
    main(use_skrub=args.skrub, polars=args.polars)
