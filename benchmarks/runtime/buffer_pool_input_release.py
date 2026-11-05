"""Buffer-pool benchmark; CPU/RSS timelines are saved under runtime/profiles/.

Requires the stratum-ai[hardware] extra. Each run produces a profiler JSON report.
"""
import os
from pathlib import Path

from sklearn.model_selection import ShuffleSplit

import stratum as skrub
import numpy as np
import pandas as pd
from time import sleep, perf_counter
from sklearn.dummy import DummyRegressor
from stratum._profiler import record_run
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

        backend = 'skrub' if use_skrub else 'stratum'
        selector = 'greedy' if polars else 'default'
        diagnostics_dir = Path(__file__).parent / "profiles" / f"input_release_{backend}_{selector}"
        run_search = pred.skb.make_grid_search
        if use_skrub:
            # Skrub bypasses Stratum's API profiling hooks.
            run_search = record_run("benchmark")(run_search)
        t0 = perf_counter()
        with skrub.config(
            scheduler=not use_skrub, stats=True, DEBUG=False,
            implementation_selector=selector,
        ), skrub.dev_config(
            hardware_utilization=True, hardware_sampling_interval=0.02,
            diagnostics_dir=diagnostics_dir,
        ):
            search = run_search(cv=cv, n_jobs=1, scoring="r2", fitted=True, refit=False)
        t1 = perf_counter()

    print(f"Time taken: {t1 - t0:.2f}s")
    print(search.results_)
    print(f"Profiler reports: {diagnostics_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skrub", action="store_true")
    parser.add_argument("--polars", action="store_true")
    args = parser.parse_args()
    main(use_skrub=args.skrub, polars=args.polars)
