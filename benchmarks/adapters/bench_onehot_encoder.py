import gc
import time
import numpy as np
import pandas as pd

import stratum as skrub
from stratum import OneHotEncoder

# Create synthetic features
def make_categorical_df(
        n_rows: int,
        n_features: int,
        n_dists: int, #nuber of distinct items in each feature
        seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    data = {}
    #for j, k in enumerate(cardinals):
    for j in range(n_features):
        # Create category pool: e.g., ['c0_000001', ..., 'c0_00xxxx']
        cats = np.array([f"c{j}_{i:06d}" for i in range(n_dists)], dtype=object)
        idx = rng.integers(0, n_dists, size=n_rows)
        col = cats[idx].copy()

        data[f"col{j}"] = col
    return pd.DataFrame(data)

def OHE_benchmark(X, sparse_output):
    # Build one hot encoder
    enc = OneHotEncoder(
        drop="if_binary",
        dtype=np.float32,
        handle_unknown="ignore",
        sparse_output=sparse_output,
    )

    # Warm-up small runs
    skrub.set_config(rust_backend=False)
    X_small = X.iloc[: min(2048, len(X))]
    _ = enc.fit_transform(X_small)
    gc.collect()
    skrub.set_config(rust_backend=True)
    X_small = X.iloc[: min(2048, len(X))]
    _ = enc.fit_transform(X_small)
    gc.collect()

    # Main benchmark. Run on the entire dataset
    print("\nStarting main benchmark")
    skrub.set_config(rust_backend=False, debug_timing=True) #sklearn backend
    t0 = time.perf_counter()
    Z = enc.fit_transform(X)
    t1 = time.perf_counter()
    exec_time = t1 - t0
    print(f"Shape = {Z.shape}")
    print(f"skrub - Execution time = {exec_time:8.3f}s\n")
    del Z #optimize memory, especially for dense outputs
    gc.collect()

    skrub.set_config(rust_backend=True, debug_timing=True, num_threads=0) #rust backend
    t0 = time.perf_counter()
    Z = enc.fit_transform(X)
    t1 = time.perf_counter()
    exec_time = t1 - t0
    print(f"Shape = {Z.shape}")
    print(f"stratum - Execution time = {exec_time:8.3f}s\n")



def main():
    print("Generate synthetic data for sparse output")
    n_rows = 2_000_000
    n_features = 4
    n_dists = 200_000 # num of distinct items in each feature
    X = make_categorical_df(n_rows=n_rows, n_features=n_features, n_dists=n_dists)
    print(X.head(), "\n")
    OHE_benchmark(X, sparse_output=True)

    print("Generate synthetic data for dense output")
    n_rows = 200_000
    n_features = 4
    n_dists = 10_000 # num of distinct items in each feature
    X = make_categorical_df(n_rows=n_rows, n_features=n_features, n_dists=n_dists)
    print(X.head(), "\n")
    OHE_benchmark(X, sparse_output=False)

if __name__ == "__main__":
    main()
