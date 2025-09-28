"""
This script runs StringEncoder on a synthetic dataset and compares Sklearn and Rust backends.
It is used to show the performance benefits of the Rust backend (~5.8x w/ 24*2 cores).
This script also shows the use various config flags related to the Rust backend.
"""
import gc
import time
import numpy as np
import pandas as pd
import stratum as skrub
from stratum import StringEncoder

# Create a synthetic test column
def make_series (n_rows, seed, vocab_size, avg_words, words_len_range=(3, 10)) -> pd.Series:
    rng = np.random.default_rng(seed)

    # Create a random lowercase word from ascii characters
    def rand_word():
        size = rng.integers(words_len_range[0], words_len_range[1])
        return ''.join(rng.choice(list('abcdefghijklmnopqrstuvwxyz'), size=size)) #use ascii

    # Build a vocabulary of unique words
    vocab = [rand_word() for _ in range(vocab_size)]

    # Randomly generate number of words (around avg_words) in each row
    n_per_row = np.maximum(1, rng.poisson(avg_words, size=n_rows))

    rows = []
    for k in n_per_row:
        idx = rng.integers(0, vocab_size, size=k)
        rows.append(' '.join(vocab[i] for i in idx))

    return pd.Series(rows, name="text")

def main():
    n_rows = 100_000        #number of rows (=200K)
    vocab_size = 20000      #number of unique words (=5K). Large -> more distinct tokens -> sparser matrix
    avg_words = 8           #average number of words per row (=8)
    words_len = (3, 10)     #length of each word (low to high)

    # Generate synthetic data
    print("Generate synthetic data")
    X = make_series(n_rows, 42, vocab_size, avg_words, words_len)
    print(X)

    # Build encoder
    enc = StringEncoder(
        vectorizer="hashing", #hashing->tfidf
        analyzer="char",
        ngram_range=(3, 4),
        n_components= 30,
        random_state=0
    )

    # Warm-up small runs to load code paths, JIT caches inside SciPy, etc.
    skrub.set_config(rust_backend=False) #sklearn backend
    X_small = X.iloc[: min(2048, len(X))]
    _ = enc.fit_transform(X_small)
    gc.collect()
    skrub.set_config(rust_backend=True) #rust backend
    X_small = X.iloc[: min(2048, len(X))]
    _ = enc.fit_transform(X_small)
    gc.collect()

    # Main benchmark: Run on the entire dataset
    print("\nStarting main benchmark")
    skrub.set_config(rust_backend=False) #sklearn
    t0 = time.perf_counter()
    X_enc = enc.fit_transform(X)
    print(f"Shape = {X_enc.shape}")
    t1 = time.perf_counter()
    exec_time = t1 - t0
    print(f"skrub - Execution time = {exec_time:8.3f}s\n")

    skrub.set_config(rust_backend=True, debug_timing=True, num_threads=0) #rust
    t0 = time.perf_counter()
    X_enc = enc.fit_transform(X)
    print(f"Shape = {X_enc.shape}")
    t1 = time.perf_counter()
    exec_time = t1 - t0
    print(f"stratum - Execution time = {exec_time:8.3f}s\n")


if __name__ == '__main__':
    main()
