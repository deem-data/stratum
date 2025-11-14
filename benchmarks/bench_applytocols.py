import gc
import time
import numpy as np
import pandas as pd
from joblib import parallel_backend

import stratum as skrub
from skrub import ApplyToCols, StringEncoder

# Create a synthetic test column
def make_data (n_rows, seed, vocab_size,
                 avg_words, words_len_range=(3, 10), n_features=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Create a random lowercase word from ascii characters
    def rand_word():
        size = rng.integers(words_len_range[0], words_len_range[1])
        return ''.join(rng.choice(list('abcdefghijklmnopqrstuvwxyz'), size=size)) #use ascii

    # Build a vocabulary of unique words
    vocab = [rand_word() for _ in range(vocab_size)]

    # Function to generate a single text series
    def gen_series():
        # Randomly generate number of words (around avg_words) in each row
        n_per_row = np.maximum(1, rng.poisson(avg_words, size=n_rows))
        rows = []
        for k in n_per_row:
            idx = rng.integers(0, vocab_size, size=k)
            rows.append(' '.join(vocab[i] for i in idx))
        return pd.Series(rows)

    # Generate n_features columns
    data = {f"text_{i+1}": gen_series() for i in range(n_features)}
    return pd.DataFrame(data)

def main():
    n_rows = 100_000        #number of rows (=200K)
    vocab_size = 20000      #number of unique words (=5K). Large -> more distinct tokens -> sparser matrix
    avg_words = 8           #average number of words per row (=8)
    words_len = (3, 10)     #length of each word (low to high)
    n_features = 2          #number of features

    # Generate synthetic data
    print("Generate synthetic data")
    X = make_data(n_rows, 42, vocab_size, avg_words, words_len, n_features)
    print(X)

    # Build encoder
    enc = StringEncoder(
        vectorizer="hashing", #hashing->tfidf
        analyzer="char",
        ngram_range=(3, 4),
        n_components= 30,
        random_state=0
    )

    # Main benchmark. Run on the entire dataset
    print("\nStarting main benchmark")
    skrub.set_config(rust_backend=True, debug_timing=True) #sklearn backend
    t0 = time.perf_counter()
    with parallel_backend('threading'):
        enc_cols = ApplyToCols(enc, n_jobs=n_features) #apply one encoder on all columns
        Z = enc_cols.fit_transform(X)
    t1 = time.perf_counter()
    exec_time = t1 - t0
    print(f"Shape = {Z.shape}")
    print(f"skrub - Execution time = {exec_time:8.3f}s\n")
    del Z #optimize memory, especially for dense outputs
    gc.collect()

if __name__ == '__main__':
    main()
