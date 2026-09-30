"""Shared helpers for the CountVectorizer macrobenchmark."""
import gc
import random
import string

from sklearn.feature_extraction.text import CountVectorizer as SKCountVectorizer
from stratum import set_config
from stratum.adapters.count_vectorizer import RustyCountVectorizer

SEED = 67


def get_synthetic_data(n_unique_words, max_length, n_sentences, max_sentence_len, seed=SEED):
    rng = random.Random(seed)
    words = [
        "".join(rng.choices(string.ascii_letters, k=rng.choice(range(2, max_length))))
        for _ in range(n_unique_words)
    ]
    sentences = [
        " ".join(rng.choices(words, k=rng.choice(range(1, max_sentence_len)))) + "."
        for _ in range(n_sentences)
    ]
    return sentences

def get_synthetic_data(n_unique_words, max_length, n_sentences, max_sentence_len, seed=SEED):
    rng = random.Random(seed)
    words = [
        "".join(rng.choices(string.ascii_letters, k=rng.choice(range(2, max_length))))
        for _ in range(n_unique_words)
    ]
    sentences = [
        " ".join(rng.choices(words, k=rng.choice(range(1, max_sentence_len)))) + "."
        for _ in range(n_sentences)
    ]
    return sentences


def warmup(n_jobs: int) -> None:
    """Touch both backends once so import/JIT overhead doesn't skew timings.

    Also locks in the Rust thread pool at `n_jobs` threads: it is built once
    per process (see _rust/src/threads.rs) from whatever SKRUB_RUST_THREADS
    is set to on first use, so this must run before any timed rust call.
    """
    data = get_synthetic_data(
        n_unique_words=100, max_length=10, n_sentences=100, max_sentence_len=20
    )

    set_config(rust_backend=False)
    SKCountVectorizer().fit_transform(data)

    set_config(rust_backend=True, num_threads=n_jobs)
    RustyCountVectorizer(n_jobs=n_jobs).fit_transform(data)

    gc.collect()
