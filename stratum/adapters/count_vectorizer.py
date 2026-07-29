from __future__ import annotations
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer as _SKCountVectorizer
from sklearn.utils.validation import check_is_fitted
from scipy.sparse import csr_matrix
import logging
from .._config import get_config
from .. import _rust_backend as rb

# File-internal config flags
_DEBUG_INFO = False
logger = logging.getLogger(__name__)
MIN_BLOCK_LEN = 1
_DEFAULT_TOKEN_PATTERN = r"(?u)\b\w\w+\b"


def supports_rust_count_vectorizer(estimator) -> tuple[bool, str]:
    """Only sklearn's default CountVectorizer parameter set is supported by the
    Rust kernel: word analyzer, unigrams, forced lowercasing, no vocabulary/df
    pruning. Anything else must fall back to vanilla sklearn."""
    if not isinstance(estimator, _SKCountVectorizer):
        return False, "estimator is not a sklearn CountVectorizer"
    if getattr(estimator, "analyzer", None) != "word":
        return False, "analyzer must be 'word'"
    if getattr(estimator, "tokenizer", None) is not None:
        return False, "custom tokenizer not supported"
    if getattr(estimator, "preprocessor", None) is not None:
        return False, "custom preprocessor not supported"
    if getattr(estimator, "lowercase", True) is not True:
        return False, "lowercase=False not supported"
    if getattr(estimator, "token_pattern", _DEFAULT_TOKEN_PATTERN) != _DEFAULT_TOKEN_PATTERN:
        return False, "custom token_pattern not supported"
    if getattr(estimator, "ngram_range", (1, 1)) != (1, 1):
        return False, "ngram_range must be (1, 1)"
    if getattr(estimator, "max_df", 1.0) != 1.0 or getattr(estimator, "min_df", 1) != 1:
        return False, "max_df/min_df pruning not supported"
    if getattr(estimator, "max_features", None) is not None:
        return False, "max_features not supported"
    if getattr(estimator, "binary", False):
        return False, "binary=True not supported"
    if getattr(estimator, "vocabulary", None) is not None:
        return False, "fixed vocabulary not supported"
    if getattr(estimator, "strip_accents", None) is not None:
        return False, "strip_accents not supported"
    if getattr(estimator, "input", "content") != "content":
        return False, "input must be 'content'"
    return True, ""


class RustyCountVectorizer(_SKCountVectorizer):
    """Drop-in CountVectorizer that prefers the Rust fastpath where supported."""

    def __init__(self, n_jobs=None, **kwargs):
        super().__init__(**kwargs)
        cores = os.cpu_count()
        if n_jobs is None:
            self.n_jobs = cores
        elif n_jobs > cores:
            logger.warning(
                f"n_jobs {n_jobs} > core count {cores}, setting n_jobs to {cores}"
            )
            self.n_jobs = cores
        else:
            self.n_jobs = n_jobs

    def _n_chunks(self, corpus):
        blocks = max(1, len(corpus) // MIN_BLOCK_LEN)
        return min(blocks, self.n_jobs)

    def _stopwords_set(self):
        # just use builtin method to cast to list, otherwise
        sw = self.get_stop_words()
        return set() if sw is None else set(sw)

    def _rust_ready(self, fn_name):
        rc = get_config()
        if not (
            rc.get("allow_patch", False)
            and rc.get("rust_backend", False)
            and rb.HAVE_RUST
            and getattr(rb, fn_name, None) is not None
        ):
            return False
        supported, reason = supports_rust_count_vectorizer(self)
        if not supported:
            logger.debug(f"Rust fastpath not eligible: {reason}")
            return False
        return True

    def fit(self, raw_documents, y=None):
        if not self._rust_ready("count_vectorize_fit"):
            logger.warning("Rust disabled, fallback to scikit for fit")
            return super().fit(raw_documents, y)
        t0 = rb.start_timing()
        corpus = list(raw_documents)
        try:
            vocab = rb.count_vectorize_fit(
                corpus,
                self._stopwords_set(),
                self._n_chunks(corpus),
            )
            if len(vocab) == 0:
                raise ValueError(
                    "empty vocabulary; perhaps the documents only contain stop words"
                )
        except Exception as e:
            logger.warning(f"Rust count_vectorize_fit failed, falling back: {e}")
            return super().fit(raw_documents, y)

        self.vocabulary_ = vocab
        rb.print_timing("count_vectorize_transform", t0)
        return self

    def transform(self, raw_documents):
        if not self._rust_ready("count_vectorize_transform"):
            logger.debug("Rust disabled, fallback to scikit for transform")
            return super().transform(raw_documents)

        check_is_fitted(self)
        t0 = rb.start_timing()
        corpus = list(raw_documents)
        
        try:
            data, indices, indptr = rb.count_vectorize_transform(
                corpus,
                self.vocabulary_,
                self._stopwords_set(),
                self._n_chunks(corpus),
            )
        except Exception as e:
            logger.warning(f"Rust count_vectorize_transform failed, falling back: {e}")
            return super().transform(raw_documents)
        rb.print_timing("count_vectorize_transform", t0)

        return csr_matrix(
            (data, indices, indptr),
            shape=(len(corpus), len(self.vocabulary_)),
        )

    def fit_transform(self, raw_documents, y=None):
        if not self._rust_ready("count_vectorize_fit_transform"):
            logger.debug("Rust disabled, fallback to scikit for fit_transform")
            return super().fit_transform(raw_documents, y)
        t0 = rb.start_timing()
        corpus = list(raw_documents)
        
        try:
            vocab, data, indices, indptr = rb.count_vectorize_fit_transform(
                corpus,
                self._stopwords_set(),
                self._n_chunks(corpus),
            )
            if len(vocab) == 0:
                raise ValueError(
                    "empty vocabulary; perhaps the documents only contain stop words"
                )
        except Exception as e:
            logger.warning(
                f"Rust count_vectorize_fit_transform failed, falling back: {e}"
            )
            return super().fit_transform(raw_documents, y)

        self.vocabulary_ = vocab
        rb.print_timing("count_vectorize_fit_transform", t0)

        return csr_matrix(
            (data, indices, indptr),
            shape=(len(corpus), len(self.vocabulary_)),
        )
