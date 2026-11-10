"""Rust implementation for CountVectorizer.

- Default word unigrams use the native vocabulary and CSR builder.
- Unsupported parameters, Unicode and non-string documents retain sklearn semantics.
- Fitted vocabularies stay in Rust.
"""

from __future__ import annotations

import logging
import numbers
import os

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer as _SKCountVectorizer

from .. import _rust_backend as rb
from .._config import get_config

logger = logging.getLogger(__name__)
MIN_BLOCK_LEN = 1
_DEFAULT_TOKEN_PATTERN = r"(?u)\b\w\w+\b"


def supports_rust_count_vectorizer(estimator) -> tuple[bool, str]:
    """Accept only known estimator types and literal supported parameters."""
    if type(estimator) not in (_SKCountVectorizer, RustyCountVectorizer):
        return False, "estimator is not a supported CountVectorizer"
    if not (
        rb.HAVE_RUST
        and rb.count_vectorize_fit
        and rb.count_vectorize_transform
        and rb.count_vectorize_fit_transform
    ):
        return False, "Rust CountVectorizer runtime is not available"
    # Check types before equality: DataOp parameters must not be evaluated here.
    defaults = {
        "analyzer": "word",
        "token_pattern": _DEFAULT_TOKEN_PATTERN,
        "input": "content",
        "ngram_range": (1, 1),
        "lowercase": True,
        "binary": False,
        "max_df": 1.0,
        "min_df": 1,
    }
    for name, default in defaults.items():
        value = getattr(estimator, name)
        if name == "ngram_range" and (
            type(value) is not tuple or any(type(n) is not int for n in value)
        ):
            return False, "unsupported ngram_range"
        if type(value) is not type(default) or value != default:
            return False, f"unsupported {name}"
    for name in (
        "tokenizer",
        "preprocessor",
        "max_features",
        "vocabulary",
        "strip_accents",
    ):
        if getattr(estimator, name) is not None:
            return False, f"unsupported {name}"
    if estimator.stop_words is not None and not (
        type(estimator.stop_words) is str
        and estimator.stop_words == "english"
        or type(estimator.stop_words) is list
        and all(type(word) is str for word in estimator.stop_words)
    ):
        return False, "unsupported stop_words"
    if not any(
        estimator.dtype is dtype
        for dtype in (np.int32, np.int64, np.float32, np.float64)
    ):
        return False, "unsupported dtype"
    return True, ""


class RustyCountVectorizer(_SKCountVectorizer):
    """Sklearn-compatible estimator with native counting for ASCII documents."""

    def __init__(
        self,
        *,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=_DEFAULT_TOKEN_PATTERN,
        ngram_range=(1, 1),
        analyzer="word",
        max_df=1.0,
        min_df=1,
        max_features=None,
        vocabulary=None,
        binary=False,
        dtype=np.int64,
        n_jobs=None,
    ):
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            analyzer=analyzer,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )
        cores = os.cpu_count() or 1
        if n_jobs is not None and (
            not isinstance(n_jobs, numbers.Integral) or n_jobs < 1
        ):
            raise ValueError("n_jobs must be a positive integer or None")
        if n_jobs is not None and n_jobs > cores:
            logger.warning(
                "n_jobs %s > core count %s, setting n_jobs to %s", n_jobs, cores, cores
            )
        self.n_jobs = cores if n_jobs is None else min(n_jobs, cores)

    def _bind_native_worker_budget(self, workers):
        if not isinstance(workers, numbers.Integral) or workers < 1:
            raise ValueError("native worker budget must be a positive integer")
        self._stratum_worker_budget = int(workers)
        return self

    def __sklearn_clone__(self):
        cloned = super().__sklearn_clone__()
        if getattr(self, "_stratum_force_rust", False):
            cloned._stratum_force_rust = True
        if hasattr(self, "_stratum_worker_budget"):
            cloned._bind_native_worker_budget(self._stratum_worker_budget)
        return cloned

    @property
    def vocabulary_(self):
        # TODO: Synchronize in-place edits of the exported sklearn dictionary.
        # Native state remains authoritative until refit or explicit replacement.
        if "vocabulary_" not in self.__dict__:
            model = self.__dict__.get("_rust_model_")
            if model is None:
                raise AttributeError("vocabulary_")
            self.__dict__["vocabulary_"] = model.vocabulary()
        return self.__dict__["vocabulary_"]

    @vocabulary_.setter
    def vocabulary_(self, vocabulary):
        # A reference fit or explicit vocabulary replacement invalidates the handle.
        self.__dict__.pop("_rust_model_", None)
        self.__dict__["vocabulary_"] = vocabulary

    def _native_workers(self):
        if hasattr(self, "_stratum_worker_budget"):
            return self._stratum_worker_budget
        return min(self.n_jobs, get_config()["num_threads"] or self.n_jobs)

    def _n_chunks(self, corpus):
        return min(max(1, len(corpus) // MIN_BLOCK_LEN), self._native_workers())

    def _stopwords_set(self):
        words = self.get_stop_words()
        return set() if words is None else set(words)

    def _rust_ready(self):
        rc = get_config()
        return (
            getattr(self, "_stratum_force_rust", False)
            or rc["allow_patch"]
            and rc["rust_backend"]
        ) and supports_rust_count_vectorizer(self)[0]

    @staticmethod
    def _supported_corpus(corpus):
        # Python and Rust have different Unicode word/lowercase rules. ASCII is
        # unambiguous; sklearn also owns bytes decoding and invalid-input errors.
        return all(isinstance(doc, str) and doc.isascii() for doc in corpus)

    def fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents, y)
        return self

    def fit_transform(self, raw_documents, y=None):
        # Refit must never reuse native state from a previous corpus.
        self.__dict__.pop("_rust_model_", None)
        if not self._rust_ready() or isinstance(raw_documents, str):
            return super().fit_transform(raw_documents, y)
        self._validate_params()
        corpus = list(raw_documents)  # allocates a list of reference
        # FIXME: The corpus of strings are copied to Rust; double allocations till fit ends
        if not self._supported_corpus(corpus):
            return super().fit_transform(corpus, y)
        try:
            model, data, indices, indptr = rb.count_vectorize_fit_transform(
                corpus,
                self._stopwords_set(),
                self._n_chunks(corpus),
                worker_budget=self._native_workers(),
            )
        except Exception as error:
            logger.warning("Rust CountVectorizer failed, falling back: %s", error)
            # The original iterable may be a consumed generator.
            return super().fit_transform(corpus, y)
        if not model.n_features:
            raise ValueError(
                "empty vocabulary; perhaps the documents only contain stop words"
            )
        self.__dict__.pop("vocabulary_", None)
        self._rust_model_ = model
        self.fixed_vocabulary_ = False
        return csr_matrix(
            (data, indices, indptr),
            shape=(len(corpus), model.n_features),
            dtype=self.dtype,
        )

    def transform(self, raw_documents):
        if (
            not self._rust_ready()
            or isinstance(raw_documents, str)
            or not hasattr(self, "_rust_model_")
        ):
            return super().transform(raw_documents)
        corpus = list(raw_documents)
        if not self._supported_corpus(corpus):
            return super().transform(corpus)
        try:
            data, indices, indptr = rb.count_vectorize_transform(
                self._rust_model_,
                corpus,
                self._stopwords_set(),
                self._n_chunks(corpus),
                worker_budget=self._native_workers(),
            )
        except Exception as error:
            logger.warning("Rust CountVectorizer failed, falling back: %s", error)
            return super().transform(corpus)
        return csr_matrix(
            (data, indices, indptr),
            shape=(len(corpus), self._rust_model_.n_features),
            dtype=self.dtype,
        )
