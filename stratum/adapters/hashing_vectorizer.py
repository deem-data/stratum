from __future__ import annotations

import logging
import operator
from numbers import Integral

import numpy as np
import scipy.sparse as sp

from sklearn.feature_extraction.text import HashingVectorizer as _HV

from .. import _rust_backend as rb
from .._config import get_config

logger = logging.getLogger(__name__)

# The one token pattern we replicate exactly in Rust (sklearn's default).
_DEFAULT_TOKEN_PATTERN = r"(?u)\b\w\w+\b"

_MAX_SKLEARN_FEATURES = np.iinfo(np.int32).max
_MAX_RUST_USIZE = np.iinfo(np.uintp).max
_SKLEARN_FALLBACK = object()


def _supported_integral(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _rust_supported_subset(enc: _HV) -> tuple[bool, str]:
    """Gate: only route to Rust for the parameter subset the kernel replicates
    bit-for-bit. Anything else falls back to sklearn."""
    if getattr(enc, "analyzer", "word") != "word":
        return False, "analyzer != 'word'"
    if getattr(enc, "token_pattern", _DEFAULT_TOKEN_PATTERN) != _DEFAULT_TOKEN_PATTERN:
        return False, "non-default token_pattern"
    if getattr(enc, "stop_words", None) is not None:
        return False, "stop_words not supported"
    if getattr(enc, "preprocessor", None) is not None:
        return False, "custom preprocessor not supported"
    if getattr(enc, "tokenizer", None) is not None:
        return False, "custom tokenizer not supported"
    if getattr(enc, "strip_accents", None) is not None:
        return False, "strip_accents not supported"
    if getattr(enc, "input", "content") != "content":
        return False, "input != 'content'"
    if getattr(enc, "norm", "l2") not in (None, "l1", "l2"):
        return False, "norm not in {None, 'l1', 'l2'}"
    try:
        dtype = np.dtype(getattr(enc, "dtype", np.float64))
    except (TypeError, ValueError):
        return False, "invalid dtype"
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        return False, "dtype is not float32 or float64"
    if not isinstance(getattr(enc, "lowercase", True), bool):
        return False, "non-bool lowercase"
    if not isinstance(getattr(enc, "binary", False), bool):
        return False, "non-bool binary"
    if not isinstance(getattr(enc, "alternate_sign", True), bool):
        return False, "non-bool alternate_sign"
    n_features = getattr(enc, "n_features", 2**20)
    if not _supported_integral(n_features):
        return False, "non-integral n_features"
    n_features = operator.index(n_features)
    if not 1 <= n_features <= _MAX_SKLEARN_FEATURES:
        return False, "n_features outside sklearn's supported range"
    ngr = getattr(enc, "ngram_range", (1, 1))
    if not (
        isinstance(ngr, tuple)
        and len(ngr) == 2
        and all(_supported_integral(value) for value in ngr)
    ):
        return False, f"invalid ngram_range {ngr!r}"
    ngram_min, ngram_max = map(operator.index, ngr)
    if not 1 <= ngram_min <= ngram_max:
        return False, f"invalid ngram_range {ngr!r}"
    if ngram_max > _MAX_RUST_USIZE:
        return False, "ngram_range exceeds Rust's supported integer range"
    return True, ""


def _materialize_documents(raw_documents):
    """Materialize containers using their optimized conversion methods.

    Plain lists are returned unchanged so the all-string hot path creates no
    second Python list.
    """
    if isinstance(raw_documents, list):
        return raw_documents
    return rb._to_list(raw_documents)


def _to_docs(raw_documents, encoding="utf-8", decode_error="strict"):
    """Decode bytes while avoiding a copy for ``list[str]``.

    Callers must route unsupported document types to sklearn before using this
    helper. The explicit error prevents accidental semantic coercion if that
    invariant is broken later.
    """
    documents = _materialize_documents(raw_documents)
    converted = None

    for index, value in enumerate(documents):
        if isinstance(value, str):
            if converted is not None:
                converted.append(value)
            continue

        if isinstance(value, bytes):
            replacement = value.decode(encoding, errors=decode_error)
        else:
            raise TypeError("documents must contain only str or bytes values")

        if converted is None:
            converted = list(documents[:index])
        converted.append(replacement)

    return documents if converted is None else converted


class RustyHashingVectorizer(_HV):
    """Drop-in sklearn ``HashingVectorizer`` that prefers a Rust fast path for the
    default word-analyzer configuration and falls back to sklearn otherwise."""

    def _rust_enabled(self) -> bool:
        rc = get_config()
        return bool(
            rc["allow_patch"]
            and rc["rust_backend"]
            and rb.HAVE_RUST
            and rb.hashing_vectorizer_transform is not None
        )

    def _rust_transform(self, X):
        """Return a Rust CSR matrix or the private pre-materialization sentinel."""
        ok, reason = _rust_supported_subset(self)
        if not ok:
            logger.debug("HashingVectorizer Rust fallback: %s", reason)
            return _SKLEARN_FALLBACK
        if not self._rust_enabled():
            return _SKLEARN_FALLBACK
        if isinstance(X, (str, bytes)):
            # Let sklearn raise its standard "iterable over raw text" error.
            return _SKLEARN_FALLBACK

        t_mat = rb.start_timing()
        documents = _materialize_documents(X)
        if not documents or any(
            not isinstance(document, (str, bytes)) for document in documents
        ):
            return super().transform(documents)

        docs = _to_docs(documents, self.encoding, self.decode_error)
        rb.print_timing("hv py_materialize", t_mat)
        t_ascii = rb.start_timing()
        non_ascii = any(not document.isascii() for document in docs)
        rb.print_timing("hv ascii_prescan", t_ascii)
        if non_ascii:
            return super().transform(docs)

        t_prep = rb.start_timing()
        n_features = operator.index(self.n_features)
        ngram_min, ngram_max = map(operator.index, self.ngram_range)
        norm = "none" if self.norm is None else self.norm
        output_dtype = np.dtype(self.dtype)
        rb.print_timing("hv prep", t_prep)
        try:
            data, indices, indptr, n_rows, n_cols = rb.hashing_vectorizer_transform(
                docs,
                n_features,
                ngram_min,
                ngram_max,
                self.binary,
                self.alternate_sign,
                self.lowercase,
                norm,
            )
        except MemoryError:
            raise
        except Exception:
            logger.warning(
                "Rust hashing_vectorizer_transform failed; falling back to sklearn",
                exc_info=True,
            )
            # ``docs`` also preserves values from one-shot iterators that were
            # consumed during materialization.
            return super().transform(docs)

        # Kernel always emits float64; cast at the boundary for float32.
        t_csr = rb.start_timing()
        X_out = sp.csr_matrix(
            (np.asarray(data, dtype=output_dtype), indices, indptr),
            shape=(n_rows, n_cols),
            dtype=output_dtype,
        )
        rb.print_timing("hv csr_wrap", t_csr)
        return X_out

    # HashingVectorizer is stateless: fit() only validates and transform()
    # does the work. We route transform (and thus fit_transform) through Rust.
    def transform(self, X):
        t_total = rb.start_timing()
        out = self._rust_transform(X)
        if out is not _SKLEARN_FALLBACK:
            rb.print_timing("hv transform", t_total)
            return out
        return super().transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
