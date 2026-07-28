from __future__ import annotations

import logging
import operator
import warnings
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer as _TV
from sklearn.utils.validation import check_is_fitted

from .. import _rust_backend as rb
from .._config import get_config


logger = logging.getLogger(__name__)

_DEFAULT_TOKEN_PATTERN = r"(?u)\b\w\w+\b"
_NATIVE_USIZE_MAX = int(np.iinfo(np.uintp).max)


def _supported_integral(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _valid_document_frequency(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, Integral):
        return value >= 1
    if isinstance(value, Real):
        return 0.0 <= value <= 1.0
    return False


def _fits_native_usize(value) -> bool:
    """Return whether an integer can cross the native ``usize`` boundary."""
    return _supported_integral(value) and 0 <= operator.index(value) <= _NATIVE_USIZE_MAX


def _rust_supported_subset(vectorizer: _TV) -> tuple[bool, str]:
    """Return whether the vectorizer params are supported by the Rust kernel."""
    if getattr(vectorizer, "analyzer", "word") != "word":
        return False, "analyzer != 'word'"
    if getattr(vectorizer, "token_pattern", _DEFAULT_TOKEN_PATTERN) != _DEFAULT_TOKEN_PATTERN:
        return False, "non-default token_pattern"
    if getattr(vectorizer, "stop_words", None) is not None:
        return False, "stop_words not supported"
    if getattr(vectorizer, "preprocessor", None) is not None:
        return False, "custom preprocessor not supported"
    if getattr(vectorizer, "tokenizer", None) is not None:
        return False, "custom tokenizer not supported"
    if getattr(vectorizer, "strip_accents", None) is not None:
        return False, "strip_accents not supported"
    if getattr(vectorizer, "input", "content") != "content":
        return False, "input != 'content'"
    if getattr(vectorizer, "norm", "l2") not in (None, "l1", "l2"):
        return False, "unsupported norm"

    for parameter in ("lowercase", "binary", "use_idf", "smooth_idf", "sublinear_tf"):
        # Plain Python bool only (rejects numpy bools), matching HashingVectorizer.
        if not isinstance(getattr(vectorizer, parameter), bool):
            return False, f"non-bool {parameter}"

    ngram_range = getattr(vectorizer, "ngram_range", (1, 1))
    if not (
        isinstance(ngram_range, tuple)
        and len(ngram_range) == 2
        and all(_fits_native_usize(value) for value in ngram_range)
    ):
        return False, f"invalid ngram_range {ngram_range!r}"
    ngram_min, ngram_max = map(operator.index, ngram_range)
    if not 1 <= ngram_min <= ngram_max:
        return False, f"invalid ngram_range {ngram_range!r}"

    if not _valid_document_frequency(getattr(vectorizer, "min_df", 1)):
        return False, "invalid min_df"
    if not _valid_document_frequency(getattr(vectorizer, "max_df", 1.0)):
        return False, "invalid max_df"

    max_features = getattr(vectorizer, "max_features", None)
    if max_features is not None and not (
        _fits_native_usize(max_features)
        and operator.index(max_features) >= 1
    ):
        return False, "invalid max_features"

    try:
        dtype = np.dtype(getattr(vectorizer, "dtype", np.float64))
    except (TypeError, ValueError):
        return False, "invalid dtype"
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        return False, "dtype is not float32 or float64"

    vocabulary = getattr(vectorizer, "vocabulary", None)
    if isinstance(vocabulary, Mapping) and not all(
        isinstance(term, str) for term in vocabulary
    ):
        return False, "fixed vocabulary contains non-string terms"
    return True, ""


def _materialize_documents(raw_documents):
    """Turn raw_documents into a list. Lists are returned as-is; pandas/NumPy/
    Polars use ``tolist`` / ``to_list`` via ``_to_list``.
    """
    if isinstance(raw_documents, list):
        return raw_documents
    return rb._to_list(raw_documents)


def _prepare_documents(raw_documents, encoding: str, decode_error: str):
    """Materialize and decode content input.

    All-``str`` lists are reused for Rust and sklearn (no copy). A decoded
    copy is built only after the first ``bytes`` element. Non-str/non-bytes
    force sklearn fallback with the materialized values kept (needed for
    consumed generators).
    """
    if isinstance(raw_documents, str):
        return None, raw_documents

    documents = _materialize_documents(raw_documents)
    decoded = None
    for index, document in enumerate(documents):
        if isinstance(document, str):
            if decoded is not None:
                decoded.append(document)
        elif isinstance(document, bytes):
            if decoded is None:
                decoded = list(documents[:index])
            decoded.append(document.decode(encoding, errors=decode_error))
        else:
            # sklearn owns the error behavior for non-content objects.
            return None, documents
    if decoded is None:
        return documents, documents
    return decoded, documents


def _documents_are_ascii(documents) -> bool:
    """Return whether every decoded document is safe for Rust tokenization."""
    return all(document.isascii() for document in documents)


def _transform_signature(vectorizer: _TV):
    """Parameters that sklearn consults while transforming fitted data."""
    return (
        vectorizer.analyzer,
        vectorizer.token_pattern,
        vectorizer.stop_words,
        vectorizer.preprocessor,
        vectorizer.tokenizer,
        vectorizer.strip_accents,
        vectorizer.input,
        bool(vectorizer.lowercase),
        tuple(vectorizer.ngram_range),
        bool(vectorizer.binary),
        np.dtype(vectorizer.dtype),
        vectorizer.norm,
        bool(vectorizer.use_idf),
        bool(vectorizer.smooth_idf),
        bool(vectorizer.sublinear_tf),
    )


def _fitted_state_matches_native_snapshot(vectorizer: _TV) -> bool:
    """Return whether mutable sklearn fitted state still matches the Rust model."""
    vocabulary = getattr(vectorizer, "vocabulary_", None)
    vocabulary_snapshot = getattr(vectorizer, "_rust_vocabulary_snapshot_", None)
    try:
        if vocabulary_snapshot is None or vocabulary != vocabulary_snapshot:
            return False
    except (TypeError, ValueError):
        return False

    if not vectorizer.use_idf:
        return True

    transformer = getattr(vectorizer, "_tfidf", None)
    idf = getattr(transformer, "idf_", None)
    idf_snapshot = getattr(vectorizer, "_rust_idf_snapshot_", None)
    if idf is None or idf_snapshot is None:
        return False
    try:
        idf = np.asarray(idf)
        return (
            idf.shape == idf_snapshot.shape
            and idf.dtype == idf_snapshot.dtype
            and np.array_equal(idf, idf_snapshot)
        )
    except (TypeError, ValueError):
        return False


class RustyTfidfVectorizer(_TV):
    """sklearn ``TfidfVectorizer`` with a Rust backend for the default word
    analyzer on ASCII content. Everything else (Unicode, custom analyzers,
    tokenizers, preprocessors, accent stripping, stop words) uses sklearn.
    """

    def _rust_enabled(self) -> bool:
        config = get_config()
        return bool(
            config["allow_patch"]
            and config["rust_backend"]
            and rb.HAVE_RUST
            and rb.tfidf_vectorizer_fit is not None
            and rb.tfidf_vectorizer_transform is not None
        )

    def _invalidate_rust_model(self) -> None:
        self._rust_tfidf_model_ = None
        self.__dict__.pop("_rust_vocabulary_snapshot_", None)
        self.__dict__.pop("_rust_idf_snapshot_", None)

    def _fit_transform_with_rust(self, raw_documents):
        supported, reason = _rust_supported_subset(self)
        if not supported:
            logger.debug("TfidfVectorizer Rust fallback: %s", reason)
            return None, raw_documents
        if not self._rust_enabled():
            return None, raw_documents

        # Native fit skips sklearn's decorated fit, so validate params ourselves.
        self._validate_params()

        t_mat = rb.start_timing()
        documents, fallback_documents = _prepare_documents(
            raw_documents, self.encoding, self.decode_error
        )
        rb.print_timing("tv py_materialize", t_mat)
        if documents is None:
            return None, fallback_documents
        t_ascii = rb.start_timing()
        non_ascii = not _documents_are_ascii(documents)
        rb.print_timing("tv ascii_prescan", t_ascii)
        if non_ascii:
            return None, fallback_documents
        if not documents:
            return None, fallback_documents

        # Match CountVectorizer/TfidfVectorizer's validation and fitted state.
        self._check_params()
        self._validate_ngram_range()
        self._warn_for_unused_params()
        self._validate_vocabulary()

        fixed_terms = None
        if self.fixed_vocabulary_:
            if not all(isinstance(term, str) for term in self.vocabulary_):
                return None, fallback_documents
            fixed_terms = [None] * len(self.vocabulary_)
            for term, feature in self.vocabulary_.items():
                fixed_terms[feature] = term
            if self.lowercase and any(any(map(str.isupper, term)) for term in fixed_terms):
                warnings.warn(
                    "Upper case characters found in vocabulary while 'lowercase' is True. "
                    "These entries will not be matched with any documents",
                    UserWarning,
                    stacklevel=2,
                )

        document_count = len(documents)
        max_document_count = (
            float(self.max_df)
            if isinstance(self.max_df, Integral)
            else float(self.max_df) * document_count
        )
        min_document_count = (
            float(self.min_df)
            if isinstance(self.min_df, Integral)
            else float(self.min_df) * document_count
        )
        if not self.fixed_vocabulary_ and max_document_count < min_document_count:
            raise ValueError("max_df corresponds to < documents than min_df")

        t_prep = rb.start_timing()
        ngram_min, ngram_max = map(operator.index, self.ngram_range)
        max_features = (
            None if self.max_features is None else operator.index(self.max_features)
        )
        norm = "none" if self.norm is None else self.norm
        output_dtype = np.dtype(self.dtype)
        rb.print_timing("tv prep", t_prep)
        try:
            (
                model,
                data,
                indices,
                indptr,
                n_rows,
                n_cols,
                terms,
                vocabulary_order,
                idf,
            ) = rb.tfidf_vectorizer_fit(
                documents,
                ngram_min,
                ngram_max,
                self.lowercase,
                self.binary,
                norm,
                self.use_idf,
                self.smooth_idf,
                self.sublinear_tf,
                min_document_count,
                max_document_count,
                max_features,
                fixed_terms,
                output_dtype.name,
            )
        except rb.TfidfVectorizerFallback:
            return None, fallback_documents

        if self.fixed_vocabulary_:
            self.vocabulary_ = dict(self.vocabulary_)
        else:
            self.vocabulary_ = {
                terms[feature]: feature for feature in vocabulary_order
            }
        self._tfidf = TfidfTransformer(
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf,
        )
        self._tfidf.n_features_in_ = n_cols
        if self.use_idf:
            self._tfidf.idf_ = np.asarray(idf, dtype=output_dtype)

        self._rust_tfidf_model_ = model
        self._rust_transform_signature_ = _transform_signature(self)
        native_transform_is_safe = not (
            self.use_idf
            and self.norm is not None
            and not np.isfinite(self._tfidf.idf_).all()
        )
        if native_transform_is_safe:
            self._rust_vocabulary_snapshot_ = self.vocabulary_.copy()
            self._rust_idf_snapshot_ = (
                np.array(self._tfidf.idf_, copy=True) if self.use_idf else None
            )
        else:
            self._invalidate_rust_model()

        t_csr = rb.start_timing()
        matrix = sp.csr_matrix(
            (
                np.asarray(data, dtype=output_dtype),
                indices,
                indptr,
            ),
            shape=(n_rows, n_cols),
            dtype=output_dtype,
        )
        rb.print_timing("tv csr_wrap", t_csr)
        return matrix, fallback_documents

    def fit(self, raw_documents, y=None):
        matrix, fallback_documents = self._fit_transform_with_rust(raw_documents)
        if matrix is not None:
            return self
        self._invalidate_rust_model()
        return super().fit(fallback_documents, y)

    def fit_transform(self, raw_documents, y=None):
        t_total = rb.start_timing()
        matrix, fallback_documents = self._fit_transform_with_rust(raw_documents)
        if matrix is not None:
            rb.print_timing("tv fit_transform total", t_total)
            return matrix
        self._invalidate_rust_model()
        return super().fit_transform(fallback_documents, y)

    def transform(self, raw_documents):
        check_is_fitted(self, msg="The TF-IDF vectorizer is not fitted")

        model = getattr(self, "_rust_tfidf_model_", None)
        if model is not None and not _fitted_state_matches_native_snapshot(self):
            self._invalidate_rust_model()
            model = None
        supported = True
        reason = ""
        if model is not None:
            supported, reason = _rust_supported_subset(self)
            if not supported:
                logger.debug("TfidfVectorizer Rust fallback: %s", reason)
        if (
            model is None
            or not self._rust_enabled()
            or not supported
            or _transform_signature(self) != getattr(self, "_rust_transform_signature_", None)
        ):
            return super().transform(raw_documents)

        t_total = rb.start_timing()
        t_mat = rb.start_timing()
        documents, fallback_documents = _prepare_documents(
            raw_documents, self.encoding, self.decode_error
        )
        rb.print_timing("tv py_materialize", t_mat)
        if documents is None:
            return super().transform(fallback_documents)
        t_ascii = rb.start_timing()
        non_ascii = not _documents_are_ascii(documents)
        rb.print_timing("tv ascii_prescan", t_ascii)
        if not documents:
            return super().transform(fallback_documents)
        if non_ascii:
            return super().transform(fallback_documents)

        try:
            data, indices, indptr, n_rows, n_cols = rb.tfidf_vectorizer_transform(
                model, documents
            )
        except rb.TfidfVectorizerFallback:
            return super().transform(fallback_documents)

        output_dtype = np.dtype(self.dtype)
        t_csr = rb.start_timing()
        matrix = sp.csr_matrix(
            (np.asarray(data, dtype=output_dtype), indices, indptr),
            shape=(n_rows, n_cols),
            dtype=output_dtype,
        )
        rb.print_timing("tv csr_wrap", t_csr)
        rb.print_timing("tv transform total", t_total)
        return matrix

    @property
    def idf_(self):
        return _TV.idf_.fget(self)

    @idf_.setter
    def idf_(self, value):
        _TV.idf_.fset(self, value)
        self._invalidate_rust_model()

    def __getstate__(self):
        state = super().__getstate__()
        # PyO3 handles aren't picklable; sklearn state alone is enough to restore.
        state["_rust_tfidf_model_"] = None
        state.pop("_rust_vocabulary_snapshot_", None)
        state.pop("_rust_idf_snapshot_", None)
        return state
