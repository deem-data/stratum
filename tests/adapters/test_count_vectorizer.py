import logging
import os
import random
import string
import sys

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer as SKCountVectorizer

from stratum import _rust_backend as rb
from stratum import config, set_config
from stratum._config import get_config
from stratum.adapters.count_vectorizer import (
    MIN_BLOCK_LEN,
    RustyCountVectorizer,
    supports_rust_count_vectorizer,
)

requires_rust = pytest.mark.skipif(not rb.HAVE_RUST, reason="Rust backend not built")
REAL_SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "A journey of a thousand miles begins with a single step",
    "To be or not to be that is the question",
    "All that glitters is not gold",
    "The rain in Spain falls mainly on the plain",
]

random.seed(67)


@pytest.fixture(autouse=True)
def isolated_config():
    with config(num_threads=4):
        yield


def get_synthetic_data(n_unique_words, max_length, n_sentences, max_sentence_len):
    words = []
    for _ in range(n_unique_words):
        word_len = random.choice(list(range(2, max_length)))
        word = "".join(random.choices(string.ascii_letters, k=word_len))
        words.append(word)
        sentences = []

    for _ in range(n_sentences):
        sentence_len = random.choice(list(range(1, max_sentence_len)))
        sentence = " ".join(random.choices(words, k=sentence_len)) + "."
        sentences.append(sentence)

    return sentences


def capture_std_out(capfd):
    # Capture timing output
    sys.stdout.flush()
    sys.stderr.flush()
    captured = capfd.readouterr()
    combined_output = (captured.out or "") + (captured.err or "")
    return combined_output


@requires_rust
@pytest.mark.parametrize("n_sentences", [10, 500, 5000, 25_000])
def test_fit_transform_matches_sklearn_various_sizes(n_sentences, capfd):

    corpus = get_synthetic_data(
        n_unique_words=42, max_length=8, n_sentences=n_sentences, max_sentence_len=8
    )
    set_config(rust_backend=True, allow_patch=True, debug_timing=True, num_threads=4)

    rv = RustyCountVectorizer(n_jobs=4)
    Z = rv.fit_transform(corpus)

    sk = SKCountVectorizer()
    Z_ref = sk.fit_transform(corpus)

    assert rv.vocabulary_ == sk.vocabulary_
    assert Z.shape == Z_ref.shape
    np.testing.assert_array_equal(Z.toarray(), Z_ref.toarray())

    assert "[rust]" in capture_std_out(capfd)


@requires_rust
def test_fit_only_matches_sklearn_vocabulary():
    set_config(rust_backend=True, allow_patch=True, debug_timing=False)
    corpus = get_synthetic_data(
        n_unique_words=42, max_length=8, n_sentences=5, max_sentence_len=8
    )

    rv = RustyCountVectorizer()
    rv.fit(corpus)

    sk = SKCountVectorizer()
    sk.fit(corpus)
    assert rv.vocabulary_ == sk.vocabulary_


@requires_rust
def test_transform_after_fit_with_unseen_tokens(capfd):
    set_config(rust_backend=True, allow_patch=True, debug_timing=True)
    train = get_synthetic_data(
        n_unique_words=42, max_length=8, n_sentences=5, max_sentence_len=8
    )
    # different seed and differrent config will produce unknown words
    random.seed(123)
    test_docs = get_synthetic_data(
        n_unique_words=100, max_length=100, n_sentences=2, max_sentence_len=8
    )
    random.seed(67)

    rv = RustyCountVectorizer()
    rv.fit(train)
    Z = rv.transform(test_docs)

    sk = SKCountVectorizer()
    sk.fit(train)
    Z_ref = sk.transform(test_docs)

    np.testing.assert_array_equal(Z.toarray(), Z_ref.toarray())
    assert "[rust]" in capture_std_out(capfd)


# ---------------------------------------------------------------------------
# Correctness: stop_words configurations
# ---------------------------------------------------------------------------


@requires_rust
@pytest.mark.parametrize(
    "stop_words",
    [None, "english", ["the", "is"], ["a", "of", "to"]],
    ids=["none", "english", "custom_list_1", "custom_list_2"],
)
def test_fit_transform_matches_sklearn_stopwords(stop_words, capfd):
    set_config(rust_backend=True, allow_patch=True, debug_timing=True)

    rv = RustyCountVectorizer(stop_words=stop_words)
    Z = rv.fit_transform(REAL_SENTENCES)

    sk = SKCountVectorizer(stop_words=stop_words)
    Z_ref = sk.fit_transform(REAL_SENTENCES)

    assert rv.vocabulary_ == sk.vocabulary_
    np.testing.assert_array_equal(Z.toarray(), Z_ref.toarray())
    assert "[rust]" in capture_std_out(capfd)


def test_stopwords_set_custom_list():
    rv = RustyCountVectorizer(stop_words=["foo", "bar"])
    assert rv._stopwords_set() == {"foo", "bar"}

    rv = RustyCountVectorizer(stop_words={"foo", "bar"})
    assert rv._stopwords_set() == {"foo", "bar"}


# ---------------------------------------------------------------------------
# n_jobs / chunking
# ---------------------------------------------------------------------------


def test_n_jobs_defaults_to_cpu_count():
    rv = RustyCountVectorizer()
    assert rv.n_jobs == os.cpu_count()


def test_n_jobs_explicit_value_kept():
    rv = RustyCountVectorizer(n_jobs=1)
    assert rv.n_jobs == 1
    assert rv._n_chunks(["hello"] * 10) == 1


def test_unknown_cpu_count_uses_one_worker(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    rv = RustyCountVectorizer()
    assert rv.n_jobs == 1
    assert rv._n_chunks(["hello"] * 10) == 1


@pytest.mark.parametrize("n_jobs", [0, -1, 1.5])
def test_invalid_n_jobs_rejected(n_jobs):
    with pytest.raises(ValueError, match="n_jobs"):
        RustyCountVectorizer(n_jobs=n_jobs)


def test_n_jobs_above_cpu_count_capped_with_warning(caplog):
    cores = os.cpu_count()
    with caplog.at_level(logging.WARNING):
        rv = RustyCountVectorizer(n_jobs=cores + 100)
    assert rv.n_jobs == cores
    assert any("n_jobs" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "corpus_len,n_jobs,expected",
    [
        (MIN_BLOCK_LEN, 4, 1),  # corpus smaller than n_jobs -> capped by corpus size
        (3, 4, 3),  # corpus smaller than n_jobs -> capped by corpus size
        (4, 4, 4),  # corpus equal to n_jobs -> full parallelism
        (5, 4, 4),  # corpus bigger than n_jobs -> capped by n_jobs
        (100, 4, 4),  # corpus bigger than n_jobs -> capped by n_jobs
    ],
)
def test_n_chunks_formula(corpus_len, n_jobs, expected):
    # With MIN_BLOCK_LEN=1, n_chunks = min(corpus_len, n_jobs): parallelism is
    # only limited by corpus size (never below it), otherwise by n_jobs.
    rv = RustyCountVectorizer(n_jobs=n_jobs)
    corpus = [""] * corpus_len
    assert rv._n_chunks(corpus) == expected


# ---------------------------------------------------------------------------
# Fallback behavior (rust disabled / unavailable / erroring)
# ---------------------------------------------------------------------------


def test_fallback_when_rust_backend_disabled(capfd):
    set_config(rust_backend=False, debug_timing=True)
    corpus = ["hello world", "world of code"]

    rv = RustyCountVectorizer()
    _ = rv.fit_transform(corpus)

    assert "[rust]" not in capture_std_out(capfd)


@requires_rust
def test_fallback_when_allow_patch_disabled(capfd):
    set_config(rust_backend=True, allow_patch=False, debug_timing=True)
    rv = RustyCountVectorizer()
    Z = rv.fit_transform(["hello world", "world of code"])
    assert Z.shape[0] == 2
    assert "[rust]" not in capture_std_out(capfd)


def test_fallback_when_rust_backend_unreachable(monkeypatch, capfd):
    """Simulates an un-built/older native extension, i.e. the exact bug that
    made this fastpath dead code: getattr(rb, fn_name, None) is None for all
    three functions must route cleanly through the sklearn fallback rather
    than erroring."""
    set_config(rust_backend=True, allow_patch=True, debug_timing=True)
    monkeypatch.setattr(rb, "HAVE_RUST", False)

    rv = RustyCountVectorizer()
    Z = rv.fit_transform(["hello world", "world of code"])
    assert Z.shape[0] == 2
    assert "[rust]" not in capture_std_out(capfd)


@requires_rust
def test_fallback_on_rust_exception(monkeypatch, caplog):
    """If the rust call raises at runtime, the adapter should log a warning
    and fall back to sklearn rather than propagating the error."""
    set_config(rust_backend=True, allow_patch=True, debug_timing=False)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(rb, "count_vectorize_fit_transform", boom)

    rv = RustyCountVectorizer()
    with caplog.at_level(logging.WARNING):
        Z = rv.fit_transform(["hello world", "world of code"])
    assert Z.shape[0] == 2
    assert any("falling back" in r.message for r in caplog.records)


# Like sklearn we raise valuerror with empty resulting vocab
@requires_rust
def test_fit_transform_empty_corpus():
    set_config(rust_backend=True, allow_patch=True, debug_timing=False)
    rv = RustyCountVectorizer()
    with pytest.raises(ValueError):
        _ = rv.fit_transform([])


@requires_rust
def test_transform_without_fit_raises():
    set_config(rust_backend=True, allow_patch=True, debug_timing=False)
    rv = RustyCountVectorizer()
    with pytest.raises(NotFittedError):
        rv.transform(["hello world"])


@requires_rust
def test_transform_empty_documents_preserves_shape():
    set_config(rust_backend=True, allow_patch=True)
    estimator = RustyCountVectorizer().fit(["hello world"])
    for documents in ([], ["", "a !", "unseen"]):
        actual = estimator.transform(documents)
        assert actual.shape == (len(documents), 2)
        assert actual.nnz == 0
        assert actual.dtype == np.int64


# ---------------------------------------------------------------------------
# Fastpath eligibility gating (non-default CountVectorizer parameters)
# ---------------------------------------------------------------------------


@requires_rust
def test_fastpath_eligible_for_default_params(capfd):
    set_config(rust_backend=True, allow_patch=True, debug_timing=True)
    rv = RustyCountVectorizer()
    supported, reason = supports_rust_count_vectorizer(rv)
    assert supported, reason

    _ = rv.fit_transform(REAL_SENTENCES)
    assert "[rust]" in capture_std_out(capfd)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"analyzer": "char"},
        {"ngram_range": (1, 2)},
        {"tokenizer": str.split},
        {"preprocessor": str.lower},
        {"lowercase": False},
        {"token_pattern": r"(?u)\b\w+\b"},
        {"max_df": 0.9},
        {"min_df": 2},
        {"max_features": 5},
        {"binary": True},
        {"vocabulary": {"the": 0, "is": 1}},
        {"strip_accents": "unicode"},
    ],
    ids=[
        "analyzer_char",
        "ngram_range",
        "custom_tokenizer",
        "custom_preprocessor",
        "lowercase_false",
        "custom_token_pattern",
        "max_df",
        "min_df",
        "max_features",
        "binary",
        "fixed_vocabulary",
        "strip_accents",
    ],
)
@requires_rust
@pytest.mark.filterwarnings(
    "ignore:The parameter 'token_pattern' will not be used since 'tokenizer' is not None:UserWarning"
)
def test_fastpath_falls_back_for_nondefault_params(kwargs, capfd):
    set_config(rust_backend=True, allow_patch=True, debug_timing=True)

    rv = RustyCountVectorizer(**kwargs)
    supported, _ = supports_rust_count_vectorizer(rv)
    assert not supported

    Z = rv.fit_transform(REAL_SENTENCES)
    assert Z.shape[0] == len(REAL_SENTENCES)
    assert "[rust]" not in capture_std_out(capfd)




@requires_rust
@pytest.mark.filterwarnings(
    "ignore:The parameter 'token_pattern' will not be used since 'tokenizer' is not None:UserWarning"
)
def test_automatic_fallback_with_custom_tokenizer(capfd):
    """A custom `tokenizer` callable can't be executed by the Rust kernel, so
    fit_transform must transparently fall back to sklearn and still produce
    the correct (sklearn-matching) result."""
    set_config(rust_backend=True, allow_patch=True, debug_timing=True)

    rv = RustyCountVectorizer(tokenizer=str.split)
    Z = rv.fit_transform(REAL_SENTENCES)

    sk = SKCountVectorizer(tokenizer=str.split)
    Z_ref = sk.fit_transform(REAL_SENTENCES)

    assert rv.vocabulary_ == sk.vocabulary_
    np.testing.assert_array_equal(Z.toarray(), Z_ref.toarray())
    assert "[rust]" not in capture_std_out(capfd)


@requires_rust
def test_automatic_fallback_with_custom_preprocessor(capfd):
    """A custom `preprocessor` callable is likewise unsupported by the Rust
    kernel and must trigger a transparent fallback to sklearn."""
    set_config(rust_backend=True, allow_patch=True, debug_timing=True)

    rv = RustyCountVectorizer(preprocessor=str.lower)
    Z = rv.fit_transform(REAL_SENTENCES)

    sk = SKCountVectorizer(preprocessor=str.lower)
    Z_ref = sk.fit_transform(REAL_SENTENCES)

    assert rv.vocabulary_ == sk.vocabulary_
    np.testing.assert_array_equal(Z.toarray(), Z_ref.toarray())
    assert "[rust]" not in capture_std_out(capfd)


@requires_rust
def test_automatic_fallback_with_non_default_ngram_range(capfd):
    """ngram_range other than (1, 1) isn't implemented in the Rust kernel
    (unigrams only), so it must trigger a transparent fallback to sklearn."""
    set_config(rust_backend=True, allow_patch=True, debug_timing=True)

    rv = RustyCountVectorizer(ngram_range=(1, 2))
    Z = rv.fit_transform(REAL_SENTENCES)

    sk = SKCountVectorizer(ngram_range=(1, 2))
    Z_ref = sk.fit_transform(REAL_SENTENCES)

    assert rv.vocabulary_ == sk.vocabulary_
    np.testing.assert_array_equal(Z.toarray(), Z_ref.toarray())
    assert "[rust]" not in capture_std_out(capfd)


@requires_rust
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_dtype_fitted_state_and_clone_parity(dtype):
    from sklearn.base import clone

    set_config(rust_backend=True, allow_patch=True)
    estimator = RustyCountVectorizer(dtype=dtype, stop_words=["the"], n_jobs=2)
    cloned = clone(estimator)
    assert cloned.get_params() == estimator.get_params()
    assert cloned.stop_words is not estimator.stop_words
    actual = cloned.fit_transform(REAL_SENTENCES)
    reference = SKCountVectorizer(dtype=dtype, stop_words=["the"])
    expected = reference.fit_transform(REAL_SENTENCES)
    assert actual.dtype == expected.dtype
    assert cloned.fixed_vocabulary_ is False
    np.testing.assert_array_equal(cloned.get_feature_names_out(), reference.get_feature_names_out())
    np.testing.assert_array_equal(actual.toarray(), expected.toarray())
    transformed = cloned.transform(["unknown fox fox", "", "The dog"])
    assert transformed.dtype == dtype
    np.testing.assert_array_equal(
        transformed.toarray(), reference.transform(["unknown fox fox", "", "The dog"]).toarray(),
    )


@requires_rust
@pytest.mark.parametrize("documents", [
    ["İSTANBUL Straße ΟΣ", "café naïve 中文"],
    [b"hello world", b"world code"],
])
def test_unicode_and_bytes_use_sklearn(monkeypatch, documents):
    set_config(rust_backend=True, allow_patch=True)

    def unexpected(*args, **kwargs):
        pytest.fail("unsupported documents reached native kernel")

    monkeypatch.setattr(rb, "count_vectorize_fit_transform", unexpected)
    monkeypatch.setattr(rb, "count_vectorize_transform", unexpected)
    estimator = RustyCountVectorizer()
    reference = SKCountVectorizer()
    np.testing.assert_array_equal(
        estimator.fit_transform(iter(documents)).toarray(),
        reference.fit_transform(documents).toarray(),
    )
    assert estimator.vocabulary_ == reference.vocabulary_
    np.testing.assert_array_equal(
        estimator.transform(iter(documents)).toarray(), reference.transform(documents).toarray(),
    )


@requires_rust
@pytest.mark.parametrize("method", ["fit", "fit_transform", "transform"])
def test_native_failure_preserves_generator(monkeypatch, method):
    set_config(rust_backend=True, allow_patch=True)
    estimator = RustyCountVectorizer()
    reference = SKCountVectorizer().fit(REAL_SENTENCES)
    if method == "transform":
        estimator.fit(REAL_SENTENCES)

    def boom(*args, **kwargs):
        raise RuntimeError("native failure")

    monkeypatch.setattr(rb, "count_vectorize_fit_transform", boom)
    monkeypatch.setattr(rb, "count_vectorize_transform", boom)
    result = getattr(estimator, method)(iter(REAL_SENTENCES))
    if method == "fit":
        assert result is estimator
    else:
        np.testing.assert_array_equal(result.toarray(), reference.transform(REAL_SENTENCES).toarray())
    assert estimator.vocabulary_ == reference.vocabulary_


@requires_rust
@pytest.mark.parametrize("documents", [[], ["", "a _ !"], [None], [float("nan")], "hello world"])
def test_invalid_or_empty_inputs_match_reference(documents):
    set_config(rust_backend=True, allow_patch=True)
    with pytest.raises(Exception) as expected:
        SKCountVectorizer().fit_transform(documents)
    with pytest.raises(type(expected.value)):
        RustyCountVectorizer().fit_transform(documents)


@requires_rust
def test_integer_max_df_falls_back_and_preserves_pruning():
    set_config(rust_backend=True, allow_patch=True)
    documents = ["common first", "common second"]
    estimator = RustyCountVectorizer(max_df=1)
    assert not supports_rust_count_vectorizer(estimator)[0]
    reference = SKCountVectorizer(max_df=1)
    np.testing.assert_array_equal(
        estimator.fit_transform(documents).toarray(), reference.fit_transform(documents).toarray(),
    )
    assert "common" not in estimator.vocabulary_


@requires_rust
@pytest.mark.parametrize("chunks", [1, 2, 4, 100])
def test_native_entry_points_preserve_row_order(chunks):
    documents = ["", "one TWO two", "_under 42 x", "third", "", "last last"]
    reference = SKCountVectorizer()
    expected = reference.fit_transform(documents)
    model, data, indices, indptr = rb.count_vectorize_fit_transform(
        documents, set(), chunks, worker_budget=2,
    )
    from scipy.sparse import csr_matrix

    assert type(model).__name__ == "_CountVectorizerModelHandle"
    assert model.vocabulary() == reference.vocabulary_
    assert model.n_features == expected.shape[1]
    fitted = rb.count_vectorize_fit(documents, set(), chunks, worker_budget=2)
    assert fitted.vocabulary() == reference.vocabulary_
    np.testing.assert_array_equal(
        csr_matrix((data, indices, indptr), shape=expected.shape).toarray(), expected.toarray(),
    )
    data, indices, indptr = rb.count_vectorize_transform(
        model, documents, set(), chunks, worker_budget=2,
    )
    np.testing.assert_array_equal(
        csr_matrix((data, indices, indptr), shape=expected.shape).toarray(), expected.toarray(),
    )


@requires_rust
@pytest.mark.parametrize("function,args", [
    ("count_vectorize_fit", ([], set())),
    ("count_vectorize_fit_transform", ([], set())),
    ("count_vectorize_transform", None),
])
@pytest.mark.parametrize("kwargs", [{"n_chunks": 0}, {"worker_budget": 0}])
def test_native_rejects_zero_budgets(function, args, kwargs):
    if args is None:
        args = (rb.count_vectorize_fit(["hello"], set()), [], set())
    with pytest.raises(ValueError):
        getattr(rb, function)(*args, **kwargs)


@requires_rust
def test_repeated_transforms_reuse_native_model_without_export(monkeypatch):
    from sklearn.base import clone

    set_config(rust_backend=True, allow_patch=True)
    estimator = RustyCountVectorizer().fit(REAL_SENTENCES)
    model = estimator._rust_model_
    assert "vocabulary_" not in vars(estimator)
    native_transform = rb.count_vectorize_transform
    calls = []

    def transform(handle, documents, *args, **kwargs):
        assert handle is model
        calls.append(handle)
        return native_transform(handle, documents, *args, **kwargs)

    monkeypatch.setattr(rb, "count_vectorize_transform", transform)
    reference = SKCountVectorizer().fit(REAL_SENTENCES)
    for documents in (["quick fox", "unseen"], ["rain Spain", ""]):
        np.testing.assert_array_equal(
            estimator.transform(documents).toarray(), reference.transform(documents).toarray(),
        )
        assert "vocabulary_" not in vars(estimator)
        assert estimator._rust_model_ is model
    assert len(calls) == 2
    assert estimator.vocabulary_ == reference.vocabulary_
    # Introspection must not send the exported dictionary back to Rust either.
    estimator.transform(["quick fox"])
    assert len(calls) == 3
    assert not hasattr(clone(estimator), "_rust_model_")


@requires_rust
def test_refit_replaces_native_state_and_clears_it_for_reference_fit():
    set_config(rust_backend=True, allow_patch=True)
    estimator = RustyCountVectorizer().fit(["first vocabulary"])
    old_model = estimator._rust_model_
    estimator.fit(["second corpus"])
    assert estimator._rust_model_ is not old_model
    assert "vocabulary_" not in vars(estimator)
    np.testing.assert_array_equal(estimator.transform(["first second"]).toarray(), [[0, 1]])
    estimator.fit(["café Unicode"])
    assert not hasattr(estimator, "_rust_model_")
    assert estimator.vocabulary_ == {"café": 0, "unicode": 1}


@requires_rust
def test_explicit_vocabulary_replacement_invalidates_native_state():
    set_config(rust_backend=True, allow_patch=True)
    estimator = RustyCountVectorizer().fit(["hello world"])
    estimator.vocabulary_ = {"world": 0, "hello": 1}
    assert not hasattr(estimator, "_rust_model_")
    np.testing.assert_array_equal(estimator.transform(["hello hello world"]).toarray(), [[1, 2]])


@requires_rust
def test_reference_transform_exports_native_vocabulary_only_when_needed():
    set_config(rust_backend=True, allow_patch=True)
    estimator = RustyCountVectorizer().fit(REAL_SENTENCES)
    assert "vocabulary_" not in vars(estimator)
    documents = ["café quick fox", "İSTANBUL rain"]
    reference = SKCountVectorizer().fit(REAL_SENTENCES)
    np.testing.assert_array_equal(
        estimator.transform(documents).toarray(), reference.transform(documents).toarray(),
    )
    assert estimator.vocabulary_ == reference.vocabulary_
