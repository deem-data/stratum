import os

os.environ.setdefault("SKRUB_RUST", "1")  # opt-in fastpath before any imports

import pickle
import warnings

import numpy as np
import pytest
import scipy.sparse as sp

import stratum
from stratum import _rust_backend as rb
from stratum.adapters.tfidf_vectorizer import (
    RustyTfidfVectorizer,
    _TV as SklearnTfidfVectorizer,
    _fitted_state_matches_native_snapshot,
    _materialize_documents,
    _prepare_documents,
    _rust_supported_subset,
    _valid_document_frequency,
)


pytestmark = pytest.mark.skipif(not rb.HAVE_RUST, reason="Rust backend not built")

_DOCUMENTS = [
    "The quick brown fox jumps over the lazy dog",
    "the Lazy dog, the CAT!!!",
    "",
    "foo a b cc dd foo foo",
    "Hello, WORLD! hello.",
    "unicode cafe test cafe",
    "a",
    "123 45 6 seven_eight under_score",
]

_TRANSFORM_DOCUMENTS = ["foo lazy dog", "unknown only", "CAFE cafe", "a"]


def _fit_rust(documents, **params):
    vectorizer = RustyTfidfVectorizer(**params)
    with stratum.config(rust_backend=True, allow_patch=True):
        matrix = vectorizer.fit_transform(documents).tocsr()
    return vectorizer, matrix


def _fit_sklearn(documents, **params):
    vectorizer = SklearnTfidfVectorizer(**params)
    return vectorizer, vectorizer.fit_transform(documents).tocsr()


def _assert_sparse_close(got, expected, *, atol=2e-12):
    assert sp.isspmatrix_csr(got)
    assert sp.isspmatrix_csr(expected)
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    assert got.indices.dtype == expected.indices.dtype
    assert got.indptr.dtype == expected.indptr.dtype
    np.testing.assert_array_equal(got.indices, expected.indices)
    np.testing.assert_array_equal(got.indptr, expected.indptr)
    np.testing.assert_allclose(got.data, expected.data, rtol=0, atol=atol)


def _assert_matching_exceptions(expected_error, got_error):
    assert type(got_error.value) is type(expected_error.value)
    assert str(got_error.value).replace(
        "RustyTfidfVectorizer", "TfidfVectorizer"
    ) == str(expected_error.value)


def _assert_vectorizers_match(documents=_DOCUMENTS, **params):
    expected_vectorizer, expected = _fit_sklearn(documents, **params)
    got_vectorizer, got = _fit_rust(documents, **params)
    tolerance = 2e-6 if np.dtype(params.get("dtype", np.float64)) == np.float32 else 2e-12

    assert got_vectorizer.vocabulary_ == expected_vectorizer.vocabulary_
    assert np.array_equal(
        got_vectorizer.get_feature_names_out(),
        expected_vectorizer.get_feature_names_out(),
    )
    _assert_sparse_close(got, expected, atol=tolerance)
    if params.get("use_idf", True):
        np.testing.assert_allclose(
            got_vectorizer.idf_, expected_vectorizer.idf_, atol=tolerance, rtol=tolerance
        )

    with stratum.config(rust_backend=True, allow_patch=True):
        transformed = got_vectorizer.transform(_TRANSFORM_DOCUMENTS).tocsr()
    expected_transformed = expected_vectorizer.transform(_TRANSFORM_DOCUMENTS).tocsr()
    _assert_sparse_close(transformed, expected_transformed, atol=tolerance)
    return got_vectorizer


def test_default_configuration_matches_sklearn():
    vectorizer = _assert_vectorizers_match()
    assert vectorizer._rust_tfidf_model_ is not None


@pytest.mark.parametrize(
    ("documents", "params"),
    [
        (["one two three four"], {}),
        (["two one three"], {"ngram_range": (1, 2)}),
        (
            ["zebra apple", "zebra banana apple"],
            {"min_df": 2},
        ),
        (
            [
                "zebra banana zebra zebra zebra",
                "banana banana apple",
                "apple carrot",
            ],
            {"max_features": 2},
        ),
    ],
)
def test_learned_fit_transform_matches_sklearn_csr_layout(documents, params):
    _, expected = _fit_sklearn(documents, **params)
    got_vectorizer, got = _fit_rust(documents, **params)

    assert not expected.has_sorted_indices
    _assert_sparse_close(got, expected)
    assert got.has_sorted_indices == expected.has_sorted_indices
    assert got.has_canonical_format == expected.has_canonical_format
    assert got_vectorizer._rust_tfidf_model_ is not None


@pytest.mark.parametrize(
    "params",
    [
        {"norm": None, "use_idf": False},
        {"norm": "l1"},
        {"binary": True},
        {"smooth_idf": False},
        {"sublinear_tf": True},
        {"binary": True, "sublinear_tf": True, "norm": None},
        {"dtype": np.float32},
    ],
)
def test_weighting_parameter_grid_matches_sklearn(params):
    vectorizer = _assert_vectorizers_match(**params)
    assert vectorizer._rust_tfidf_model_ is not None


def test_float32_high_count_and_idf_stress_matches_with_explicit_tolerance():
    documents = [
        " ".join(["common"] * 10_000 + ["rare"]),
        *(f"common filler{index}" for index in range(1, 128)),
    ]
    params = {
        "dtype": np.float32,
        "norm": "l2",
        "sublinear_tf": True,
    }
    expected_vectorizer, expected = _fit_sklearn(documents, **params)
    got_vectorizer, got = _fit_rust(documents, **params)
    tolerance = 2e-6

    assert got.dtype == expected.dtype == np.dtype(np.float32)
    assert got_vectorizer.vocabulary_ == expected_vectorizer.vocabulary_
    _assert_sparse_close(got, expected, atol=tolerance)
    np.testing.assert_allclose(
        got_vectorizer.idf_,
        expected_vectorizer.idf_,
        atol=tolerance,
        rtol=tolerance,
    )
    assert got_vectorizer._rust_tfidf_model_ is not None


@pytest.mark.parametrize("ngram_range", [(1, 2), (2, 3), (1, 3)])
def test_word_ngram_ranges_match_sklearn(ngram_range):
    vectorizer = _assert_vectorizers_match(ngram_range=ngram_range)
    assert vectorizer._rust_tfidf_model_ is not None


def test_large_word_ngram_upper_bound_matches_sklearn():
    documents = ["one token", "token"]
    ngram_range = (1, 1_000_000)
    expected_vectorizer, expected = _fit_sklearn(
        documents, ngram_range=ngram_range
    )
    got_vectorizer, got = _fit_rust(documents, ngram_range=ngram_range)

    assert got_vectorizer.vocabulary_ == expected_vectorizer.vocabulary_
    _assert_sparse_close(got, expected)
    assert got_vectorizer._rust_tfidf_model_ is not None


@pytest.mark.parametrize(
    "params",
    [
        {"ngram_range": (1, 2**128)},
        {"max_features": 2**128},
    ],
)
def test_oversized_native_integer_falls_back_and_matches_sklearn(params):
    documents = ["one token", "token"]
    expected_vectorizer, expected = _fit_sklearn(documents, **params)
    got_vectorizer, got = _fit_rust(documents, **params)

    assert got_vectorizer.vocabulary_ == expected_vectorizer.vocabulary_
    _assert_sparse_close(got, expected)
    assert got_vectorizer._rust_tfidf_model_ is None


def test_lowercase_false_and_bytes_match_sklearn():
    documents = [b"Hello WORLD", "hello world", b"CAFE cafe"]
    vectorizer = _assert_vectorizers_match(documents, lowercase=False)
    assert vectorizer._rust_tfidf_model_ is not None


@pytest.mark.parametrize("lowercase", [True, False])
@pytest.mark.parametrize("ngram_range", [(1, 1), (1, 2), (2, 3)])
@pytest.mark.parametrize("as_generator", [False, True])
def test_non_ascii_fit_falls_back_and_matches_sklearn(
    lowercase, ngram_range, as_generator
):
    documents = [
        "plain ascii text",
        "café déjà vu",
        "שָׁלוֹם עולם",
        "اَلْعَرَبِيَّة العربية",
        "A\u0345B",
        "中文 混合 English",
    ]
    expected_vectorizer, expected = _fit_sklearn(
        documents, lowercase=lowercase, ngram_range=ngram_range
    )
    raw_documents = (document for document in documents) if as_generator else documents
    got_vectorizer, got = _fit_rust(
        raw_documents, lowercase=lowercase, ngram_range=ngram_range
    )

    assert got_vectorizer.vocabulary_ == expected_vectorizer.vocabulary_
    assert np.array_equal(
        got_vectorizer.get_feature_names_out(),
        expected_vectorizer.get_feature_names_out(),
    )
    _assert_sparse_close(got, expected)
    np.testing.assert_allclose(got_vectorizer.idf_, expected_vectorizer.idf_)
    assert got_vectorizer._rust_tfidf_model_ is None


@pytest.mark.parametrize("as_generator", [False, True])
def test_non_ascii_fixed_vocabulary_fit_falls_back_and_matches_sklearn(as_generator):
    documents = ["plain cafe", "café déjà vu", "A\u0345B"]
    vocabulary = {"plain": 0, "café": 1, "vu": 2, "ab": 3, "unseen": 4}
    expected_vectorizer, expected = _fit_sklearn(documents, vocabulary=vocabulary)
    raw_documents = (document for document in documents) if as_generator else documents
    got_vectorizer, got = _fit_rust(raw_documents, vocabulary=vocabulary)

    assert got_vectorizer.vocabulary_ == expected_vectorizer.vocabulary_
    _assert_sparse_close(got, expected)
    np.testing.assert_allclose(got_vectorizer.idf_, expected_vectorizer.idf_)
    assert got_vectorizer._rust_tfidf_model_ is None


def test_non_ascii_transform_bypasses_native_binding_and_matches_sklearn(monkeypatch):
    documents = ["apple banana", "banana carrot"]
    expected = SklearnTfidfVectorizer().fit(documents)
    got, _ = _fit_rust(documents)
    assert got._rust_tfidf_model_ is not None

    native_called = False
    native_transform = rb.tfidf_vectorizer_transform

    def recording_native_transform(*args, **kwargs):
        nonlocal native_called
        native_called = True
        return native_transform(*args, **kwargs)

    monkeypatch.setattr(rb, "tfidf_vectorizer_transform", recording_native_transform)
    probes = (document for document in ["café déjà vu", "A\u0345B"])
    with stratum.config(rust_backend=True, allow_patch=True):
        transformed = got.transform(probes).tocsr()

    _assert_sparse_close(
        transformed,
        expected.transform(["café déjà vu", "A\u0345B"]).tocsr(),
    )
    assert not native_called
    assert got._rust_tfidf_model_ is not None


def test_native_boundary_rejects_non_ascii_fit_and_transform():
    native_args = (
        1,
        1,
        True,
        False,
        "l2",
        True,
        True,
        False,
        1.0,
        2.0,
        None,
        None,
    )
    with pytest.raises(rb.TfidfVectorizerFallback, match="non-ASCII"):
        rb.tfidf_vectorizer_fit(["plain ascii", "café"], *native_args)

    model = rb.tfidf_vectorizer_fit(["plain ascii", "more text"], *native_args)[0]
    with pytest.raises(rb.TfidfVectorizerFallback, match="non-ASCII"):
        rb.tfidf_vectorizer_transform(model, ["café"])


def test_native_boundary_rejects_duplicate_fixed_terms_without_panicking():
    native_args = (
        1,
        1,
        True,
        False,
        "l2",
        True,
        True,
        False,
        1.0,
        1.0,
        None,
        ["duplicate", "duplicate"],
    )
    with pytest.raises(ValueError, match="Duplicate term in vocabulary"):
        rb.tfidf_vectorizer_fit(["duplicate"], *native_args)


@pytest.mark.parametrize(
    "params",
    [
        {"min_df": 2},
        {"min_df": 0.25},
        {"max_df": 0.5},
    ],
)
def test_document_frequency_pruning_matches_sklearn(params):
    vectorizer = _assert_vectorizers_match(**params)
    assert vectorizer._rust_tfidf_model_ is not None


def test_max_features_uses_rust_when_boundary_is_unambiguous():
    documents = [
        "alpha alpha alpha alpha beta beta beta gamma gamma delta",
        "alpha beta gamma epsilon",
    ]
    vectorizer = _assert_vectorizers_match(documents, max_features=3)
    assert vectorizer._rust_tfidf_model_ is not None


def test_float32_max_features_uses_rust_while_frequencies_are_exact():
    documents = [
        "alpha alpha alpha alpha beta beta beta gamma gamma delta",
        "alpha beta gamma epsilon",
    ]
    vectorizer = _assert_vectorizers_match(
        documents,
        dtype=np.float32,
        max_features=3,
    )
    assert vectorizer._rust_tfidf_model_ is not None


def test_tied_max_features_falls_back_for_numpy_exactness():
    vectorizer = _assert_vectorizers_match(max_features=5)
    assert vectorizer._rust_tfidf_model_ is None


def test_tied_max_features_fallback_preserves_materialized_generator(
    monkeypatch,
):
    expected = SklearnTfidfVectorizer(max_features=5).fit_transform(_DOCUMENTS).tocsr()
    original_fit_transform = SklearnTfidfVectorizer.fit_transform
    fallback_documents = None

    def recording_fit_transform(self, raw_documents, y=None):
        nonlocal fallback_documents
        fallback_documents = raw_documents
        return original_fit_transform(self, raw_documents, y)

    monkeypatch.setattr(
        SklearnTfidfVectorizer,
        "fit_transform",
        recording_fit_transform,
    )
    vectorizer = RustyTfidfVectorizer(max_features=5)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with stratum.config(rust_backend=True, allow_patch=True):
            got = vectorizer.fit_transform(
                document for document in _DOCUMENTS
            ).tocsr()

    _assert_sparse_close(got, expected)
    assert fallback_documents == _DOCUMENTS
    assert isinstance(fallback_documents, list)
    assert vectorizer._rust_tfidf_model_ is None
    assert not recorded


@pytest.mark.parametrize(
    "vocabulary",
    [
        {"dog": 0, "foo": 1, "unseen": 2},
        ["dog", "foo", "unseen"],
    ],
)
def test_fixed_vocabulary_matches_sklearn(vocabulary):
    vectorizer = _assert_vectorizers_match(vocabulary=vocabulary)
    assert vectorizer._rust_tfidf_model_ is not None


def test_fixed_vocabulary_mapping_preserves_insertion_order():
    vocabulary = {"banana": 1, "apple": 0}
    documents = ["apple banana"]

    expected_vectorizer, _ = _fit_sklearn(documents, vocabulary=vocabulary)
    got_vectorizer, _ = _fit_rust(documents, vocabulary=vocabulary)

    assert list(got_vectorizer.vocabulary_.items()) == list(
        expected_vectorizer.vocabulary_.items()
    )
    assert got_vectorizer._rust_tfidf_model_ is not None


def test_learned_vocabulary_preserves_discovery_order():
    documents = ["zebra apple", "banana zebra"]

    expected_vectorizer, _ = _fit_sklearn(documents)
    got_vectorizer, _ = _fit_rust(documents)

    assert list(got_vectorizer.vocabulary_.items()) == list(
        expected_vectorizer.vocabulary_.items()
    )
    assert got_vectorizer._rust_tfidf_model_ is not None


@pytest.mark.parametrize("norm", ["l1", "l2", None])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("probe", ["seen", "unseen"])
def test_nonfinite_fixed_vocabulary_idf_uses_safe_transform_path(
    norm, dtype, probe
):
    documents = ["seen"]
    params = {
        "vocabulary": ["seen", "unseen"],
        "smooth_idf": False,
        "norm": norm,
        "dtype": dtype,
    }
    with np.errstate(divide="ignore"):
        expected_vectorizer, expected = _fit_sklearn(documents, **params)
        got_vectorizer, got = _fit_rust(documents, **params)

    _assert_sparse_close(got, expected)
    np.testing.assert_allclose(
        got_vectorizer.idf_,
        expected_vectorizer.idf_,
        rtol=0,
        atol=0,
    )

    if norm is None:
        assert got_vectorizer._rust_tfidf_model_ is not None
        with np.errstate(invalid="ignore"):
            expected_transformed = expected_vectorizer.transform([probe]).toarray()
            with stratum.config(rust_backend=True, allow_patch=True):
                got_transformed = got_vectorizer.transform([probe]).toarray()
        np.testing.assert_allclose(
            got_transformed,
            expected_transformed,
            rtol=0,
            atol=0,
        )
    else:
        assert got_vectorizer._rust_tfidf_model_ is None
        try:
            expected_transformed = expected_vectorizer.transform([probe]).tocsr()
        except Exception as error:
            expected_error = error
        else:
            expected_error = None

        with stratum.config(rust_backend=True, allow_patch=True):
            if expected_error is None:
                got_transformed = got_vectorizer.transform([probe]).tocsr()
                _assert_sparse_close(got_transformed, expected_transformed)
            else:
                with pytest.raises(type(expected_error)) as got_error:
                    got_vectorizer.transform([probe])
                assert str(got_error.value) == str(expected_error)


def test_randomized_differential_includes_nonfinite_fixed_vocabulary_idf():
    rng = np.random.default_rng(20260726)
    vocabulary = ["seen", "unseen", *(f"term{index}" for index in range(12))]
    documents = [
        " ".join(rng.choice(vocabulary[2:], size=8, replace=True))
        for _ in range(24)
    ]
    documents[0] += " seen"
    params = {
        "vocabulary": vocabulary,
        "smooth_idf": False,
        "norm": "l2",
    }

    with np.errstate(divide="ignore"):
        expected_vectorizer, expected = _fit_sklearn(documents, **params)
        got_vectorizer, got = _fit_rust(documents, **params)
    _assert_sparse_close(got, expected)
    np.testing.assert_allclose(got_vectorizer.idf_, expected_vectorizer.idf_)
    assert got_vectorizer._rust_tfidf_model_ is None

    probe = [" ".join(rng.choice(vocabulary, size=6, replace=True)) + " unseen"]
    with pytest.raises(Exception) as expected_error:
        expected_vectorizer.transform(probe)
    with stratum.config(rust_backend=True, allow_patch=True):
        with pytest.raises(Exception) as got_error:
            got_vectorizer.transform(probe)
    _assert_matching_exceptions(expected_error, got_error)


@pytest.mark.parametrize("use_idf", [False, True])
def test_large_mostly_unseen_fixed_vocabulary_matches_sklearn(use_idf):
    vocabulary = [f"term{index:05d}" for index in range(10_000)]
    documents = [
        "term00001 term00001 term05000",
        "term05000 unknown",
        "term09999",
    ]
    params = {
        "vocabulary": vocabulary,
        "use_idf": use_idf,
        "norm": None,
        "binary": False,
        "sublinear_tf": True,
    }
    expected_vectorizer, expected = _fit_sklearn(documents, **params)
    got_vectorizer, got = _fit_rust(documents, **params)

    assert got_vectorizer.vocabulary_ == expected_vectorizer.vocabulary_
    _assert_sparse_close(got, expected)
    if use_idf:
        np.testing.assert_allclose(got_vectorizer.idf_, expected_vectorizer.idf_)
    assert got_vectorizer._rust_tfidf_model_ is not None


def test_fixed_vocabulary_above_dense_counter_limit_matches_sklearn():
    # The Rust kernel switches from dense to sparse feature counting above
    # 2**18 features. Exercise that selection through the public adapter.
    vocabulary_size = (1 << 18) + 1
    vocabulary = [f"term{index:06d}" for index in range(vocabulary_size)]
    documents = [
        "term000001 term000001 term131072 unknown",
        "term131072 term262144",
    ]
    params = {
        "vocabulary": vocabulary,
        "use_idf": False,
        "norm": None,
    }

    expected_vectorizer, expected = _fit_sklearn(documents, **params)
    got_vectorizer, got = _fit_rust(documents, **params)

    _assert_sparse_close(got, expected)
    assert got_vectorizer.vocabulary_ == expected_vectorizer.vocabulary_
    assert got_vectorizer._rust_tfidf_model_ is not None

    probes = ["term262144 term262144 term000001 unseen", ""]
    with stratum.config(rust_backend=True, allow_patch=True):
        transformed = got_vectorizer.transform(probes).tocsr()
    _assert_sparse_close(
        transformed,
        expected_vectorizer.transform(probes).tocsr(),
    )


def test_fit_then_transform_matches_sklearn():
    expected = SklearnTfidfVectorizer().fit(_DOCUMENTS)
    got = RustyTfidfVectorizer()
    with stratum.config(rust_backend=True, allow_patch=True):
        returned = got.fit((document for document in _DOCUMENTS))
        transformed = got.transform(_TRANSFORM_DOCUMENTS).tocsr()

    assert returned is got
    assert got._rust_tfidf_model_ is not None
    _assert_sparse_close(transformed, expected.transform(_TRANSFORM_DOCUMENTS).tocsr())


@pytest.mark.parametrize(
    "params",
    [
        {"analyzer": "char"},
        {"stop_words": "english"},
        {"token_pattern": r"(?u)\b\w+\b"},
        {"preprocessor": str.strip},
        {"tokenizer": str.split},
        {"strip_accents": "unicode"},
        {"dtype": np.int32},
    ],
)
def test_unsupported_configuration_falls_back(params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        vectorizer = _assert_vectorizers_match(**params)
    assert vectorizer._rust_tfidf_model_ is None


def test_filename_input_falls_back_and_matches_sklearn(tmp_path):
    paths = []
    for index, document in enumerate(_DOCUMENTS):
        path = tmp_path / f"document-{index}.txt"
        path.write_text(document)
        paths.append(path)

    expected_vectorizer, expected = _fit_sklearn(paths, input="filename")
    got_vectorizer, got = _fit_rust(paths, input="filename")

    assert got_vectorizer.vocabulary_ == expected_vectorizer.vocabulary_
    _assert_sparse_close(got, expected)
    assert got_vectorizer._rust_tfidf_model_ is None


def test_fit_after_configuration_fallback_matches_sklearn():
    params = {"preprocessor": str.strip}
    expected = SklearnTfidfVectorizer(**params).fit(_DOCUMENTS)
    got = RustyTfidfVectorizer(**params)

    with stratum.config(rust_backend=True, allow_patch=True):
        returned = got.fit(document for document in _DOCUMENTS)
        transformed = got.transform(_TRANSFORM_DOCUMENTS).tocsr()

    assert returned is got
    _assert_sparse_close(
        transformed,
        expected.transform(_TRANSFORM_DOCUMENTS).tocsr(),
    )
    assert got._rust_tfidf_model_ is None


def test_disabled_backend_falls_back_to_sklearn():
    vectorizer = RustyTfidfVectorizer()
    with stratum.config(rust_backend=False, allow_patch=True):
        got = vectorizer.fit_transform(_DOCUMENTS).tocsr()
    expected = SklearnTfidfVectorizer().fit_transform(_DOCUMENTS).tocsr()
    _assert_sparse_close(got, expected)
    assert vectorizer._rust_tfidf_model_ is None


def test_transform_before_fit_raises_sklearn_error():
    with pytest.raises(Exception, match="not fitted"):
        RustyTfidfVectorizer().transform(_DOCUMENTS)


def test_empty_vocabulary_raises_sklearn_error_without_rust_warning():
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="empty vocabulary"):
            _fit_rust(["a", "b", ""])
    assert not recorded


def test_empty_documents_bypass_native_fit_and_raise_sklearn_error(monkeypatch):
    native_called = False
    native_fit = rb.tfidf_vectorizer_fit

    def recording_native_fit(*args, **kwargs):
        nonlocal native_called
        native_called = True
        return native_fit(*args, **kwargs)

    monkeypatch.setattr(rb, "tfidf_vectorizer_fit", recording_native_fit)
    with pytest.raises(
        ValueError,
        match="empty vocabulary; perhaps the documents only contain stop words",
    ):
        _fit_rust([])
    assert not native_called


@pytest.mark.parametrize(
    "container",
    ["list", "tuple", "generator", "numpy"],
)
def test_empty_transform_matches_sklearn_without_native_call(monkeypatch, container):
    documents = ["one token"]
    expected = SklearnTfidfVectorizer().fit(documents)
    got, _ = _fit_rust(documents)
    model = got._rust_tfidf_model_

    def make_empty():
        if container == "list":
            return []
        if container == "tuple":
            return ()
        if container == "generator":
            return (document for document in ())
        return np.array([], dtype=object)

    native_called = False

    def recording_native_transform(*args, **kwargs):
        nonlocal native_called
        native_called = True
        raise AssertionError("native transform must not receive an empty corpus")

    monkeypatch.setattr(
        rb,
        "tfidf_vectorizer_transform",
        recording_native_transform,
    )
    with pytest.raises(Exception) as expected_error:
        expected.transform(make_empty())
    with stratum.config(rust_backend=True, allow_patch=True):
        with pytest.raises(Exception) as got_error:
            got.transform(make_empty())

    _assert_matching_exceptions(expected_error, got_error)
    assert not native_called
    assert got._rust_tfidf_model_ is model


@pytest.mark.parametrize(
    "documents",
    [
        [],
        [None],
        [np.nan],
        ["valid", 42],
    ],
)
@pytest.mark.parametrize("as_generator", [False, True])
def test_invalid_documents_match_sklearn_exception(documents, as_generator):
    expected_documents = (
        (document for document in documents) if as_generator else documents
    )
    got_documents = (
        (document for document in documents) if as_generator else documents
    )

    with pytest.raises(Exception) as expected_error:
        SklearnTfidfVectorizer().fit_transform(expected_documents)
    vectorizer = RustyTfidfVectorizer()
    with stratum.config(rust_backend=True, allow_patch=True):
        with pytest.raises(Exception) as got_error:
            vectorizer.fit_transform(got_documents)

    assert type(got_error.value) is type(expected_error.value)
    assert str(got_error.value).replace(
        "RustyTfidfVectorizer", "TfidfVectorizer"
    ) == str(expected_error.value)
    assert vectorizer._rust_tfidf_model_ is None


@pytest.mark.parametrize("method", ["fit_transform", "transform"])
def test_zero_dimensional_numpy_input_matches_sklearn_exception(method):
    expected = SklearnTfidfVectorizer()
    got = RustyTfidfVectorizer()
    if method == "transform":
        expected.fit(_DOCUMENTS)
        with stratum.config(rust_backend=True, allow_patch=True):
            got.fit(_DOCUMENTS)

    raw_documents = np.array("hello world")
    with pytest.raises(Exception) as expected_error:
        getattr(expected, method)(raw_documents)
    with stratum.config(rust_backend=True, allow_patch=True):
        with pytest.raises(Exception) as got_error:
            getattr(got, method)(raw_documents)

    _assert_matching_exceptions(expected_error, got_error)


@pytest.mark.parametrize(
    "params",
    [
        {"binary": 1},
        {"lowercase": 1},
        {"ngram_range": (1.0, 2.0)},
        {"max_features": 1.5},
        {"min_df": np.nan},
        {"max_df": np.inf},
        {"norm": "invalid"},
        {"encoding": None},
        {"encoding": 123},
        {"decode_error": None},
        {"decode_error": 123},
        {"decode_error": "bogus"},
    ],
)
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_invalid_parameter_types_match_sklearn_exception(params, method):
    documents = ["valid text", "other text"]
    with pytest.raises(Exception) as expected_error:
        getattr(SklearnTfidfVectorizer(**params), method)(documents)

    vectorizer = RustyTfidfVectorizer(**params)
    with stratum.config(rust_backend=True, allow_patch=True):
        with pytest.raises(Exception) as got_error:
            getattr(vectorizer, method)(documents)

    assert type(got_error.value) is type(expected_error.value)
    assert str(got_error.value).replace(
        "RustyTfidfVectorizer", "TfidfVectorizer"
    ) == str(expected_error.value)
    assert getattr(vectorizer, "_rust_tfidf_model_", None) is None


@pytest.mark.parametrize(
    "vocabulary",
    [
        {1: 0, "one": 1},
        [1, "one"],
    ],
)
def test_non_string_fixed_vocabulary_falls_back_to_sklearn(vocabulary):
    documents = ["one token"]
    with pytest.raises(Exception) as expected_error:
        SklearnTfidfVectorizer(vocabulary=vocabulary).fit_transform(documents)

    vectorizer = RustyTfidfVectorizer(vocabulary=vocabulary)
    with stratum.config(rust_backend=True, allow_patch=True):
        with pytest.raises(Exception) as got_error:
            vectorizer.fit_transform(documents)

    _assert_matching_exceptions(expected_error, got_error)
    assert vectorizer._rust_tfidf_model_ is None


def test_fixed_vocabulary_uppercase_warning_matches_sklearn():
    documents = ["apple"]
    params = {"vocabulary": ["Apple", "apple"]}

    with warnings.catch_warnings(record=True) as expected_warnings:
        warnings.simplefilter("always")
        expected_vectorizer, expected = _fit_sklearn(documents, **params)
    with warnings.catch_warnings(record=True) as got_warnings:
        warnings.simplefilter("always")
        got_vectorizer, got = _fit_rust(documents, **params)

    assert [(warning.category, str(warning.message)) for warning in got_warnings] == [
        (warning.category, str(warning.message)) for warning in expected_warnings
    ]
    _assert_sparse_close(got, expected)
    assert got_vectorizer.vocabulary_ == expected_vectorizer.vocabulary_
    assert got_vectorizer._rust_tfidf_model_ is not None


def test_max_df_below_min_df_matches_sklearn_exception():
    documents = ["one token", "other token"]
    params = {"min_df": 2, "max_df": 0.5}

    with pytest.raises(Exception) as expected_error:
        SklearnTfidfVectorizer(**params).fit_transform(documents)
    vectorizer = RustyTfidfVectorizer(**params)
    with stratum.config(rust_backend=True, allow_patch=True):
        with pytest.raises(Exception) as got_error:
            vectorizer.fit_transform(documents)

    _assert_matching_exceptions(expected_error, got_error)


@pytest.mark.parametrize("error_type", [MemoryError, RuntimeError])
def test_unexpected_native_fit_error_propagates_without_sklearn_retry(
    monkeypatch, error_type
):
    sklearn_called = False
    original_fit_transform = SklearnTfidfVectorizer.fit_transform

    def failing_native_fit(*args, **kwargs):
        raise error_type("native fit failed")

    def recording_sklearn_fit_transform(self, raw_documents, y=None):
        nonlocal sklearn_called
        sklearn_called = True
        return original_fit_transform(self, raw_documents, y)

    monkeypatch.setattr(rb, "tfidf_vectorizer_fit", failing_native_fit)
    monkeypatch.setattr(
        SklearnTfidfVectorizer,
        "fit_transform",
        recording_sklearn_fit_transform,
    )

    with pytest.raises(error_type, match="native fit failed"):
        _fit_rust(["apple banana", "banana carrot"])
    assert not sklearn_called


@pytest.mark.parametrize("error_type", [MemoryError, RuntimeError])
def test_unexpected_native_transform_error_propagates_without_sklearn_retry(
    monkeypatch, error_type
):
    vectorizer, _ = _fit_rust(["apple banana", "banana carrot"])
    sklearn_called = False
    original_transform = SklearnTfidfVectorizer.transform

    def failing_native_transform(*args, **kwargs):
        raise error_type("native transform failed")

    def recording_sklearn_transform(self, raw_documents):
        nonlocal sklearn_called
        sklearn_called = True
        return original_transform(self, raw_documents)

    monkeypatch.setattr(rb, "tfidf_vectorizer_transform", failing_native_transform)
    monkeypatch.setattr(
        SklearnTfidfVectorizer,
        "transform",
        recording_sklearn_transform,
    )

    with stratum.config(rust_backend=True, allow_patch=True):
        with pytest.raises(error_type, match="native transform failed"):
            vectorizer.transform(["apple"])
    assert not sklearn_called


def test_native_transform_fallback_exception_delegates_to_sklearn(monkeypatch):
    documents = ["apple banana", "banana carrot"]
    probe = ["apple carrot"]
    expected = SklearnTfidfVectorizer().fit(documents)
    got, _ = _fit_rust(documents)

    def unsupported_native_transform(*args, **kwargs):
        raise rb.TfidfVectorizerFallback("use sklearn")

    monkeypatch.setattr(
        rb,
        "tfidf_vectorizer_transform",
        unsupported_native_transform,
    )
    with stratum.config(rust_backend=True, allow_patch=True):
        transformed = got.transform(probe).tocsr()

    _assert_sparse_close(transformed, expected.transform(probe).tocsr())
    assert got._rust_tfidf_model_ is not None


@pytest.mark.parametrize(
    ("fit_dtype", "transform_dtype"),
    [
        (np.float64, np.float32),
        (np.float32, np.float64),
        (np.float64, np.int32),
        (np.float64, np.bool_),
        (np.float64, np.longdouble),
        (np.float64, np.complex128),
        (np.float64, ("i4", (-1,))),
    ],
)
def test_transform_dtype_mutation_bypasses_native_and_matches_sklearn(
    monkeypatch, fit_dtype, transform_dtype
):
    # On Windows and macOS, longdouble is an alias of float64, so mutating
    # dtype to longdouble is a no-op for np.dtype() and the native path is
    # correctly kept.
    try:
        dtypes_equal = np.dtype(fit_dtype) == np.dtype(transform_dtype)
    except (TypeError, ValueError):
        dtypes_equal = False
    if dtypes_equal:
        pytest.skip(
            f"{fit_dtype!r} and {transform_dtype!r} normalize to the same dtype"
        )

    documents = ["apple apple banana", "banana carrot"]
    probe = ["apple banana"]
    expected = SklearnTfidfVectorizer(dtype=fit_dtype).fit(documents)
    got = RustyTfidfVectorizer(dtype=fit_dtype)
    with stratum.config(rust_backend=True, allow_patch=True):
        got.fit(documents)
    model = got._rust_tfidf_model_

    expected.dtype = transform_dtype
    got.dtype = transform_dtype
    native_called = False

    def unexpected_native_transform(*args, **kwargs):
        nonlocal native_called
        native_called = True
        raise AssertionError("a changed dtype must use sklearn")

    monkeypatch.setattr(
        rb,
        "tfidf_vectorizer_transform",
        unexpected_native_transform,
    )

    try:
        expected_transformed = expected.transform(probe).tocsr()
    except Exception as error:
        expected_error = error
    else:
        expected_error = None

    with stratum.config(rust_backend=True, allow_patch=True):
        if expected_error is None:
            got_transformed = got.transform(probe).tocsr()
            _assert_sparse_close(got_transformed, expected_transformed)
        else:
            with pytest.raises(type(expected_error)) as got_error:
                got.transform(probe)
            assert str(got_error.value) == str(expected_error)

    assert not native_called
    assert got._rust_tfidf_model_ is model


@pytest.mark.parametrize("container", ["list", "generator", "string"])
def test_invalid_transform_input_matches_sklearn_without_native_call(
    monkeypatch, container
):
    documents = ["apple banana", "banana carrot"]
    expected = SklearnTfidfVectorizer().fit(documents)
    got, _ = _fit_rust(documents)
    model = got._rust_tfidf_model_

    def make_invalid():
        if container == "list":
            return [None]
        if container == "generator":
            return (document for document in [None])
        return "not a document collection"

    native_called = False

    def recording_native_transform(*args, **kwargs):
        nonlocal native_called
        native_called = True
        raise AssertionError("invalid objects must be validated by sklearn")

    monkeypatch.setattr(
        rb,
        "tfidf_vectorizer_transform",
        recording_native_transform,
    )
    with pytest.raises(Exception) as expected_error:
        expected.transform(make_invalid())
    with stratum.config(rust_backend=True, allow_patch=True):
        with pytest.raises(Exception) as got_error:
            got.transform(make_invalid())

    _assert_matching_exceptions(expected_error, got_error)
    assert not native_called
    assert got._rust_tfidf_model_ is model


def test_pickle_round_trip_retains_correct_sklearn_fallback():
    vectorizer, _ = _fit_rust(_DOCUMENTS)
    state = vectorizer.__getstate__()
    assert "_rust_vocabulary_snapshot_" not in state
    assert "_rust_idf_snapshot_" not in state
    restored = pickle.loads(pickle.dumps(vectorizer))
    assert restored._rust_tfidf_model_ is None
    _assert_sparse_close(
        restored.transform(_TRANSFORM_DOCUMENTS).tocsr(),
        vectorizer.transform(_TRANSFORM_DOCUMENTS).tocsr(),
    )


def test_assigning_idf_invalidates_native_model():
    vectorizer, _ = _fit_rust(_DOCUMENTS)
    vectorizer.idf_ = np.ones_like(vectorizer.idf_)
    assert vectorizer._rust_tfidf_model_ is None


@pytest.mark.parametrize("mutation", ["slice", "alias", "ufunc"])
def test_in_place_idf_mutation_invalidates_native_model_and_matches_sklearn(mutation):
    documents = ["apple apple banana", "banana carrot"]
    expected = SklearnTfidfVectorizer(norm=None).fit(documents)
    got, _ = _fit_rust(documents, norm=None)
    assert got._rust_tfidf_model_ is not None

    if mutation == "slice":
        expected.idf_[:] = 1.0
        got.idf_[:] = 1.0
    elif mutation == "alias":
        expected_idf = expected.idf_
        got_idf = got.idf_
        expected_idf[0] = 1.0
        got_idf[0] = 1.0
    else:
        np.add(expected.idf_, 1.0, out=expected.idf_)
        np.add(got.idf_, 1.0, out=got.idf_)

    probe = ["apple banana"]
    with stratum.config(rust_backend=True, allow_patch=True):
        transformed = got.transform(probe).tocsr()
    _assert_sparse_close(transformed, expected.transform(probe).tocsr())
    assert got._rust_tfidf_model_ is None


def test_in_place_vocabulary_mutation_invalidates_native_model_and_matches_sklearn():
    documents = ["apple banana", "banana carrot"]
    expected = SklearnTfidfVectorizer(norm=None, use_idf=False).fit(documents)
    got, _ = _fit_rust(documents, norm=None, use_idf=False)
    assert got._rust_tfidf_model_ is not None

    for vectorizer in (expected, got):
        vocabulary = vectorizer.vocabulary_
        vocabulary["apple"], vocabulary["banana"] = (
            vocabulary["banana"],
            vocabulary["apple"],
        )

    probe = ["apple"]
    with stratum.config(rust_backend=True, allow_patch=True):
        transformed = got.transform(probe).tocsr()
    _assert_sparse_close(transformed, expected.transform(probe).tocsr())
    assert got._rust_tfidf_model_ is None


def test_unmodified_fitted_state_continues_using_native_transform(monkeypatch):
    vectorizer, _ = _fit_rust(_DOCUMENTS)
    native_transform = rb.tfidf_vectorizer_transform
    call_count = 0

    def recording_native_transform(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return native_transform(*args, **kwargs)

    monkeypatch.setattr(rb, "tfidf_vectorizer_transform", recording_native_transform)
    with stratum.config(rust_backend=True, allow_patch=True):
        first = vectorizer.transform(_TRANSFORM_DOCUMENTS).tocsr()
        second = vectorizer.transform(_TRANSFORM_DOCUMENTS).tocsr()

    _assert_sparse_close(first, second)
    assert call_count == 2
    assert vectorizer._rust_tfidf_model_ is not None


def test_refit_after_fitted_state_mutation_restores_native_model():
    documents = ["apple apple banana", "banana carrot"]
    expected = SklearnTfidfVectorizer(norm=None).fit(documents)
    got, _ = _fit_rust(documents, norm=None)
    got.idf_[:] = 1.0

    with stratum.config(rust_backend=True, allow_patch=True):
        got.transform(["apple"])
        assert got._rust_tfidf_model_ is None
        returned = got.fit(documents)
        transformed = got.transform(["apple banana"]).tocsr()

    assert returned is got
    assert got._rust_tfidf_model_ is not None
    _assert_sparse_close(
        transformed,
        expected.transform(["apple banana"]).tocsr(),
    )


def test_supported_subset_gate():
    assert _rust_supported_subset(RustyTfidfVectorizer())[0]
    supported, reason = _rust_supported_subset(RustyTfidfVectorizer(analyzer="char"))
    assert not supported
    assert "analyzer" in reason
    supported, reason = _rust_supported_subset(
        RustyTfidfVectorizer(lowercase=np.bool_(True))
    )
    assert not supported
    assert "non-bool lowercase" in reason
    assert issubclass(rb.TfidfVectorizerFallback, Exception)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, True),
        (0.5, True),
        (True, False),
        (False, False),
        (np.bool_(True), False),
        ("x", False),
        (None, False),
        (object(), False),
    ],
)
def test_valid_document_frequency_rejects_non_numeric_and_bool(value, expected):
    assert _valid_document_frequency(value) is expected


@pytest.mark.parametrize(
    ("params", "reason_fragment"),
    [
        ({"min_df": True}, "invalid min_df"),
        ({"max_df": False}, "invalid max_df"),
        ({"min_df": "x"}, "invalid min_df"),
        ({"ngram_range": (2, 1)}, "invalid ngram_range"),
        ({"ngram_range": (0, 1)}, "invalid ngram_range"),
    ],
)
def test_invalid_df_and_ngram_gates_fall_back(params, reason_fragment):
    supported, reason = _rust_supported_subset(RustyTfidfVectorizer(**params))
    assert not supported
    assert reason_fragment in reason

    documents = ["one two", "two three"]
    expected_vectorizer = SklearnTfidfVectorizer(**params)
    try:
        expected = expected_vectorizer.fit_transform(documents).tocsr()
    except Exception as error:
        expected_error = error
        expected = None
    else:
        expected_error = None

    vectorizer = RustyTfidfVectorizer(**params)
    with stratum.config(rust_backend=True, allow_patch=True):
        if expected_error is None:
            got = vectorizer.fit_transform(documents).tocsr()
            _assert_sparse_close(got, expected)
            assert vectorizer.vocabulary_ == expected_vectorizer.vocabulary_
        else:
            with pytest.raises(type(expected_error)) as got_error:
                vectorizer.fit_transform(documents)
            assert type(got_error.value) is type(expected_error)
            assert str(got_error.value).replace(
                "RustyTfidfVectorizer", "TfidfVectorizer"
            ) == str(expected_error)
    assert getattr(vectorizer, "_rust_tfidf_model_", None) is None


def test_fitted_state_snapshot_defensive_paths():
    class RaisingEq:
        def __eq__(self, other):
            raise TypeError("cannot compare vocabulary")

        def __bool__(self):
            return True

    class BrokenAsArray:
        def __array__(self, dtype=None, copy=None):
            raise ValueError("cannot convert idf")

    vectorizer = RustyTfidfVectorizer()
    vectorizer.vocabulary_ = RaisingEq()
    vectorizer._rust_vocabulary_snapshot_ = {"a": 0}
    assert _fitted_state_matches_native_snapshot(vectorizer) is False

    vectorizer = RustyTfidfVectorizer()
    vectorizer.use_idf = True
    vectorizer.vocabulary_ = {"a": 0}
    vectorizer._rust_vocabulary_snapshot_ = {"a": 0}
    vectorizer._tfidf = type("T", (), {})()
    vectorizer._rust_idf_snapshot_ = np.ones(1)
    assert _fitted_state_matches_native_snapshot(vectorizer) is False

    vectorizer._tfidf = type("T", (), {"idf_": BrokenAsArray()})()
    assert _fitted_state_matches_native_snapshot(vectorizer) is False


def test_unsupported_gate_is_logged(caplog):
    vectorizer = RustyTfidfVectorizer(analyzer="char")
    with (
        caplog.at_level("DEBUG", logger="stratum.adapters.tfidf_vectorizer"),
        stratum.config(rust_backend=True, allow_patch=True),
    ):
        vectorizer.fit_transform(["alpha beta"])
    assert "TfidfVectorizer Rust fallback" in caplog.text
    assert "analyzer" in caplog.text


def test_sklearn_symbol_is_patched():
    import sklearn.feature_extraction.text as text

    assert text.TfidfVectorizer is RustyTfidfVectorizer
    assert stratum.TfidfVectorizer is RustyTfidfVectorizer


class TestMaterializeDocuments:
    def test_list_is_returned_as_is(self):
        documents = ["a", "b"]
        assert _materialize_documents(documents) is documents

    def test_tuple_and_generator(self):
        assert _materialize_documents(("a", "b")) == ["a", "b"]
        assert _materialize_documents(document for document in ("a", "b")) == ["a", "b"]

    def test_empty_input(self):
        assert _materialize_documents([]) == []
        assert _materialize_documents(()) == []

    def test_pandas_series(self):
        pd = pytest.importorskip("pandas")
        series = pd.Series(["foo", "bar", "baz"])
        assert _materialize_documents(series) == ["foo", "bar", "baz"]

    def test_numpy_object_and_string_array(self):
        object_array = np.array(["foo", "bar"], dtype=object)
        assert _materialize_documents(object_array) == ["foo", "bar"]
        string_array = np.array(["foo", "bar"])
        assert _materialize_documents(string_array) == ["foo", "bar"]

    def test_polars_series(self):
        pl = pytest.importorskip("polars")
        series = pl.Series(["foo", "bar", "baz"])
        assert _materialize_documents(series) == ["foo", "bar", "baz"]


class TestPrepareDocuments:
    def test_string_input_forces_sklearn_fallback(self):
        assert _prepare_documents("not a corpus", "utf-8", "strict") == (
            None,
            "not a corpus",
        )

    def test_all_string_list_reuses_same_object(self):
        documents = ["hello", "world"]
        rust_docs, fallback = _prepare_documents(documents, "utf-8", "strict")
        assert rust_docs is documents
        assert fallback is documents
        assert rust_docs is fallback

    def test_empty_list_reuses_same_object(self):
        documents = []
        rust_docs, fallback = _prepare_documents(documents, "utf-8", "strict")
        assert rust_docs is documents
        assert fallback is documents

    def test_tuple_of_strings(self):
        rust_docs, fallback = _prepare_documents(("a", "b"), "utf-8", "strict")
        assert rust_docs == ["a", "b"]
        assert rust_docs is fallback

    def test_bytes_are_decoded_lazily(self):
        documents = ["hello", b"world", "again"]
        rust_docs, fallback = _prepare_documents(documents, "utf-8", "strict")
        assert rust_docs == ["hello", "world", "again"]
        assert fallback is documents
        assert rust_docs is not fallback
        assert fallback[1] == b"world"

    def test_mixed_string_bytes_encoding_and_decode_error(self):
        documents = [b"caf\xe9", "plain"]
        rust_docs, fallback = _prepare_documents(documents, "latin-1", "strict")
        assert rust_docs == ["café", "plain"]
        assert fallback is documents

        replaced, _ = _prepare_documents([b"\xff"], "utf-8", "replace")
        assert replaced == ["\ufffd"]

        ignored, _ = _prepare_documents([b"a\xffb"], "utf-8", "ignore")
        assert ignored == ["ab"]

    def test_invalid_object_falls_back_with_original_values(self):
        documents = ["ok", 123, "also"]
        rust_docs, fallback = _prepare_documents(documents, "utf-8", "strict")
        assert rust_docs is None
        assert fallback is documents
        assert fallback[1] == 123

    def test_missing_values_fall_back(self):
        pd = pytest.importorskip("pandas")
        series = pd.Series(["ok", None, "also"])
        rust_docs, fallback = _prepare_documents(series, "utf-8", "strict")
        assert rust_docs is None
        assert fallback[0] == "ok"
        assert fallback[1] is None or (isinstance(fallback[1], float) and np.isnan(fallback[1]))

    def test_consumed_generator_preserves_fallback_values(self):
        generator = (document for document in ("alpha", "beta", b"gamma"))
        rust_docs, fallback = _prepare_documents(generator, "utf-8", "strict")
        assert rust_docs == ["alpha", "beta", "gamma"]
        assert fallback == ["alpha", "beta", b"gamma"]
        assert list(generator) == []

    def test_invalid_in_generator_preserves_materialized_fallback(self):
        generator = (document for document in ("alpha", object(), "beta"))
        rust_docs, fallback = _prepare_documents(generator, "utf-8", "strict")
        assert rust_docs is None
        assert fallback[0] == "alpha"
        assert not isinstance(fallback[1], (str, bytes))

    def test_pandas_series_all_strings(self):
        pd = pytest.importorskip("pandas")
        series = pd.Series(["foo", "bar"])
        rust_docs, fallback = _prepare_documents(series, "utf-8", "strict")
        assert rust_docs == ["foo", "bar"]
        assert rust_docs is fallback

    def test_numpy_object_array(self):
        array = np.array(["foo", "bar"], dtype=object)
        rust_docs, fallback = _prepare_documents(array, "utf-8", "strict")
        assert rust_docs == ["foo", "bar"]
        assert rust_docs is fallback

    def test_polars_series_all_strings(self):
        pl = pytest.importorskip("polars")
        series = pl.Series(["foo", "bar"])
        rust_docs, fallback = _prepare_documents(series, "utf-8", "strict")
        assert rust_docs == ["foo", "bar"]
        assert rust_docs is fallback


def test_fit_transform_accepts_pandas_series():
    pd = pytest.importorskip("pandas")
    series = pd.Series(_DOCUMENTS)
    _assert_vectorizers_match(series)


def test_fit_transform_accepts_generator():
    vectorizer = RustyTfidfVectorizer()
    with stratum.config(rust_backend=True, allow_patch=True):
        matrix = vectorizer.fit_transform(document for document in _DOCUMENTS).tocsr()
    expected = SklearnTfidfVectorizer().fit_transform(_DOCUMENTS).tocsr()
    _assert_sparse_close(matrix, expected)
    assert vectorizer._rust_tfidf_model_ is not None
