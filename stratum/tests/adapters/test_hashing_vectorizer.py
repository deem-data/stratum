import os

os.environ.setdefault("SKRUB_RUST", "1")  # opt-in fastpath before any imports

import numpy as np
import pytest
import scipy.sparse as sp

import stratum  # noqa: F401  (ensures patching + config side effects)
from stratum import _rust_backend as rb
from stratum.adapters.hashing_vectorizer import (
    _HV as OriginalHashingVectorizer,
    RustyHashingVectorizer,
    _materialize_documents,
    _rust_supported_subset,
    _to_docs,
)

pytestmark = pytest.mark.skipif(not rb.HAVE_RUST, reason="Rust backend not built")

_DOCS = [
    "The quick brown fox jumps over the lazy dog",
    "the Lazy dog, the CAT!!!",
    "",
    "foo a b cc dd foo foo",           # repeated tokens + single-char tokens dropped
    "Hello, WORLD! hello.",
    "ünïcodé café tëst café",          # non-ASCII word chars
    "a",                                # single char only -> empty row
    "123 45 6 seven_eight under_score",
]
_ASCII_DOCS = [document for document in _DOCS if document.isascii()]


def _rust_csr(docs, **params):
    enc = RustyHashingVectorizer(**params)
    with stratum.config(rust_backend=True, allow_patch=True):
        X = enc.transform(docs)
    assert sp.issparse(X)
    return X.tocsr()


def _sklearn_csr(docs, **params):
    return OriginalHashingVectorizer(**params).transform(docs).tocsr()


def _assert_csr_equal(got, ref, *, atol=1e-12):
    got = got.tocsr()
    ref = ref.tocsr()
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    assert got.indptr.dtype == ref.indptr.dtype
    assert got.indices.dtype == ref.indices.dtype
    np.testing.assert_array_equal(got.indptr, ref.indptr)
    np.testing.assert_array_equal(got.indices, ref.indices)
    np.testing.assert_allclose(got.data, ref.data, rtol=0, atol=atol)


def _assert_matches_sklearn_behavior(
    documents_factory, *, method="transform", params=None, atol=1e-12
):
    params = {} if params is None else params

    try:
        reference = getattr(OriginalHashingVectorizer(**params), method)(
            documents_factory()
        )
    except Exception as exc:
        reference_error = exc
    else:
        reference_error = None

    try:
        with stratum.config(rust_backend=True, allow_patch=True):
            actual = getattr(RustyHashingVectorizer(**params), method)(
                documents_factory()
            )
    except Exception as exc:
        actual_error = exc
    else:
        actual_error = None

    if reference_error is not None or actual_error is not None:
        assert reference_error is not None, "sklearn succeeded but the adapter raised"
        assert actual_error is not None, "sklearn raised but the adapter succeeded"
        assert type(actual_error) is type(reference_error)
        return

    _assert_csr_equal(actual, reference, atol=atol)


def _assert_close(docs, atol=1e-12, **params):
    got = _rust_csr(docs, **params)
    ref = _sklearn_csr(docs, **params)
    _assert_csr_equal(got, ref, atol=atol)


def test_rust_config_controls_native_dispatch(monkeypatch):
    native_transform = rb.hashing_vectorizer_transform
    native_calls = 0

    def counted_transform(*args, **kwargs):
        nonlocal native_calls
        native_calls += 1
        return native_transform(*args, **kwargs)

    monkeypatch.setattr(rb, "hashing_vectorizer_transform", counted_transform)
    vectorizer = RustyHashingVectorizer()

    with stratum.config(rust_backend=True, allow_patch=True):
        actual = vectorizer.transform(["plain ascii text"])
    assert native_calls == 1
    reference = OriginalHashingVectorizer().transform(["plain ascii text"])
    _assert_csr_equal(actual, reference)

    with stratum.config(rust_backend=False, allow_patch=True):
        vectorizer.transform(["plain ascii text"])
    assert native_calls == 1


@pytest.mark.parametrize(
    ("dtype", "norm"),
    [
        (np.float64, "l2"),
        (np.float64, "l1"),
        (np.float32, "l2"),
        (np.float32, "l1"),
        (np.float32, None),
        (np.int64, "l2"),
        (np.int64, None),
        (np.bool_, "l2"),
        (np.float16, "l2"),
        (np.longdouble, "l2"),
        (np.complex128, "l2"),
        (object, "l2"),
    ],
)
def test_dtype_behavior_matches_sklearn(dtype, norm):
    _assert_matches_sklearn_behavior(
        lambda: ["alpha beta"],
        params={"dtype": dtype, "norm": norm, "n_features": 2**10},
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        np.longdouble,
        np.int64,
        np.bool_,
        np.complex128,
        object,
    ],
)
def test_unverified_dtypes_are_rejected_by_native_gate(dtype):
    # On Windows and macOS, longdouble is an alias of float64, so the native
    # path correctly accepts it as float64.
    try:
        if np.dtype(dtype) in (np.dtype(np.float32), np.dtype(np.float64)):
            pytest.skip(f"{dtype!r} normalizes to a supported float dtype")
    except (TypeError, ValueError):
        pass
    supported, _ = _rust_supported_subset(RustyHashingVectorizer(dtype=dtype))
    assert not supported


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_dtypes_are_supported_by_native_gate(dtype):
    supported, reason = _rust_supported_subset(
        RustyHashingVectorizer(dtype=dtype)
    )
    assert supported, reason


@pytest.mark.parametrize("norm", [None, "l1", "l2"])
def test_norm_options_are_supported_by_native_gate(norm):
    supported, reason = _rust_supported_subset(
        RustyHashingVectorizer(norm=norm, n_features=2**10)
    )
    assert supported, reason


def test_unsupported_norm_is_rejected_by_native_gate():
    # sklearn's normalize also accepts 'max'; the Rust kernel only does l1/l2.
    supported, reason = _rust_supported_subset(
        RustyHashingVectorizer(norm="max", n_features=2**10)
    )
    assert not supported
    assert reason == "norm not in {None, 'l1', 'l2'}"

@pytest.mark.parametrize(
    "documents_factory",
    [
        lambda: [],
        lambda: [None],
        lambda: [np.nan],
        lambda: ["valid", None],
        lambda: ["valid", 42],
        lambda: [bytearray(b"text")],
        lambda: [object()],
    ],
    ids=[
        "empty",
        "none",
        "nan",
        "mixed-none",
        "mixed-int",
        "bytearray",
        "object",
    ],
)
def test_invalid_document_behavior_matches_sklearn(documents_factory):
    _assert_matches_sklearn_behavior(documents_factory)


@pytest.mark.parametrize(
    "values",
    [
        [None],
        [np.nan],
        ["valid", None],
        ["valid", 42],
        [bytearray(b"text")],
        [object()],
    ],
    ids=["none", "nan", "mixed-none", "mixed-int", "bytearray", "object"],
)
def test_invalid_generator_behavior_matches_sklearn(values):
    _assert_matches_sklearn_behavior(lambda: (value for value in values))


@pytest.mark.parametrize(
    "document",
    [
        "plain ascii text",
        "café déjà vu",
        "שָׁלוֹם עולם",
        "بِسْمِ الله",
        "A\u0345B",
        "中文 混合 English",
    ],
)
@pytest.mark.parametrize("lowercase", [True, False])
@pytest.mark.parametrize("ngram_range", [(1, 1), (1, 2)])
def test_unicode_behavior_matches_sklearn(document, lowercase, ngram_range):
    _assert_matches_sklearn_behavior(
        lambda: [document],
        params={
            "lowercase": lowercase,
            "ngram_range": ngram_range,
            "n_features": 2**18,
        },
    )


@pytest.mark.parametrize(
    "document",
    [
        "café déjà vu",
        "שָׁלוֹם עולם",
        "بِسْمِ الله",
        "A\u0345B",
        "中文 混合 English",
    ],
)
def test_non_ascii_uses_sklearn_fallback(document, monkeypatch):
    def unexpected_native_call(*args, **kwargs):
        raise AssertionError("non-ASCII input was sent to Rust")

    monkeypatch.setattr(rb, "hashing_vectorizer_transform", unexpected_native_call)
    _assert_matches_sklearn_behavior(lambda: [document])


def test_native_boundary_rejects_non_ascii_input():
    with pytest.raises(ValueError, match="non-ASCII"):
        rb.hashing_vectorizer_transform(
            ["plain ascii", "café"],
            2**18,
            1,
            1,
            False,
            True,
            True,
            "l2",
        )


def test_native_boundary_accepts_max_i32_feature_count():
    docs = ["plain ascii"]
    n_features = 2**31 - 1
    supported, reason = _rust_supported_subset(
        RustyHashingVectorizer(n_features=n_features)
    )
    assert supported, reason

    data, indices, indptr, n_rows, n_cols = rb.hashing_vectorizer_transform(
        docs,
        n_features,
        1,
        1,
        False,
        True,
        True,
        "l2",
    )
    actual = sp.csr_matrix(
        (data, indices, indptr),
        shape=(n_rows, n_cols),
        dtype=np.float64,
    )
    expected = OriginalHashingVectorizer(n_features=n_features).transform(docs)
    _assert_csr_equal(actual, expected)


def test_native_boundary_rejects_feature_count_above_i32_range():
    with pytest.raises(ValueError, match="at most 2147483647"):
        rb.hashing_vectorizer_transform(
            ["plain ascii"],
            2**31,
            1,
            1,
            False,
            True,
            True,
            "l2",
        )


@pytest.mark.parametrize(
    "params",
    [
        {"n_features": 8.9},
        {"n_features": "8"},
        {"n_features": 0},
        {"n_features": 2**31},
        {"binary": 1},
        {"alternate_sign": 1},
        {"ngram_range": (1.0, 2.0)},
        {"ngram_range": (2, 1)},
        {"ngram_range": (1, 2**100)},
    ],
)
def test_fit_transform_parameter_validation_matches_sklearn(params):
    _assert_matches_sklearn_behavior(
        lambda: ["hello world"], method="fit_transform", params=params
    )


@pytest.mark.parametrize(
    "params",
    [
        {"n_features": 8.9},
        {"n_features": "8"},
        {"n_features": 2**31},
        {"binary": 1},
        {"alternate_sign": 1},
        {"ngram_range": (1.0, 2.0)},
        {"ngram_range": (2, 1)},
        {"ngram_range": (1, 2**100)},
    ],
)
def test_transform_parameter_behavior_matches_sklearn(params):
    _assert_matches_sklearn_behavior(lambda: ["hello world"], params=params)


@pytest.mark.parametrize(
    "params",
    [
        {"n_features": 0},
        {"n_features": 2**31},
        {"n_features": 8.9},
        {"n_features": "8"},
        {"binary": 1},
        {"alternate_sign": 1},
        {"lowercase": 1},
        {"ngram_range": (1.0, 2.0)},
        {"ngram_range": (2, 1)},
        {"ngram_range": (1, 2**100)},
    ],
)
def test_invalid_native_parameters_are_rejected_by_gate(params):
    supported, _ = _rust_supported_subset(RustyHashingVectorizer(**params))
    assert not supported


def test_invalid_dtype_is_rejected_by_gate():
    # Mutate after init so the gate's np.dtype(...) exception path is hit.
    vectorizer = RustyHashingVectorizer()
    vectorizer.dtype = "not_a_real_dtype_xyz"
    supported, reason = _rust_supported_subset(vectorizer)
    assert not supported
    assert reason == "invalid dtype"


def test_transform_rejects_y_argument():
    vectorizer = RustyHashingVectorizer()
    with pytest.raises(TypeError):
        vectorizer.transform(["hello world"], None)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("alternate_sign", [False, True])
@pytest.mark.parametrize("norm", [None, "l1", "l2"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matches_sklearn_param_grid(binary, alternate_sign, norm, dtype):
    atol = 1e-6 if dtype == np.float32 else 1e-12
    _assert_close(
        _ASCII_DOCS,
        atol=atol,
        n_features=2**18,
        binary=binary,
        alternate_sign=alternate_sign,
        norm=norm,
        dtype=dtype,
    )


@pytest.mark.parametrize("ngram_range", [(1, 1), (1, 2), (2, 3), (1, 3)])
def test_matches_sklearn_ngram_ranges(ngram_range):
    _assert_close(_ASCII_DOCS, n_features=2**18, ngram_range=ngram_range)


def test_huge_ngram_upper_bound_matches_sklearn(monkeypatch):
    native_transform = rb.hashing_vectorizer_transform
    native_calls = 0

    def counted_transform(*args, **kwargs):
        nonlocal native_calls
        native_calls += 1
        return native_transform(*args, **kwargs)

    monkeypatch.setattr(rb, "hashing_vectorizer_transform", counted_transform)
    _assert_matches_sklearn_behavior(
        lambda: ["one token"],
        params={"n_features": 2**10, "ngram_range": (1, 1_000_000_000)},
    )
    assert native_calls == 1


def test_default_config_matches_sklearn():
    # Full defaults, incl. n_features=2**20 and norm="l2".
    _assert_close(_ASCII_DOCS)


@pytest.mark.parametrize("n_features", [1, 2, 4, 8])
def test_hash_collisions_match_sklearn(n_features):
    docs = [
        "alpha beta gamma delta epsilon zeta eta theta iota kappa alpha beta",
        "collision heavy repeated repeated one two three four five six seven",
    ]
    _assert_close(
        docs,
        n_features=n_features,
        binary=True,
        alternate_sign=True,
    )


def test_lowercase_false_matches_sklearn():
    _assert_close(_ASCII_DOCS, n_features=2**16, lowercase=False)


def test_rows_are_l2_normalized():
    X = _rust_csr(_ASCII_DOCS, n_features=2**18)
    row_norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    # Empty rows have norm 0; non-empty rows must be unit norm.
    nnz_per_row = np.diff(X.indptr)
    for norm, nnz in zip(row_norms, nnz_per_row):
        if nnz > 0:
            assert abs(norm - 1.0) < 1e-6


def test_rows_are_l1_normalized():
    X = _rust_csr(_ASCII_DOCS, n_features=2**18, norm="l1")
    row_norms = np.asarray(np.abs(X).sum(axis=1)).ravel()
    nnz_per_row = np.diff(X.indptr)
    for norm, nnz in zip(row_norms, nnz_per_row):
        if nnz > 0:
            assert abs(norm - 1.0) < 1e-6


def test_float32_output_dtype_uses_native_path(monkeypatch):
    native_transform = rb.hashing_vectorizer_transform
    native_calls = 0

    def counted_transform(*args, **kwargs):
        nonlocal native_calls
        native_calls += 1
        return native_transform(*args, **kwargs)

    monkeypatch.setattr(rb, "hashing_vectorizer_transform", counted_transform)
    X = _rust_csr(_ASCII_DOCS, n_features=2**12, dtype=np.float32, norm="l1")
    assert native_calls == 1
    assert X.dtype == np.float32
    _assert_csr_equal(X, _sklearn_csr(_ASCII_DOCS, n_features=2**12, dtype=np.float32, norm="l1"), atol=1e-6)

def test_empty_rows():
    for docs in ([""], ["", ""], ["a", "b"]):
        X = _rust_csr(docs, n_features=2**15)
        assert X.shape[0] == len(docs)
        assert X.nnz == X.indptr[-1]


@pytest.mark.parametrize("method", ["fit_transform", "transform"])
def test_zero_dimensional_numpy_input_matches_sklearn_exception(method):
    raw_documents = np.array("hello world")
    reference = OriginalHashingVectorizer()
    vectorizer = RustyHashingVectorizer()

    with pytest.raises(Exception) as expected_error:
        getattr(reference, method)(raw_documents)
    with stratum.config(rust_backend=True, allow_patch=True):
        with pytest.raises(Exception) as actual_error:
            getattr(vectorizer, method)(raw_documents)

    assert type(actual_error.value) is type(expected_error.value)
    assert str(actual_error.value) == str(expected_error.value)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("alternate_sign", [False, True])
def test_long_ngram_row_sparse_accumulator_matches_sklearn(
    binary, alternate_sign
):
    # More than COMPACT_FEATURE_LIMIT generated n-grams forces the Rust
    # accumulator onto its bounded sparse path.
    document = " ".join(f"token{index}" for index in range(300))
    _assert_close(
        [document],
        n_features=2**12,
        ngram_range=(1, 3),
        norm=None,
        binary=binary,
        alternate_sign=alternate_sign,
    )


def test_all_string_list_is_not_copied():
    docs = ["hello", "world"]
    assert _materialize_documents(docs) is docs
    assert _to_docs(docs) is docs


def test_materializes_common_containers():
    assert _materialize_documents(("hello", "world")) == ["hello", "world"]
    assert _materialize_documents(doc for doc in ("hello", "world")) == [
        "hello",
        "world",
    ]
    assert _materialize_documents(np.array(["hello", "world"])) == ["hello", "world"]

    pl = pytest.importorskip("polars")
    assert _materialize_documents(pl.Series(["hello", "world"])) == ["hello", "world"]


def test_to_docs_decodes_bytes_without_coercing_other_values():
    docs = ["hello", b"world"]
    converted = _to_docs(docs)
    assert converted == ["hello", "world"]
    assert converted is not docs

    for value in (None, np.nan, 42, object()):
        with pytest.raises(TypeError, match="only str or bytes"):
            _to_docs(["hello", value])


def test_bytes_and_encoding_match_sklearn():
    _assert_close([b"Hello WORLD", "hello world", b"caf\xc3\xa9 caf\xc3\xa9"])
    _assert_close([b"caf\xe9 plain", "café plain"], encoding="latin-1")


@pytest.mark.parametrize("decode_error", ["strict", "ignore", "replace"])
def test_bytes_decode_error_matches_sklearn(decode_error):
    _assert_matches_sklearn_behavior(
        lambda: [b"invalid \xff bytes"],
        params={"encoding": "utf-8", "decode_error": decode_error},
    )


def test_pandas_series_matches_sklearn():
    pd = pytest.importorskip("pandas")
    docs = pd.Series(_DOCS)
    _assert_close(docs, n_features=2**18)


def test_polars_series_matches_sklearn():
    pl = pytest.importorskip("polars")
    docs = pl.Series(_ASCII_DOCS)
    _assert_close(docs, n_features=2**18)


def test_generator_matches_sklearn():
    got = _rust_csr((doc for doc in _DOCS), n_features=2**18)
    ref = _sklearn_csr((doc for doc in _DOCS), n_features=2**18)
    diff = abs(got - ref)
    assert (diff.max() if diff.nnz else 0.0) <= 1e-6


@pytest.mark.parametrize(
    ("values", "expected_error"),
    [
        (["valid ascii text", "another document"], None),
        (["café déjà vu", "another document"], None),
        (["valid ascii text", None], AttributeError),
    ],
    ids=["native", "unicode-fallback", "invalid-fallback"],
)
def test_one_shot_iterable_is_consumed_once(values, expected_error):
    class OneShotDocuments:
        def __init__(self, documents):
            self.documents = documents
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("documents were consumed more than once")
            return iter(self.documents)

    documents = OneShotDocuments(values)
    vectorizer = RustyHashingVectorizer()
    with stratum.config(rust_backend=True, allow_patch=True):
        if expected_error is None:
            vectorizer.transform(documents)
        else:
            with pytest.raises(expected_error):
                vectorizer.transform(documents)
    assert documents.iterations == 1


def test_native_failure_is_logged_and_falls_back(
    monkeypatch, caplog, capsys
):
    def failing_transform(*args, **kwargs):
        raise RuntimeError("native test failure")

    monkeypatch.setattr(rb, "hashing_vectorizer_transform", failing_transform)
    vectorizer = RustyHashingVectorizer(n_features=2**10)
    with (
        caplog.at_level("WARNING"),
        stratum.config(rust_backend=True, allow_patch=True),
    ):
        actual = vectorizer.transform(["plain ascii text"])

    reference = OriginalHashingVectorizer(n_features=2**10).transform(
        ["plain ascii text"]
    )
    _assert_csr_equal(actual, reference)
    assert "falling back to sklearn" in caplog.text
    assert "native test failure" in caplog.text
    assert "WARNING" not in capsys.readouterr().out


def test_native_memory_error_is_not_retried(monkeypatch):
    def failing_transform(*args, **kwargs):
        raise MemoryError("native allocation failed")

    monkeypatch.setattr(rb, "hashing_vectorizer_transform", failing_transform)
    vectorizer = RustyHashingVectorizer()
    with (
        stratum.config(rust_backend=True, allow_patch=True),
        pytest.raises(MemoryError, match="native allocation failed"),
    ):
        vectorizer.transform(["plain ascii text"])


def test_raw_string_uses_sklearn_validation():
    enc = RustyHashingVectorizer()
    with stratum.config(rust_backend=True, allow_patch=True):
        with pytest.raises(ValueError, match="Iterable over raw text documents"):
            enc.transform("not a corpus")


@pytest.mark.parametrize(
    "params",
    [
        {"analyzer": "char"},
        {"stop_words": "english"},
        {"token_pattern": r"(?u)\b\w+\b"},
        {"preprocessor": str.strip},
        {"tokenizer": str.split},
        {"strip_accents": "unicode"},
        {"norm": "max"},
        {"dtype": np.float16},
    ],
)
def test_unsupported_configuration_falls_back(params):
    params = {**params, "n_features": 2**16}
    supported, reason = _rust_supported_subset(RustyHashingVectorizer(**params))
    assert not supported, reason

    _assert_matches_sklearn_behavior(
        lambda: _ASCII_DOCS,
        params=params,
        atol=1e-6,
    )


def test_filename_input_falls_back_and_matches_sklearn(tmp_path):
    paths = []
    for index, document in enumerate(_ASCII_DOCS):
        path = tmp_path / f"document-{index}.txt"
        path.write_text(document)
        paths.append(str(path))

    supported, reason = _rust_supported_subset(
        RustyHashingVectorizer(input="filename", n_features=2**12)
    )
    assert not supported
    assert "input" in reason

    _assert_matches_sklearn_behavior(
        lambda: paths,
        params={"input": "filename", "n_features": 2**12},
    )


def test_unsupported_gate_is_logged(caplog):
    vectorizer = RustyHashingVectorizer(analyzer="char", n_features=2**10)
    with (
        caplog.at_level("DEBUG", logger="stratum.adapters.hashing_vectorizer"),
        stratum.config(rust_backend=True, allow_patch=True),
    ):
        vectorizer.transform(["alpha beta"])
    assert "HashingVectorizer Rust fallback" in caplog.text
    assert "analyzer" in caplog.text


@pytest.mark.parametrize(
    "params",
    [
        {"encoding": None},
        {"encoding": 123},
        {"decode_error": None},
        {"decode_error": 123},
        {"decode_error": "bogus"},
    ],
)
def test_invalid_encoding_parameters_match_sklearn(params):
    _assert_matches_sklearn_behavior(
        lambda: ["hello world"],
        params=params,
    )


def test_patched_symbol_is_adapter():
    import sklearn.feature_extraction.text as text_mod
    import skrub
    import skrub._string_encoder as string_encoder

    assert text_mod.HashingVectorizer is RustyHashingVectorizer
    assert skrub.HashingVectorizer is RustyHashingVectorizer
    assert string_encoder.HashingVectorizer is RustyHashingVectorizer
    assert stratum.HashingVectorizer is RustyHashingVectorizer
