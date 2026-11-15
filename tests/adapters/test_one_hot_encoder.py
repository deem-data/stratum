import sys
import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
import sklearn.preprocessing
from stratum import set_config, OneHotEncoder
from stratum.adapters.one_hot_encoder import _codes_from_categories
from stratum import _rust_backend as rb
from stratum.adapters.one_hot_encoder import RustyOneHotEncoder

def capture_std_out(capfd):
    # Capture timing output
    sys.stdout.flush()
    sys.stderr.flush()
    captured = capfd.readouterr()
    combined_output = (captured.out or "") + (captured.err or "")
    return combined_output

@pytest.mark.parametrize("sparse_output", [True, False])
def test_ohe_compare(sparse_output, capfd):
    X = pd.DataFrame({
        #"a": ["x","y",None,"x","z",None], #failing case due to Pandas categorical/codes vs sklearn _encode
        "a": ["x","y","x","x","z","y"],
        "b": ["c","c","d","e","d","c"],
    })
    set_config(rust_backend=True, debug_timing=True)
    enc = OneHotEncoder(sparse_output=sparse_output) #stratum
    Z = enc.fit_transform(X)
    #print(Z)

    sk = sklearn.preprocessing.OneHotEncoder(drop="if_binary", dtype=np.float32, handle_unknown="ignore", sparse_output=sparse_output)
    Z_ref = sk.fit_transform(X)
    #print(Z_ref)

    if sparse_output:
        assert_array_equal(Z.data, Z_ref.data)
        assert_array_equal(Z.indices, Z_ref.indices)
        assert_array_equal(Z.indptr, Z_ref.indptr)
    else: #dense
        assert_array_equal(Z, Z_ref)
        # Assert if rust timing appeared (verifies that rust code is executed)
        assert "[rust]" in capture_std_out(capfd)


def test_ohe_preserves_fitted_missing_category():
    X = pd.DataFrame({"a": ["x", None, None]})
    reference_encoder = sklearn.preprocessing.OneHotEncoder(
        drop="if_binary",
        dtype=np.float32,
        handle_unknown="ignore",
        sparse_output=False,
    ).set_output(transform="pandas")
    reference = reference_encoder.fit_transform(X)

    encoder = OneHotEncoder(
        drop="if_binary",
        dtype=np.float32,
        handle_unknown="ignore",
        sparse_output=False,
    )
    encoder.set_output(transform="pandas")
    encoder._stratum_force_rust = True
    actual = encoder.fit_transform(X)

    pd.testing.assert_frame_equal(actual, reference)
    assert actual.shape == (3, 1)
    codes, n_cats = _codes_from_categories(X, encoder.categories_)
    assert_array_equal(codes[0], np.array([0, 1, 1], dtype=np.int32))
    assert n_cats == [2]


@pytest.mark.skipif(not rb.HAVE_RUST, reason="native OHE unavailable")
@pytest.mark.parametrize("drop", [None, "if_binary"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, "float32", np.dtype("float64")])
@pytest.mark.parametrize("sparse_output", [False, True])
@pytest.mark.parametrize("kind", ["strings", "numeric", "categorical"])
def test_native_options_match_sklearn(monkeypatch, drop, dtype, sparse_output, kind):
    # Include fitted missing categories, binary and constant columns, and values
    # seen only at prediction time. Category discovery/lookup stays in sklearn.
    if kind == "numeric":
        train = pd.DataFrame({"a": [1., 2., np.nan, 1.], "b": [0, 1, 0, 1]})
        test = pd.DataFrame({"a": [3., np.nan, 2.], "b": [2, 1, 0]})
    else:
        train = pd.DataFrame({"a": ["x", "y", None, "x"], "b": ["L", "R", "L", "R"]})
        test = pd.DataFrame({"a": ["unseen", None, "y"], "b": ["new", "R", "L"]})
        if kind == "categorical":
            train = train.astype("category")
            test = test.astype("category")
    train["constant"] = "fixed"
    test["constant"] = ["fixed", "new", "fixed"]
    params = dict(drop=drop, dtype=dtype, sparse_output=sparse_output, handle_unknown="ignore")
    reference = sklearn.preprocessing.OneHotEncoder(**params)
    encoder = RustyOneHotEncoder(**params)
    encoder._stratum_force_rust = True
    calls = []
    native = rb.ohe_transform

    def emit(*args, **kwargs):
        result = native(*args, **kwargs)
        assert result[0].dtype == np.dtype(dtype)
        calls.append(True)
        return result

    monkeypatch.setattr(rb, "ohe_transform", emit)
    for actual, expected in (
        (encoder.fit_transform(train), reference.fit_transform(train)),
        (encoder.transform(test), reference.transform(test)),
    ):
        assert actual.dtype == expected.dtype == np.dtype(dtype)
        assert actual.shape == expected.shape
        if sparse_output:
            assert_array_equal(actual.data, expected.data)
            assert_array_equal(actual.indices, expected.indices)
            assert_array_equal(actual.indptr, expected.indptr)
        else:
            assert_array_equal(actual, expected)
    assert len(calls) == 2
    assert_array_equal(encoder.get_feature_names_out(), reference.get_feature_names_out())
    for actual, expected in zip(encoder.categories_, reference.categories_):
        pd.testing.assert_index_equal(pd.Index(actual), pd.Index(expected))


@pytest.mark.parametrize("drop", [None, "if_binary"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_dataframe_output(drop, dtype):
    train = pd.DataFrame({"deck": ["A", "B", None], "side": ["P", "S", "P"]}, index=[9, 4, 7])
    params = dict(drop=drop, dtype=dtype, handle_unknown="ignore", sparse_output=False)
    reference = sklearn.preprocessing.OneHotEncoder(**params).set_output(transform="pandas")
    encoder = RustyOneHotEncoder(**params).set_output(transform="pandas")
    encoder._stratum_force_rust = True
    pd.testing.assert_frame_equal(encoder.fit_transform(train), reference.fit_transform(train))


@pytest.mark.parametrize("drop", [None, "if_binary"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse_output", [False, True])
def test_all_unknown_rows_emit_zeros(drop, dtype, sparse_output):
    params = dict(drop=drop, dtype=dtype, handle_unknown="ignore", sparse_output=sparse_output)
    train = np.array([["x"], ["y"]])
    test = np.array([["unseen"], ["other"]])
    reference = sklearn.preprocessing.OneHotEncoder(**params).fit(train)
    encoder = RustyOneHotEncoder(**params).fit(train)
    encoder._stratum_force_rust = True
    actual, expected = encoder.transform(test), reference.transform(test)
    assert actual.dtype == expected.dtype == np.dtype(dtype)
    assert actual.shape == expected.shape
    if sparse_output:
        assert actual.nnz == 0
        assert_array_equal(actual.indptr, expected.indptr)
    else:
        assert_array_equal(actual, expected)


@pytest.mark.parametrize("drop", [None, "if_binary"])
def test_empty_fit_and_feature_count_errors(drop):
    params = dict(drop=drop, dtype=np.float64, handle_unknown="ignore", sparse_output=False)
    for cls in (sklearn.preprocessing.OneHotEncoder, RustyOneHotEncoder):
        encoder = cls(**params)
        encoder._stratum_force_rust = True
        with pytest.raises(ValueError):
            encoder.fit(np.empty((0, 2)))
        encoder.fit(np.array([["x", "a"], ["y", "b"]]))
        with pytest.raises(ValueError):
            encoder.transform(np.array([["x"]]))
