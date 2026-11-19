"""SimpleImputer parity, native dispatch, and reference fallbacks."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer

from stratum import _rust_backend as rb
from stratum._config import config
from stratum.adapters.simple_imputer import RustySimpleImputer


def _native(**params):
    estimator = RustySimpleImputer(**params)._bind_native_worker_budget(2)
    estimator._stratum_force_rust = True
    return estimator


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent", "constant"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("indicator", [False, True])
@pytest.mark.parametrize("keep", [False, True])
def test_fitted_state_and_transform(strategy, dtype, indicator, keep):
    rng = np.random.default_rng(30)
    X = rng.integers(-5, 6, size=(51, 6)).astype(dtype)
    X[::3, 0] = np.nan
    X[::5, 1] = np.nan
    X[:, 2] = np.nan
    X[:, 3] = 7
    test = X[:7].copy()
    test[:, 4] = np.nan  # New missingness must not add an indicator column.
    test[:, 2] = 100  # A fitted-empty column stays dropped or retained at transform.
    params = dict(strategy=strategy, add_indicator=indicator, keep_empty_features=keep)
    if strategy == "constant":
        params["fill_value"] = -999
    expected = SimpleImputer(**params).fit(X)
    actual = _native(**params).fit(X)
    np.testing.assert_allclose(np.asarray(actual.statistics_, float),
                               np.asarray(expected.statistics_, float), rtol=1e-6, equal_nan=True)
    assert actual.statistics_.dtype == expected.statistics_.dtype
    assert actual.n_features_in_ == expected.n_features_in_
    np.testing.assert_array_equal(actual.get_feature_names_out(), expected.get_feature_names_out())
    assert (actual.indicator_ is None) == (expected.indicator_ is None)
    if indicator:
        np.testing.assert_array_equal(actual.indicator_.features_, expected.indicator_.features_)
    for source in (X, test):
        original = source.copy()
        result = actual.transform(source)
        target = expected.transform(source)
        assert result.dtype == target.dtype
        np.testing.assert_allclose(result, target, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("layout", ["C", "F", "strided", "readonly", "frame"])
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
def test_layouts_and_names(layout, strategy):
    X = np.random.default_rng(1).normal(size=(50, 6))
    X[::3, ::2] = np.nan
    if layout == "F":
        X = np.asfortranarray(X)
    elif layout == "strided":
        X = X[::-2, ::-1]
    elif layout == "readonly":
        X.setflags(write=False)
    elif layout == "frame":
        X = pd.DataFrame(X, columns=list("abcdef"), index=np.arange(50) * 2)
    reference = SimpleImputer(strategy=strategy, add_indicator=True)
    actual = _native(strategy=strategy, add_indicator=True)
    if layout == "frame":
        actual.set_output(transform="pandas")
        reference.set_output(transform="pandas")
        pd.testing.assert_frame_equal(actual.fit_transform(X), reference.fit_transform(X))
        with pytest.raises(ValueError, match="feature names"):
            actual.transform(X[list("bacdef")])
    else:
        np.testing.assert_allclose(actual.fit_transform(X), reference.fit_transform(X), atol=1e-12)


@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent", "constant"])
@pytest.mark.parametrize("indicator", [False, True])
def test_all_empty_and_no_missing(strategy, indicator):
    params = dict(strategy=strategy, add_indicator=indicator)
    for X in (np.full((5, 3), np.nan), np.ones((5, 3))):
        actual, expected = _native(**params), SimpleImputer(**params)
        np.testing.assert_array_equal(actual.fit_transform(X), expected.fit_transform(X))


@pytest.mark.parametrize("kind", ["strings", "string_constant", "sparse", "sentinel", "integer", "copy_false", "callable"])
def test_reference_fallbacks(monkeypatch, kind):
    def unexpected(*args, **kwargs):
        pytest.fail("unsupported input reached a native kernel")
    monkeypatch.setattr(rb, "simple_imputer_fit", unexpected)
    monkeypatch.setattr(rb, "simple_imputer_transform", unexpected)
    X = np.array([[1., np.nan], [2., 4.], [np.nan, 6.]])
    params = dict(strategy="median", add_indicator=True)
    if kind in ("strings", "string_constant"):
        X = np.array([["a", np.nan], ["b", "z"]], dtype=object)
        params = dict(strategy="most_frequent") if kind == "strings" else dict(
            strategy="constant", fill_value="missing")
    elif kind == "sparse":
        X = sparse.csc_matrix(X)
    elif kind == "sentinel":
        X = np.nan_to_num(X, nan=-1)
        params["missing_values"] = -1
    elif kind == "integer":
        X = np.nan_to_num(X).astype(int)
    elif kind == "copy_false":
        params["copy"] = False
    else:
        params["strategy"] = np.nanmax
    actual = _native(**params).fit_transform(X.copy())
    expected = SimpleImputer(**params).fit_transform(X.copy())
    if sparse.issparse(expected):
        actual, expected = actual.toarray(), expected.toarray()
    np.testing.assert_array_equal(actual, expected)


def test_native_calls_binding_and_refit(monkeypatch):
    calls = []
    for name in ("simple_imputer_fit", "simple_imputer_transform"):
        native = getattr(rb, name)
        def record(*args, _native=native, _name=name, **kwargs):
            assert kwargs["worker_budget"] == 2
            calls.append(_name)
            return _native(*args, **kwargs)
        monkeypatch.setattr(rb, name, record)
    estimator = clone(_native(strategy="median", add_indicator=True))
    assert estimator._stratum_worker_budget == 2
    X = np.array([[1., np.nan], [3., 4.]])
    with config(rust_backend=False, allow_patch=False, num_threads=8):
        estimator.fit_transform(X)
        estimator.fit_transform(X[:, :1])
        assert estimator.n_features_in_ == 1
        assert estimator.indicator_.features_.size == 0
    assert calls == ["simple_imputer_fit", "simple_imputer_transform"] * 2


@pytest.mark.parametrize("workers", [1, 2, 1])
def test_standalone_worker_config(workers):
    with config(rust_backend=True, allow_patch=True, num_threads=workers):
        estimator = RustySimpleImputer(strategy="median")
        assert estimator._native_workers() == workers
        np.testing.assert_array_equal(estimator.fit_transform([[1., np.nan], [3., 4.]]),
                                      [[1., 4.], [3., 4.]])


@pytest.mark.parametrize("X", [np.empty((0, 2)), np.empty((2, 0)), np.ones(3), [[np.inf]]])
def test_invalid_input(X):
    with pytest.raises(ValueError):
        _native().fit(X)


def test_unfitted_feature_count_and_dtype_drift():
    with pytest.raises(NotFittedError):
        _native().transform(np.ones((2, 2)))
    estimator = _native(strategy="median").fit(np.array([[1., np.nan], [3., 4.]], dtype=np.float32))
    with pytest.raises(ValueError, match="features"):
        estimator.transform(np.ones((2, 3)))
    test = np.array([[np.nan, 6.]], dtype=np.float64)
    reference = SimpleImputer(strategy="median").fit(np.array([[1., np.nan], [3., 4.]], dtype=np.float32))
    np.testing.assert_array_equal(estimator.transform(test), reference.transform(test))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["C", "F", "one_column", "strided"])
def test_mean_cancellation_matches_reference(dtype, layout):
    X = np.tile([1e20, 1., -1e20, 1.], 256).astype(dtype).reshape(-1, 1)
    X = np.concatenate([X, X], axis=1)
    if layout == "F":
        X = np.asfortranarray(X)
    elif layout == "one_column":
        X = X[:, :1]
    elif layout == "strided":
        X = X[::-1]
    actual, expected = _native(strategy="mean"), SimpleImputer(strategy="mean")
    actual.fit(X)
    expected.fit(X)
    np.testing.assert_array_equal(actual.statistics_, expected.statistics_)
    test = np.full_like(X[:3], np.nan)
    np.testing.assert_array_equal(actual.transform(test), expected.transform(test))


def test_ignored_fill_value_and_reference_refit():
    X = np.array([[1., np.nan], [3., 4.]])
    estimator = _native(strategy="median", fill_value="unused")
    np.testing.assert_array_equal(estimator.fit_transform(X), SimpleImputer(
        strategy="median", fill_value="unused").fit_transform(X))
    estimator.set_params(strategy="constant", fill_value="missing", add_indicator=True)
    objects = np.array([["a", np.nan], ["b", "c"]], dtype=object)
    np.testing.assert_array_equal(estimator.fit_transform(objects), SimpleImputer(
        strategy="constant", fill_value="missing", add_indicator=True).fit_transform(objects))
    assert not estimator._stratum_native_fit


@pytest.mark.parametrize("workers", [0, -1, 1.5, None])
def test_invalid_worker_budget(workers):
    with pytest.raises(ValueError, match="native worker budget"):
        RustySimpleImputer()._bind_native_worker_budget(workers)


def test_disabled_native_and_native_validation(monkeypatch):
    X = np.ones((4, 2))
    fit = rb.simple_imputer_fit
    with pytest.raises(ValueError, match="worker budget"):
        fit(X, strategy="median", keep_empty=False, fill_value=0., worker_budget=0)
    with pytest.raises(ValueError, match="nonempty"):
        fit(X[:0], strategy="median", keep_empty=False, fill_value=0., worker_budget=1)
    with pytest.raises(ValueError, match="strategy"):
        fit(X, strategy="unknown", keep_empty=False, fill_value=0., worker_budget=1)
    with pytest.raises(ValueError, match="indices mismatch"):
        rb.simple_imputer_transform(X, np.zeros(2), [2], [], 1)
    def unexpected(*args, **kwargs):
        pytest.fail("disabled native kernel called")
    monkeypatch.setattr(rb, "simple_imputer_fit", unexpected)
    with config(rust_backend=False, allow_patch=False):
        np.testing.assert_array_equal(RustySimpleImputer().fit_transform(X), X)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_inverse_and_random_even_median(dtype):
    X = np.random.default_rng(8).normal(size=(60, 4)).astype(dtype)
    X[::2, 0] = np.nan
    X[::3, 1] = np.nan
    actual = _native(strategy="median", add_indicator=True)
    reference = SimpleImputer(strategy="median", add_indicator=True)
    result = actual.fit_transform(X)
    np.testing.assert_array_equal(actual.statistics_, reference.fit(X).statistics_)
    np.testing.assert_array_equal(result, reference.transform(X))
    np.testing.assert_array_equal(actual.inverse_transform(result), reference.inverse_transform(result))
