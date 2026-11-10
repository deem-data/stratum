"""StandardScaler parity, native dispatch and conservative fallback regressions."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from stratum import _rust_backend as rb
from stratum._config import config
from stratum.adapters.standard_scaler import NumpyStandardScaler, RustyStandardScaler


@pytest.fixture(params=[NumpyStandardScaler, RustyStandardScaler])
def scaler_cls(request):
    return request.param


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
@pytest.mark.parametrize("flags", [(True, True), (False, True), (True, False), (False, False)])
def test_parameters_and_fitted_state(scaler_cls, dtype, flags):
    X = np.random.default_rng(4).integers(-10, 10, size=(70, 5)).astype(dtype)
    X[:, 0] = 3
    params = dict(with_mean=flags[0], with_std=flags[1])
    expected = StandardScaler(**params).fit(X)
    with config(allow_patch=True, rust_backend=True):
        actual = scaler_cls(**params).fit(X)
        out = actual.transform(X)
    np.testing.assert_allclose(out, expected.transform(X), rtol=1e-6, atol=1e-12)
    assert out.dtype == expected.transform(X).dtype
    for name in ("mean_", "var_", "scale_", "n_samples_seen_", "n_features_in_"):
        if getattr(expected, name) is None:
            assert getattr(actual, name) is None
        else:
            np.testing.assert_allclose(getattr(actual, name), getattr(expected, name))
    np.testing.assert_allclose(actual.inverse_transform(out), X, atol=1e-6)


@pytest.mark.parametrize("kind", ["nan", "weighted", "frame", "sparse", "float16"])
def test_reference_fallbacks(scaler_cls, kind):
    X = np.random.default_rng(5).normal(size=(32, 3))
    kwargs = {}
    params = {}
    if kind == "nan":
        X[::2, 0] = np.nan
        X[:, 1] = np.nan
    elif kind == "weighted":
        kwargs["sample_weight"] = np.arange(1, len(X) + 1)
    elif kind == "frame":
        X = pd.DataFrame(X, columns=["a", "b", "c"])
    elif kind == "sparse":
        X = sparse.csr_matrix(X)
        params["with_mean"] = False
    else:
        X = X.astype(np.float16)
    expected = StandardScaler(**params).fit(X, **kwargs)
    with config(allow_patch=True, rust_backend=True):
        actual = scaler_cls(**params).fit(X, **kwargs)
        result = actual.transform(X)
    target = expected.transform(X)
    if sparse.issparse(target):
        result, target = result.toarray(), target.toarray()
    np.testing.assert_allclose(result, target, rtol=1e-6, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(actual.n_samples_seen_, expected.n_samples_seen_)
    if kind == "frame":
        np.testing.assert_array_equal(actual.feature_names_in_, expected.feature_names_in_)
        with pytest.raises(ValueError, match="feature names"):
            actual.transform(X[["b", "a", "c"]])


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_offsets_and_layouts(scaler_cls, layout):
    X = np.random.default_rng(7).normal(size=(1001, 6))
    X[:, 0] = 1e9 + X[:, 0] * 1e-3
    X[:, 1] = 1e12
    if layout == "F":
        X = np.asfortranarray(X)
    elif layout == "strided":
        X = X[::-2, ::-1]
    expected = StandardScaler().fit(X)
    with config(allow_patch=True, rust_backend=True):
        actual = scaler_cls().fit(X)
        np.testing.assert_allclose(actual.transform(X), expected.transform(X), atol=1e-3)
    np.testing.assert_allclose(actual.var_, expected.var_, rtol=1e-8, atol=1e-12)
    assert np.isfinite(actual.scale_).all()


@pytest.mark.parametrize("copy", [None, True, False])
def test_copy_override(scaler_cls, copy):
    X = np.random.default_rng(8).normal(size=(30, 2))
    original = X.copy()
    with config(allow_patch=True, rust_backend=True):
        scaler = scaler_cls(copy=True).fit(X)
        result = scaler.transform(X, copy=copy)
    assert np.shares_memory(result, X) == (copy is False)
    if copy is not False:
        np.testing.assert_array_equal(X, original)
    np.testing.assert_allclose(result, StandardScaler().fit_transform(original), atol=1e-12)


def test_incremental_fit_and_refit(scaler_cls):
    X = np.random.default_rng(9).normal(size=(50, 4))
    with config(allow_patch=True, rust_backend=True):
        actual = scaler_cls().fit(X[:20]).partial_fit(X[20:])
        expected = StandardScaler().fit(X)
        np.testing.assert_allclose(actual.transform(X), expected.transform(X), atol=1e-12)
        actual.fit(X[:, :2])
    assert actual.n_features_in_ == 2
    assert actual.n_samples_seen_ == 50


@pytest.mark.parametrize("reuse_mean", [False, True])
def test_numpy_reuse_mean_and_inplace_copy(reuse_mean):
    X = np.random.default_rng(13).normal(size=(100, 4))
    X[:, 0] = 2
    expected = StandardScaler().fit_transform(X)
    actual = NumpyStandardScaler(copy=False, reuse_mean=reuse_mean)
    assert actual.fit_transform(X) is X
    np.testing.assert_allclose(X, expected, atol=1e-12)


@pytest.mark.parametrize("workers", [1, 2, np.int64(3)])
def test_worker_budget_and_clone(workers):
    scaler = clone(RustyStandardScaler()._bind_native_worker_budget(workers))
    assert scaler._stratum_worker_budget == workers
    assert "n_jobs" not in scaler.get_params()
    with config(num_threads=8):
        assert scaler._native_workers() == workers


@pytest.mark.parametrize("workers", [0, -1, 1.5, None])
def test_invalid_worker_budget(workers):
    with pytest.raises(ValueError, match="native worker budget"):
        RustyStandardScaler()._bind_native_worker_budget(workers)


def test_no_public_n_jobs():
    with pytest.raises(TypeError, match="n_jobs"):
        RustyStandardScaler(n_jobs=2)


def test_standalone_worker_budget_uses_config():
    with config(num_threads=2):
        assert RustyStandardScaler()._native_workers() == 2


@pytest.mark.parametrize("X", [np.empty((0, 2)), np.empty((2, 0)), np.ones(3), [[np.inf]]])
def test_invalid_inputs(scaler_cls, X):
    with config(allow_patch=True, rust_backend=True), pytest.raises(ValueError):
        scaler_cls().fit(X)


def test_unfitted_and_feature_count(scaler_cls):
    with pytest.raises(NotFittedError):
        scaler_cls().transform(np.ones((2, 2)))
    with config(allow_patch=True, rust_backend=True):
        scaler = scaler_cls().fit(np.ones((3, 2)))
        with pytest.raises(ValueError, match="features"):
            scaler.transform(np.ones((3, 3)))


@pytest.mark.skipif(rb.standard_scale_fit is None, reason="native scaler unavailable")
def test_native_kernels_are_reached_and_binding_survives_clone(monkeypatch):
    calls = []
    fit, transform = rb.standard_scale_fit, rb.standard_scale_transform

    def record_fit(*args, **kwargs):
        calls.append("fit")
        assert kwargs["worker_budget"] == 2
        return fit(*args, **kwargs)

    def record_transform(*args, **kwargs):
        calls.append("transform")
        assert kwargs["worker_budget"] == 2
        return transform(*args, **kwargs)

    monkeypatch.setattr(rb, "standard_scale_fit", record_fit)
    monkeypatch.setattr(rb, "standard_scale_transform", record_transform)
    scaler = RustyStandardScaler()._bind_native_worker_budget(2)
    scaler._stratum_force_rust = True
    scaler = clone(scaler)
    X = np.random.default_rng(10).normal(size=(21001, 3))
    with config(allow_patch=False, rust_backend=False):
        result = scaler.fit_transform(X)
    assert calls == ["fit", "transform"]
    np.testing.assert_allclose(result, StandardScaler().fit_transform(X), atol=1e-12)


@pytest.mark.skipif(rb.standard_scale_fit is None, reason="native scaler unavailable")
def test_native_validation():
    X = np.ones((3, 2))
    with pytest.raises(ValueError):
        rb.standard_scale_fit(X, 0)
    with pytest.raises(ValueError):
        rb.standard_scale_fit(np.empty((0, 2)), 1)
    with pytest.raises(ValueError):
        rb.standard_scale_transform(X, np.zeros(1), np.ones(2), 1)
    with pytest.raises(ValueError, match="worker budget"):
        rb.standard_scale_transform(X, np.zeros(2), np.ones(2), 0)


@pytest.mark.skipif(rb.standard_scale_fit is None, reason="native scaler unavailable")
def test_native_worker_budgets_can_change_between_calls():
    X = np.random.default_rng(24).normal(size=(21001, 4))
    expected = StandardScaler().fit_transform(X)
    with config(allow_patch=True, rust_backend=True):
        for workers in (1, 2, 1):
            scaler = RustyStandardScaler()._bind_native_worker_budget(workers)
            np.testing.assert_allclose(scaler.fit_transform(X), expected, atol=1e-12)


def test_disabled_rust_does_not_dispatch(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("disabled native kernel called")
    monkeypatch.setattr(rb, "standard_scale_fit", unexpected)
    monkeypatch.setattr(rb, "standard_scale_transform", unexpected)
    X = np.arange(12, dtype=float).reshape(6, 2)
    with config(allow_patch=True, rust_backend=False):
        np.testing.assert_allclose(RustyStandardScaler().fit_transform(X),
                                   StandardScaler().fit_transform(X))
