"""Tests for row normalization, kernel and adapter.

Three layers, in the order a value passes through them:

  1. the Rust kernels, called directly through ``_rust_backend`` — do they
     compute the right numbers;
  2. the ``normalize`` function — does the fastpath match sklearn, and does
     every input the kernels cannot serve fall back cleanly;
  3. ``RustyNormalizer`` — does the transformer behave like sklearn's.

The property most of these turn on: a fallback must be invisible apart from
being slower. Same values, same dtype, same copy semantics as sklearn.
"""
import os
os.environ.setdefault("SKRUB_RUST", "1")

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer as SKNormalizer
from sklearn.preprocessing import normalize as sk_normalize

from stratum import _rust_backend as rb
from stratum import set_config
from stratum.adapters.normalizer import (
    _KERNELS,
    _fastpath_kernels,
    RustyNormalizer,
    normalize,
)

pytestmark = pytest.mark.skipif(not rb.HAVE_RUST, reason="Rust backend not built")

EPS = 1e-5
NORMS = ["l1", "l2", "max"]
RTOL, ATOL = 1e-5, 1e-6


def _f32(arr):
    return np.asarray(arr, dtype=np.float32)


@pytest.fixture(autouse=True)
def _rust_on():
    """Pin the Rust path on for every test, and restore it afterwards.

    Setting SKRUB_RUST at module import is not enough: it only takes effect if
    nothing imported stratum first. Without this the tests still pass, because
    the fallback returns the same values, and the fastpath goes unexercised.
    """
    set_config(rust_backend=True, allow_patch=True)
    yield
    set_config(rust_backend=True, allow_patch=True)


def used_rust(monkeypatch, norm):
    """Record which kernel for `norm` is invoked; empty means sklearn."""
    calls = []
    copy_name, inplace_name = _KERNELS[norm]
    for name in (copy_name, inplace_name):
        real = getattr(rb, name)

        def spy(*a, _real=real, _name=name, **kw):
            calls.append(_name)
            return _real(*a, **kw)

        monkeypatch.setattr(rb, name, spy)
    return calls


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((257, 37)).astype(np.float32)
    data[3] = 0.0  # zero row: norm is 0, must be left alone rather than dividing
    return data


# ══ 1. the Rust kernels ═══════════════════════════════════════════════════════

class TestNormalizeL2:
    def test_basic_row_unit_norm(self):
        data = _f32([[3, 4], [1, 0], [5, 12]])
        out = rb.normalize_l2(data)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=EPS)

    def test_zero_row_unchanged(self):
        data = _f32([[0, 0, 0], [1, 0, 0]])
        out = rb.normalize_l2(data)
        np.testing.assert_allclose(out[0], [0, 0, 0], atol=EPS)
        np.testing.assert_allclose(out[1], [1, 0, 0], atol=EPS)

    def test_negative_values(self):
        data = _f32([[-3, 4]])
        out = rb.normalize_l2(data)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0], atol=EPS)
        assert out[0, 0] < 0  # sign preserved

    def test_does_not_modify_input(self):
        data = _f32([[3, 4]])
        original = data.copy()
        rb.normalize_l2(data)
        np.testing.assert_array_equal(data, original)

    def test_single_row(self):
        data = _f32([[0, 2, 0]])
        out = rb.normalize_l2(data)
        np.testing.assert_allclose(out, [[0, 1, 0]], atol=EPS)

    def test_output_dtype_is_float32(self):
        data = _f32([[1, 2]])
        out = rb.normalize_l2(data)
        assert out.dtype == np.float32


class TestNormalizeL1:
    def test_basic_row_unit_l1_norm(self):
        data = _f32([[3, 1, 2], [0, 4, 0]])
        out = rb.normalize_l1(data)
        l1_norms = np.abs(out).sum(axis=1)
        np.testing.assert_allclose(l1_norms, 1.0, atol=EPS)

    def test_negative_values_absolute_sum(self):
        data = _f32([[-2, 2]])
        out = rb.normalize_l1(data)
        np.testing.assert_allclose(out, [[-0.5, 0.5]], atol=EPS)

    def test_zero_row_unchanged(self):
        data = _f32([[0, 0]])
        out = rb.normalize_l1(data)
        np.testing.assert_allclose(out[0], [0, 0], atol=EPS)

    def test_known_values(self):
        data = _f32([[6, 3, 1]])
        out = rb.normalize_l1(data)
        np.testing.assert_allclose(out, [[0.6, 0.3, 0.1]], atol=EPS)


class TestNormalizeMax:
    def test_max_abs_becomes_one(self):
        data = _f32([[2, -6, 3], [10, 1, 0]])
        out = rb.normalize_max(data)
        max_abs = np.abs(out).max(axis=1)
        np.testing.assert_allclose(max_abs, 1.0, atol=EPS)

    def test_negative_max(self):
        data = _f32([[1, -4]])
        out = rb.normalize_max(data)
        np.testing.assert_allclose(np.abs(out).max(), 1.0, atol=EPS)
        assert out[0, 1] == pytest.approx(-1.0, abs=EPS)

    def test_zero_row_unchanged(self):
        data = _f32([[0, 0]])
        out = rb.normalize_max(data)
        np.testing.assert_allclose(out[0], [0, 0], atol=EPS)

    def test_known_values(self):
        data = _f32([[2, -6, 3]])
        out = rb.normalize_max(data)
        np.testing.assert_allclose(out, [[2/6, -1.0, 3/6]], atol=EPS)


# ══ 2. the normalize() function ═══════════════════════════════════════════════

class TestFastpath:
    """Inputs the kernels serve directly: dense, C-order, float32, axis=1."""

    @pytest.mark.parametrize("norm", NORMS)
    def test_matches_sklearn(self, X, norm):
        got = normalize(X.copy(), norm=norm)
        ref = sk_normalize(X.copy(), norm=norm)
        assert got.dtype == ref.dtype
        np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("norm", NORMS)
    def test_zero_row_preserved(self, X, norm):
        assert np.all(normalize(X.copy(), norm=norm)[3] == 0)

    @pytest.mark.parametrize("norm", NORMS)
    def test_copy_true_does_not_mutate_input(self, X, norm):
        buf = X.copy()
        normalize(buf, norm=norm, copy=True)
        np.testing.assert_array_equal(buf, X)

    @pytest.mark.parametrize("norm", NORMS)
    def test_copy_false_normalizes_in_place(self, X, norm):
        buf = X.copy()
        out = normalize(buf, norm=norm, copy=False)
        assert out is buf, "copy=False must return the caller's own array"
        np.testing.assert_allclose(buf, sk_normalize(X.copy(), norm=norm),
                                   rtol=RTOL, atol=ATOL)


class TestFallback:
    """Everything the kernels cannot serve must reach sklearn unchanged."""

    def test_float64_keeps_dtype(self, X):
        X64 = X.astype(np.float64)
        got = normalize(X64.copy())
        assert got.dtype == np.float64
        np.testing.assert_allclose(got, sk_normalize(X64.copy()))

    def test_axis0(self, X):
        np.testing.assert_allclose(
            normalize(X.copy(), axis=0), sk_normalize(X.copy(), axis=0),
            rtol=RTOL, atol=ATOL,
        )

    def test_return_norm(self, X):
        got, got_n = normalize(X.copy(), return_norm=True)
        ref, ref_n = sk_normalize(X.copy(), return_norm=True)
        np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(got_n, ref_n, rtol=RTOL, atol=ATOL)

    def test_sparse(self, X):
        S = sp.csr_matrix(X)
        np.testing.assert_allclose(
            normalize(S).toarray(), sk_normalize(S).toarray(), rtol=RTOL, atol=ATOL
        )

    def test_fortran_order(self, X):
        Xf = np.asfortranarray(X)
        np.testing.assert_allclose(normalize(Xf), sk_normalize(Xf),
                                   rtol=RTOL, atol=ATOL)

    def test_readonly_input_with_copy_false(self, X):
        ro = X.copy()
        ro.flags.writeable = False
        # sklearn cannot write through it either, so it returns a normalized copy
        np.testing.assert_allclose(
            normalize(ro, copy=False), sk_normalize(X.copy()), rtol=RTOL, atol=ATOL
        )

    def test_unknown_norm_raises_like_sklearn(self, X):
        with pytest.raises(ValueError):
            normalize(X.copy(), norm="l3")


# ══ 3. the transformer ════════════════════════════════════════════════════════

class TestTransformer:
    @pytest.mark.parametrize("norm", NORMS)
    def test_matches_sklearn(self, X, norm):
        np.testing.assert_allclose(
            RustyNormalizer(norm=norm).fit_transform(X.copy()),
            SKNormalizer(norm=norm).fit_transform(X.copy()),
            rtol=RTOL, atol=ATOL,
        )

    def test_copy_false_in_place(self, X):
        est = RustyNormalizer(norm="l2", copy=False).fit(X)
        buf = X.copy()
        assert est.transform(buf) is buf

    def test_checks_n_features(self, X):
        est = RustyNormalizer().fit(X)
        with pytest.raises(ValueError, match="features"):
            est.transform(np.zeros((4, 5), dtype=np.float32))

    def test_in_pipeline(self, X):
        pipe = make_pipeline(RustyNormalizer(norm="l1"))
        np.testing.assert_allclose(
            pipe.fit_transform(X.copy()),
            SKNormalizer(norm="l1").fit_transform(X.copy()),
            rtol=RTOL, atol=ATOL,
        )

    def test_get_params_roundtrip(self):
        assert RustyNormalizer(norm="max", copy=False).get_params() == {
            "norm": "max", "copy": False}


class TestFastpathIsActuallyTaken:
    """Guard against the fastpath silently going unexercised.

    The other tests compare against sklearn, so a fallback passes just as well.
    Only a spy on the kernel shows whether the Rust code ran.
    """

    @pytest.mark.parametrize("norm", NORMS)
    def test_copy_path_calls_the_kernel(self, X, norm, monkeypatch):
        calls = used_rust(monkeypatch, norm)
        normalize(X.copy(), norm=norm, copy=True)
        assert calls == [_KERNELS[norm][0]]

    @pytest.mark.parametrize("norm", NORMS)
    def test_inplace_path_calls_the_inplace_kernel(self, X, norm, monkeypatch):
        calls = used_rust(monkeypatch, norm)
        normalize(X.copy(), norm=norm, copy=False)
        assert calls == [_KERNELS[norm][1]]

    @pytest.mark.parametrize("norm", NORMS)
    def test_transformer_reaches_the_kernel(self, X, norm, monkeypatch):
        est = RustyNormalizer(norm=norm).fit(X)
        calls = used_rust(monkeypatch, norm)
        est.transform(X.copy())
        assert calls, "RustyNormalizer.transform fell back to sklearn"

    def test_fallback_does_not_call_any_kernel(self, X, monkeypatch):
        calls = used_rust(monkeypatch, "l2")
        normalize(X.astype(np.float64), norm="l2")
        assert calls == []


class TestFastpathGuards:
    """The conditions under which _fastpath_kernels declines to dispatch."""

    @pytest.mark.parametrize("flag", ["rust_backend", "allow_patch"])
    def test_disabled_by_config(self, flag):
        set_config(**{flag: False})
        assert _fastpath_kernels("l2") is None

    def test_unknown_norm(self):
        assert _fastpath_kernels("nope") is None

    def test_missing_binding(self, monkeypatch):
        """An extension built without the in-place kernels must fall back."""
        monkeypatch.setattr(rb, _KERNELS["l2"][1], None, raising=True)
        assert _fastpath_kernels("l2") is None

    def test_transform_falls_back_when_disabled(self, X):
        est = RustyNormalizer(norm="l2").fit(X)
        set_config(rust_backend=False)
        np.testing.assert_allclose(
            est.transform(X.copy()), sk_normalize(X.copy(), norm="l2"),
            rtol=RTOL, atol=ATOL)

    def test_debug_info_announces_the_dispatch(self, X, capsys, monkeypatch):
        monkeypatch.setattr(
            "stratum.adapters.normalizer._DEBUG_INFO", True, raising=True)
        normalize(X.copy(), norm="l2")
        assert "Dispatching normalize(norm='l2'" in capsys.readouterr().out
