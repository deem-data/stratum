"""Tests for the RustyElasticNet *adapter*.

test_elastic_net.py covers the Rust kernel through ``rb.elastic_net_fit`` /
``rb.elastic_net_predict`` directly. This file covers the estimator wrapped
around it: which inputs reach the Rust path, which fall back to sklearn, what
the caller is warned about, and whether the fitted attributes look like
sklearn's.

The distinction that most of these tests turn on: a fallback must be
*invisible* apart from being slower. Same attributes, same dtypes sklearn would
produce, same predictions.
"""
import os
os.environ.setdefault("SKRUB_RUST", "1")

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import ElasticNet as SKElasticNet

from stratum import ElasticNet, set_config
from stratum import _rust_backend as rb
from stratum.adapters.elastic_net import (
    RustyElasticNet,
    UnsupportedParameterWarning,
    _is_supported_array,
    _rust_available,
)

pytestmark = pytest.mark.skipif(not rb.HAVE_RUST, reason="Rust backend not built")


@pytest.fixture(autouse=True)
def _rust_on():
    """Every test starts with the Rust path enabled and leaves it that way.

    The adapter reads the config on each call, so a test that flips it has to
    restore it or it leaks into whatever runs next.
    """
    set_config(rust_backend=True, allow_patch=True)
    yield
    set_config(rust_backend=True, allow_patch=True)


def make_regression(n=200, p=10, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    true_coef = rng.standard_normal(p).astype(np.float32)
    y = (X @ true_coef + 0.1 * rng.standard_normal(n)).astype(np.float32)
    return X, y, true_coef


def took_rust_path(est):
    """A model id is registered only by the Rust branch of fit."""
    return getattr(est, "_rust_model_id", None) is not None


# ---- the exported name ----

class TestExport:
    def test_stratum_elasticnet_is_the_subclass(self):
        assert ElasticNet is RustyElasticNet

    def test_subclasses_sklearn(self):
        assert issubclass(RustyElasticNet, SKElasticNet)

    def test_is_clonable(self):
        """get_params/set_params must round-trip or sklearn tooling breaks."""
        est = ElasticNet(alpha=0.3, l1_ratio=0.7, max_iter=123, tol=1e-5)
        cloned = clone(est)
        assert type(cloned) is RustyElasticNet
        assert cloned.get_params() == est.get_params()

    def test_no_extra_constructor_params(self):
        """The adapter must not add parameters sklearn does not have."""
        assert set(ElasticNet().get_params()) == set(SKElasticNet().get_params())


# ---- the Rust path ----

class TestRustPath:
    def test_supported_input_uses_rust(self):
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)
        assert took_rust_path(est)

    def test_matches_the_kernel_called_directly(self):
        """The adapter must not perturb what it forwards to the kernel."""
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-4).fit(X, y)
        _, coef, intercept, n_iter = rb.elastic_net_fit(
            X, y, 0.1, 0.5, 1000, 1e-4, True)
        np.testing.assert_allclose(est.coef_, coef, rtol=1e-5, atol=1e-6)
        assert est.intercept_ == pytest.approx(float(intercept), abs=1e-5)
        assert est.n_iter_ == n_iter

    def test_fitted_attribute_dtypes(self):
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.1).fit(X, y)
        assert est.coef_.dtype == np.float32
        assert est.coef_.shape == (X.shape[1],)
        assert np.ndim(est.intercept_) == 0
        assert isinstance(est.n_iter_, int) and est.n_iter_ >= 1

    def test_dual_gap_is_nan_not_faked(self):
        """The Jacobi solver computes no duality gap, so it reports none."""
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.1).fit(X, y)
        assert np.isnan(est.dual_gap_)

    def test_n_features_in_recorded(self):
        X, y, _ = make_regression(100, 7)
        est = ElasticNet(alpha=0.1).fit(X, y)
        assert est.n_features_in_ == 7

    def test_no_intercept_gives_zero_intercept(self):
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.1, fit_intercept=False).fit(X, y)
        assert took_rust_path(est)
        assert est.intercept_ == pytest.approx(0.0, abs=1e-6)

    def test_predict_agrees_with_sklearn(self):
        """End to end: same estimator API, same answers within f32 tolerance."""
        X, y, _ = make_regression(300, 20)
        ours = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10_000, tol=1e-6)
        theirs = SKElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10_000, tol=1e-6)
        ours.fit(X, y)
        theirs.fit(X.astype(np.float64), y.astype(np.float64))
        assert took_rust_path(ours)
        np.testing.assert_allclose(
            ours.predict(X), theirs.predict(X.astype(np.float64)),
            rtol=0.05, atol=0.1)

    def test_refit_replaces_previous_model(self):
        """A second fit must not keep serving the first model's coefficients."""
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.001, max_iter=5000, tol=1e-6).fit(X, y)
        first_id, first_coef = est._rust_model_id, est.coef_.copy()
        est.fit(X, 10.0 * y)
        assert est._rust_model_id != first_id
        assert not np.allclose(est.coef_, first_coef)
        np.testing.assert_allclose(est.coef_, 10.0 * first_coef, rtol=0.05, atol=1e-3)


# ---- fallbacks ----

class TestFallback:
    """Each of these inputs must reach sklearn instead of the kernel.

    Nothing is silently converted: converting would return an answer for a
    different input than the caller passed.
    """

    def _fit_and_assert_fallback(self, X, y, **kw):
        est = ElasticNet(alpha=0.1, **kw).fit(X, y)
        assert not took_rust_path(est)
        return est

    def test_rust_backend_disabled(self):
        X, y, _ = make_regression()
        set_config(rust_backend=False)
        self._fit_and_assert_fallback(X, y)

    def test_allow_patch_disabled(self):
        X, y, _ = make_regression()
        set_config(allow_patch=False)
        self._fit_and_assert_fallback(X, y)

    def test_float64_input(self):
        X, y, _ = make_regression()
        self._fit_and_assert_fallback(X.astype(np.float64), y.astype(np.float64))

    def test_fortran_order_input(self):
        X, y, _ = make_regression()
        assert not np.asfortranarray(X).flags.c_contiguous
        self._fit_and_assert_fallback(np.asfortranarray(X), y)

    def test_sparse_input(self):
        X, y, _ = make_regression()
        self._fit_and_assert_fallback(sp.csc_matrix(X), y)

    def test_dataframe_input(self):
        pd = pytest.importorskip("pandas")
        X, y, _ = make_regression()
        self._fit_and_assert_fallback(pd.DataFrame(X), y)

    def test_sample_weight_reweights_the_objective(self):
        """sample_weight changes the loss, so it cannot be quietly dropped."""
        X, y, _ = make_regression()
        w = np.abs(np.random.default_rng(0).standard_normal(X.shape[0]))
        est = ElasticNet(alpha=0.1).fit(X, y, sample_weight=w)
        assert not took_rust_path(est)

    def test_multi_output_y(self):
        """A 2-D target is a different solver entirely."""
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.1).fit(X, np.column_stack([y, 2.0 * y]))
        assert not took_rust_path(est)
        assert est.coef_.shape == (2, X.shape[1])

    def test_fallback_result_matches_sklearn_exactly(self):
        """A fallback is sklearn, so it should be bit-for-bit sklearn."""
        X, y, _ = make_regression()
        X64, y64 = X.astype(np.float64), y.astype(np.float64)
        ours = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X64, y64)
        theirs = SKElasticNet(alpha=0.1, l1_ratio=0.5).fit(X64, y64)
        assert not took_rust_path(ours)
        np.testing.assert_array_equal(ours.coef_, theirs.coef_)
        np.testing.assert_array_equal(ours.predict(X64), theirs.predict(X64))

    def test_fallback_preserves_sklearn_dtype(self):
        """float64 in, float64 out. The adapter must not downcast on this path."""
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.1).fit(X.astype(np.float64), y.astype(np.float64))
        assert est.coef_.dtype == np.float64

    def test_fallback_reports_a_real_dual_gap(self):
        """Only the Rust path has no duality gap to report."""
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.1).fit(X.astype(np.float64), y.astype(np.float64))
        assert not np.isnan(est.dual_gap_)


# ---- input validation happens on both paths ----

class TestValidation:
    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_X_raises(self, bad):
        X, y, _ = make_regression()
        X = X.copy()
        X[0, 0] = bad
        with pytest.raises(ValueError):
            ElasticNet(alpha=0.1).fit(X, y)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_y_raises(self, bad):
        X, y, _ = make_regression()
        y = y.copy()
        y[0] = bad
        with pytest.raises(ValueError):
            ElasticNet(alpha=0.1).fit(X, y)

    def test_length_mismatch_raises(self):
        X, y, _ = make_regression()
        with pytest.raises(ValueError):
            ElasticNet(alpha=0.1).fit(X, y[:-1])


# ---- warnings for options the solver cannot honour ----

class TestUnsupportedParameterWarnings:
    @pytest.mark.parametrize("kw", [
        {"precompute": True},
        {"warm_start": True},
        {"selection": "random"},
    ])
    def test_tuning_options_warn_but_still_use_rust(self, kw):
        """These change only how the answer is reached, so Rust still runs."""
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.1, **kw)
        with pytest.warns(UnsupportedParameterWarning, match="ignored"):
            est.fit(X, y)
        assert took_rust_path(est)

    def test_tuning_warning_does_not_promise_an_identical_result(self):
        """The warning must not claim equivalence it cannot guarantee."""
        X, y, _ = make_regression()
        with pytest.warns(UnsupportedParameterWarning,
                          match="can still move the coefficients"):
            ElasticNet(alpha=0.1, warm_start=True).fit(X, y)

    def test_positive_warns_that_the_result_differs(self):
        """positive=True changes what the solution IS, so the warning must be loud."""
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.1, positive=True)
        with pytest.warns(UnsupportedParameterWarning, match="DIFFERS"):
            est.fit(X, y)
        assert took_rust_path(est)

    def test_positive_is_actually_honoured_on_the_fallback(self):
        """The warning tells the caller to disable Rust. That advice must work."""
        X, y, _ = make_regression()
        set_config(rust_backend=False)
        est = ElasticNet(alpha=0.1, positive=True).fit(X, y)
        assert not took_rust_path(est)
        assert (est.coef_ >= 0).all()

    def test_random_state_warns_only_with_random_selection(self):
        X, y, _ = make_regression()
        with pytest.warns(UnsupportedParameterWarning, match="random_state"):
            ElasticNet(alpha=0.1, selection="random", random_state=0).fit(X, y)

        with warnings_as_errors():
            ElasticNet(alpha=0.1, random_state=0).fit(X, y)

    def test_default_parameters_warn_about_nothing(self):
        X, y, _ = make_regression()
        with warnings_as_errors():
            ElasticNet(alpha=0.1).fit(X, y)

    def test_no_warning_on_the_fallback_path(self):
        """The fallback honours everything, so there is nothing to warn about."""
        X, y, _ = make_regression()
        set_config(rust_backend=False)
        with warnings_as_errors():
            ElasticNet(alpha=0.1, warm_start=True, positive=True).fit(X, y)


class warnings_as_errors:
    """Fail if any UnsupportedParameterWarning is raised in the block."""

    def __enter__(self):
        import warnings
        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("error", UnsupportedParameterWarning)
        return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


# ---- predict ----

class TestPredict:
    def test_predict_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            ElasticNet(alpha=0.1).predict(make_regression()[0])

    def test_predict_dtype_and_shape(self):
        X, y, _ = make_regression(150, 6)
        preds = ElasticNet(alpha=0.1).fit(X, y).predict(X)
        assert preds.shape == (150,)
        assert preds.dtype == np.float32

    def test_predict_matches_manual_computation(self):
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.01, max_iter=5000, tol=1e-6).fit(X, y)
        manual = X @ est.coef_ + float(est.intercept_)
        np.testing.assert_allclose(est.predict(X), manual, rtol=1e-4, atol=1e-4)

    def test_predict_on_unseen_rows(self):
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.1).fit(X, y)
        X_new = np.ascontiguousarray(
            np.random.default_rng(1).standard_normal((17, X.shape[1])),
            dtype=np.float32)
        assert est.predict(X_new).shape == (17,)

    def test_predict_wrong_n_features_raises(self):
        X, y, _ = make_regression(100, 5)
        est = ElasticNet(alpha=0.1).fit(X, y)
        with pytest.raises(ValueError):
            est.predict(np.zeros((10, 3), dtype=np.float32))

    def test_predict_falls_back_for_unsupported_dtype(self):
        """A Rust-fitted model still has to serve a float64 matrix."""
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.1).fit(X, y)
        np.testing.assert_allclose(
            est.predict(X.astype(np.float64)), est.predict(X),
            rtol=1e-4, atol=1e-4)

    def test_config_is_read_per_call_not_cached_at_fit(self):
        """Disabling Rust after fitting must route predict back to sklearn."""
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.1).fit(X, y)
        rust_preds = est.predict(X)
        set_config(rust_backend=False)
        np.testing.assert_allclose(est.predict(X), rust_preds, rtol=1e-4, atol=1e-4)

    def test_sklearn_fitted_model_predicts_without_rust(self):
        """No model id was registered, so predict cannot use the Rust path."""
        X, y, _ = make_regression()
        set_config(rust_backend=False)
        est = ElasticNet(alpha=0.1).fit(X, y)
        set_config(rust_backend=True)
        assert not took_rust_path(est)
        assert est.predict(X).shape == (X.shape[0],)


# ---- helpers ----

class TestIsSupportedArray:
    def test_accepts_c_contiguous_float32_2d(self):
        assert _is_supported_array(np.zeros((4, 3), dtype=np.float32))

    @pytest.mark.parametrize("X", [
        np.zeros((4, 3), dtype=np.float64),
        np.zeros((4, 3), dtype=np.int32),
        np.zeros(4, dtype=np.float32),
        np.asfortranarray(np.zeros((4, 3), dtype=np.float32)),
        sp.csr_matrix((4, 3), dtype=np.float32),
        [[1.0, 2.0], [3.0, 4.0]],
    ])
    def test_rejects_everything_else(self, X):
        assert not _is_supported_array(X)

    def test_rejects_a_non_contiguous_view(self):
        assert not _is_supported_array(np.zeros((4, 6), dtype=np.float32)[:, ::2])


class TestRustAvailable:
    def test_true_by_default(self):
        assert _rust_available()

    @pytest.mark.parametrize("flag", ["rust_backend", "allow_patch"])
    def test_false_when_a_config_flag_is_off(self, flag):
        set_config(**{flag: False})
        assert not _rust_available()


class TestWarningsMatchTheChosenPath:
    """A warning must describe the path taken, not the one nearly taken."""

    def test_no_warning_when_multi_output_falls_back(self):
        X, y, _ = make_regression()
        y2d = np.column_stack([y, y * 2]).astype(np.float32)
        est = ElasticNet(alpha=0.1, positive=True)
        with warnings_as_errors():
            est.fit(X, y2d)
        assert not took_rust_path(est)

    def test_multi_output_fallback_still_honours_positive(self):
        """The claim the warning would have made is false here."""
        X, y, _ = make_regression()
        y2d = np.column_stack([y, y * 2]).astype(np.float32)
        est = ElasticNet(alpha=0.1, positive=True).fit(X, y2d)
        assert (est.coef_ >= 0).all()

    def test_no_warning_when_unsupported_dtype_falls_back(self):
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.1, positive=True)
        with warnings_as_errors():
            est.fit(X.astype(np.float64), y.astype(np.float64))
        assert not took_rust_path(est)


class TestKernelErrorsAreNotPanics:
    """Bad kernel input must raise a Python error, not panic across FFI.

    Only reachable through rb.* directly; the adapter filters both cases first.
    """

    def test_mismatched_y_length_raises_value_error(self):
        X = np.ones((10, 3), dtype=np.float32)
        y = np.ones(7, dtype=np.float32)
        with pytest.raises(ValueError, match="y length must match"):
            rb.elastic_net_fit(X, y, 0.1, 0.5, 100, 1e-4, True)

    def test_non_contiguous_x_raises_value_error(self):
        X = np.asfortranarray(np.ones((10, 3), dtype=np.float32))
        y = np.ones(10, dtype=np.float32)
        with pytest.raises(ValueError, match="C-contiguous"):
            rb.elastic_net_fit(X, y, 0.1, 0.5, 100, 1e-4, True)


class TestModelRegistryIsReleased:
    """Fitted models must leave the process-global Rust registry.

    Otherwise a refit loop grows by n_features * 4 bytes per iteration.
    """

    def test_refit_releases_the_previous_model(self):
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.1)
        est.fit(X, y)
        first = est._rust_model_id
        est.fit(X, y)
        assert est._rust_model_id != first
        # Released by the refit, so freeing it again is a no-op.
        assert rb.elastic_net_free(first) is False
        assert rb.elastic_net_free(est._rust_model_id) is True

    def test_freeing_makes_the_id_unusable(self):
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.1).fit(X, y)
        model_id = est._rust_model_id
        assert rb.elastic_net_free(model_id) is True
        with pytest.raises(KeyError):
            rb.elastic_net_predict(model_id, X)

    def test_discarding_an_estimator_releases_its_model(self):
        import gc
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.1).fit(X, y)
        model_id = est._rust_model_id
        del est
        gc.collect()
        assert rb.elastic_net_free(model_id) is False


class TestDiagnosticsAndCleanupPaths:
    """The branches that only run under a debug flag or a failed cleanup."""

    def test_debug_info_announces_both_dispatches(self, capsys, monkeypatch):
        monkeypatch.setattr(
            "stratum.adapters.elastic_net._DEBUG_INFO", True, raising=True)
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.1).fit(X, y)
        est.predict(X)
        out = capsys.readouterr().out
        assert "Dispatching ElasticNet fit to Rust backend" in out
        assert "Dispatching ElasticNet predict to Rust backend" in out

    def test_cleanup_survives_a_failing_free(self, monkeypatch):
        """__del__ runs this, so it must never raise."""
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.1).fit(X, y)

        def boom(_model_id):
            raise RuntimeError("registry unavailable")

        monkeypatch.setattr(rb, "elastic_net_free", boom, raising=True)
        est._free_rust_model()          # must not propagate
        assert est._rust_model_id is not None  # not cleared on failure

    def test_cleanup_is_a_noop_without_the_binding(self, monkeypatch):
        """An extension built without elastic_net_free must still work."""
        X, y, _ = make_regression()
        est = ElasticNet(alpha=0.1).fit(X, y)
        monkeypatch.setattr(rb, "elastic_net_free", None, raising=True)
        est._free_rust_model()
        assert est._rust_model_id is not None

    def test_cleanup_on_an_unfitted_estimator(self):
        ElasticNet(alpha=0.1)._free_rust_model()
