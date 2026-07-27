import os
os.environ.setdefault("SKRUB_RUST", "1")
import numpy as np
import pytest
from sklearn.linear_model import ElasticNet, Lasso, Ridge

from stratum import _rust_backend as rb

pytestmark = pytest.mark.skipif(not rb.HAVE_RUST, reason="Rust backend not built")

EPS = 1e-2   # f32 precision + CD approximation tolerance


def _f32(arr):
    return np.asarray(arr, dtype=np.float32)


def make_regression(n=200, p=10, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    true_coef = rng.standard_normal(p).astype(np.float32)
    y = (X @ true_coef + 0.1 * rng.standard_normal(n)).astype(np.float32)
    return X, y, true_coef


# ---- fit return shape / dtype ----

class TestFitReturnValues:
    def test_returns_four_tuple(self):
        X, y, _ = make_regression(50, 5)
        result = rb.elastic_net_fit(X, y, 0.1, 0.5, 1000, 1e-4, True)
        assert len(result) == 4

    def test_model_id_is_positive_int(self):
        X, y, _ = make_regression(50, 5)
        model_id, *_ = rb.elastic_net_fit(X, y, 0.1, 0.5, 1000, 1e-4, True)
        assert isinstance(model_id, int) and model_id > 0

    def test_coef_shape(self):
        X, y, _ = make_regression(50, 5)
        _, coef, intercept, _ = rb.elastic_net_fit(X, y, 0.1, 0.5, 1000, 1e-4, True)
        assert coef.shape == (5,)

    def test_coef_dtype_float32(self):
        X, y, _ = make_regression(50, 5)
        _, coef, _, _ = rb.elastic_net_fit(X, y, 0.1, 0.5, 1000, 1e-4, True)
        assert coef.dtype == np.float32

    def test_intercept_is_scalar(self):
        X, y, _ = make_regression(50, 5)
        _, _, intercept, _ = rb.elastic_net_fit(X, y, 0.1, 0.5, 1000, 1e-4, True)
        assert np.isscalar(intercept) or intercept.ndim == 0

    def test_n_iter_positive(self):
        X, y, _ = make_regression(50, 5)
        _, _, _, n_iter = rb.elastic_net_fit(X, y, 0.1, 0.5, 1000, 1e-4, True)
        assert n_iter >= 1


# ---- predict ----

class TestPredict:
    def test_predict_shape(self):
        X, y, _ = make_regression(100, 8)
        model_id, *_ = rb.elastic_net_fit(X, y, 0.1, 0.5, 1000, 1e-4, True)
        preds = rb.elastic_net_predict(model_id, X)
        assert preds.shape == (100,)

    def test_predict_dtype_float32(self):
        X, y, _ = make_regression(50, 5)
        model_id, *_ = rb.elastic_net_fit(X, y, 0.1, 0.5, 1000, 1e-4, True)
        preds = rb.elastic_net_predict(model_id, X)
        assert preds.dtype == np.float32

    def test_predict_ncols_mismatch_raises(self):
        X, y, _ = make_regression(50, 5)
        model_id, *_ = rb.elastic_net_fit(X, y, 0.1, 0.5, 1000, 1e-4, True)
        bad = _f32(np.random.randn(10, 3))
        with pytest.raises(Exception):
            rb.elastic_net_predict(model_id, bad)

    def test_invalid_model_id_raises(self):
        X = _f32(np.random.randn(10, 3))
        with pytest.raises(Exception):
            rb.elastic_net_predict(999_999_999, X)

    def test_manual_prediction_matches(self):
        X, y, _ = make_regression(50, 5)
        model_id, coef, intercept, _ = rb.elastic_net_fit(X, y, 0.01, 0.5, 5000, 1e-6, True)
        preds_rust = rb.elastic_net_predict(model_id, X)
        preds_manual = X.astype(np.float32) @ coef.astype(np.float32) + float(intercept)
        np.testing.assert_allclose(preds_rust, preds_manual, atol=1e-4)


# ---- agreement with sklearn ----

class TestSklearnAgreement:
    """
    ElasticNet implementations can differ in convergence criterion and
    normalisation, so we compare predictions rather than coefficients.
    """

    def _compare_predictions(self, X, y, alpha=0.01, l1_ratio=0.5,
                              fit_intercept=True, rtol=0.05):
        sk = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                        fit_intercept=fit_intercept,
                        max_iter=10_000, tol=1e-6)
        sk.fit(X.astype(np.float64), y.astype(np.float64))
        sk_preds = sk.predict(X.astype(np.float64)).astype(np.float32)

        model_id, *_ = rb.elastic_net_fit(X, y, alpha, l1_ratio, 10_000, 1e-6, fit_intercept)
        ru_preds = rb.elastic_net_predict(model_id, X)

        np.testing.assert_allclose(ru_preds, sk_preds, rtol=rtol, atol=0.1)

    def test_elasticnet_medium_alpha(self):
        X, y, _ = make_regression(300, 20)
        self._compare_predictions(X, y, alpha=0.1, l1_ratio=0.5)

    def test_lasso_limit(self):
        X, y, _ = make_regression(300, 10)
        self._compare_predictions(X, y, alpha=0.05, l1_ratio=1.0)

    def test_ridge_limit(self):
        X, y, _ = make_regression(300, 10)
        self._compare_predictions(X, y, alpha=0.05, l1_ratio=0.0)

    def test_no_intercept(self):
        X, y, _ = make_regression(300, 10)
        self._compare_predictions(X, y, alpha=0.05, l1_ratio=0.5, fit_intercept=False)


# ---- regularisation behaviour ----

class TestRegularisation:
    def test_large_l1_ratio_sparsity(self):
        """Pure Lasso with large alpha should zero out many coefficients."""
        X, y, _ = make_regression(200, 20, seed=7)
        _, coef, _, _ = rb.elastic_net_fit(X, y, 2.0, 1.0, 5000, 1e-6, True)
        n_zero = np.sum(np.abs(coef) < 1e-5)
        assert n_zero > 5, f"Expected sparse solution; only {n_zero} zeros out of 20"

    def test_zero_alpha_smaller_residual_than_large_alpha(self):
        """Less regularisation should give smaller training residual."""
        X, y, _ = make_regression(100, 5)
        mid_id, _, _, _ = rb.elastic_net_fit(X, y, 1.0, 0.5, 5000, 1e-6, True)
        low_id, _, _, _ = rb.elastic_net_fit(X, y, 0.001, 0.5, 5000, 1e-6, True)
        mse_mid = np.mean((rb.elastic_net_predict(mid_id, X) - y) ** 2)
        mse_low = np.mean((rb.elastic_net_predict(low_id, X) - y) ** 2)
        assert mse_low < mse_mid

    def test_fit_intercept_moves_intercept(self):
        """Shifting y by a constant should be absorbed in the intercept."""
        X, y, _ = make_regression(100, 5)
        y_shifted = y + 50.0
        _, _, intercept_base, _ = rb.elastic_net_fit(X, y, 0.01, 0.5, 5000, 1e-6, True)
        _, _, intercept_shift, _ = rb.elastic_net_fit(X, y_shifted, 0.01, 0.5, 5000, 1e-6, True)
        assert abs((intercept_shift - intercept_base) - 50.0) < 1.0
