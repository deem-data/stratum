"""StandardScaler implementations with conservative dense fast paths.

- Float64, finite dense inputs use NumPy or parallel Rust statistics.
- Unsupported inputs, weights and configurations retain sklearn semantics.
- Validation and constant-feature detection use sklearn's reference helpers.
"""

from __future__ import annotations

import numbers
import os

import numpy as np
from sklearn.preprocessing import StandardScaler as _SKStandardScaler
from sklearn.preprocessing._data import _handle_zeros_in_scale, _is_constant_feature
from sklearn.utils.validation import check_is_fitted, validate_data

from .. import _rust_backend as rb
from .._config import get_config


def supports_rust_standard_scaler(estimator) -> tuple[bool, str]:
    if type(estimator) not in (_SKStandardScaler, RustyStandardScaler):
        return False, "estimator is not a supported StandardScaler"
    if not (rb.HAVE_RUST and rb.standard_scale_fit and rb.standard_scale_transform):
        return False, "Rust StandardScaler runtime is not available"
    if not (estimator.with_mean and estimator.with_std and estimator.copy):
        return False, "Rust requires with_mean=True, with_std=True and copy=True"
    return True, ""


class NumpyStandardScaler(_SKStandardScaler):
    """Dense NumPy statistics with sklearn validation and fallback behavior."""

    def __init__(self, *, with_mean=True, with_std=True, copy=True, reuse_mean=False):
        super().__init__(with_mean=with_mean, with_std=with_std, copy=copy)
        self.reuse_mean = reuse_mean

    def _fast_fit_supported(self, X, sample_weight):
        # Do not coerce dtype, sparse inputs, missing values or dataframe metadata
        # merely to enter a fast path. sklearn handles those cases unchanged.
        return (
            sample_weight is None
            and self.with_mean
            and self.with_std
            and isinstance(X, np.ndarray)
            and X.ndim == 2
            and X.dtype == np.float64
            and X.size > 0
            and np.isfinite(X).all()
        )

    def _statistics(self, X):
        mean = X.mean(axis=0)
        if self.reuse_mean:
            centered = X - mean
            var = np.mean(centered * centered, axis=0)
        else:
            var = X.var(axis=0)
        return mean, var

    def fit(self, X, y=None, sample_weight=None):
        if not self._fast_fit_supported(X, sample_weight):
            return super().fit(X, y, sample_weight=sample_weight)
        self._validate_params()
        self._reset()
        # The eligibility check already verified finiteness; do not scan twice.
        X = validate_data(self, X, dtype=np.float64, ensure_all_finite=False)
        self.mean_, self.var_ = self._statistics(X)
        self.n_samples_seen_ = np.int64(X.shape[0])
        self.scale_ = _handle_zeros_in_scale(
            np.sqrt(self.var_),
            copy=False,
            constant_mask=_is_constant_feature(
                self.var_, self.mean_, self.n_samples_seen_
            ),
        )
        return self


class RustyStandardScaler(NumpyStandardScaler):
    """Parallel Rust float64 kernels; other inputs use sklearn unchanged."""

    def __init__(self, *, with_mean=True, with_std=True, copy=True):
        super().__init__(with_mean=with_mean, with_std=with_std, copy=copy)

    def _bind_native_worker_budget(self, workers):
        """Set the propagated number of threads from config"""
        if not isinstance(workers, numbers.Integral) or workers <= 0:
            raise ValueError("native worker budget must be a positive integer")
        self._stratum_worker_budget = int(workers)
        return self

    def __sklearn_clone__(self):
        cloned = type(self)(**self.get_params(deep=False))
        if getattr(self, "_stratum_force_rust", False):
            cloned._stratum_force_rust = True
        if hasattr(self, "_stratum_worker_budget"):
            cloned._bind_native_worker_budget(self._stratum_worker_budget)
        return cloned

    def _use_rust(self):
        rc = get_config()
        enabled = getattr(self, "_stratum_force_rust", False) or (
            rc["allow_patch"] and rc["rust_backend"]
        )
        return enabled and supports_rust_standard_scaler(self)[0]

    def _native_workers(self):
        # Use num_threads from the config
        workers = getattr(self, "_stratum_worker_budget", get_config()["num_threads"])
        return int(workers) or (os.cpu_count() or 1)

    def _fast_fit_supported(self, X, sample_weight):
        return self._use_rust() and super()._fast_fit_supported(X, sample_weight)

    def _statistics(self, X):
        return rb.standard_scale_fit(X, worker_budget=self._native_workers())

    def transform(self, X, copy=None):
        check_is_fitted(self)
        effective_copy = self.copy if copy is None else copy
        if not (
            self._use_rust()
            and effective_copy
            and isinstance(X, np.ndarray)
            and X.dtype == np.float64
        ):
            return super().transform(X, copy=copy)
        # NaNs are propagated by the transform kernel just as in sklearn.
        X = validate_data(
            self,
            X,
            reset=False,
            dtype=np.float64,
            ensure_all_finite="allow-nan",
            force_writeable=False,
        )
        return rb.standard_scale_transform(
            X,
            np.asarray(self.mean_, dtype=np.float64),
            np.asarray(self.scale_, dtype=np.float64),
            worker_budget=self._native_workers(),
        )
