"""SimpleImputer for dense inputs.

- Optimize for float32/float64 data with NaN missing values.
- Keep sklearn validation, fitted metadata, names and indicator semantics.
- Object/string, sparse, non-default sentinels and copy=False use sklearn.
- Fused imputation, empty column removal, and indicator output under one allocation.
"""
from __future__ import annotations

import numbers
import os
import warnings

import numpy as np
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.utils._missing import is_scalar_nan
from sklearn.utils.validation import check_is_fitted, validate_data

from .. import _rust_backend as rb
from .._config import get_config


def supports_rust_simple_imputer(estimator):
    if type(estimator) not in (SimpleImputer, RustySimpleImputer):
        return False, "estimator is not a supported SimpleImputer"
    if not (rb.HAVE_RUST and rb.simple_imputer_fit and rb.simple_imputer_transform):
        return False, "Rust SimpleImputer runtime is not available"
    if not isinstance(estimator.strategy, str) or estimator.strategy not in (
        "mean", "median", "most_frequent", "constant",
    ):
        return False, "unsupported imputation strategy"
    if not is_scalar_nan(estimator.missing_values):
        return False, "requires missing_values=np.nan"
    if not all(isinstance(v, (bool, np.bool_)) for v in (
        estimator.copy, estimator.add_indicator, estimator.keep_empty_features,
    )) or not estimator.copy:
        return False, "requires copy=True and boolean indicator/empty-feature options"
    if estimator.strategy == "constant" and estimator.fill_value is not None and (
        not isinstance(estimator.fill_value, numbers.Real) or not np.isfinite(estimator.fill_value)
    ):
        return False, "native constant imputation requires a finite numeric fill_value"
    return True, ""


class RustySimpleImputer(SimpleImputer):

    def _bind_native_worker_budget(self, workers):
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

    def _native_workers(self):
        workers = getattr(self, "_stratum_worker_budget", get_config()["num_threads"])
        return int(workers) or (os.cpu_count() or 1)

    def _native_input(self, X):
        rc = get_config()
        if not (getattr(self, "_stratum_force_rust", False) or (
            rc["allow_patch"] and rc["rust_backend"]
        )) or not supports_rust_simple_imputer(self)[0]:
            return False
        # Inspect dtype without coercing object arrays or materializing sparse data.
        array = np.asarray(X)
        return array.ndim == 2 and array.dtype in (np.dtype("float32"), np.dtype("float64"))

    def fit(self, X, y=None):
        self._stratum_native_fit = False
        if not self._native_input(X):
            return super().fit(X, y)
        self._validate_params()
        original = X
        X = validate_data(self, X, dtype=None, ensure_all_finite="allow-nan")
        # NumPy uses a different (pairwise) mean reduction on contiguous columns.
        # Preserve its cancellation/overflow behavior rather than substitute our
        # sequential column reduction. Fused native transform is still available.
        if self.strategy == "mean" and (not X.flags.c_contiguous or X.shape[1] == 1):
            super().fit(original, y)
            self._stratum_native_fit = True
            return self
        self._fit_dtype = self._fill_dtype = X.dtype
        stats, missing = rb.simple_imputer_fit(
            X, strategy=self.strategy, keep_empty=bool(self.keep_empty_features),
            fill_value=float(self.fill_value) if (
                self.strategy == "constant" and self.fill_value is not None
            ) else 0.0,
            worker_budget=self._native_workers(),
        )
        # Match sklearn's public fitted statistics dtype for each strategy.
        dtype = object if self.strategy == "constant" else (
            X.dtype if self.strategy == "median" else np.float64
        )
        self.statistics_ = stats.astype(dtype, copy=False)
        self.indicator_ = None
        if self.add_indicator:
            self.indicator_ = MissingIndicator(missing_values=self.missing_values, error_on_new=False)
            # Only column-level flags are needed to fit the reference indicator.
            self.indicator_._fit(missing[None, :], precomputed=True)
        self._stratum_native_fit = True
        return self

    def transform(self, X):
        check_is_fitted(self, "statistics_")
        if not (getattr(self, "_stratum_native_fit", False) and self._native_input(X)):
            return super().transform(X)
        X = validate_data(self, X, reset=False, dtype=None, ensure_all_finite="allow-nan")
        stats = np.asarray(self.statistics_, dtype=self._fill_dtype).astype(np.float64, copy=False)
        valid = np.ones(len(stats), dtype=bool) if self.keep_empty_features else ~np.isnan(stats)
        if not valid.all():
            invalid = np.flatnonzero(~valid)
            if hasattr(self, "feature_names_in_"):
                invalid = self.feature_names_in_[invalid]
            warnings.warn(
                f"Skipping features without any observed values: {invalid}. "
                "At least one non-missing value is needed"
                f" for imputation with strategy='{self.strategy}'."
            )
        return rb.simple_imputer_transform(
            X, stats, columns=np.flatnonzero(valid).tolist(),
            indicators=self.indicator_.features_.tolist() if self.add_indicator else [],
            worker_budget=self._native_workers(),
        )
