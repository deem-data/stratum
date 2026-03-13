from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler as _SKStandardScaler
from sklearn.utils.validation import check_is_fitted
import logging
from .._config import get_config
from .. import _rust_backend as rb

logger = logging.getLogger(__name__)

class NumpyStandardScaler(_SKStandardScaler):
    """Drop-in StandardScaler that uses numpy to compute the mean and scale."""
    def __init__(self, with_mean: bool = True, with_std: bool = True, copy: bool = True, reuse_mean: bool = False):
        super().__init__(with_mean=with_mean, with_std=with_std, copy=copy)
        self.reuse_mean = reuse_mean

    def fit(self, X, y=None, sample_weight=None):
        if sample_weight is not None:
            self.mean_ = np.average(X, axis=0, weights=sample_weight)
            scale = np.sqrt(np.average(((X - self.mean_) ** 2), axis=0, weights=sample_weight))
            self.scale_ = np.where(scale > 0, scale, 1.0)
            return self

        # mean
        self.mean_ = X.mean(axis=0)

        # variance
        if self.reuse_mean:
            scale = np.sqrt(((X - self.mean_) ** 2).mean(axis=0))
            self.scale_ = np.where(scale > 0, scale, 1.0)
        else:
            self.scale_ = X.std(axis=0)
        return self

    def transform(self, X, copy=None):
        check_is_fitted(self)
        if self.copy or copy:
            X = X.copy()
        X -= self.mean_
        X /= self.scale_
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

class RustyStandardScaler(_SKStandardScaler):
    """Drop-in StandardScaler that prefers the Rust fastpath where supported.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True, copy: bool = True, **kwargs):
        super().__init__(with_mean=with_mean, with_std=with_std, copy=copy)
        self._supported_params = (
            with_mean
            and with_std
            and copy
        )

    def fit(self, X, y=None, sample_weight=None,):
        X_arr = np.asarray(X, dtype=np.float32)
        # Check Rust kernel availability
        if getattr(rb, "standard_scale_fit", None) is None or not self._supported_params or sample_weight is not None:
            logger.debug("Fallback to scikit for fit")
            print("fall1")
            return super().fit(X, y, sample_weight=sample_weight)
        mean, scale = rb.standard_scale_fit(X_arr)
        self.mean_ = mean
        self.scale_ = scale
        return self


    def transform(self, X, copy=None):
        rc = get_config()
        # Global kill-switch / feature flags
        if not (rc.get("allow_patch", False) and rc.get("rust_backend", False) and rb.HAVE_RUST):
            logger.debug("Rust disabled, fallback to scikit for fit")
            return super().transform(X, copy=copy)

        # Check Rust kernel availability and supported parameters
        if getattr(rb, "standard_scale_transform", None) is None or not self._supported_params:
            logger.debug("Rust not available, fallback to scikit for fit")
            print("fall2")
            return super().transform(X, copy=copy)

        check_is_fitted(self)

        # Coerce to float32 array for Rust
        X_arr = np.asarray(X, dtype=np.float32)
        mean = np.asarray(self.mean_, dtype=np.float32)
        scale = np.asarray(self.scale_, dtype=np.float32)

        t0 = rb.start_timing()
        try:
            logger.debug("Calling Rust standard_scale_transform")
            out = rb.standard_scale_transform(X_arr, mean, scale)
        except Exception as e:
            # Never fail user code because of Rust; just fall back
            print(f"WARNING: Rust standard_scale failed, falling back. Error: {e}")
            return super().transform(X, copy=copy)
        rb.print_timing("standard_scale", t0)
        return out

    def fit_transform(self, X, y=None, **fit_params):
        # Use base class for fitting, then reuse our transform fastpath
        return self.fit(X, y, **fit_params).transform(X)