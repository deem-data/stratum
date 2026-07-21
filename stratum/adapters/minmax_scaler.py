from __future__ import annotations
import os

import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler as _SKMinMaxScaler
from sklearn.utils.validation import check_is_fitted
import logging
from .._config import get_config
from .. import _rust_backend as rb

# File-internal config flags
_DEBUG_INFO = False
logger = logging.getLogger(__name__)
MIN_BLOCK_LEN = 10_000


class RustyMinMaxScaler(_SKMinMaxScaler):
    """Drop-in MinMaxScaler that prefers the Rust fastpath where supported.

    Supported params: feature_range=(0, 1). copy and clip are always honored
    (clip is passed straight through to the Rust kernel)."""

    def __init__(self, feature_range=(0, 1), copy=True, n_jobs=None, clip=False):
        super().__init__(feature_range=feature_range, copy=copy, clip=clip)
        lo, hi = self.feature_range
        self._supported_params = (lo == 0 and hi == 1)

        cores = os.cpu_count()

        if n_jobs is None:
            self.n_jobs = cores
        elif n_jobs > cores:
            logger.warning(
                f"n_jobs {n_jobs} > core count {cores}, setting n_jobs to {cores}"
            )
            self.n_jobs = cores
        else:
            self.n_jobs = n_jobs

    def _n_chunks(self, X):
        print(f"N_rows = {X.shape[0]}")

        blocks = max(1, X.shape[0] // MIN_BLOCK_LEN)
        print(f"Number of blocks: {blocks}")
        return min(blocks, self.n_jobs)

    def fit(self, X, y=None):
        rc = get_config()
        if not (
            rc.get("allow_patch", False)
            and rc.get("rust_backend", False)
            and rb.HAVE_RUST
            and getattr(rb, "minmax_scale_fit", None)
            and self._supported_params
        ):
            logger.debug("Rust disabled, fallback to scikit for fit")
            return super().fit(X, y)

        X_arr = np.asarray(X, dtype=np.float32)

        data_min, data_max = rb.minmax_scale_fit(X_arr, self._n_chunks(X_arr))

        self.data_min_ = data_min.astype(np.float64)
        self.data_max_ = data_max.astype(np.float64)

        return self

    def transform(self, X, copy=None):
        rc = get_config()
        if not (
            rc.get("allow_patch", False)
            and rc.get("rust_backend", False)
            and rb.HAVE_RUST
            and getattr(rb, "minmax_scale_transform", None)
            and self._supported_params
        ):
            logger.debug("Rust disabled, fallback to scikit for fit")
            return super().transform(X)

        check_is_fitted(self)

        # Coerce to float32 array for Rust
        X_arr = np.asarray(X, dtype=np.float32)
        data_min = self.data_min_.astype(np.float32)
        data_max = self.data_max_.astype(np.float32)
        t0 = rb.start_timing()
        try:
            out = rb.minmax_scale_transform(
                X_arr, data_min, data_max, self._n_chunks(X_arr), self.clip
            )
        except Exception as e:
            logger.warning(f"Rust minmax_scale_transform failed, falling back: {e}")
            return super().transform(X)
        rb.print_timing("minmax_scale_transform", t0)

        return out

    def fit_transform(self, X, y=None, **fit_params):
        # Use base class for fitting, then reuse our transform fastpath
        return self.fit(X, y, **fit_params).transform(X)
