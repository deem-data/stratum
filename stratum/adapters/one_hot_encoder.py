from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder as _SKOneHot
from sklearn.utils._encode import _encode, _check_unknown  # private, mirrors sklearn’s own path

from ..config import get_config
from .. import _rust_backend as rb
from .._rust_backend import _to_list

def _iter_columns(X):
    if hasattr(X, "ndim") and getattr(X, "ndim", 1) == 2 and hasattr(X, "shape"):
        n_cols = X.shape[1]
        for j in range(n_cols):
            try:
                yield X.iloc[:, j]
            except Exception:
                yield X[:, j]
    else:
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 1:
            yield arr
        else:
            for j in range(arr.shape[1]):
                yield arr[:, j]

# Recode X using categories
def _codes_from_categories(X, categories_):
    cols = list(_iter_columns(X))
    if len(cols) != len(categories_):
        raise ValueError(f"X has {len(cols)} columns, expected {len(categories_)}")
    codes, n_cats = [], []

    for j, cats in enumerate(categories_):
        # Remove NaNs/Nones from categories
        # Note: This may differ output from sklearn. Sklearn allows None to be a real category.
        cats_is_na = pd.isna(cats)
        cats_wo_na = cats[~cats_is_na]
        Xi = np.asarray(_to_list(cols[j]), dtype=object)
        # Fast vectorized recode (C-level). 3x faster than sklearn's _encode.
        codes_j = pd.Categorical(Xi, categories=cats_wo_na, ordered=True).codes.astype(np.int32, copy=False) #recode
        n_cats.append(len(cats_wo_na))
        codes.append(codes_j)

    return codes, n_cats


# Create a subclass of sklearn's one hot encoder
class RustyOneHotEncoder(_SKOneHot):
    """Drop-in OneHotEncoder that prefers the Rust fastpath where supported

    Supported params:
        drop='if_binary', dtype=float32, handle_unknown='ignore', sparse_output=True|False"""

    def __init__(self,
                 drop="if_binary",
                 dtype=np.float32,
                 handle_unknown="ignore",
                 sparse_output=False,
                 **kwargs):
        super().__init__(drop=drop, dtype=dtype, handle_unknown=handle_unknown, sparse_output=sparse_output, **kwargs)
        self._supported_params = (drop == "if_binary"
                                  and np.dtype(self.dtype) == np.dtype(np.float32) #'float32'/np.float32/np.dtype('float32)
                                  and handle_unknown == "ignore" and len(kwargs) == 0)

    def fit(self, X, y=None):
        # Fit the sklearn OHE for exact categories/drop parity
        super().fit(X, y)
        return self

    def transform(self, X):
        # Check kill-switch and feature flag at call time
        rc = get_config()
        if not (rc["allow_patch"] and rc["rust_backend"] and rb.HAVE_RUST):
            return super().transform(X)
        # Check if the rust modules are available
        if getattr(rb, "ohe_transform_csr", None) is None or getattr(rb, "csr_to_dense", None) is None:
            return super().transform(X)
        # Check if called with supported parameters
        if not (self._supported_params and hasattr(self, "categories_")):
            return super().transform(X)

        # Recode (integer encoding) the input features. codes[i] is recoded feature i
        t0 = rb.start_timing()
        codes, n_cats = _codes_from_categories(X, self.categories_)
        drop_idx = [None if d is None else int(d) for d in self.drop_idx_]
        rb.print_timing("Recoding", t0)

        # Dispatch the apply phase to Rust (returns CSR)
        print("INFO: Dispatching OneHotEncoder transform to Rust backend") #TODO: proper logging
        t0 = rb.start_timing()
        data, indices, indptr, n_rows, n_cols = rb.ohe_transform_csr(codes, n_cats, drop_idx)
        rb.print_timing("ohe_transform_csr", t0)

        out_dt = np.dtype(self.dtype)
        if self.sparse_output:
            # Make a SciPy CSR
            sparse_mat = sp.csr_matrix(
                (np.asarray(data, dtype=np.float32),
                 np.asarray(indices, dtype=np.int32),
                 np.asarray(indptr, dtype=np.int64)),
                shape=(int(n_rows), int(n_cols)),
                dtype=out_dt,
            )
            return sparse_mat

        # Else, use Rust densifier
        #print("INFO: Using Rust densifier")
        t0 = rb.start_timing()
        dense_mat = rb.csr_to_dense(data, indices, indptr, n_rows, n_cols) #2.5x faster than scipy toarray
        if out_dt != np.dtype(np.float32):
            dense_mat = dense_mat.astype(out_dt, copy=False)
        rb.print_timing("csr_to_dense", t0)
        return dense_mat

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
