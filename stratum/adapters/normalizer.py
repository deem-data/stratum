from __future__ import annotations
import numpy as np
from sklearn.preprocessing import normalize as _sk_normalize
from sklearn.preprocessing import Normalizer as _SKNormalizer
from sklearn.utils.validation import check_array, validate_data

from .._config import get_config
from .. import _rust_backend as rb

# File-internal config flags
_DEBUG_INFO = False

# norm -> (copy-returning kernel, in-place kernel)
_KERNELS = {
    "l1": ("normalize_l1", "normalize_l1_inplace"),
    "l2": ("normalize_l2", "normalize_l2_inplace"),
    "max": ("normalize_max", "normalize_max_inplace"),
}


def _fastpath_kernels(norm):
    """Return (copy_fn, inplace_fn) if the Rust backend can serve this norm, else None."""
    rc = get_config()
    if not (rc["allow_patch"] and rc["rust_backend"] and rb.HAVE_RUST):
        return None
    names = _KERNELS.get(norm)
    if names is None:
        return None
    copy_fn = getattr(rb, names[0], None)
    inplace_fn = getattr(rb, names[1], None)
    if copy_fn is None or inplace_fn is None:
        return None
    return copy_fn, inplace_fn


def _is_supported_array(X, copy):
    """The kernels take a 2-D, C-contiguous, float32 dense array.

    Anything else (sparse, float64, F-order, a DataFrame, a list) goes back to
    sklearn rather than being silently converted, so the output dtype and layout
    stay exactly what sklearn would have produced.
    """
    if not isinstance(X, np.ndarray):
        return False
    if X.ndim != 2 or X.dtype != np.float32:
        return False
    if not X.flags.c_contiguous:
        return False
    # In place means we write through the caller's buffer, so it must be writable
    # and must not share memory with anything we would corrupt.
    if not copy and not X.flags.writeable:
        return False
    return True


def normalize(X, norm="l2", *, axis=1, copy=True, return_norm=False):
    """Drop-in replacement for :func:`sklearn.preprocessing.normalize`.

    Scale each row (``axis=1``) of ``X`` to unit norm. Dispatches to the Rust
    backend when the input is a 2-D C-contiguous ``float32`` array and
    ``norm`` is one of ``l1``, ``l2``, ``max``. ``copy=False`` selects the
    in-place kernel, which writes through the caller's buffer instead of
    allocating an output array. Every other case falls back to sklearn.
    """
    kernels = _fastpath_kernels(norm)
    if kernels is None:
        return _sk_normalize(X, norm=norm, axis=axis, copy=copy, return_norm=return_norm)

    # The kernels are row-wise and do not report the norms they divided by.
    if axis != 1 or return_norm:
        return _sk_normalize(X, norm=norm, axis=axis, copy=copy, return_norm=return_norm)

    if not _is_supported_array(X, copy):
        return _sk_normalize(X, norm=norm, axis=axis, copy=copy, return_norm=return_norm)

    # sklearn's normalize runs check_array first, so NaN and inf raise rather
    # than propagating into the output. Skipping it would be both a behaviour
    # difference and an unearned speed advantage in any benchmark. copy=False
    # here because the kernels below already honour `copy`, and the array is
    # known to be 2-D C-contiguous float32 so nothing is converted.
    check_array(X, dtype=np.float32, order="C", copy=False, estimator="normalize")

    copy_fn, inplace_fn = kernels
    if _DEBUG_INFO:
        print(f"INFO: Dispatching normalize(norm={norm!r}, copy={copy}) to Rust backend")

    t0 = rb.start_timing()
    if copy:
        out = copy_fn(X)
    else:
        inplace_fn(X)
        out = X
    rb.print_timing(f"normalize_{norm}{'' if copy else '_inplace'}", t0)
    return out


class RustyNormalizer(_SKNormalizer):
    """Drop-in :class:`sklearn.preprocessing.Normalizer` backed by the Rust kernels.

    Stateless like sklearn's: ``fit`` only validates and records the feature
    count, and ``transform`` does the work through :func:`normalize` above.
    """

    def transform(self, X, copy=None):
        copy = self.copy if copy is None else copy
        if _fastpath_kernels(self.norm) is None or not _is_supported_array(X, copy):
            return super().transform(X, copy=copy)
        # Validate exactly the way sklearn's Normalizer.transform does: one
        # validate_data here (which also confirms the feature count seen during
        # fit and yields estimator-named error messages), and one check_array
        # inside normalize() below, because that is a public entry point in its
        # own right and cannot assume its caller validated. Anything less would
        # both weaken the error contract and understate our cost relative to
        # sklearn, which pays for both passes.
        #
        # copy=False here because normalize() honours `copy` itself through the
        # kernel choice; sklearn instead copies during validation and then
        # normalizes in place. The observable behaviour is the same, but the
        # copy-returning kernel writes its output in a single fused pass.
        X = validate_data(self, X, accept_sparse="csr", force_writeable=True,
                          copy=False, reset=False)
        return normalize(X, norm=self.norm, axis=1, copy=copy)
