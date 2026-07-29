from __future__ import annotations
import warnings

import numpy as np
from sklearn.linear_model import ElasticNet as _SKElasticNet
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

from .._config import get_config
from .. import _rust_backend as rb

# File-internal config flags
_DEBUG_INFO = False


class UnsupportedParameterWarning(UserWarning):
    """A parameter the Rust solver cannot honour was set and has been ignored."""


# Options the Jacobi solver does not implement. The Rust path runs anyway and
# ignores them, warning the caller. Split by whether ignoring them can change
# the answer:
#   * "tuning" options only affect how the solution is reached, so ignoring them
#     yields the same coefficients sklearn would produce.
#   * "semantic" options change what the solution IS. Ignoring them returns a
#     different model from the one that was asked for.
_IGNORED_TUNING = {
    "precompute": (False, "the solver has no Gram-matrix path"),
    "warm_start": (False, "the Jacobi solver keeps no state between fits"),
    "selection":  ("cyclic", "Jacobi updates every coordinate from one snapshot, "
                             "so coordinate ordering has no meaning"),
}
_IGNORED_SEMANTIC = {
    "positive": (False, "the solver has no non-negativity constraint, so "
                        "coefficients MAY BE NEGATIVE despite positive=True"),
}


def _warn_ignored(est):
    """Warn about set-but-unhonoured parameters. Returns nothing; never raises."""
    for name, (default, why) in _IGNORED_TUNING.items():
        if getattr(est, name) != default:
            warnings.warn(
                f"{type(est).__name__}: {name}={getattr(est, name)!r} is not supported "
                f"by the Rust backend and was ignored ({why}). It affects how "
                f"the solution is reached, not what is optimised, but it can "
                f"still move the coefficients if max_iter or tol stops the "
                f"solver before convergence.",
                UnsupportedParameterWarning, stacklevel=3,
            )
    for name, (default, why) in _IGNORED_SEMANTIC.items():
        if getattr(est, name) != default:
            warnings.warn(
                f"{type(est).__name__}: {name}={getattr(est, name)!r} is not supported "
                f"by the Rust backend and was ignored ({why}). THE RESULT DIFFERS "
                f"from sklearn. Use set_config(rust_backend=False) to honour it.",
                UnsupportedParameterWarning, stacklevel=3,
            )
    # random_state only has an effect together with selection='random'
    if est.random_state is not None and est.selection == "random":
        warnings.warn(
            f"{type(est).__name__}: random_state is not used by the Rust backend, "
            f"because coordinate selection is not randomised.",
            UnsupportedParameterWarning, stacklevel=3,
        )


def _rust_available():
    """True when the config allows the Rust backend and both kernels are present."""
    rc = get_config()
    if not (rc["allow_patch"] and rc["rust_backend"] and rb.HAVE_RUST):
        return False
    return getattr(rb, "elastic_net_fit", None) is not None \
        and getattr(rb, "elastic_net_predict", None) is not None


def _is_supported_array(X):
    """The kernel takes a 2-D, C-contiguous, float32 dense design matrix.

    Anything else (sparse, float64, F-order, a DataFrame) goes back to sklearn
    rather than being silently converted, so the result matches what sklearn
    would have produced for that input.
    """
    return (isinstance(X, np.ndarray) and X.ndim == 2
            and X.dtype == np.float32 and X.flags.c_contiguous)


class RustyElasticNet(_SKElasticNet):
    """Drop-in :class:`sklearn.linear_model.ElasticNet` backed by the Rust solver.

    The Rust path covers the common case: a dense ``float32`` design matrix, a
    single ``float32`` target, and the default solver options. Everything else
    (sparse input, ``float64``, multi-output ``y``, ``precompute``,
    ``warm_start``, ``positive``, random ``selection``, or sample weights) falls
    back to sklearn, so behaviour and dtypes stay what sklearn would produce.

    The config is read on every call, so ``set_config(rust_backend=False)``
    routes back to sklearn without rebuilding the estimator.
    """

    def fit(self, X, y, sample_weight=None, check_input=True):
        # The registry never collects on its own, so release the previous model
        # before its id is overwritten.
        self._free_rust_model()
        self._rust_model_id = None
        # sample_weight is not a constructor option and reweighting the loss
        # changes the objective outright, so that one still defers to sklearn.
        if not _rust_available() or sample_weight is not None \
                or not _is_supported_array(X):
            return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)

        # Validate the way sklearn does, so NaN/inf raise here rather than
        # propagating into the solver and so n_features_in_ is recorded.
        X, y = validate_data(self, X, y, accept_sparse="csc", order="C",
                             dtype=np.float32, y_numeric=True, multi_output=True)
        # sklearn's ElasticNet.fit scans three times: validate_data covers X and
        # y, then it re-checks y alone to coerce it to whichever dtype X ended up
        # with (it accepts float32 or float64, so it cannot know in advance).
        # Mirror that third scan exactly, over y and not X, so both backends do
        # identical validation work.
        y = check_array(y, order="C", copy=False, dtype=X.dtype.type,
                        ensure_2d=False, estimator=type(self).__name__)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 1:
            # Multi-output is a different solver; hand it back to sklearn.
            return super().fit(X, y, sample_weight=sample_weight, check_input=check_input)

        # Warn only once the Rust path is committed to; the fallbacks above do
        # honour these options.
        _warn_ignored(self)

        if _DEBUG_INFO:
            print("INFO: Dispatching ElasticNet fit to Rust backend")

        t0 = rb.start_timing()
        model_id, coef, intercept, n_iter = rb.elastic_net_fit(
            X, y, float(self.alpha), float(self.l1_ratio),
            int(self.max_iter), float(self.tol), bool(self.fit_intercept),
        )
        rb.print_timing("elastic_net_fit", t0)

        self._rust_model_id = model_id
        self.coef_      = np.asarray(coef, dtype=np.float32)
        self.intercept_ = np.float32(intercept)
        self.n_iter_    = int(n_iter)
        # sklearn exposes the coordinate-descent duality gap; the Jacobi solver
        # does not compute one, so report it as unavailable rather than faking it.
        self.dual_gap_  = np.nan
        return self

    def predict(self, X):
        check_is_fitted(self)
        if (getattr(self, "_rust_model_id", None) is None
                or not _rust_available() or not _is_supported_array(X)):
            return super().predict(X)

        X = validate_data(self, X, accept_sparse="csr", order="C",
                          dtype=np.float32, reset=False)
        if _DEBUG_INFO:
            print("INFO: Dispatching ElasticNet predict to Rust backend")

        t0 = rb.start_timing()
        out = rb.elastic_net_predict(self._rust_model_id, X)
        rb.print_timing("elastic_net_predict", t0)
        return np.asarray(out, dtype=np.float32)

    def _free_rust_model(self):
        """Release this estimator's model from the Rust-side registry.

        The registry is process-global and only ever inserted into, so an
        unreleased model lives until the process exits. Safe to call repeatedly
        and on an unfitted estimator.
        """
        try:
            model_id = getattr(self, "_rust_model_id", None)
            if model_id is None or rb.elastic_net_free is None:
                return
            rb.elastic_net_free(model_id)
        except Exception:
            # Cleanup must not raise: __del__ can run during interpreter
            # shutdown, with module globals already torn down.
            return
        self._rust_model_id = None

    def __del__(self):
        # Covers fit-and-discard, where no refit comes along to release it.
        self._free_rust_model()