"""Standalone Rust adapters for exact tree and bounded forest classification.

Exact and histogram forests share bootstrap orchestration, tree growth, compact
models, NaN routing, and prediction. Bootstrap samples stay in native code as
multiplicity weights rather than duplicated rows.
"""

from __future__ import annotations

import numbers
import os

import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
    check_is_fitted,
    column_or_1d,
    validate_data,
)

from .. import _rust_backend as rb
from .._config import get_config


# Keep the supported configuration narrow so the Rust path stays parity-safe.
def supports_rust_tree_classifier(estimator) -> tuple[bool, str]:
    """Return whether an estimator belongs to the initial exact-tree subset."""
    if not isinstance(estimator, DecisionTreeClassifier):
        return False, "estimator is not a sklearn DecisionTreeClassifier"
    if not rb.HAVE_RUST or rb.tree_fit_exact is None:
        return False, "Rust decision-tree runtime is not available"
    checks = (
        (estimator.criterion == "gini", "criterion must be 'gini'"),
        (estimator.splitter == "best", "splitter must be 'best'"),
        (estimator.min_weight_fraction_leaf == 0.0, "min_weight_fraction_leaf must be 0"),
        (estimator.class_weight is None, "class_weight is not supported"),
        (estimator.ccp_alpha == 0.0, "ccp_alpha must be 0"),
        (estimator.monotonic_cst is None, "monotonic_cst is not supported"),
        (
            estimator.random_state is None
            or isinstance(estimator.random_state, numbers.Integral),
            "random_state must be None or an integer",
        ),
    )
    for supported, reason in checks:
        if not supported:
            return False, reason
    return True, ""


class RustDecisionTreeClassifier(DecisionTreeClassifier):
    """Sklearn-style adapter backed by the Rust exact tree."""

    def fit(self, X, y, sample_weight=None, check_input=True):
        self._validate_params()
        supported, reason = supports_rust_tree_classifier(self)
        if not supported:
            raise ValueError(f"unsupported Rust decision-tree configuration: {reason}")
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported by the Rust decision tree")
        if not check_input:
            raise ValueError("check_input=False is not supported during fit")

        # Validate and normalize the input exactly once before crossing into Rust.
        X, y = validate_data(
            self,
            X,
            y,
            validate_separately=(
                dict(dtype=np.float32, accept_sparse=False, ensure_all_finite="allow-nan"),
                dict(ensure_2d=False, dtype=None),
            ),
        )
        if sparse.issparse(X):
            raise TypeError("sparse matrices are not supported by the Rust decision tree")
        if np.isinf(X).any():
            raise ValueError("Input X contains infinity")
        y = column_or_1d(y, warn=True)
        check_classification_targets(y)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_outputs_ = 1
        n_samples, self.n_features_in_ = X.shape

        # Derive sklearn-style integer hyperparameters from the public estimator API.
        min_samples_leaf = (
            int(self.min_samples_leaf)
            if isinstance(self.min_samples_leaf, numbers.Integral)
            else int(np.ceil(self.min_samples_leaf * n_samples))
        )
        min_samples_split = (
            int(self.min_samples_split)
            if isinstance(self.min_samples_split, numbers.Integral)
            else max(2, int(np.ceil(self.min_samples_split * n_samples)))
        )
        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)
        max_depth = np.iinfo(np.int32).max if self.max_depth is None else int(self.max_depth)
        if self.max_features is None:
            self.max_features_ = self.n_features_in_
        elif self.max_features == "sqrt":
            self.max_features_ = max(1, int(np.sqrt(self.n_features_in_)))
        elif self.max_features == "log2":
            self.max_features_ = max(1, int(np.log2(self.n_features_in_)))
        elif isinstance(self.max_features, numbers.Integral):
            self.max_features_ = int(self.max_features)
        else:
            self.max_features_ = max(1, int(self.max_features * self.n_features_in_))

        # Convert inputs to contiguous native arrays before calling the Rust runtime.
        random_state = check_random_state(self.random_state)
        tree_seed = int(random_state.randint(0, np.iinfo(np.int32).max))
        X = np.ascontiguousarray(X, dtype=np.float32)
        encoded = np.ascontiguousarray(encoded, dtype=np.int64)
        self._tree_model_handle_ = rb.tree_fit_exact(
            X,
            encoded,
            int(self.n_classes_),
            max_depth,
            min_samples_split,
            min_samples_leaf,
            self.max_features_,
            float(self.min_impurity_decrease),
            tree_seed,
            self.max_leaf_nodes,
        )
        return self

    # Shared prediction guard: allow NaNs but reject infinities and sparse input.
    def _validated_prediction_data(self, X, check_input):
        check_is_fitted(self, "_tree_model_handle_")
        X = self._validate_X_predict(X, check_input)
        if sparse.issparse(X):
            raise TypeError("sparse matrices are not supported by the Rust decision tree")
        if np.isinf(X).any():
            raise ValueError("Input X contains infinity")
        return np.ascontiguousarray(X, dtype=np.float32)

    # Native probability prediction and sklearn-style array conversion.
    def predict_proba(self, X, check_input=True):
        X = self._validated_prediction_data(X, check_input)
        probabilities, _ = rb.tree_predict(self._tree_model_handle_, X)
        return np.asarray(probabilities, dtype=np.float64)

    # Class labels are reconstructed from the native probability matrix.
    def predict(self, X, check_input=True):
        probabilities = self.predict_proba(X, check_input=check_input)
        return self.classes_.take(np.argmax(probabilities, axis=1), axis=0)

    def predict_log_proba(self, X):
        with np.errstate(divide="warn"):
            return np.log(self.predict_proba(X))

    # Expose leaf indices for parity checks and downstream diagnostics.
    def apply(self, X, check_input=True):
        X = self._validated_prediction_data(X, check_input)
        _, leaves = rb.tree_predict(self._tree_model_handle_, X)
        return np.asarray(leaves, dtype=np.intp)

    # Mirror sklearn's tree introspection helpers.
    def get_depth(self):
        return int(self._inspect_model_arrays()["max_depth"])

    def get_n_leaves(self):
        return int(self._inspect_model_arrays()["n_leaves"])

    @property
    def feature_importances_(self):
        return np.asarray(
            self._inspect_model_arrays()["feature_importances"], dtype=np.float64
        )

    # Pull the serialized native tree arrays for inspection and testing.
    def _inspect_model_arrays(self):
        check_is_fitted(self, "_tree_model_handle_")
        return rb.tree_model_arrays(self._tree_model_handle_)


def _supports_rust_random_forest_parameters(estimator) -> tuple[bool, str]:
    if not isinstance(estimator, RandomForestClassifier):
        return False, "estimator is not a sklearn RandomForestClassifier"
    checks = (
        (estimator.criterion == "gini", "criterion must be 'gini'"),
        (
            estimator.bootstrap or estimator.max_samples is None,
            "max_samples requires bootstrap=True",
        ),
        (not estimator.oob_score, "oob_score is not supported"),
        (not estimator.warm_start, "warm_start is not supported"),
        (estimator.verbose == 0, "verbose must be 0"),
        (estimator.min_weight_fraction_leaf == 0.0, "min_weight_fraction_leaf must be 0"),
        (estimator.class_weight is None, "class_weight is not supported"),
        (estimator.ccp_alpha == 0.0, "ccp_alpha must be 0"),
        (estimator.monotonic_cst is None, "monotonic_cst is not supported"),
        (
            estimator.random_state is None
            or isinstance(estimator.random_state, numbers.Integral),
            "random_state must be None or an integer",
        ),
    )
    for supported, reason in checks:
        if not supported:
            return False, reason
    return True, ""


def supports_rust_random_forest_classifier(estimator) -> tuple[bool, str]:
    """Return whether the exact forest can run this estimator."""
    supported, reason = _supports_rust_random_forest_parameters(estimator)
    if not supported:
        return supported, reason
    if not rb.HAVE_RUST or rb.forest_fit_exact is None:
        return False, "Rust exact random-forest runtime is not available"
    return True, ""


def supports_rust_histogram_random_forest_classifier(estimator) -> tuple[bool, str]:
    """Return whether the histogram forest can run this estimator."""
    supported, reason = _supports_rust_random_forest_parameters(estimator)
    if not supported:
        return supported, reason
    if not rb.HAVE_RUST or rb.forest_fit_hist is None:
        return False, "Rust histogram random-forest runtime is not available"
    return True, ""


class RustRandomForestClassifier(RandomForestClassifier):
    """Sklearn-style adapter backed by a bounded parallel Rust forest."""

    def _bind_native_worker_budget(self, workers):
        """Bind a plan-time worker snapshot for a future physical operator."""
        if not isinstance(workers, numbers.Integral) or workers <= 0:
            raise ValueError("native worker budget must be a positive integer")
        self._stratum_worker_budget = int(workers)
        return self

    def _bind_forest_backend(self, backend, native_fit, native_fit_args=()):
        """Bind one native split backend before execution.

        Physical selection calls this once. Refusing a different second binding
        keeps the selected algorithm immutable for the estimator's lifetime.
        """
        binding = (backend, native_fit, tuple(native_fit_args))
        current = getattr(self, "_stratum_forest_binding", None)
        if current is not None and current != binding:
            raise ValueError("Rust random-forest split backend is already bound")
        self._stratum_forest_binding = binding
        self._stratum_forest_fit = native_fit
        self._stratum_forest_fit_args = tuple(native_fit_args)
        return self

    def _bind_exact_backend(self):
        """Bind the exact native split finder for physical execution."""
        if not rb.HAVE_RUST or rb.forest_fit_exact is None:
            raise ValueError("Rust exact random-forest runtime is not available")
        return self._bind_forest_backend("exact", rb.forest_fit_exact)

    def _bind_histogram_backend(self, n_bins=128):
        """Bind the standalone approximate histogram implementation."""
        if not isinstance(n_bins, numbers.Integral) or not 2 <= n_bins <= 255:
            raise ValueError("n_bins must be an integer in 2..=255")
        if not rb.HAVE_RUST or rb.forest_fit_hist is None:
            raise ValueError("Rust histogram random-forest runtime is not available")
        return self._bind_forest_backend(
            "histogram", rb.forest_fit_hist, (int(n_bins),)
        )

    def __sklearn_clone__(self):
        clone = type(self)(**self.get_params(deep=False))
        if hasattr(self, "_stratum_forest_fit"):
            clone._bind_forest_backend(*self._stratum_forest_binding)
        if hasattr(self, "_stratum_worker_budget"):
            clone._stratum_worker_budget = self._stratum_worker_budget
        return clone

    def fit(self, X, y, sample_weight=None):
        self._validate_params()
        supported, reason = _supports_rust_random_forest_parameters(self)
        if not supported:
            raise ValueError(f"unsupported Rust random-forest configuration: {reason}")
        native_fit = getattr(self, "_stratum_forest_fit", rb.forest_fit_exact)
        if native_fit is None:
            raise ValueError("bound Rust random-forest runtime is not available")
        if sample_weight is not None:
            raise ValueError("sample_weight is not supported by the Rust random forest")

        X, y = validate_data(
            self,
            X,
            y,
            validate_separately=(
                dict(dtype=np.float32, accept_sparse=False, ensure_all_finite="allow-nan"),
                dict(ensure_2d=False, dtype=None),
            ),
        )
        if sparse.issparse(X):
            raise TypeError("sparse matrices are not supported by the Rust random forest")
        if np.isinf(X).any():
            raise ValueError("Input X contains infinity")
        y = column_or_1d(y, warn=True)
        check_classification_targets(y)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_outputs_ = 1
        n_samples, self.n_features_in_ = X.shape

        min_samples_leaf = (
            int(self.min_samples_leaf)
            if isinstance(self.min_samples_leaf, numbers.Integral)
            else int(np.ceil(self.min_samples_leaf * n_samples))
        )
        min_samples_split = (
            int(self.min_samples_split)
            if isinstance(self.min_samples_split, numbers.Integral)
            else max(2, int(np.ceil(self.min_samples_split * n_samples)))
        )
        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)
        max_depth = np.iinfo(np.int32).max if self.max_depth is None else int(self.max_depth)
        if self.max_features is None:
            self.max_features_ = self.n_features_in_
        elif self.max_features == "sqrt":
            self.max_features_ = max(1, int(np.sqrt(self.n_features_in_)))
        elif self.max_features == "log2":
            self.max_features_ = max(1, int(np.log2(self.n_features_in_)))
        elif isinstance(self.max_features, numbers.Integral):
            self.max_features_ = int(self.max_features)
        else:
            self.max_features_ = max(1, int(self.max_features * self.n_features_in_))

        if not self.bootstrap:
            n_bootstrap = n_samples
        elif self.max_samples is None:
            n_bootstrap = n_samples
        elif isinstance(self.max_samples, numbers.Integral):
            n_bootstrap = int(self.max_samples)
            if n_bootstrap > n_samples:
                raise ValueError(
                    f"`max_samples` must be <= n_samples={n_samples} but got "
                    f"value {n_bootstrap}"
                )
        else:
            n_bootstrap = max(round(n_samples * self.max_samples), 1)

        random_state = check_random_state(self.random_state)
        tree_seeds = np.ascontiguousarray(
            random_state.randint(
                0, np.iinfo(np.int32).max, size=self.n_estimators, dtype=np.int64
            ),
            dtype=np.int64,
        )
        configured_workers = getattr(
            self, "_stratum_worker_budget", get_config()["num_threads"]
        )
        self._native_worker_budget_ = int(configured_workers) or (os.cpu_count() or 1)
        X = np.ascontiguousarray(X, dtype=np.float32)
        encoded = np.ascontiguousarray(encoded, dtype=np.int64)
        self._forest_model_handle_ = native_fit(
            X,
            encoded,
            tree_seeds,
            int(self.n_classes_),
            max_depth,
            min_samples_split,
            min_samples_leaf,
            self.max_features_,
            float(self.min_impurity_decrease),
            self.max_leaf_nodes,
            bool(self.bootstrap),
            n_bootstrap,
            self._native_worker_budget_,
            *getattr(self, "_stratum_forest_fit_args", ()),
        )
        return self

    def _validated_prediction_data(self, X):
        check_is_fitted(self, "_forest_model_handle_")
        X = validate_data(
            self,
            X,
            reset=False,
            dtype=np.float32,
            accept_sparse=False,
            ensure_all_finite="allow-nan",
        )
        if sparse.issparse(X):
            raise TypeError("sparse matrices are not supported by the Rust random forest")
        if np.isinf(X).any():
            raise ValueError("Input X contains infinity")
        return np.ascontiguousarray(X, dtype=np.float32)

    def predict_proba(self, X):
        X = self._validated_prediction_data(X)
        return np.asarray(
            rb.forest_predict(self._forest_model_handle_, X), dtype=np.float64
        )

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_.take(np.argmax(probabilities, axis=1), axis=0)

    def predict_log_proba(self, X):
        with np.errstate(divide="warn"):
            return np.log(self.predict_proba(X))

    @property
    def feature_importances_(self):
        return np.asarray(
            self._inspect_model()["feature_importances"], dtype=np.float64
        )

    def _inspect_model(self):
        check_is_fitted(self, "_forest_model_handle_")
        return rb.forest_model_info(self._forest_model_handle_)
