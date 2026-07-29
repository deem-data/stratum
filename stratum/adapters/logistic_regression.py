import warnings
from numbers import Integral, Real

import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

try:
    from stratum import _rust_backend_native as _native

    _HAVE_RUST = True
except ImportError:  # extension not built for this platform/interpreter
    _native = None
    _HAVE_RUST = False

# The only solver the native kernel implements. A solver scikit-learn accepts but this
# one does not is a throwback, not an error, which is why `_check_params` rejects it
# rather than running the lbfgs kernel under another solver's name.
_SOLVER = "lbfgs"


class LogisticRegression(SklearnLogisticRegression):
    def __init__(
        self,
        penalty="deprecated",
        *,
        C=1.0,
        l1_ratio=0.0,
        dual=False,
        tol=0.0001,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        verbose=0,
        warm_start=False,
        n_jobs=None,
    ):
        super().__init__(
            penalty=penalty,
            C=C,
            l1_ratio=l1_ratio,
            dual=dual,
            tol=tol,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
        )

    def _check_params(self):
        if self.solver != _SOLVER:
            raise NotImplementedError(
                f"solver={self.solver!r} has no native kernel "
                f"(supported: [{_SOLVER!r}])."
            )

        if self.penalty != "deprecated":
            raise NotImplementedError(
                f"penalty={self.penalty!r} is not read by the fast path, which derives "
                "an L2 penalty from C."
            )

        # Class weights multiply into the per-sample weights; the kernel takes
        # `sample_weight` as given and does no reweighting of its own.
        if self.class_weight is not None:
            raise NotImplementedError("class_weight is not applied by the fast path.")

        # The kernel minimises the primal objective only.
        if self.dual:
            raise NotImplementedError("dual=True has no native kernel.")

        # The kernel always starts LBFGS from w = 0, so warm_start would do nothing.
        if self.warm_start:
            raise NotImplementedError("warm_start is not supported by the fast path.")

        
        if not isinstance(self.C, Real) or not self.C > 0:
            raise NotImplementedError(f"C={self.C!r} is outside the fast path's range.")

        # Only the pure-L2 setting has the kernel.
        if not isinstance(self.l1_ratio, Real) or self.l1_ratio != 0:
            raise NotImplementedError(
                f"l1_ratio={self.l1_ratio!r}: the fast path has no L1 kernel, only L2."
            )

        if not isinstance(self.tol, Real) or self.tol < 0:
            raise NotImplementedError(
                f"tol={self.tol!r} is outside the fast path's range."
            )

        if not isinstance(self.max_iter, Integral) or self.max_iter < 0:
            raise NotImplementedError(
                f"max_iter={self.max_iter!r} is outside the fast path's range."
            )

        if not isinstance(self.fit_intercept, (bool, np.bool_)):
            raise NotImplementedError(
                f"fit_intercept={self.fit_intercept!r} is not a boolean."
            )

    def _check_before_predict(self, X):
        if not _HAVE_RUST:
            raise NotImplementedError("The native backend is unavailable.")

        # Doubles as the not-fitted check: `n_classes_` is set only by a successful kernel
        # fit, so a model trained through the fallback (or not trained at all) lands here.
        if not hasattr(self, "n_classes_"):
            raise NotImplementedError(
                "This model was not fitted by the native fast path; predicting with scikit-learn."
            )

        # The kernel reads the buffer directly. An exact type check rejects DataFrames,
        # lists, sparse matrices and masked arrays in one comparison, without touching a
        # single element and without materialising a conversion.
        if type(X) is not np.ndarray:
            raise NotImplementedError(
                f"X is {type(X).__name__}; the fast path needs a plain ndarray."
            )

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise NotImplementedError(
                f"X has shape {X.shape}; the fast path needs (n_samples, {self.n_features_in_})."
            )

        if X.dtype != np.float32:
            raise NotImplementedError(f"X is {X.dtype}; the fast path needs float32 X.")

        if not X.flags["C_CONTIGUOUS"]:
            raise NotImplementedError("X is not C-contiguous.")

        # Guards the kernel's contract on the model's OWN coef_. A fast-fitted model always
        # satisfies these; they can only fail if coef_ was mutated externally to a
        # dtype/layout the kernel cannot consume — sklearn can still use it, so fall back.
        if self.coef_.dtype != np.float32 or not self.coef_.flags["F_CONTIGUOUS"]:
            raise NotImplementedError(
                "coef_ is not float32/F-contiguous; the fast path needs both."
            )

    def _check_before_fit(self, X, y, sample_weight):
        if not _HAVE_RUST:
            raise NotImplementedError("The native backend is unavailable.")

        self._check_params()

        if type(X) is not np.ndarray or type(y) is not np.ndarray:
            raise NotImplementedError(
                f"X is {type(X).__name__} and y is {type(y).__name__}; the fast path "
                "needs plain ndarrays."
            )

        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise NotImplementedError(
                f"X has shape {X.shape} and y has shape {y.shape}; the fast path needs a "
                "2-D X and a matching 1-D y."
            )

        if X.dtype != np.float32:
            raise NotImplementedError(f"X is {X.dtype}; the fast path needs float32 X.")

        if y.dtype != np.uint32:
            raise NotImplementedError(f"y is {y.dtype}; the fast path needs uint32 labels.")

        if not X.flags["C_CONTIGUOUS"] or not y.flags["C_CONTIGUOUS"]:
            raise NotImplementedError("X and y must both be C-contiguous.")

        # Here happens a O(n) check, since otherwise the solver will not halt.
        if not np.isfinite(X).all():
            raise NotImplementedError(
                "X contains NaN or inf; the fast path cannot handle that."
            )

        if sample_weight is not None:
            if (
                type(sample_weight) is not np.ndarray
                or sample_weight.ndim != 1
                or sample_weight.shape[0] != y.shape[0]
                or sample_weight.dtype != np.float32
                or not sample_weight.flags["C_CONTIGUOUS"]
            ):
                raise NotImplementedError(
                    "sample_weight must be a C-contiguous 1-D float32 ndarray matching y."
                )
            # Here happens a O(n) check, since otherwise the solver will not halt
            if not np.isfinite(sample_weight).all():
                raise NotImplementedError(
                    "sample_weight contains NaN or inf; the fast path cannot handle that."
                )

        # Checked before `y.max()`, which has no answer on an empty y and would raise a
        # numpy ValueError straight past the NotImplementedError contract `fit` catches.
        if y.shape[0] == 0:
            raise NotImplementedError("y is empty; the fast path needs a sample to fit.")


        n_classes = max(int(y.max()) + 1, 2)
        if n_classes > 2:
            raise NotImplementedError(
                "Only binary classification (labels {0, 1}) is supported by the fast path; "
                "training multiclass with scikit-learn."
            )

        self.n_classes_ = n_classes
        self.n_features_in_ = X.shape[1]

        # A bare ndarray carries no column names. Clearing any left over from an earlier
        # DataFrame fit on this instance keeps `feature_names_in_` from going stale.
        if hasattr(self, "feature_names_in_"):
            del self.feature_names_in_

    def predict_proba(self, X):
        try:
            self._check_before_predict(X)
        except NotImplementedError as e:
            warnings.warn(
                f"{e} Falling back to scikit-learn's predict_proba implementation.",
                stacklevel=2,
            )
            return super().predict_proba(X)

        # coef_ is stored in sklearn's (1, n_features) shape (see fit); the kernel wants
        # a 1-D weight vector and a scalar bias, so pass a row view and a Python float.
        try:
            return _native.binary_predict_proba(
                X,
                self.coef_[0],
                float(np.ravel(self.intercept_)[0]),
                intercept=self.fit_intercept,
            )
        except Exception as e:
            warnings.warn(
                f"The native kernel failed to predict probabilities: {e}; "
                "falling back to scikit-learn's predict_proba implementation.",
                stacklevel=2,
            )
            return super().predict_proba(X)

    def predict(self, X):
        try:
            self._check_before_predict(X)
        except NotImplementedError as e:
            warnings.warn(
                f"{e} Falling back to scikit-learn's predict implementation.",
                stacklevel=2,
            )
            return super().predict(X)

        try:
            return _native.binary_predict(
                X,
                self.coef_[0],
                float(np.ravel(self.intercept_)[0]),
                intercept=self.fit_intercept,
            )
        except Exception as e:
            warnings.warn(
                f"The native kernel failed to predict: {e}; "
                "falling back to scikit-learn's predict implementation.",
                stacklevel=2,
            )
            return super().predict(X)

    def fit(self, X, y, sample_weight=None):
        try:
            self._check_before_fit(X, y, sample_weight)
        except NotImplementedError as e:
            if hasattr(self, "n_classes_"):
                del self.n_classes_
            warnings.warn(
                f"{e} Falling back to scikit-learn's fit implementation.",
                stacklevel=2,
            )
            # The parent receives the caller's inputs exactly as given — unconverted, so a
            # DataFrame still arrives as a DataFrame and keeps its `feature_names_in_`.
            return super().fit(X, y, sample_weight=sample_weight)

        # `_check_params` has already forced l1_ratio to 0, so l1_reg is always 0.0 and
        # l2_reg is 1 / C. The elasticnet split is written out anyway: it is the mapping the
        # kernel expects, and it keeps this call correct on the day a solver that can
        # actually do L1 replaces lbfgs.
        try:
            all_weights = _native.binary_lbfgs_fit(
                x=X,
                y=y,
                l1_reg=1 / self.C * self.l1_ratio,
                l2_reg=1 / self.C * (1 - self.l1_ratio),
                intercept=self.fit_intercept,
                max_iters=self.max_iter,
                m=10,
                tolerance=self.tol,
                sample_weights=sample_weight,
            )
        except Exception as e:
            print(
                f"The native kernel failed to fit the model: {e}; falling back to scikit-learn's fit implementation."
            )
            return super().fit(X, y, sample_weight=sample_weight)

        # Store the solution in scikit-learn's canonical layout so a model trained on the
        # fast path is also servable by the inherited (fallback) predict/predict_proba and
        # by downstream sklearn tooling:
        #   - coef_       shape (1, n_features)   (binary case is a single row)
        #   - intercept_  shape (1,)
        #   - classes_    the ordered class labels used by LinearClassifierMixin.predict (but unused by the rust kernel)
        # refs: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_logistic.py
        #       https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/linear_model/_base.py  (decision_function / predict)
        self.classes_ = np.array([0, 1], dtype=y.dtype)

        if self.fit_intercept:
            self.coef_ = all_weights[:-1].reshape(1, -1)
            self.intercept_ = all_weights[-1:].copy()
        else:
            self.coef_ = all_weights.reshape(1, -1)
            # sklearn guarantees `intercept_` is an ndarray of shape (1,) in the binary
            # case even when the intercept is not fitted. A bare 0.0 float happens to
            # broadcast correctly inside decision_function, but breaks every caller that
            # reads `.shape` or `.dtype` off it.
            self.intercept_ = np.zeros(1, dtype=self.coef_.dtype)

        # scikit-learn's fit contract: return the fitted estimator so callers can chain
        # `model.fit(X, y).predict(...)` and Pipeline/clone work as expected.
        return self
