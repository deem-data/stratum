import warnings
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral

import numpy as np
from scipy import sparse
from scipy.special import ndtr, ndtri
from sklearn.preprocessing import QuantileTransformer as SklearnQuantileTransformer
from sklearn.utils import resample
from sklearn.utils._param_validation import (
    Interval,
    StrOptions,
)

try:
    from stratum import _rust_backend_native as _native

    _HAVE_RUST = True
except ImportError:  # extension not built for this platform/interpreter
    _native = None
    _HAVE_RUST = False

BOUNDS_THRESHOLD = 1e-7


class QuantileTransformer(SklearnQuantileTransformer):
    _parameter_constraints: dict = {
        "n_quantiles": [Interval(Integral, 1, None, closed="left")],
        "output_distribution": [StrOptions({"uniform", "normal"})],
        "ignore_implicit_zeros": ["boolean"],
        "subsample": [Interval(Integral, 1, None, closed="left"), None],
        "random_state": ["random_state"],
        "copy": ["boolean"],
    }

    def __init__(
        self,
        n_quantiles=1000,
        output_distribution="uniform",
        ignore_implicit_zeros=False,
        subsample=10000,
        random_state=None,
        copy=True,
        n_jobs=1,
    ):
        super().__init__(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            ignore_implicit_zeros=ignore_implicit_zeros,
            subsample=subsample,
            random_state=random_state,
            copy=copy,
        )

        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.ignore_implicit_zeros = ignore_implicit_zeros
        self.subsample = subsample
        self.random_state = random_state
        self.copy = copy
        self.n_jobs = n_jobs

    def _check_before_fit(self, X, references):

        if not _HAVE_RUST:
            raise NotImplementedError(
                "the native extension is not built for this interpreter."
            )

        # The kernel reads the buffer directly. An exact type check rejects DataFrames,
        # lists, sparse matrices and masked arrays in one comparison, without touching a
        # single element and without materialising a conversion.
        if type(X) is not np.ndarray:
            raise NotImplementedError(
                f"X is {type(X).__name__}; the fast path needs a plain ndarray."
            )

        if X.ndim != 2:
            raise NotImplementedError(
                f"X has shape {X.shape}; the fast path needs a 2-D X."
            )

        # scikit-learn's input validation admits any of FLOAT_DTYPES, so a float32 X
        # reaches here unconverted; the kernel is compiled for f64 only.
        if X.dtype != np.float64:
            raise NotImplementedError(f"X is {X.dtype}; the fast path needs float64 X.")

        # Deliberately no layout check: the kernel gathers each column element by element
        # through a strided view, so F-order and non-contiguous X are as correct as C-order
        # (verified against np.nanpercentile down to 1e-15). Demanding C-contiguity here
        # would drop the whole DataFrame path, which validation hands over column-major.

        # An empty column has no percentile: the kernel fills that feature with NaN
        # rather than failing, which would poison `quantiles_` silently.
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise NotImplementedError(
                f"X has shape {X.shape}; the fast path needs at least one sample and "
                "one feature."
            )

        # `references` is built by scikit-learn (linspace over [0, 1]), so these hold for
        # any untampered estimator — they are here because the kernel trusts them.
        if type(references) is not np.ndarray or references.ndim != 1:
            raise NotImplementedError(
                "references must be a 1-D ndarray of quantile levels."
            )

        if references.dtype != np.float64:
            raise NotImplementedError(
                f"references is {references.dtype}; the fast path needs float64."
            )

        if references.shape[0] == 0:
            raise NotImplementedError(
                "references is empty; the fast path needs at least one quantile level."
            )

        # The kernel turns each level q into the index q * (n_samples - 1) of the sorted
        # column and indexes with it unchecked, so a level outside [0, 1] (or a NaN) is
        # an out-of-bounds read, not a wrong number. O(n_quantiles), which is small.
        if not np.isfinite(references).all() or references.min() < 0 or references.max() > 1:
            raise NotImplementedError(
                "references must be finite and inside [0, 1]; the fast path indexes "
                "the sorted column with them directly."
            )

    def _check_before_transform(self, X):
        # Doubles as the not-fitted check: `_transform` interpolates against these two,
        # and both are set only by a completed fit.
        if not hasattr(self, "quantiles_") or not hasattr(self, "references_"):
            raise NotImplementedError(
                "This transformer is not fitted; transforming with scikit-learn."
            )

        if self.output_distribution not in ("uniform", "normal"):
            raise NotImplementedError(
                f"output_distribution={self.output_distribution!r} has no fast path "
                "(supported: 'uniform', 'normal')."
            )

        # The threaded path fans columns out across a ThreadPoolExecutor sized by n_jobs;
        # anything below one worker has no pool to run in.
        if not isinstance(self.n_jobs, Integral) or self.n_jobs < 1:
            raise NotImplementedError(
                f"n_jobs={self.n_jobs!r} is outside the fast path's range."
            )

        # Sparse input takes the per-column scikit-learn route inside `_transform`; the
        # fast path slices X as a dense 2-D buffer.
        if type(X) is not np.ndarray:
            raise NotImplementedError(
                f"X is {type(X).__name__}; the fast path needs a plain ndarray."
            )

        # `quantiles_` is (n_quantiles, n_features) and is indexed column by column
        # against X, so its width — not `n_features_in_` — is what has to match.
        if X.ndim != 2 or X.shape[1] != self.quantiles_.shape[1]:
            raise NotImplementedError(
                f"X has shape {X.shape}; the fast path needs "
                f"(n_samples, {self.quantiles_.shape[1]})."
            )

        # Each column is transformed through float64 intermediates (np.interp, ndtri) and
        # written straight back; a narrower X would round-trip through a silent downcast.
        if X.dtype != np.float64:
            raise NotImplementedError(f"X is {X.dtype}; the fast path needs float64 X.")

        # The transform assigns into X (and into views of it) in place.
        if not X.flags["WRITEABLE"]:
            raise NotImplementedError(
                "X is read-only; the fast path transforms it in place."
            )

        # Guards the kernel's contract on the model's OWN fitted state. A fast-fitted
        # transformer always satisfies these; they can only fail if `quantiles_` or
        # `references_` was mutated externally — sklearn can still use them, so fall back.
        if self.quantiles_.dtype != np.float64 or self.references_.dtype != np.float64:
            raise NotImplementedError(
                "quantiles_/references_ are not float64; the fast path needs both to be."
            )

        if self.references_.shape[0] != self.quantiles_.shape[0]:
            raise NotImplementedError(
                f"references_ has {self.references_.shape[0]} levels but quantiles_ has "
                f"{self.quantiles_.shape[0]} rows; np.interp needs them paired."
            )

    def _transform(self, X, inverse=False):
        """Forward and inverse transform.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.

        inverse : bool, default=False
            If False, apply forward transform. If True, apply
            inverse transform.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Projected data.
        """
        # Sparse X lands here too: the guard rejects it as "not a plain ndarray" and the
        # parent's per-column CSC loop takes it, which is what this override used to
        # duplicate verbatim.
        try:
            self._check_before_transform(X)
        except NotImplementedError as e:
            warnings.warn(
                f"{e} Falling back to scikit-learn's _transform implementation.",
                stacklevel=2,
            )
            return super()._transform(X, inverse=inverse)

        # Which features have repeated quantiles, computed once for the whole matrix
        # rather than per column. `_transform_col_fast` needs this to decide whether the
        # reverse interpolation pass is load-bearing; see the comment there. One
        # vectorised pass over (n_quantiles, n_features) costs ~100us and saves a
        # per-column np.diff.
        if inverse:
            # The inverse direction interpolates against `references_`, which is a
            # linspace and never repeats, so the reverse pass never runs.
            has_duplicates = np.zeros(self.quantiles_.shape[1], dtype=bool)
        else:
            has_duplicates = (np.diff(self.quantiles_, axis=0) == 0).any(axis=0)

        # No try/except around the loops below, unlike `_dense_fit`: they write the
        # transformed columns straight back into X, so a failure part-way through leaves X
        # half-transformed and there is nothing clean left to hand the parent. A raise
        # from here is a real error, not a fallback condition.
        def transform_feature(feature_idx):
            # The worker writes its own column instead of handing the result back for
            # the parent to store: features are disjoint slices of X, so there is no
            # race, and it keeps the strided scatter off the serialising main thread.
            X[:, feature_idx] = self._transform_col_fast(
                X[:, feature_idx],
                self.quantiles_[:, feature_idx],
                inverse,
                has_duplicates[feature_idx],
            )

        if self.n_jobs == 1:
            for feature_idx in range(X.shape[1]):
                transform_feature(feature_idx)
        else:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                # Consume the iterator: `map` is lazy and would otherwise not run.
                for _ in executor.map(transform_feature, range(X.shape[1])):
                    pass

        return X

    

    def _transform_col_fast(self, X_col, quantiles, inverse, has_duplicates):
        """Transform a single feature, for either output distribution.

        Mirrors `QuantileTransformer._transform_col` exactly, but goes through the
        `ndtr`/`ndtri` ufuncs rather than `scipy.stats.norm`, and skips work that
        provably cannot change the result — see the comments on each branch.
        """
        normal = self.output_distribution == "normal"

        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1
        else:
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]
            # for inverse transform, match a uniform distribution
            if normal:
                # `ndtr` is `scipy.stats.norm.cdf` without the frozen-distribution
                # dispatch, which dominates the call at this array size.
                with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
                    X_col = ndtr(X_col)
                # else output distribution is already a uniform distribution

        # find index for lower and higher bounds
        with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
            if normal:
                # `X_col - THRESHOLD < bound` rearranged: same predicate, but it
                # compares against a scalar instead of materialising a shifted copy
                # of the whole column first.
                lower_bounds_idx = X_col < lower_bound_x + BOUNDS_THRESHOLD
                upper_bounds_idx = X_col > upper_bound_x - BOUNDS_THRESHOLD
            else:
                lower_bounds_idx = X_col == lower_bound_x
                upper_bounds_idx = X_col == upper_bound_x

        # A column with no NaN needs neither the gather nor the scatter, and running
        # on the compacted result also takes the ops below off X's strided column
        # view (X is usually C-ordered, so that view has an n_features stride).
        nan_mask = np.isnan(X_col)
        any_nan = nan_mask.any()
        if any_nan:
            isfinite_mask = ~nan_mask
            X_col_finite = X_col[isfinite_mask]
        else:
            X_col_finite = X_col

        if not inverse:
            if has_duplicates:
                # Interpolate in one direction and in the other and take the
                # mean. This is in case of repeated values in the features
                # and hence repeated quantiles
                #
                # If we don't do this, only one extreme of the duplicated is
                # used (the upper when we do ascending, and the
                # lower for descending). We take the mean of these two
                transformed = 0.5 * (
                    np.interp(X_col_finite, quantiles, self.references_)
                    - np.interp(
                        -X_col_finite, -quantiles[::-1], -self.references_[::-1]
                    )
                )
            else:
                # Strictly increasing quantiles have no repeated value for the two
                # passes to straddle, so the reverse pass agrees with the forward one
                # (~1e-16) and averaging them is a no-op. Interpolation is ~80% of
                # this function, so skipping it is the difference that matters.
                transformed = np.interp(X_col_finite, quantiles, self.references_)
        else:
            transformed = np.interp(X_col_finite, self.references_, quantiles)

        if any_nan:
            X_col[isfinite_mask] = transformed
        else:
            X_col = transformed

        X_col[upper_bounds_idx] = upper_bound_y
        X_col[lower_bounds_idx] = lower_bound_y
        # for forward transform, match the output distribution
        if not inverse:
            if normal:
                with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
                    X_col = ndtri(X_col)
                    # find the value to clip the data to avoid mapping to
                    # infinity. Clip such that the inverse transform will be
                    # consistent
                    clip_min = ndtri(BOUNDS_THRESHOLD - np.spacing(1))
                    clip_max = ndtri(1 - (BOUNDS_THRESHOLD - np.spacing(1)))
                    X_col = np.clip(X_col, clip_min, clip_max)
            # else output distribution is uniform and the ppf is the
            # identity function so we let X_col unchanged

        return X_col


    def _dense_fit(self, X, random_state):
        """Compute percentiles for dense matrices.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.
        """
        try:
            self._check_before_fit(X, self.references_)
        except NotImplementedError as e:
            warnings.warn(
                f"{e} Falling back to scikit-learn's _dense_fit implementation.",
                stacklevel=2,
            )
            # The parent receives X exactly as given and draws its own subsample, so
            # `random_state` is consumed once either way.
            return super()._dense_fit(X, random_state)

        if self.ignore_implicit_zeros:
            warnings.warn(
                "'ignore_implicit_zeros' takes effect only with"
                " sparse matrix. This parameter has no effect."
            )

        n_samples, _ = X.shape
        references = self.references_

        if self.subsample is not None and self.subsample < n_samples:
            # Take a subsample of `X`
            X = resample(
                X, replace=False, n_samples=self.subsample, random_state=random_state
            )

        # `references` is passed in [0, 1] — the kernel scales each level by the column
        # length itself, unlike np.nanpercentile, which wants percents.
        try:
            self.quantiles_ = _native.rust_fit_dense(X, references)
        except Exception as e:
            warnings.warn(
                f"The native kernel failed to compute the quantiles: {e}; "
                "falling back to scikit-learn's _dense_fit implementation.",
                stacklevel=2,
            )
            # X is the subsampled array here, so `self.subsample` is no longer smaller
            # than its row count and the parent will not draw a second time.
            return super()._dense_fit(X, random_state)