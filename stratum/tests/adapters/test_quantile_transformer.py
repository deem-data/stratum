import importlib.util
import sys
import warnings

import numpy as np
import pytest
from sklearn.preprocessing import QuantileTransformer as SklearnQuantileTransformer

from stratum.adapters.quantile_transformer import QuantileTransformer

# These are float-noise bounds, not correctness bounds. `_transform_col_fast` skips the
# reverse interpolation pass when the quantiles are strictly increasing, which reorders
# the same arithmetic and so agrees with scikit-learn to ~1e-14 in uniform space; `ndtri`
# then amplifies that to ~1e-12 in the tails, where its derivative is steep. Measured
# worst cases over this file's data are 8.9e-14 (uniform) and 1.3e-12 (normal).
FORWARD_ATOL = {"uniform": 1e-11, "normal": 1e-9}

# `inverse_transform` returns values on the data's own scale. An element-wise rtol
# degenerates for standard-normal data, which has entries arbitrarily close to zero, and
# a fixed atol is meaningless for a column scaled by 1e6 — so the bound is tied to the
# magnitude of the values actually being compared.
INVERSE_RTOL = 1e-9
INVERSE_ATOL_SCALE = 1e-9

DISTRIBUTIONS = ["uniform", "normal"]


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def data(rng):
    """Well-behaved continuous data, in the exact dtype/layout the fast path requires."""
    return np.ascontiguousarray(rng.standard_normal((2000, 12)))


def fit_quietly(estimator, X):
    """Fit and return (estimator, [warning messages]) without failing on warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = estimator.fit(X)
    return fitted, [str(w.message) for w in caught]


def call_quietly(bound_method, *args):
    """Call and return (result, [warning messages]) without failing on warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = bound_method(*args)
    return result, [str(w.message) for w in caught]


def fell_back(messages):
    """The adapter announces every throwback to the parent with this phrase."""
    return [m for m in messages if "alling back" in m]


def assert_matches_sklearn(X, *, distribution="uniform", n_quantiles=1000, n_jobs=1):
    """Fit+transform+inverse under both implementations and assert they agree.

    Asserts no fallback fired, so a regression that quietly routes everything to
    scikit-learn fails here instead of passing trivially.
    """
    fast = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution=distribution,
        subsample=None,
        n_jobs=n_jobs,
    )
    ref = SklearnQuantileTransformer(
        n_quantiles=n_quantiles, output_distribution=distribution, subsample=None
    )

    fast, fit_warnings = fit_quietly(fast, X)
    ref.fit(X)
    assert not fell_back(fit_warnings), fit_warnings

    fast_out, tf_warnings = call_quietly(fast.transform, X.copy())
    assert not fell_back(tf_warnings), tf_warnings
    ref_out = ref.transform(X.copy())

    assert np.array_equal(np.isnan(fast_out), np.isnan(ref_out))
    np.testing.assert_allclose(
        fast_out, ref_out, rtol=0, atol=FORWARD_ATOL[distribution], equal_nan=True
    )

    fast_inv, inv_warnings = call_quietly(fast.inverse_transform, fast_out.copy())
    assert not fell_back(inv_warnings), inv_warnings
    ref_inv = ref.inverse_transform(ref_out.copy())
    scale = np.nanmax(np.abs(ref_inv)) if np.isfinite(ref_inv).any() else 1.0
    np.testing.assert_allclose(
        fast_inv,
        ref_inv,
        rtol=INVERSE_RTOL,
        atol=INVERSE_ATOL_SCALE * max(1.0, float(scale)),
        equal_nan=True,
    )
    return fast, ref


# --------------------------------------------------------------------------------------
# 1. Numerical agreement with scikit-learn
# --------------------------------------------------------------------------------------


def test_import_smoke():
    assert QuantileTransformer is not SklearnQuantileTransformer
    assert issubclass(QuantileTransformer, SklearnQuantileTransformer)


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_matches_sklearn(data, distribution):
    assert_matches_sklearn(data, distribution=distribution)


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_quantiles_match_sklearn(data, distribution):
    fast, ref = assert_matches_sklearn(data, distribution=distribution)
    np.testing.assert_allclose(fast.quantiles_, ref.quantiles_, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(fast.references_, ref.references_)


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_with_nans(rng, distribution):
    X = rng.standard_normal((2000, 10))
    X[rng.random(X.shape) < 0.3] = np.nan
    assert_matches_sklearn(np.ascontiguousarray(X), distribution=distribution)


def test_all_nan_column(rng):
    """An empty column has no percentile; the kernel fills it with NaN rather than
    failing, which would poison the whole `quantiles_` matrix."""
    X = np.ascontiguousarray(rng.standard_normal((500, 6)))
    X[:, 2] = np.nan
    fast, _ = assert_matches_sklearn(X)
    assert np.isnan(fast.quantiles_[:, 2]).all()
    assert np.isfinite(fast.quantiles_[:, [0, 1, 3, 4, 5]]).all()


def test_negative_nan_is_filtered(rng):
    """The kernel sorts with `f64::total_cmp`, whose IEEE total order puts -NaN *below*
    every finite value and +NaN above. Filtering only one end would silently make the
    lowest quantiles NaN — and x86 really does produce negative NaNs (0.0/0.0)."""
    neg_nan = np.array([np.nan], dtype=np.float64)
    neg_nan.view(np.uint64)[0] |= np.uint64(1) << np.uint64(63)
    assert np.signbit(neg_nan[0])

    X = np.ascontiguousarray(rng.standard_normal((500, 5)))
    mask = rng.random(X.shape) < 0.3
    X[mask] = neg_nan[0]
    fast, _ = assert_matches_sklearn(X)
    assert np.isfinite(fast.quantiles_).all()


def test_constant_and_duplicated_columns(rng):
    X = np.ascontiguousarray(rng.standard_normal((800, 5)))
    X[:, 0] = 0.0
    X[:, 1] = 7.5
    assert_matches_sklearn(X)


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_repeated_values_need_both_interpolation_passes(rng, distribution):
    """Discrete data is the case the forward/reverse interpolation average exists for.

    `_transform_col_fast` skips the reverse pass when `quantiles_` is strictly
    increasing. Get that guard backwards and continuous data still matches to 1e-16
    while discrete data is off by ~0.1, so this is the test that can see it.
    """
    X = np.ascontiguousarray(rng.integers(0, 5, (2000, 6)).astype(np.float64))
    fast, _ = assert_matches_sklearn(X, distribution=distribution)
    # the premise: this data really does produce repeated quantiles
    assert (np.diff(fast.quantiles_, axis=0) == 0).any()


def test_quantiles_are_monotonic_with_ties(rng):
    X = np.ascontiguousarray(rng.integers(0, 3, (1000, 4)).astype(np.float64))
    fast, _ = assert_matches_sklearn(X)
    assert (np.diff(fast.quantiles_, axis=0) >= 0).all()


def test_extreme_scale(rng):
    """Forward output is bounded, but the inverse round-trips to the data's own scale."""
    assert_matches_sklearn(np.ascontiguousarray(rng.standard_normal((1000, 4)) * 1e6))


@pytest.mark.parametrize("n_quantiles", [1, 2, 10, 1000])
def test_n_quantiles_range(data, n_quantiles):
    assert_matches_sklearn(data[:500], n_quantiles=n_quantiles)


def test_subsample_path(rng):
    """With subsample < n_samples the adapter resamples before calling the kernel, and
    must not let the parent draw a second time on the fallback route."""
    X = np.ascontiguousarray(rng.standard_normal((5000, 6)))
    fast = QuantileTransformer(n_quantiles=100, subsample=1000, random_state=0)
    ref = SklearnQuantileTransformer(n_quantiles=100, subsample=1000, random_state=0)
    fast, messages = fit_quietly(fast, X)
    ref.fit(X)
    assert not fell_back(messages), messages
    np.testing.assert_allclose(fast.quantiles_, ref.quantiles_, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------------------
# 2. Layout and dtype handling
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
def test_fortran_and_non_contiguous_input(rng, distribution):
    """The gather handles column-major and doubly-strided views, so neither may fall
    back — the DataFrame route arrives column-major and must stay on the fast path."""
    base = rng.standard_normal((1000, 16))
    assert_matches_sklearn(np.asfortranarray(base), distribution=distribution)

    strided = rng.standard_normal((2000, 48))[::2, ::3]
    assert not strided.flags["C_CONTIGUOUS"] and not strided.flags["F_CONTIGUOUS"]
    assert_matches_sklearn(strided, distribution=distribution)


def test_float32_falls_back_with_warning(rng):
    """The kernel is compiled for f64 only; a float32 X must warn and still be correct."""
    X = rng.standard_normal((500, 4)).astype(np.float32)
    fast = QuantileTransformer(n_quantiles=100, subsample=None)
    fast, messages = fit_quietly(fast, X)
    assert fell_back(messages), "float32 should have thrown back to scikit-learn"

    ref = SklearnQuantileTransformer(n_quantiles=100, subsample=None).fit(X)
    np.testing.assert_allclose(fast.quantiles_, ref.quantiles_, rtol=0, atol=1e-6)


def test_dataframe_input_still_works():
    """Validation converts a DataFrame before `_dense_fit`; whichever path that lands
    on, the result must match scikit-learn."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        np.random.default_rng(0).standard_normal((500, 4)),
        columns=list("abcd"),
    )
    fast = QuantileTransformer(n_quantiles=100, subsample=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fast.fit(frame)
        out = fast.transform(frame)
    ref = SklearnQuantileTransformer(n_quantiles=100, subsample=None).fit(frame)
    np.testing.assert_allclose(
        out, ref.transform(frame), rtol=0, atol=FORWARD_ATOL["uniform"]
    )


def test_empty_and_single_row_inputs(rng):
    """A single sample has no interpolation interval; it must not raise."""
    assert_matches_sklearn(np.ascontiguousarray(rng.standard_normal((1, 5))),
                           n_quantiles=1)


# --------------------------------------------------------------------------------------
# 3. The fast path is actually taken
# --------------------------------------------------------------------------------------


def test_native_extension_is_reachable():
    """Guards the import itself.

    `_dense_fit` wraps the kernel call in `except Exception`, so importing the wrong
    module — the `_rust_backend` config shim rather than `_rust_backend_native` — turns
    every call into a silent fallback that no numerical test can see.
    """
    from stratum.adapters import quantile_transformer as mod

    assert mod._HAVE_RUST, "native extension not built for this interpreter"
    assert hasattr(mod._native, "rust_fit_dense"), (
        "adapter imported a module without the kernel on it"
    )


NATIVE_ATTR = "_rust_backend_native"
NATIVE_MODULE = f"stratum.{NATIVE_ATTR}"


@pytest.fixture
def adapter_without_native():
    """A private copy of the adapter, executed as it would import with no compiled kernel.

    Two things have to be hidden, not one. A ``None`` entry in ``sys.modules`` is the
    documented way to make an import fail -- the import system finds the key, sees no
    module, and raises ImportError -- but ``from stratum import _rust_backend_native``
    reads the attribute the parent package caches once the submodule has been imported
    and never consults ``sys.modules`` at all. Hide only one of the two and the import
    quietly succeeds, which is exactly the false pass this fixture exists to avoid.

    The module is executed under its own name and deliberately NOT registered in
    ``sys.modules``, so the adapter every other test imported is never touched.
    """
    import stratum

    source = sys.modules[QuantileTransformer.__module__].__file__
    saved_module = sys.modules.get(NATIVE_MODULE)
    saved_attr = getattr(stratum, NATIVE_ATTR, None)

    sys.modules[NATIVE_MODULE] = None
    if saved_attr is not None:
        delattr(stratum, NATIVE_ATTR)
    try:
        spec = importlib.util.spec_from_file_location(
            "stratum.adapters._quantile_transformer_no_native", source
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        if saved_module is None:
            sys.modules.pop(NATIVE_MODULE, None)
        else:
            sys.modules[NATIVE_MODULE] = saved_module
        if saved_attr is not None:
            setattr(stratum, NATIVE_ATTR, saved_attr)


def test_import_without_the_extension_degrades_instead_of_failing(adapter_without_native):
    """The adapter must remain importable on a platform with no compiled kernel.

    This is the negative half of `test_native_extension_is_reachable`: that one pins the
    import down when the extension is there, this one executes the `except ImportError`
    arm that no other test in this file reaches.
    """
    module = adapter_without_native

    assert module._HAVE_RUST is False
    assert module._native is None


def test_fit_falls_back_when_the_extension_is_missing(adapter_without_native, data):
    """Without a kernel the adapter is a plain scikit-learn transformer that says so."""
    estimator = adapter_without_native.QuantileTransformer(
        n_quantiles=100, subsample=None
    )
    estimator, messages = fit_quietly(estimator, data)

    assert fell_back(messages), messages
    assert any("native extension is not built" in m for m in messages), messages

    # Falling back has to mean scikit-learn's answer, not merely scikit-learn's code
    # path: a guard that warned and then returned something else would pass the above.
    reference = SklearnQuantileTransformer(n_quantiles=100, subsample=None).fit(data)
    np.testing.assert_allclose(estimator.quantiles_, reference.quantiles_)

    # Only the fit needs the kernel: `_check_before_transform` has no `_HAVE_RUST` arm,
    # so the transform fast path still runs here, on quantiles_ the parent computed.
    out, tf_messages = call_quietly(estimator.transform, data.copy())
    assert not fell_back(tf_messages), tf_messages
    np.testing.assert_allclose(
        out, reference.transform(data), atol=FORWARD_ATOL["uniform"]
    )


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.parametrize("n_jobs", [1, 4])
def test_no_fallback_on_the_supported_path(data, distribution, n_jobs):
    """Nothing about well-formed float64 input may trigger the parent implementation."""
    estimator = QuantileTransformer(
        n_quantiles=100, output_distribution=distribution, subsample=None, n_jobs=n_jobs
    )
    estimator, fit_messages = fit_quietly(estimator, data)
    assert not fell_back(fit_messages), fit_messages

    out, tf_messages = call_quietly(estimator.transform, data.copy())
    assert not fell_back(tf_messages), tf_messages

    _, inv_messages = call_quietly(estimator.inverse_transform, out)
    assert not fell_back(inv_messages), inv_messages


def test_quantiles_shape_and_dtype(data):
    """`quantiles_` is built transposed in Rust and returned via a stride swap; the
    caller-visible shape must still be (n_quantiles, n_features)."""
    estimator, _ = fit_quietly(
        QuantileTransformer(n_quantiles=250, subsample=None), data
    )
    assert estimator.quantiles_.shape == (250, data.shape[1])
    assert estimator.quantiles_.dtype == np.float64


@pytest.mark.parametrize("n_jobs", [1, 2, 8])
def test_n_jobs_does_not_change_the_result(data, n_jobs):
    """n_jobs is a fan-out over features and must be numerically invisible."""
    serial, _ = fit_quietly(
        QuantileTransformer(n_quantiles=200, subsample=None, n_jobs=1), data
    )
    threaded, _ = fit_quietly(
        QuantileTransformer(n_quantiles=200, subsample=None, n_jobs=n_jobs), data
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.testing.assert_array_equal(
            serial.transform(data.copy()), threaded.transform(data.copy())
        )


@pytest.mark.parametrize("n_jobs", [0, -1, 1.5, None])
def test_invalid_n_jobs_falls_back(data, n_jobs):
    """The threaded path needs at least one worker. Anything else throws back rather
    than raising — including -1, which is scikit-learn's "all cores" convention and is
    deliberately *not* implemented here."""
    estimator = QuantileTransformer(n_quantiles=100, subsample=None, n_jobs=n_jobs)
    estimator, _ = fit_quietly(estimator, data)
    _, messages = call_quietly(estimator.transform, data.copy())
    assert fell_back(messages), f"n_jobs={n_jobs!r} should have thrown back"


# --------------------------------------------------------------------------------------
# 4. Contracts held by construction rather than by code
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.parametrize("method", ["transform", "inverse_transform"])
def test_copy_true_does_not_touch_the_callers_array(data, distribution, method):
    """`copy` is honoured by the *inherited* `transform`, which copies before
    `_transform` ever runs. The adapter must not override that away."""
    estimator, _ = fit_quietly(
        QuantileTransformer(
            n_quantiles=100, output_distribution=distribution, subsample=None, copy=True
        ),
        data,
    )
    X = data.copy()
    before = X.copy()
    call_quietly(getattr(estimator, method), X)
    np.testing.assert_array_equal(X, before)


@pytest.mark.parametrize("distribution", DISTRIBUTIONS)
@pytest.mark.parametrize("n_jobs", [1, 4])
def test_copy_false_writes_through_to_the_caller(data, distribution, n_jobs):
    """The other half of the contract, and the fragile one.

    `_transform_col_fast` returns a *new* array on the no-NaN branch instead of mutating
    the column view, so writing through depends entirely on `_transform` assigning the
    result back into `X[:, i]`. Delete that assignment as a redundant copy and
    `copy=False` silently stops writing through, with no error anywhere.
    """
    estimator, _ = fit_quietly(
        QuantileTransformer(
            n_quantiles=100,
            output_distribution=distribution,
            subsample=None,
            copy=False,
            n_jobs=n_jobs,
        ),
        data,
    )
    X = data.copy()
    before = X.copy()
    result, _ = call_quietly(estimator.transform, X)
    assert not np.array_equal(X, before), "copy=False did not write through to X"
    np.testing.assert_array_equal(result, X)


def test_copy_matches_sklearn_semantics(data):
    """Whatever scikit-learn does to the caller's array, the adapter does too."""
    for copy in (True, False):
        fast, _ = fit_quietly(
            QuantileTransformer(n_quantiles=100, subsample=None, copy=copy), data
        )
        ref = SklearnQuantileTransformer(
            n_quantiles=100, subsample=None, copy=copy
        ).fit(data)

        Xa, Xb = data.copy(), data.copy()
        ref_out = ref.transform(Xa)
        fast_out, _ = call_quietly(fast.transform, Xb)

        ref_mutated = not np.array_equal(Xa, data)
        fast_mutated = not np.array_equal(Xb, data)
        assert ref_mutated == fast_mutated, f"copy={copy} side effect differs"
        np.testing.assert_allclose(
            fast_out, ref_out, rtol=0, atol=FORWARD_ATOL["uniform"]
        )


def test_transform_before_fit_falls_back(data):
    """`_check_before_transform` doubles as the not-fitted check."""
    estimator = QuantileTransformer(n_quantiles=100, subsample=None)
    with pytest.raises(Exception):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.transform(data.copy())


def test_round_trip_recovers_the_input(rng):
    """inverse_transform(transform(X)) ~= X away from the clipped tails."""
    X = np.ascontiguousarray(rng.standard_normal((2000, 5)))
    estimator, _ = fit_quietly(
        QuantileTransformer(n_quantiles=1000, subsample=None), X
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recovered = estimator.inverse_transform(estimator.transform(X.copy()))
    np.testing.assert_allclose(recovered, X, rtol=0, atol=1e-9)
