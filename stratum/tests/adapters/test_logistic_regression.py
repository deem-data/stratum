import contextlib
import importlib.util
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.utils._param_validation import InvalidParameterError

from stratum.adapters.logistic_regression import LogisticRegression

# f32 LBFGS against sklearn's f64 lbfgs: the solutions agree to a few 1e-4, not to
# machine precision. Tight enough to catch a wrong objective, loose enough to survive
# the dtype difference.
COEF_ATOL = 5e-3
PROBA_ATOL = 5e-3


@pytest.fixture(scope="module")
def data():
    """A separable-ish binary problem, in the exact dtypes the fast path requires."""
    X, y = make_classification(
        n_samples=2000,
        n_features=12,
        n_informative=8,
        n_redundant=0,
        random_state=0,
    )
    return np.ascontiguousarray(X, dtype=np.float32), y.astype(np.uint32)


def fit_quietly(estimator, *args, **kwargs):
    """Fit and return (estimator, [warning messages]) without failing on warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = estimator.fit(*args, **kwargs)
    return fitted, [str(w.message) for w in caught]


def took_fast_path(estimator):
    """`n_classes_` is the marker the adapter sets only on a successful kernel fit."""
    return hasattr(estimator, "n_classes_")


@contextlib.contextmanager
def no_warnings():
    """Silence the expected "falling back" warning where it is not what is being tested."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# --------------------------------------------------------------------------------------
# 1. Numerical agreement with scikit-learn
# --------------------------------------------------------------------------------------


def test_fast_path_matches_sklearn(data):
    X, y = data
    fast = LogisticRegression().fit(X, y)
    ref = SklearnLogisticRegression().fit(X.astype(np.float64), y)

    assert took_fast_path(fast)
    assert np.allclose(fast.coef_, ref.coef_, atol=COEF_ATOL)
    assert np.allclose(fast.intercept_, ref.intercept_, atol=COEF_ATOL)
    assert np.array_equal(fast.predict(X), ref.predict(X.astype(np.float64)))
    assert np.allclose(
        fast.predict_proba(X), ref.predict_proba(X.astype(np.float64)), atol=PROBA_ATOL
    )


def test_sample_weight_matches_sklearn(data):
    X, y = data
    sw = np.random.RandomState(0).rand(len(y)).astype(np.float32)

    fast = LogisticRegression().fit(X, y, sample_weight=sw)
    ref = SklearnLogisticRegression().fit(
        X.astype(np.float64), y, sample_weight=sw.astype(np.float64)
    )

    assert took_fast_path(fast)
    assert np.allclose(fast.coef_, ref.coef_, atol=COEF_ATOL)


def test_no_intercept_matches_sklearn(data):
    X, y = data
    fast = LogisticRegression(fit_intercept=False).fit(X, y)
    ref = SklearnLogisticRegression(fit_intercept=False).fit(X.astype(np.float64), y)

    assert took_fast_path(fast)
    assert np.allclose(fast.coef_, ref.coef_, atol=COEF_ATOL)


def test_regularisation_strength_tracks_C(data):
    """Stronger regularisation must shrink the coefficients, and match sklearn's scale.

    This is the guard on the `l1_reg/l2_reg /= sample_weights_sum` normalisation in the
    kernel: get that factor wrong and the fit is still "reasonable", just regularised by
    a factor of order n_samples more than the C the caller asked for.
    """
    X, y = data
    norms = []
    for C in (10.0, 1.0, 0.01):
        fast = LogisticRegression(C=C).fit(X, y)
        ref = SklearnLogisticRegression(C=C).fit(X.astype(np.float64), y)
        assert np.allclose(fast.coef_, ref.coef_, atol=COEF_ATOL)
        norms.append(np.linalg.norm(fast.coef_))
    assert norms[0] > norms[1] > norms[2]


# --------------------------------------------------------------------------------------
# 2a. Invalid estimator parameters: hard errors, no hang
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {"C": -5},  # negative penalty -> unbounded objective -> used to hang forever
        {"C": 0},
        {"tol": -1},
        {"max_iter": -1},
        {"l1_ratio": 2.0},
        {"solver": "not-a-solver"},
    ],
)
def test_invalid_params_raise_instead_of_hanging(data, params):
    """`_validate_params` runs before the solver is ever reached, so these must fail
    immediately. The elapsed-time assertion is what distinguishes "raised" from "ground
    through 100 LBFGS iterations first"; catching a true infinite hang would need the
    pytest-timeout plugin, which this project does not depend on.
    """
    X, y = data
    start = time.perf_counter()
    with pytest.raises(InvalidParameterError):
        LogisticRegression(**params).fit(X, y)
    assert time.perf_counter() - start < 5.0


# --------------------------------------------------------------------------------------
# 2b. Invalid data: hard errors, same type as scikit-learn
# --------------------------------------------------------------------------------------


def test_1d_X_raises(data):
    _, y = data
    with pytest.raises(ValueError):
        LogisticRegression().fit(np.zeros(10, dtype=np.float32), y[:10])


def test_length_mismatch_raises(data):
    X, y = data
    with pytest.raises(ValueError):
        LogisticRegression().fit(X, y[:-1])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_X_raises(data, bad):
    """The one O(n) scan the fast path keeps. A single non-finite element makes the
    More-Thuente line search non-terminating, so this must fall back (and scikit-learn
    then raises) rather than reach the kernel. The elapsed-time assertion is what would
    catch a regression that lets it through: the symptom is a hang, not a bad answer."""
    X, y = data
    X = X.copy()
    X[0, 0] = bad
    start = time.perf_counter()
    with pytest.raises(ValueError):
        with no_warnings():
            LogisticRegression().fit(X, y)
    assert time.perf_counter() - start < 10.0


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_sample_weight_raises(data, bad):
    X, y = data
    sw = np.ones(len(y), dtype=np.float32)
    sw[0] = bad
    with pytest.raises(ValueError):
        with no_warnings():
            LogisticRegression().fit(X, y, sample_weight=sw)


def test_empty_y_raises_sklearns_error(data):
    """`y.max()` has no answer on an empty y, and numpy's "zero-size array to reduction"
    error would escape the NotImplementedError contract `fit` catches. Must be thrown back
    and rejected by the parent instead."""
    with pytest.raises(ValueError, match="0 sample"):
        with no_warnings():
            LogisticRegression().fit(
                np.zeros((0, 12), dtype=np.float32), np.zeros(0, dtype=np.uint32)
            )


def test_bad_sample_weight_length_raises(data):
    X, y = data
    with pytest.raises(ValueError):
        LogisticRegression().fit(X, y, sample_weight=np.ones(len(y) - 1, np.float32))


def test_predict_before_fit_raises(data):
    X, _ = data
    with pytest.raises(NotFittedError):
        LogisticRegression().predict(X)


def test_predict_wrong_n_features_raises(data):
    X, y = data
    model = LogisticRegression().fit(X, y)
    with pytest.raises(ValueError):
        model.predict(X[:, :-1])


# --------------------------------------------------------------------------------------
# 2c. Valid-but-unsupported input: throwback to scikit-learn, never a crash
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {"solver": "liblinear"},
        {"solver": "newton-cg"},
        {"penalty": "l2"},
        {"class_weight": "balanced"},
        {"class_weight": {0: 1.0, 1: 3.0}},
        {"warm_start": True},
    ],
)
def test_unsupported_params_fall_back_and_are_honoured(data, params):
    """The parameters the kernel cannot honour must reach scikit-learn, not be ignored."""
    X, y = data
    model, messages = fit_quietly(LogisticRegression(**params), X, y)

    assert not took_fast_path(model)
    assert any("Falling back" in m for m in messages)

    with no_warnings():
        ref = SklearnLogisticRegression(**params).fit(X.astype(np.float64), y)
    assert np.allclose(model.coef_, ref.coef_, atol=COEF_ATOL)


@pytest.mark.parametrize(
    "params",
    [
        {"penalty": "l1", "l1_ratio": 1.0},
        {"l1_ratio": 1.0},  # same request, stated the 1.8 way
        {"l1_ratio": 0.5},  # elasticnet
    ],
)
def test_unsupported_param_surfaces_the_parent_error(data, params):
    """The throwback hands the job over completely — including the parent's own errors.
    lbfgs cannot do an L1 penalty, and scikit-learn says so; the adapter must not swallow
    that behind its "falling back" warning and fit something else instead.

    The `l1_ratio`-only cases are the ones that matter: `penalty=` is deprecated in 1.8, so
    they are how a caller asks for L1 today, and they reach the kernel's own L1 term rather
    than the `penalty` guard. That term is a subgradient inside LBFGS — close to saga's
    answer, but never sparse, since a smooth optimiser cannot pin a coefficient to zero.
    """
    X, y = data
    with pytest.raises(ValueError, match="supports only 'l2' or None penalties"):
        with no_warnings():
            LogisticRegression(**params).fit(X, y)


def test_class_weight_actually_changes_the_fit(data):
    """A regression test with teeth: a silently ignored class_weight would pass the
    fallback test above by accident, since both models would then be unweighted."""
    X, y = data
    weighted, _ = fit_quietly(LogisticRegression(class_weight={0: 1.0, 1: 5.0}), X, y)
    plain = LogisticRegression().fit(X, y)
    assert not np.allclose(weighted.coef_, plain.coef_, atol=COEF_ATOL)


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(lambda X, y: (X.astype(np.float64), y), id="float64-X"),
        pytest.param(lambda X, y: (X, y.astype(np.int64)), id="int64-y"),
        pytest.param(lambda X, y: (X.astype(np.int32), y), id="int-X"),
        pytest.param(lambda X, y: (X, np.where(y == 1, 3, 0).astype(np.uint32)), id="labels-0-3"),
        pytest.param(
            lambda X, y: (X, np.where(np.arange(len(y)) % 3 == 0, 2, y).astype(np.uint32)),
            id="multiclass",
        ),
    ],
)
def test_unsupported_data_falls_back(data, transform):
    X, y = data
    Xt, yt = transform(*data)
    model, messages = fit_quietly(LogisticRegression(), Xt, yt)

    assert not took_fast_path(model)
    assert any("Falling back" in m for m in messages)
    # The fallback must still produce a usable model.
    with no_warnings():
        assert model.predict(Xt).shape == (len(yt),)


def test_sparse_falls_back(data):
    from scipy import sparse

    X, y = data
    model, messages = fit_quietly(LogisticRegression(), sparse.csr_matrix(X), y)
    assert not took_fast_path(model)
    assert any("Falling back" in m for m in messages)


@pytest.mark.parametrize(
    "make_input",
    [
        pytest.param(lambda X: pd.DataFrame(X), id="dataframe"),
        pytest.param(lambda X: X.tolist(), id="list"),
        pytest.param(lambda X: np.asfortranarray(X), id="fortran-order"),
        pytest.param(lambda X: np.ascontiguousarray(X.T).T, id="non-contiguous"),
    ],
)
def test_array_like_inputs_are_accepted(data, make_input):
    """These used to escape as AttributeError/KeyError from inside the guards."""
    X, y = data
    model, _ = fit_quietly(LogisticRegression(), make_input(X), y)
    with no_warnings():
        assert model.predict(make_input(X)).shape == (len(y),)


@pytest.mark.parametrize(
    "container",
    [
        pytest.param(lambda X, y: (pd.DataFrame(X), y), id="dataframe-X"),
        pytest.param(lambda X, y: (X, pd.Series(y)), id="series-y"),
        pytest.param(lambda X, y: (X.tolist(), y), id="list-X"),
        pytest.param(lambda X, y: (np.asfortranarray(X), y), id="fortran-X"),
        pytest.param(lambda X, y: (np.ascontiguousarray(X.T).T, y), id="strided-X"),
        pytest.param(lambda X, y: (np.ma.masked_array(X), y), id="masked-X"),
    ],
)
def test_containers_the_kernel_cannot_read_fall_back(data, container):
    """The adapter converts NOTHING. Anything the kernel cannot consume as a raw buffer is
    handed to scikit-learn whole, rather than copied into a shape the kernel likes —
    copying is the expensive half of validation and would spend the speed-up it buys."""
    Xc, yc = container(*data)
    model, messages = fit_quietly(LogisticRegression(), Xc, yc)
    assert not took_fast_path(model)
    assert any("Falling back" in m for m in messages)


def test_fast_path_does_not_copy_or_modify_its_input(data):
    """The kernel reads the caller's buffers in place; fitting must leave them untouched."""
    X, y = data
    X_before, y_before = X.copy(), y.copy()
    model = LogisticRegression().fit(X, y)
    assert took_fast_path(model)
    assert np.array_equal(X, X_before)
    assert np.array_equal(y, y_before)


def test_dataframe_keeps_feature_names(data):
    """A DataFrame throws back to sklearn (it arrives as float64); the fallback must see
    the caller's original frame, or `feature_names_in_` is silently lost."""
    X, y = data
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    model, _ = fit_quietly(LogisticRegression(), frame, y)
    assert list(model.feature_names_in_) == list(frame.columns)


def test_refit_on_an_ndarray_drops_the_feature_names(data):
    """The other direction, which goes stale silently: a bare ndarray carries no column
    names, so refitting the same instance must clear the ones the DataFrame fit left
    behind, as sklearn does."""
    X, y = data
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    model, _ = fit_quietly(LogisticRegression(), frame, y)
    model, _ = fit_quietly(model, X, y)
    assert took_fast_path(model)
    assert not hasattr(model, "feature_names_in_")


# --------------------------------------------------------------------------------------
# 2d. Deliberate divergences from scikit-learn
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_fits_instead_of_raising(data, label):
    """The one input the fast path accepts that scikit-learn rejects ("needs samples of at
    least 2 classes"). No guard, because with one label the objective is still convex and,
    since C > 0 is enforced, still penalised: LBFGS reaches a finite solution instead of
    running the intercept off to +/-inf. That is what is asserted here — converged rather
    than truncated (same answer at 100 and 1000 iterations), predicting the label shown.
    """
    X, y = data
    y_one = np.full_like(y, label)

    model = LogisticRegression(max_iter=100).fit(X, y_one)
    assert took_fast_path(model)
    assert np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
    assert np.array_equal(np.unique(model.predict(X)), [label])

    # Converged, not truncated: a tenfold iteration budget must not move the solution.
    longer = LogisticRegression(max_iter=1000).fit(X, y_one)
    assert np.allclose(model.coef_, longer.coef_, atol=COEF_ATOL)
    assert np.allclose(model.intercept_, longer.intercept_, atol=COEF_ATOL)


# --------------------------------------------------------------------------------------
# 2e. No native extension at all
# --------------------------------------------------------------------------------------

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
    ``importlib.reload`` would be the shorter route and is the wrong one: it re-executes
    in the existing namespace, rebinding the class object while the name imported at the
    top of this file still points at the old one -- after which pickling an instance
    fails with "not the same object as stratum.adapters.logistic_regression".
    """
    import stratum

    source = sys.modules[LogisticRegression.__module__].__file__
    saved_module = sys.modules.get(NATIVE_MODULE)
    saved_attr = getattr(stratum, NATIVE_ATTR, None)

    sys.modules[NATIVE_MODULE] = None
    if saved_attr is not None:
        delattr(stratum, NATIVE_ATTR)
    try:
        spec = importlib.util.spec_from_file_location(
            "stratum.adapters._logistic_regression_no_native", source
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

    Nothing in the module body may read an attribute off the extension: the kernel is
    looked up at the call site, behind the `_HAVE_RUST` guard, so a missing extension
    leaves `_native` as None instead of raising while the body runs.
    """
    module = adapter_without_native

    assert module._HAVE_RUST is False
    assert module._native is None


def test_everything_falls_back_when_the_extension_is_missing(adapter_without_native, data):
    """Without a kernel the adapter is a plain scikit-learn estimator that says so.

    All three public entry points are checked, because each reaches the `_HAVE_RUST`
    guard through a different path: `fit` via `_check_before_fit`, `predict` and
    `predict_proba` via `_check_before_predict`.
    """
    X, y = data
    model, messages = fit_quietly(adapter_without_native.LogisticRegression(), X, y)

    assert not took_fast_path(model)
    assert any("native backend is unavailable" in m for m in messages), messages

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        proba = model.predict_proba(X)
        labels = model.predict(X)
    predict_messages = [str(w.message) for w in caught]
    assert sum("native backend is unavailable" in m for m in predict_messages) == 2, (
        predict_messages
    )

    # Falling back has to mean scikit-learn's answer, not merely scikit-learn's code
    # path: a guard that warned and then returned something else would pass every
    # assertion above.
    with no_warnings():
        ref = SklearnLogisticRegression().fit(X.astype(np.float64), y)
    assert np.allclose(model.coef_, ref.coef_, atol=COEF_ATOL)
    assert np.array_equal(labels, ref.predict(X.astype(np.float64)))
    assert np.allclose(proba, ref.predict_proba(X.astype(np.float64)), atol=PROBA_ATOL)


def test_the_live_adapter_is_unaffected(data):
    """The canary on the fixture's teardown: hiding the extension is process-wide while
    it lasts, so a leak would leave every later test measuring the fallback."""
    module = sys.modules[LogisticRegression.__module__]

    assert module._HAVE_RUST is True
    assert module._native is not None
    assert took_fast_path(LogisticRegression().fit(*data))


# --------------------------------------------------------------------------------------
# 3. Fitted-attribute conventions and estimator plumbing
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_fitted_attribute_layout(data, fit_intercept):
    X, y = data
    model = LogisticRegression(fit_intercept=fit_intercept).fit(X, y)

    assert model.coef_.shape == (1, X.shape[1])
    assert model.coef_.dtype == np.float32
    # sklearn guarantees an ndarray here in the binary case, fitted intercept or not.
    assert isinstance(model.intercept_, np.ndarray)
    assert model.intercept_.shape == (1,)
    assert model.intercept_.dtype == model.coef_.dtype
    assert model.n_features_in_ == X.shape[1]
    assert list(model.classes_) == [0, 1]
    if not fit_intercept:
        assert model.intercept_[0] == 0.0


def test_predict_proba_is_a_distribution(data):
    X, y = data
    model = LogisticRegression().fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    # predict() skips the sigmoid and thresholds the score directly, so it has to be
    # checked against predict_proba rather than assumed consistent with it.
    assert np.array_equal(proba.argmax(axis=1).astype(np.uint32), model.predict(X))


def test_refit_clears_the_fast_path_marker(data):
    """fast fit -> fallback fit on the SAME instance must not leave a stale marker, or
    predict() would hand sklearn's float64 coef_ to the kernel."""
    X, y = data
    model = LogisticRegression().fit(X, y)
    assert took_fast_path(model)

    model, _ = fit_quietly(model, X.astype(np.float64), y)
    assert not took_fast_path(model)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predictions = model.predict(X)
    assert any("Falling back" in str(w.message) for w in caught)
    assert predictions.shape == (len(y),)


def test_clone_and_pickle_roundtrip(data):
    X, y = data
    model = LogisticRegression(C=0.5, tol=1e-3).fit(X, y)

    fresh = clone(model)
    assert fresh.get_params() == model.get_params()
    assert not took_fast_path(fresh)

    restored = pickle.loads(pickle.dumps(model))
    assert np.array_equal(restored.predict(X), model.predict(X))


def test_works_inside_a_pipeline(data):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = data
    # StandardScaler preserves float32, so the fast path survives the pipeline.
    pipe = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
    assert took_fast_path(pipe[-1])
    assert pipe.score(X, y) > 0.7
