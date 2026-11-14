"""Fitted state is measured during fitting and read cheaply between passes."""
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from stratum import _rust_backend as rb
from stratum.optimizer.logical._base import All
from stratum.optimizer.physical import _physical_ops
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.physical._predictor_execs import (
    SklearnDecisionTree,
    SklearnExtraTrees,
    SklearnKNeighbors,
    SklearnRandomForest,
    StratumExactRandomForestClassifier,
    StratumHistogramRandomForestClassifier,
)
from stratum.optimizer.physical._transform_execs import PassthroughTransformer
from stratum.runtime import _object_size
from stratum.runtime._object_size import UNKNOWN_SIZE
from tests.optimizer.physical._execution import response_op


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    return X, rng.integers(0, 2, len(X))


@pytest.fixture
def no_generic_sizing(monkeypatch):
    measure = Mock(side_effect=AssertionError("known models must size their own state"))
    monkeypatch.setattr(_physical_ops, "get_size_estimator", measure)
    monkeypatch.setattr(_object_size, "get_size_estimator", measure)
    yield
    measure.assert_not_called()


def test_stateless_operator_has_no_retained_state():
    assert PhysicalOp().retained_state_size() == 0


@pytest.mark.parametrize("transformer", [False, True])
@pytest.mark.parametrize("no_wrap", [False, True])
def test_size_is_refreshed_only_when_fitting(data, transformer, no_wrap):
    X, y = data
    if transformer:
        op = PassthroughTransformer(estimator=StandardScaler(), cols=All(), no_wrap=no_wrap)
    else:
        op = SklearnKNeighbors(estimator=KNeighborsClassifier(), y=y,
                               cols=All(), no_wrap=no_wrap)
    op.parallelism = 1
    assert op.retained_state_size() == 0

    with patch.object(op, "_measure_retained_state", wraps=op._measure_retained_state) as measure:
        fitted = op.process([X])
        response = response_op(op)
        expected = getattr(fitted.state, response.mode)(X)
        first_size = op.retained_state_size()
        assert first_size > UNKNOWN_SIZE
        for _ in range(3):
            assert op.retained_state_size() == first_size
            np.testing.assert_allclose(response.process([X, fitted.state]), expected)
        assert measure.call_count == 1

        # A new fold replaces the fitted state and its estimate.
        op.y = y[:100]
        op.process([X.iloc[:100].copy()])
        assert measure.call_count == 2
        if not transformer:
            assert UNKNOWN_SIZE < op.retained_state_size() < first_size


@pytest.mark.parametrize("impl, estimator", [
    (SklearnDecisionTree, DecisionTreeClassifier(random_state=0)),
    (SklearnRandomForest, RandomForestClassifier(n_estimators=3, random_state=0)),
    (SklearnExtraTrees, ExtraTreesClassifier(n_estimators=3, random_state=0)),
])
def test_sklearn_tree_storage_is_included(data, impl, estimator, no_generic_sizing):
    X, y = data
    op = impl(estimator=estimator, y=y, cols=All(), no_wrap=True)
    fitted = op.process([X])
    trees = getattr(fitted.state, "estimators_", [fitted.state])
    native_bytes = sum(
        sum(arr.nbytes for arr in tree.tree_.__getstate__().values()
            if isinstance(arr, np.ndarray))
        for tree in trees
    )
    # Fitted trees have compact storage; metadata accounting is still a TODO.
    assert op.retained_state_size() == native_bytes


@pytest.mark.skipif(not rb.HAVE_RUST, reason="Rust backend not built")
@pytest.mark.parametrize("impl", [StratumExactRandomForestClassifier,
                                  StratumHistogramRandomForestClassifier])
@pytest.mark.parametrize("no_wrap", [False, True])
def test_rust_forest_native_size_is_read_once_per_fit(data, impl, no_wrap, no_generic_sizing):
    X, y = data
    op = impl(estimator=RandomForestClassifier(n_estimators=5, random_state=0), y=y,
              cols=All(), no_wrap=no_wrap)
    op.on_impl_selected(PlanContext(pandas_query=False, rechunk=True, parallelism=1,
                                    rust_backend=True, allow_patch=True))
    native_info = rb.forest_model_info
    with patch.object(rb, "forest_model_info", wraps=native_info) as measure:
        fitted = op.process([X])
        response = response_op(op)
        expected = fitted.state.predict(X)
        assert measure.call_count == 1
        handle = measure.call_args.args[0]
        native_bytes = native_info(handle)["model_bytes"]
        assert native_bytes > 10_000
        size = op.retained_state_size()
        assert size == native_bytes

        for _ in range(3):
            assert op.retained_state_size() == size
            np.testing.assert_array_equal(response.process([X, fitted.state]), expected)
        assert measure.call_count == 1

        op.y = y[:100]
        op.process([X.iloc[:100]])
        assert measure.call_count == 2
        assert UNKNOWN_SIZE < op.retained_state_size() < size


def test_failed_measurement_does_not_break_fit_or_keep_previous_size(data):
    X, y = data
    op = SklearnKNeighbors(estimator=KNeighborsClassifier(), y=y, cols=All(), no_wrap=True)
    op.process([X])
    assert op.retained_state_size() > UNKNOWN_SIZE
    with patch.object(op, "_measure_retained_state", side_effect=RuntimeError("no estimate")):
        fitted = op.process([X])
    assert op.retained_state_size() == UNKNOWN_SIZE
    response = response_op(op)
    np.testing.assert_array_equal(response.process([X, fitted.state]), fitted.state.predict(X))
