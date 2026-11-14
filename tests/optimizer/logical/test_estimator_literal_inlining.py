"""Estimator literal inlining preserves parameter bindings and ownership."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import stratum as st
from stratum.frontend._conversion import convert_to_ops
from stratum.optimizer._op_utils import topological_iterator, validate_dag
from stratum.optimizer._optimize import OptConfig, logical_optimize
from stratum.optimizer.logical._ops import OperandRef, TransformerOp, ValueOp
from stratum.optimizer.logical.rewrites._literal_inlining import literal_inlining


@pytest.fixture(autouse=True)
def lazy_data_ops():
    with st.config_context(eager_data_ops=False):
        yield


def features():
    return st.as_data_op(pd.DataFrame({"a": [1.0, 2.0, 4.0]}))


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("mean", [False, True])
def test_parameters_are_inlined_into_owned_prototypes(nested, mean):
    parameter = st.as_data_op(mean)
    scaler = StandardScaler(with_mean=parameter, with_std=parameter)
    estimator = Pipeline([("scale", scaler)]) if nested else scaler
    root = convert_to_ops(features().skb.apply(estimator))
    source = root.inputs[1]
    prototypes = (root.estimator, root.original_estimator)

    literal_inlining(root)

    prefix = "scale__" if nested else ""
    assert root.param_refs == {} and len(root.inputs) == 1
    assert source.outputs == []
    for prototype in (root.estimator, root.original_estimator):
        assert prototype.get_params()[prefix + "with_mean"] is mean
        assert prototype.get_params()[prefix + "with_std"] is mean
    assert root.estimator is not estimator
    assert root.estimator is not root.original_estimator
    assert root.estimator is prototypes[0] and root.original_estimator is prototypes[1]
    assert scaler.with_mean is parameter and scaler.with_std is parameter
    validate_dag(root)

    literal_inlining(root)
    assert root.estimator is prototypes[0] and root.original_estimator is prototypes[1]


@pytest.mark.parametrize("value", [None, False, 7, 1.25, "constant", b"constant"])
def test_estimator_parameters_use_the_existing_scalar_policy(value):
    root = convert_to_ops(features().skb.apply(
        DummyRegressor(constant=st.as_data_op(value))))

    literal_inlining(root)

    assert root.param_refs == {}
    assert type(root.estimator.constant) is type(value) and root.estimator.constant == value
    validate_dag(root)


@pytest.mark.parametrize("value", [np.bool_(True), np.float64(2.0), np.array([2.0]), [2.0]])
def test_nonliteral_parameter_values_stay_bound(value):
    root = convert_to_ops(features().skb.apply(
        DummyRegressor(constant=st.as_data_op(value))))
    inputs = root.inputs[:]
    estimator = root.estimator

    literal_inlining(root)

    assert root.param_refs == {"constant": OperandRef(1)}
    assert root.inputs == inputs and root.estimator is estimator
    validate_dag(root)


def test_remaining_parameter_y_and_kwargs_refs_are_renumbered():
    X = features()
    intercept = X.skb.apply_func(lambda frame: False)
    weights = st.as_data_op(np.ones(3))
    root = convert_to_ops(X.skb.apply(
        Ridge(alpha=st.as_data_op(2.0), fit_intercept=intercept), y=X["a"],
        fit_kwargs={"sample_weight": weights}))
    inputs = root.inputs[:]

    literal_inlining(root)

    assert root.inputs == [inputs[0], *inputs[2:]]
    assert root.param_refs == {"fit_intercept": OperandRef(1)}
    assert root.y == OperandRef(2)
    assert root.kwargs["fit"]["sample_weight"] == OperandRef(3)
    frame = pd.DataFrame({"a": [1.0, 2.0, 4.0]})
    args = root.extract_args_from_inputs([frame, False, frame["a"], np.ones(3)])
    assert args[0].alpha == 2.0 and args[0].fit_intercept is False
    validate_dag(root)


def test_parameter_shared_with_apply_field_keeps_its_edge():
    flag = st.as_data_op(True)
    root = convert_to_ops(features().skb.apply(StandardScaler(with_mean=flag), no_wrap=flag))
    inputs = root.inputs[:]

    literal_inlining(root)

    assert root.param_refs == {} and root.estimator.with_mean is True
    assert root.no_wrap == OperandRef(1) and root.inputs == inputs
    assert inputs[1].outputs == [root]
    validate_dag(root)


def test_shared_estimator_and_literal_are_not_mutated():
    parameter = st.as_data_op(False)
    estimator = StandardScaler(with_mean=parameter)
    X = features()
    first, second = X.skb.apply(estimator), X.skb.apply(estimator)
    root = convert_to_ops(first.skb.apply_func(lambda a, b: (a, b), second))
    consumers = [op for op in topological_iterator(root) if isinstance(op, TransformerOp)]
    source = consumers[0].inputs[1]
    assert consumers[1].inputs[1] is source

    literal_inlining(root)

    assert source.outputs == [] and estimator.with_mean is parameter
    assert consumers[0].estimator is not consumers[1].estimator
    assert all(op.param_refs == {} and op.estimator.with_mean is False for op in consumers)
    validate_dag(root)


@pytest.mark.parametrize("flag", ["is_X", "is_y"])
def test_marked_parameter_sources_are_preserved(flag):
    root = convert_to_ops(features().skb.apply(StandardScaler(with_mean=st.as_data_op(True))))
    setattr(root.inputs[1], flag, True)

    literal_inlining(root)

    assert root.param_refs == {"with_mean": OperandRef(1)}
    assert len(root.inputs) == 2
    validate_dag(root)


def test_implicit_x_is_kept_when_also_used_as_a_parameter():
    source = ValueOp(True)
    root = TransformerOp(StandardScaler(), param_refs={"with_mean": OperandRef(0)})
    root.inputs = [source]
    source.add_output(root)

    literal_inlining(root)

    assert root.param_refs == {} and root.inputs == [source]
    assert root.estimator.with_mean is True
    validate_dag(root)


def test_response_methods_reflect_inlined_configuration():
    root = convert_to_ops(features().skb.apply(SVC(probability=st.as_data_op(True))))

    literal_inlining(root)

    assert "predict_proba" in root.supported_modes
    assert root.estimator.probability is True


def test_inlining_enables_cse_with_plain_estimator_parameters():
    X = features()
    wrapped = X.skb.apply(StandardScaler(with_mean=st.as_data_op(False)))
    plain = X.skb.apply(StandardScaler(with_mean=False))
    dag = wrapped.skb.apply_func(lambda a, b: a + b, plain)

    root = logical_optimize(dag, OptConfig())

    estimators = [op for op in topological_iterator(root) if isinstance(op, TransformerOp)]
    assert len(estimators) == 1 and estimators[0].param_refs == {}
    validate_dag(root)
