"""Estimator parameter dependencies survive graph extraction and conversion."""

import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from skrub._data_ops._evaluation import _Graph

import stratum as st
from stratum.frontend._conversion import convert_to_ops
from stratum.frontend._skrub_graph import build_graph, get_data
from stratum.optimizer._op_utils import validate_dag
from stratum.optimizer.logical._ops import OperandRef, ValueOp


@pytest.fixture(autouse=True)
def lazy_data_ops():
    with st.config_context(eager_data_ops=False):
        yield


def features():
    return st.as_data_op(pd.DataFrame({"a": [1.0, 2.0, 4.0]}))


def graph_edges(graph):
    nodes = graph["nodes"]
    return {(id(nodes[parent]), id(nodes[child]))
            for parent, children in graph["children"].items() for child in children}


@pytest.mark.parametrize("nested", [False, True])
def test_estimator_parameter_edges_match_skrub(nested):
    mean = st.var("mean", True)
    scaler = StandardScaler(with_mean=mean, with_std=mean)
    estimator = Pipeline([("scale", scaler)]) if nested else scaler
    dag = features().skb.apply(estimator)

    graph = build_graph(dag)

    assert graph_edges(graph) == graph_edges(_Graph().run(dag))
    assert id(mean) in graph["children"][id(dag)]
    assert graph["children"][id(dag)].count(id(mean)) == 1
    assert get_data(dag) == dag.skb.get_data() == {"mean": True}


@pytest.mark.parametrize("nested", [False, True])
def test_direct_and_named_nested_parameters_are_bound(nested):
    mean = st.as_data_op(False)
    scaler = StandardScaler(with_mean=mean, with_std=mean)
    estimator = Pipeline([("scale", scaler)]) if nested else scaler
    root = convert_to_ops(features().skb.apply(estimator))

    prefix = "scale__" if nested else ""
    assert root.param_refs == {prefix + name: OperandRef(1)
                               for name in ("with_mean", "with_std")}
    assert len(root.inputs) == 2
    assert isinstance(root.inputs[1], ValueOp) and root.inputs[1].value is False
    assert root.inputs[1].outputs == [root]
    assert scaler.with_mean is mean
    validate_dag(root)


def test_parameter_variable_uses_conversion_environment():
    dag = features().skb.apply(StandardScaler(with_mean=st.var("mean")))
    root = convert_to_ops(dag, env={"mean": False})

    assert root.inputs[root.param_refs["with_mean"].k].value is False
    validate_dag(root)


def test_estimator_choices_keep_shared_parameter_dependencies():
    alpha = st.as_data_op(2.0)
    estimators = st.choose_from([
        Ridge(alpha=alpha), Ridge(alpha=alpha, fit_intercept=False)])
    X = features()
    root = convert_to_ops(X.skb.apply(estimators, y=X["a"]))

    first, second = root.inputs
    assert first.inputs[first.param_refs["alpha"].k] is second.inputs[second.param_refs["alpha"].k]
    validate_dag(root)


@pytest.mark.parametrize("value", [
    lambda p: p,
    lambda p: [p],
    lambda p: (p,),
    lambda p: slice(None, p),
    lambda p: StandardScaler(with_mean=p),
])
def test_parameter_containers_are_rejected_even_with_a_shared_binding(value):
    parameter = st.as_data_op(True)
    # The same DataOp also has a supported binding. That must not hide its
    # unsupported use inside kw_args.
    estimator = FunctionTransformer(validate=parameter, kw_args={"option": value(parameter)})
    with pytest.raises(NotImplementedError, match="parameter containers.*kw_args"):
        convert_to_ops(features().skb.apply(estimator))


def test_container_in_a_named_nested_estimator_is_rejected():
    estimator = Pipeline([("transform", FunctionTransformer(
        kw_args={"offset": st.as_data_op(1)}))])
    with pytest.raises(NotImplementedError, match="parameter containers.*transform__kw_args"):
        convert_to_ops(features().skb.apply(estimator))


@pytest.mark.parametrize("as_value", [False, True])
def test_estimator_parameters_outside_apply_are_rejected(as_value):
    estimator = StandardScaler(with_mean=st.as_data_op(True))
    dag = (st.as_data_op(estimator) if as_value else
           features().skb.apply_func(lambda X, estimator: X, estimator))
    with pytest.raises(NotImplementedError, match="only supported.*skb.apply"):
        convert_to_ops(dag)


def test_param_choice_preserves_other_dataop_parameters():
    mean = st.as_data_op(False)
    estimator = StandardScaler(with_mean=mean, with_std=st.choose_bool(name="std"))
    root = convert_to_ops(features().skb.apply(estimator))

    first, second = root.inputs
    assert first.inputs[first.param_refs["with_mean"].k] is second.inputs[second.param_refs["with_mean"].k]
    assert {first.estimator.with_std, second.estimator.with_std} == {False, True}
    assert estimator.with_mean is mean
    validate_dag(root)
