"""DCE preserves exactly the result's data and fitted-state dependencies."""
from collections import Counter

import pytest

from stratum.optimizer._op_utils import topological_iterator, validate_dag
from stratum.optimizer.logical._base import OperandRef
from stratum.optimizer.logical._ops import Op
from stratum.optimizer.physical.rewrites._dead_code_elimination import eliminate_dead_code
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical.rewrites._linearization import linearize_dag


@pytest.fixture(params=[Op, PhysicalOp], ids=["logical", "physical"])
def node(request):
    def make(name, *inputs):
        op = request.param(name=name, inputs=list(inputs))
        for producer in inputs:
            producer.outputs.append(op)
        return op
    return make


def assert_graph(root, expected):
    ops = list(topological_iterator(root))
    assert set(ops) == set(expected) and len(ops) == len(expected)
    for op in ops:
        assert Counter(op.outputs) == Counter(
            consumer for consumer in ops for producer in consumer.inputs
            if producer is op)
    validate_dag(root)
    assert set(linearize_dag(root)[0]) == set(expected)


def test_prunes_dead_branches_and_private_producers(node):
    shared = node("shared")
    root = node("root", shared)
    private_source = node("private source")
    private = node("private", private_source)
    dead = node("dead", shared, private)
    tail = node("tail", dead)

    assert eliminate_dead_code(root) is root

    assert_graph(root, [shared, root])
    for op in (private_source, private, dead, tail):
        assert not op.inputs and not op.outputs


def test_state_dependencies_keep_fitting_alive_but_statefulness_does_not(node):
    source = node("source")
    fit = node("fit", source)
    fit.has_state = True
    fit.mode = "fit"
    predict = node("predict", source, fit)
    predict.mode = "predict"
    predict.state_ref = OperandRef(1)
    dead_fit = node("unused fit", source)
    dead_fit.has_state = True
    dead_fit.mode = "fit"

    eliminate_dead_code(predict)

    assert_graph(predict, [source, fit, predict])
    assert fit.has_state_reader() and not fit.has_data_reader()
    assert not dead_fit.inputs and not dead_fit.outputs


def test_preserves_repeated_data_and_state_edges(node):
    fit = node("fit")
    root = node("root", fit, fit)
    root.state_ref = OperandRef(1)
    dead = node("dead", fit, fit)

    eliminate_dead_code(root)

    assert_graph(root, [fit, root])
    assert root.inputs == [fit, fit] and fit.outputs == [root, root]
    assert root.state_ref == OperandRef(1)
    assert fit.has_data_reader() and fit.has_state_reader()
    assert not dead.inputs and not dead.outputs


def test_keeps_a_source_root_and_removes_its_consumers(node):
    root = node("root")
    dead = node("dead", root)

    assert eliminate_dead_code(root) is root

    assert_graph(root, [root])
    assert not dead.inputs and not dead.outputs


def test_can_run_again_after_another_rewrite(node):
    source = node("source")
    left = node("left", source)
    right = node("right", source)
    root = node("root", left, right)
    eliminate_dead_code(root)
    assert_graph(root, [source, left, right, root])

    # A rewrite makes the right branch unnecessary.
    root.inputs.remove(right)
    right.outputs.remove(root)
    eliminate_dead_code(root)
    assert_graph(root, [source, left, root])
    assert not right.inputs and not right.outputs

    edges = {op: (list(op.inputs), list(op.outputs)) for op in (source, left, root)}
    eliminate_dead_code(root)
    assert all((op.inputs, op.outputs) == old for op, old in edges.items())


def test_prunes_a_long_branch_without_recursion(node):
    root = node("root")
    branch = []
    tail = root
    for i in range(1500):
        tail = node(str(i), tail)
        branch.append(tail)

    eliminate_dead_code(root)

    assert_graph(root, [root])
    assert all(not op.inputs and not op.outputs for op in branch)
