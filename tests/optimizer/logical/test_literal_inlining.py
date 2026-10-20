"""Literal inlining on the logical Op DAG."""

from collections import defaultdict
import operator

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

import stratum as st
from stratum._api import evaluate
from stratum.optimizer._optimize import OptConfig, extract_numeric_operators, logical_optimize, optimize
from stratum.optimizer._op_utils import topological_iterator, validate_dag
from stratum.optimizer.logical._aggregation_ops import AggregateOp
from stratum.optimizer.logical._column_expr import BinOpExpr, Col, Const
from stratum.optimizer.logical._dataframe_ops import ConcatOp, MaskSelectionOp
from stratum.optimizer.logical._literal_inlining import literal_inlining
from stratum.optimizer.logical._numeric_ops import NumericOp, NumericOpType
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp
from stratum.runtime._buffer_pool import BufferPool
from stratum.optimizer.logical._ops import BinOp, CallOp, GetItemOp, MethodCallOp, OperandRef, PredictorOp, UnaryOp, ValueOp
from stratum.optimizer.physical._impl_selection import bind_op, DefaultImplementationSelector
from stratum.optimizer.physical._lowering import lower_to_physical
from stratum.optimizer.physical._plan_context import PlanContext

R0, R1, R2 = OperandRef(0), OperandRef(1), OperandRef(2)


def wire(op, *inputs):
    op.inputs = list(inputs)
    for source in inputs:
        source.add_output(op)
    return op


def call(func, *args, **kwargs):
    return CallOp(func=func, args=args, kwargs=kwargs)


def inline(make, values):
    """Wire ``make()`` over one ValueOp per value and run the pass on it."""
    sources = [ValueOp(value) for value in values]
    root = wire(make(), *sources)
    assert literal_inlining(root) is root
    return root, sources


def assert_kept(root, sources, kept):
    """Exactly ``sources[kept]`` stay inputs of ``root``, with matching output edges."""
    assert root.inputs == [sources[i] for i in kept]
    for i, source in enumerate(sources):
        assert (root in source.outputs) == (i in kept)
    validate_dag(root)


def execute(root):
    root = lower_to_physical(root, PlanContext.from_flags())
    values = {}
    for op in topological_iterator(root):
        bind_op(op, PlanContext.from_flags(), selector=DefaultImplementationSelector())
        values[op] = op.process("fit_transform", [values[p] for p in op.inputs])
    return values[root]


@pytest.mark.parametrize("value, inlined", [
    *[(v, True) for v in (None, False, 7, 1.25, "BRASS", b"BRASS")],
    # Exact built-in types only: numpy scalars and containers stay graph inputs.
    *[(v, False) for v in ([1], {"a": 1}, np.array([1]), np.float64(1.0), pd.DataFrame({"a": [1]}), object())],
])
def test_only_builtin_scalars_are_inlined(value, inlined):
    root, sources = inline(lambda: call(lambda x: x, R0), [value])

    assert_kept(root, sources, kept=[] if inlined else [0])
    assert root.args == ((value,) if inlined else (R0,))
    assert execute(root) is value


@pytest.mark.parametrize("pattern", ["LLL", "DDD", "LDL", "DLD"])
def test_inputs_compact_and_refs_remap(pattern):
    # L = literal (inlined), D = list (stays an input).
    values = [i if kind == "L" else [i] for i, kind in enumerate(pattern)]
    root, sources = inline(lambda: call(lambda *a: a, R0, R1, R2), values)

    assert_kept(root, sources, kept=[i for i, kind in enumerate(pattern) if kind == "D"])
    assert execute(root) == tuple(values)


def test_refs_in_nested_containers_and_kwargs():
    nested = [R1, slice(None, R0, None), defaultdict(lambda: "missing", {"a": R0})]
    root, sources = inline(lambda: call(lambda *a, **k: (a, k), R0, nested, k=R0, d=R1), [3, [4]])

    assert_kept(root, sources, kept=[1])
    assert root.args[:2] == (3, [R0, slice(None, 3, None), root.args[1][2]])
    assert type(root.args[1][2]) is defaultdict and root.args[1][2]["absent"] == "missing"
    assert root.kwargs == {"k": 3, "d": R0}
    args, kwargs = execute(root)
    assert args[1][0] == [4] and args[1][2]["a"] == 3 and kwargs == {"k": 3, "d": [4]}


def test_shared_literal_and_idempotence():
    source = ValueOp("a")
    method = wire(MethodCallOp("replace", args=(R0, "x"), kwargs={}), source)
    other = wire(call(lambda x: x, R0), source)
    root = wire(call(lambda a, b: (a, b), R0, R1), method, other)

    for _ in range(2):
        literal_inlining(root)
        # The method's receiver is implicit input 0, so that edge must stay.
        assert source.outputs == [method] and method.inputs == [source]
        assert method.args == (R0, "x")
        assert other.inputs == [] and other.args == ("a",)
    assert execute(root) == ("x", "a")


_UNCHECKED = object()


@pytest.mark.parametrize("make, values, kept, expected", [
    pytest.param(lambda: GetItemOp(key=R1), ["BRASS", 3], [0], "S", id="getitem-keeps-container"),
    pytest.param(lambda: MethodCallOp("endswith", args=(R1,), kwargs={}), ["BRASS", "SS"], [0], True,
                 id="method-keeps-receiver"),
    pytest.param(lambda: call(np.add, R0, R1), [2.0, 3.0], [0], 5.0, id="numpy-keeps-primary"),
    # Numeric extraction and selection folding already read a ValueOp operand as a constant.
    pytest.param(lambda: BinOp(operator.sub, R0, R1), [np.array([10, 20]), 3], [0, 1], np.array([7, 17]),
                 id="binop-unsupported"),
    pytest.param(lambda: UnaryOp(operator.neg, R0), [3], [0], -3, id="unary-unsupported"),
    pytest.param(lambda: PredictorOp(estimator=Ridge(), param_refs={"alpha": R1}),
                 [pd.DataFrame({"x": [1, 2]}), 0.5], [0, 1], _UNCHECKED, id="unsupported-consumer"),
])
def test_consumer_rules(make, values, kept, expected):
    root, sources = inline(make, values)

    assert_kept(root, sources, kept)
    if expected is not _UNCHECKED:
        np.testing.assert_equal(execute(root), expected)


@pytest.mark.parametrize("axis, inlined", [(1, True), ("columns", True), (True, False), (1.0, False), (None, False)])
def test_concat_inlines_only_int_or_str_axis(axis, inlined):
    # `first` is a literal too, but only the axis position is eligible.
    frame = pd.DataFrame({"a": [1, 2]})
    root, sources = inline(lambda: ConcatOp(first=R0, others=[R1], axis=R2), [None, frame, axis])

    assert_kept(root, sources, kept=[0, 1] if inlined else [0, 1, 2])
    if inlined:
        pd.testing.assert_frame_equal(execute(root), pd.concat([None, frame], axis=axis))


@pytest.mark.parametrize("make, values", [
    pytest.param(lambda: call(np.add, R0, R1), [2.0, 3.0], id="numpy-literal-primary"),
    pytest.param(lambda: call(np.arctan2, R0, R1), [np.array([2.0, 4.0]), 1.0], id="numpy-literal-second"),
    pytest.param(lambda: BinOp(operator.sub, R0, R1), [np.array([2, 4]), 10], id="binop-literal-right"),
    pytest.param(lambda: BinOp(operator.sub, R0, R1), [10, np.array([2, 4])], id="binop-literal-left"),
])
def test_inlined_constants_feed_numeric_extraction(make, values):
    reference = make()
    expected = reference.func(*values) if isinstance(reference, CallOp) else reference.op(*values)
    root, _ = inline(make, values)

    root = extract_numeric_operators(root)

    assert isinstance(root, NumericOp)
    np.testing.assert_allclose(execute(root), expected)


@pytest.mark.parametrize("flag", ["is_X", "is_y"])
def test_marked_x_y_values_are_not_inlined(flag):
    source = ValueOp(3)
    setattr(source, flag, True)
    root = wire(call(lambda x: x, R0), source)

    literal_inlining(root)

    assert_kept(root, [source], kept=[0])
    assert root.args == (R0,)


def test_getitem_name_reflects_inlined_key():
    root, sources = inline(lambda: GetItemOp(key=R1, name="Value"), ["BRASS", 3])

    assert_kept(root, sources, kept=[0])
    assert root.key == 3 and root.name == "3"


def test_optconfig_disables_literal_inlining():
    frame = pd.DataFrame({"part": ["BRASS", "STEEL"]})
    dag = st.as_data_op(frame)[st.as_data_op("part")]

    ops, *_ = optimize(dag, OptConfig(literal_inlining=False))

    assert any(isinstance(op, ValueOp) and op.value == "part" for op in ops)
    assert any(isinstance(op, GetItemOp) and op.key == OperandRef(1) for op in ops)


def run_plan(ops):
    pool = BufferPool()
    for op in ops:
        pool.put(op, op.process("fit_transform", [pool.pin(key) for key in op.inputs]))
    return pool.pin(ops[-1])


def test_inlining_enables_aggregation_fusion():
    frame = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3], "w": [4, 5, 6]})
    dag = st.as_data_op(frame).groupby("g").agg(st.var("spec"))
    config = OptConfig(dataframe_ops=True)

    root = logical_optimize(dag, config, env={"spec": "sum"})
    validate_dag(root)
    assert len([o for o in topological_iterator(root) if isinstance(o, AggregateOp)]) == 1

    plan, *_ = optimize(dag, config, env={"spec": "sum"})
    pd.testing.assert_frame_equal(run_plan(plan), frame.groupby("g").agg("sum"))


def test_inlining_enables_aggregation_parameter_fusion():
    frame = pd.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 2.0, 3.0], "w": [4.0, 5.0, 6.0]})
    dag = st.as_data_op(frame).groupby("g").std(ddof=st.var("d"))
    config = OptConfig(dataframe_ops=True)

    root = logical_optimize(dag, config, env={"d": 0})
    validate_dag(root)
    agg_ops = [o for o in topological_iterator(root) if isinstance(o, AggregateOp)]
    assert len(agg_ops) == 1 and agg_ops[0].aggregations[0][1].params == {"ddof": 0}

    plan, *_ = optimize(dag, config, env={"d": 0})
    pd.testing.assert_frame_equal(run_plan(plan), frame.groupby("g").std(ddof=0))


def test_inlining_enables_projection_extraction():
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    dag = st.as_data_op(frame)[st.as_data_op("x")]
    config = OptConfig(dataframe_ops=True)

    root = logical_optimize(dag, config)
    validate_dag(root)
    ops = list(topological_iterator(root))
    proj = [o for o in ops if isinstance(o, ColumnProjectionOp)]
    assert len(proj) == 1 and proj[0].key == "x"
    assert not any(isinstance(o, GetItemOp) for o in ops)

    plan, *_ = optimize(dag, config)
    assert run_plan(plan).tolist() == [1, 2, 3]


def test_inlining_enables_numeric_extraction_with_constant():
    dag = st.as_data_op(2).skb.apply_func(np.add, st.as_data_op(3))
    config = OptConfig()

    root = logical_optimize(dag, config)
    validate_dag(root)
    numeric = [o for o in topological_iterator(root) if isinstance(o, NumericOp)]
    assert len(numeric) == 1 and numeric[0].type is NumericOpType.ADD and numeric[0].constant == 3

    plan, *_ = optimize(dag, config)
    assert run_plan(plan) == 5


def test_inlining_enables_selection_folding():
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    data = st.as_data_op(frame)
    thr = st.as_data_op(1)
    dag = data[data["x"] > thr].assign(base=thr)
    config = OptConfig(dataframe_ops=True)

    root = logical_optimize(dag, config)
    validate_dag(root)
    ops = list(topological_iterator(root))
    sels = [o for o in ops if isinstance(o, MaskSelectionOp)]
    assert len(sels) == 1
    assert sels[0].predicate == BinOpExpr(operator.gt, Col("x"), Const(1))
    assert len(sels[0].inputs) == 1
    assert not any(isinstance(o, ValueOp) for o in ops)

    plan, *_ = optimize(dag, config)
    pd.testing.assert_frame_equal(run_plan(plan), frame[frame["x"] > 1].assign(base=1))


def test_optimize_inlines_constants_and_env_variables():
    frame = pd.DataFrame({"part": ["BRASS", "STEEL", None, "RED BRASS"]})
    # `evaluate` resolves `st.var` through its env, so both literals reach the pass as ValueOps.
    dag = st.as_data_op(frame)[st.as_data_op("part")].str.endswith(st.var("suffix", "BRASS"), na=False)

    ops, *_ = optimize(dag, env={"suffix": "BRASS"})

    assert not any(isinstance(op, ValueOp) and isinstance(op.value, str) for op in ops)
    pd.testing.assert_series_equal(evaluate(dag), frame["part"].str.endswith("BRASS", na=False))
