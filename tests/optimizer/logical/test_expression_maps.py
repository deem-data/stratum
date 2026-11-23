"""Maps compute values from explicit operands; assignment combines the results."""
import operator

import numpy as np
import pandas as pd
import polars as pl
import pytest
import skrub

import stratum as st
from stratum.optimizer._op_utils import topological_iterator, validate_dag
from stratum.optimizer._optimize import OptConfig, logical_optimize, optimize
from stratum.optimizer.logical._base import IndexEffect, OperandRef, OutputType
from stratum.optimizer.logical._column_expr import BinOpExpr, Const, OperandLeaf, UFUNCS, WhereExpr
from stratum.optimizer.logical._combine_ops import AssignOp, CombineOp, ConcatOp
from stratum.optimizer.logical._map_ops import ExpressionMapOp, MapOp
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp
from stratum.optimizer.physical._impl_selection import get_implementation_selector
from stratum.runtime._scheduler import SequentialScheduler


FRAME = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})


def logical(build, **options):
    with skrub.config_context(eager_data_ops=False):
        dag = build(skrub.var("frame", FRAME))
    root = logical_optimize(dag, OptConfig(**options), env={"frame": FRAME})
    validate_dag(root)
    return list(topological_iterator(root))


def evaluate(build, selector="default", frame=FRAME):
    with skrub.config_context(eager_data_ops=False):
        dag = build(skrub.var("frame", frame))
    ops, split = optimize(dag, OptConfig(map_fusion=False,
                                        selector=get_implementation_selector(selector)),
                          env={"frame": frame})
    return ops, SequentialScheduler(ops, split).evaluate()


def pandas(value):
    return value.to_pandas() if isinstance(value, (pl.Series, pl.DataFrame)) else value


def test_assign_is_a_combine_over_explicit_maps():
    ops = logical(lambda df: df.assign(c=df["a"] + df["b"]))
    [assign] = [op for op in ops if isinstance(op, AssignOp)]
    mapped = assign.inputs[assign.kwargs["c"].k]
    assert isinstance(assign, CombineOp) and not isinstance(assign, MapOp)
    assert isinstance(mapped, ExpressionMapOp)
    assert mapped.expr == BinOpExpr(operator.add, OperandLeaf(OperandRef(0)),
                                   OperandLeaf(OperandRef(1)))
    assert [op.key for op in mapped.inputs] == ["a", "b"]
    assert all(isinstance(op, ColumnProjectionOp) for op in mapped.inputs)
    assert all(op.inputs == [assign.inputs[0]] for op in mapped.inputs)
    assert isinstance(ConcatOp(), CombineOp)


def test_map_chains_keep_intermediate_dependencies():
    ops = logical(lambda df: (df["a"] + 2) * 3)
    maps = [op for op in ops if isinstance(op, ExpressionMapOp)]
    assert len(maps) == 2
    assert maps[1].inputs == [maps[0]]
    assert maps[0].expr == BinOpExpr(operator.add, OperandLeaf(OperandRef(0)), Const(2))


def test_maps_do_not_require_an_originating_frame():
    ops, result = evaluate(lambda df: df["a"].apply(lambda x: x + 1) * 2)
    mapped = next(op for op in ops if isinstance(op, ExpressionMapOp))
    assert len(mapped.inputs) == 1
    assert not isinstance(mapped.inputs[0], ColumnProjectionOp)
    pd.testing.assert_series_equal(result, (FRAME["a"] + 1) * 2)


def test_where_keeps_all_three_inputs():
    ops = logical(lambda df: df.assign(c=df["a"].where(df["a"] > 1, df["b"])))
    [where] = [op for op in ops if isinstance(op, ExpressionMapOp)
               and isinstance(op.expr, WhereExpr)]
    assert len(where.inputs) == 3
    assert isinstance(where.inputs[where.expr.cond.ref.k], ExpressionMapOp)
    assert where.inputs[where.expr.other.ref.k].key == "b"


def test_shared_maps_and_repeated_operands_remain_shared():
    def build(df):
        derived = df["a"] + 1
        return df.assign(first=derived, second=derived, doubled=derived + derived)
    ops = logical(build)
    [assign] = [op for op in ops if isinstance(op, AssignOp)]
    assert assign.kwargs["first"] == assign.kwargs["second"]
    shared = assign.inputs[assign.kwargs["first"].k]
    doubled = assign.inputs[assign.kwargs["doubled"].k]
    assert doubled.inputs == [shared]
    assert doubled.expr.left == doubled.expr.right == OperandLeaf(OperandRef(0))


def test_cse_deduplicates_computation_before_assignment():
    ops = logical(lambda df: df.assign(first=df["a"] + 1, second=df["a"] + 1))
    assert len([op for op in ops if isinstance(op, ExpressionMapOp)]) == 1
    [assign] = [op for op in ops if isinstance(op, AssignOp)]
    assert assign.kwargs["first"] == assign.kwargs["second"]


def test_assignment_order_and_scalar_types_are_not_merged():
    ops = logical(lambda df: df.assign(x=1, y=2).skb.concat(
        [df.assign(y=2, x=1), df.assign(x=1.0, y=2)], axis=0))
    assert len([op for op in ops if isinstance(op, AssignOp)]) == 3


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("build", [
    lambda df: df["a"] * 2 + 1,
    lambda df: df["a"] + df["b"],
    lambda df: -(df["a"] + df["b"]) * 2,
    lambda df: (df["a"] + df["b"]).isna(),
    lambda df: (df["a"] + df["b"]).fillna(0),
    lambda df: df.skb.apply_func(np.log) if hasattr(df, "skb") else np.log(df),
    lambda df: df["a"] - df["a"].mean(),
    lambda df: df.assign(a=df["a"] * 10, original=df["a"] * 2),
    lambda df: df.assign(c=(df["a"] + df["b"]) * 2),
    lambda df: df.assign(c=df["a"].where(df["a"] > 1, df["b"])),
    lambda df: df.assign(c=[8, 9, 10], flag=7),
    lambda df: df.assign(c=lambda frame: frame["a"] * 2,
                         d=lambda frame: frame["c"] + 1),
])
def test_names_values_order_and_inputs_match_pandas(build, selector):
    original = FRAME.copy(deep=True)
    _, result = evaluate(build, selector)
    expected = build(FRAME)
    check = pd.testing.assert_frame_equal if isinstance(expected, pd.DataFrame) else pd.testing.assert_series_equal
    check(pandas(result), expected, check_dtype=False)
    pd.testing.assert_frame_equal(FRAME, original)


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("build", [
    lambda df: df.assign(c=df["a"] * 0),
    lambda df: df > 0,
])
def test_maps_preserve_shape_and_missing_values(build, selector):
    frame = FRAME.copy()
    frame.loc[1, "a"] = np.nan
    _, result = evaluate(build, selector, frame)
    pd.testing.assert_frame_equal(pandas(result), build(frame), check_dtype=False)


@pytest.mark.parametrize("selector", ["default", "greedy"])
def test_binary_map_aligns_different_indexes(selector):
    other = pd.DataFrame({"a": [10.0, 20.0, 30.0]}, index=[1, 2, 3])
    with skrub.config_context(eager_data_ops=False):
        x, y = skrub.var("x", FRAME), skrub.var("y", other)
        dag = x["a"] + y["a"]
    ops, split = optimize(dag, OptConfig(selector=get_implementation_selector(selector)),
                          env={"x": FRAME, "y": other})
    mapped = next(op for op in ops if isinstance(op, ExpressionMapOp))
    assert mapped.index_effect is IndexEffect.COMMON
    pd.testing.assert_series_equal(SequentialScheduler(ops, split).evaluate(),
                                   FRAME["a"] + other["a"])


@pytest.mark.parametrize("selector", ["default", "greedy"])
def test_assignment_aligns_a_literal_series(selector):
    values = pd.Series([10, 20, 30], index=[2, 0, 1])
    _, result = evaluate(lambda df: df.assign(c=values), selector)
    pd.testing.assert_frame_equal(result, FRAME.assign(c=values))


@pytest.mark.parametrize("selector", ["default", "greedy"])
def test_map_aligns_a_literal_series_before_assignment(selector):
    values = pd.Series([10.0, 20.0, 30.0], index=[1, 2, 3], name="a")
    _, result = evaluate(lambda df: df.assign(c=df["a"] + values), selector)
    pd.testing.assert_frame_equal(pandas(result), FRAME.assign(c=FRAME["a"] + values))


@pytest.mark.parametrize("selector", ["default", "greedy"])
def test_assignment_aligns_a_graph_fed_literal_series(selector):
    values = pd.Series([10.0, 20.0, 30.0], index=[2, 0, 1])
    _, result = evaluate(lambda df: df.assign(c=st.as_data_op(values)), selector)
    pd.testing.assert_frame_equal(pandas(result), FRAME.assign(c=values))


def test_cse_keeps_different_scalar_operand_types():
    ops = logical(lambda df: df.assign(a=df["a"] * st.as_data_op(1)).skb.concat(
        [df.assign(a=df["a"] * st.as_data_op(1.0))], axis=0))
    assert len([op for op in ops if isinstance(op, ExpressionMapOp)]) == 2


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("name", UFUNCS)
def test_numpy_maps_keep_the_numpy_kernel(name, selector):
    frame = pd.DataFrame({"a": np.random.default_rng(0).lognormal(0, 1, 2000)})
    func = UFUNCS[name]
    ops, result = evaluate(lambda df: df["a"].skb.apply_func(func), selector, frame)
    assert any(isinstance(op, ExpressionMapOp) for op in ops)
    pd.testing.assert_series_equal(pandas(result), func(frame["a"]), check_exact=True)


def test_map_schema_and_clone_preserve_output_kind():
    ops = logical(lambda df: df["a"] > 1)
    [mapped] = [op for op in ops if isinstance(op, ExpressionMapOp)]
    assert mapped.output_schema == pl.Schema({"a": pl.Boolean})
    clone = mapped.clone()
    assert clone.expr is mapped.expr
    assert clone.output_type is OutputType.SERIES
    ops = logical(lambda df: df > 1)
    [mapped] = [op for op in ops if isinstance(op, ExpressionMapOp)]
    assert mapped.output_schema == pl.Schema({"a": pl.Boolean, "b": pl.Boolean})
    assert mapped.clone().output_type is OutputType.FRAME


def test_disabling_expression_maps_keeps_assignment_explicit():
    with st.config(make_map_op=False):
        ops = logical(lambda df: df.assign(c=df["a"] + 2))
    assert not any(isinstance(op, ExpressionMapOp) for op in ops)
    assert any(isinstance(op, AssignOp) for op in ops)
