"""Assignments preserve projected base frames and explicit value dependencies."""
import pandas as pd
import polars as pl
import pytest

import stratum as st
from stratum.optimizer._optimize import OptConfig, logical_optimize
from stratum.optimizer.logical.rewrites._schema_propagation import propagate_output_schema
from stratum.optimizer.logical._combine_ops import AssignOp
from stratum.optimizer.logical._map_ops import ExpressionMapOp
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp, DropOp
from stratum.optimizer.logical._source_ops import InMemoryFrameOp
from .test_dataframe_ops import force_polars, optimize


FRAME = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6],
                      "col3": [7, 8, 9], "col4": [10, 11, 12]})


@pytest.fixture(params=[False, True], ids=["pandas", "polars"])
def polars(request):
    with force_polars(request.param):
        yield request.param


def _run(ops):
    pool = {}
    for op in ops:
        pool[id(op)] = op.process([pool[id(i)] for i in op.inputs])
    result = pool[id(ops[-1])]
    return result.to_pandas() if isinstance(result, pl.DataFrame) else result


def _check(build, expected=None):
    """Plan ``build(df)``, run it, compare it with pandas, and return the plan."""
    ops = optimize(build(st.as_data_op(FRAME)))
    expected = build(FRAME) if expected is None else expected
    pd.testing.assert_frame_equal(expected, _run(ops), check_dtype=False)
    return ops


def _of(ops, cls):
    return [op for op in ops if isinstance(op, cls)]


def _source(ops):
    [source] = _of(ops, InMemoryFrameOp)
    return source


def test_projected_assign_keeps_separate_inputs(polars):
    ops = _check(lambda df: df[["col1", "col2"]].assign(
        new_col1=df["col3"] * 2, new_col2=df["col4"] * 3))
    [assign] = _of(ops, AssignOp)
    assert assign.inputs[0].key == ["col1", "col2"]
    assert list(assign.kwargs) == ["new_col1", "new_col2"]
    for name, column in (("new_col1", "col3"), ("new_col2", "col4")):
        mapped = assign.inputs[assign.kwargs[name].k]
        assert isinstance(mapped, ExpressionMapOp)
        assert mapped.inputs[0].key == column
        assert mapped.inputs[0].inputs == [_source(ops)]
    assert list(assign.output_schema) == ["col1", "col2", "new_col1", "new_col2"]


def test_projected_assignment_preserves_column_order(polars):
    # col1 is kept and overwritten in place; col4 was projected away, so it is
    # appended, as assign does on the projected frame.
    ops = _check(lambda df: df[["col2", "col1"]].assign(
        col1=df["col3"] * 2, col4=df["col3"] - 1))
    assert list(ops[-1].output_schema) == ["col2", "col1", "col4"]


def test_entry_reads_a_column_another_entry_overwrites(polars):
    # Entries are evaluated before any is assigned, so x sees the original col3.
    _check(lambda df: df[["col1"]].assign(col3=df["col4"], x=df["col3"] * 2))


def test_shared_column_stays_for_its_other_consumer(polars):
    def build(df):
        col3 = df["col3"]
        first = df[["col1"]].assign(a=col3 * 2)
        return first.assign(b=col3 + 1)
    ops = _check(build)
    first = next(op for op in _of(ops, AssignOp) if "a" in op.kwargs)
    shared = next(op for op in _of(ops, ColumnProjectionOp) if op.key == "col3")
    maps = _of(ops, ExpressionMapOp)
    assert len(maps) == 2 and all(shared in op.inputs for op in maps)
    assert first.inputs[0].key == ["col1"]


def test_shared_projection_stays_for_its_other_consumer(polars):
    def build(df):
        kept = df[["col1", "col2"]]
        return kept.assign(a=df["col3"] * 2)[["a"]].skb.concat([kept], axis=1)
    expected = pd.concat([FRAME[["col3"]].rename(columns={"col3": "a"}) * 2,
                          FRAME[["col1", "col2"]]], axis=1)
    ops = _check(build, expected)
    [map_op] = _of(ops, AssignOp)
    assert map_op.inputs[0].key == ["col1", "col2"]
    assert any(isinstance(op, ColumnProjectionOp) and op.key == ["col1", "col2"]
               for op in ops)


def test_column_read_through_the_projection_stays_explicit(polars):
    # The shared column projection remains an explicit input of both maps.
    def build(df):
        kept = df[["col1", "col2"]]
        col1 = kept["col1"]
        mapped = kept.assign(a=col1 * 2, b=df["col3"])
        return mapped.assign(c=col1 + 1)
    ops = _check(build)
    first = next(op for op in _of(ops, AssignOp) if "a" in op.kwargs)
    mapped = first.inputs[first.kwargs["a"].k]
    assert mapped.inputs[0].key == "col1"
    assert first.inputs[first.kwargs["b"].k].key == "col3"


def test_column_drop_remains_the_assignment_base(polars):
    ops = _check(lambda df: df.drop(columns="col2").assign(x=df["col2"] * 2))
    [drop] = _of(ops, DropOp)
    assert ops[-1].inputs[0] is drop
    assert list(ops[-1].output_schema) == ["col1", "col3", "col4", "x"]


def test_dropped_column_assigned_again_moves_to_the_end(polars):
    ops = _check(lambda df: df.drop(columns=["col2"]).assign(col2=df["col2"] + 1))
    assert list(ops[-1].output_schema) == ["col1", "col3", "col4", "col2"]


def test_final_propagation_flags_a_stale_schema():
    df = st.as_data_op(FRAME)
    root = logical_optimize(df[["col1"]], OptConfig())
    root.output_schema = pl.Schema({"stale": pl.Int64})
    with pytest.raises(AssertionError, match="stale output schema"):
        propagate_output_schema(root, check=True)
