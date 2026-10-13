"""A map over a column projection runs on the unprojected frame instead.

``df[["a", "b"]].assign(c=df["x"] * 2)`` folds against the projection, which leaves
``df["x"]`` as a separate input of the map. Pushing the projection after the map
lets ``x`` be read as a column of ``df``.
"""
import pandas as pd
import polars as pl
import pytest

import stratum as st
from stratum.optimizer._optimize import OptConfig, logical_optimize, propagate_output_schema
from stratum.optimizer.logical._column_expr import BinOpExpr, Col, Const
from stratum.optimizer.logical._map_ops import AssignMapOp
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp, DropOp
from stratum.optimizer.physical._source_execs import InMemoryFrame
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
        pool[id(op)] = op.process("fit_transform", [pool[id(i)] for i in op.inputs])
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
    [source] = _of(ops, InMemoryFrame)
    return source


def test_projected_assign_maps_the_source_frame(polars):
    ops = _check(lambda df: df[["col1", "col2"]].assign(
        new_col1=df["col3"] * 2, new_col2=df["col4"] * 3))
    [map_op] = _of(ops, AssignMapOp)
    assert map_op.inputs == [_source(ops)]
    assert map_op.entries == {
        "new_col1": BinOpExpr(map_op.entries["new_col1"].op, Col("col3"), Const(2)),
        "new_col2": BinOpExpr(map_op.entries["new_col2"].op, Col("col4"), Const(3))}
    # The single-column inputs are gone; one projection is left, after the map.
    [projection] = _of(ops, ColumnProjectionOp)
    assert projection.inputs == [map_op]
    assert projection.key == ["col1", "col2", "new_col1", "new_col2"]
    assert ops[-1] is projection
    assert list(projection.output_schema) == projection.key


def test_disabled_by_config():
    df = st.as_data_op(FRAME)
    ops = optimize(df[["col1", "col2"]].assign(x=df["col3"] * 2),
                   OptConfig(projection_pushup=False))
    [map_op] = _of(ops, AssignMapOp)
    assert map_op.inputs[0] is not _source(ops)
    assert len(map_op.inputs) == 2


def test_reordering_projection_keeps_its_order(polars):
    ops = _check(lambda df: df[["col2", "col1"]].assign(x=df["col3"] + 1))
    assert ops[-1].key == ["col2", "col1", "x"]


def test_overwritten_columns_keep_their_position(polars):
    # col1 is kept and overwritten in place; col4 was projected away, so it is
    # appended, as assign does on the projected frame.
    ops = _check(lambda df: df[["col1", "col2"]].assign(
        col1=df["col3"] * 2, col4=df["col3"] - 1))
    assert ops[-1].key == ["col1", "col2", "col4"]


def test_entry_reads_a_column_another_entry_overwrites(polars):
    # Entries are evaluated before any is assigned, so x sees the original col3.
    _check(lambda df: df[["col1"]].assign(col3=df["col4"], x=df["col3"] * 2))


def test_shared_column_stays_for_its_other_consumer(polars):
    def build(df):
        col3 = df["col3"]
        first = df[["col1"]].assign(a=col3 * 2)
        return first.assign(b=col3 + 1)
    ops = _check(build)
    first = next(op for op in _of(ops, AssignMapOp) if "a" in op.entries)
    assert first.inputs == [_source(ops)]
    assert first.entries["a"].left == Col("col3")
    # Still materialised for the second map, which reads it as an operand.
    assert any(isinstance(op, ColumnProjectionOp) and op.key == "col3" for op in ops)


def test_shared_projection_stays_for_its_other_consumer(polars):
    def build(df):
        kept = df[["col1", "col2"]]
        return kept.assign(a=df["col3"] * 2)[["a"]].skb.concat([kept], axis=1)
    expected = pd.concat([FRAME[["col3"]].rename(columns={"col3": "a"}) * 2,
                          FRAME[["col1", "col2"]]], axis=1)
    ops = _check(build, expected)
    [map_op] = _of(ops, AssignMapOp)
    assert map_op.inputs == [_source(ops)]
    assert any(isinstance(op, ColumnProjectionOp) and op.key == ["col1", "col2"]
               for op in ops)


def test_column_read_through_the_projection_folds_too(polars):
    # kept["col1"] has another consumer, so it was not absorbed against the
    # projection; it is still col1 of the source.
    def build(df):
        kept = df[["col1", "col2"]]
        col1 = kept["col1"]
        mapped = kept.assign(a=col1 * 2, b=df["col3"])
        return mapped.assign(c=col1 + 1)
    ops = _check(build)
    first = next(op for op in _of(ops, AssignMapOp) if "a" in op.entries)
    assert first.inputs == [_source(ops)]
    assert first.entries["a"].left == Col("col1")
    assert first.entries["b"] == Col("col3")


def test_column_drop_becomes_a_projection(polars):
    ops = _check(lambda df: df.drop(columns="col2").assign(x=df["col2"] * 2))
    assert not _of(ops, DropOp)
    assert ops[-1].key == ["col1", "col3", "col4", "x"]


def test_dropped_column_assigned_again_moves_to_the_end(polars):
    ops = _check(lambda df: df.drop(columns=["col2"]).assign(col2=df["col2"] + 1))
    assert ops[-1].key == ["col1", "col3", "col4", "col2"]


@pytest.mark.parametrize("build", [
    pytest.param(lambda df: df.drop(index=[0]).assign(x=df["col3"] * 2), id="row-drop"),
    pytest.param(lambda df: df[["col1", "col1"]].assign(x=df["col3"] * 2),
                 id="duplicate-keys"),
    pytest.param(lambda df: df[["col1"]].assign(x=5), id="no-column-leaf"),
])
def test_not_pushed(build):
    ops = optimize(build(st.as_data_op(FRAME)))
    [map_op] = _of(ops, AssignMapOp)
    assert map_op.inputs[0] is not _source(ops)


def test_selector_projection_is_not_pushed():
    df = st.as_data_op(FRAME)
    ops = optimize(df.skb.select(["col1"]).assign(x=df["col3"] * 2))
    [map_op] = _of(ops, AssignMapOp)
    assert map_op.inputs[0] is not _source(ops)


def test_final_propagation_flags_a_stale_schema():
    df = st.as_data_op(FRAME)
    root = logical_optimize(df[["col1"]], OptConfig())
    root.output_schema = pl.Schema({"stale": pl.Int64})
    with pytest.raises(AssertionError, match="stale output schema"):
        propagate_output_schema(root, check=True)
