"""Use label alignment unless the plan proves that operands share an index."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
import skrub

from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.logical._column_expr import WhereExpr
from stratum.optimizer.logical._combine_ops import AssignOp, ConcatOp
from stratum.optimizer.logical._map_ops import ExpressionMapOp
from stratum.optimizer.logical._source_ops import InMemoryFrameOp
from stratum.optimizer.physical._conversion import has_default_index, index_origin
from stratum.optimizer.physical._fused_map_execs import PolarsFusedAssignOp
from stratum.optimizer.physical._map_execs import PandasExpressionMapOp
from stratum.optimizer.physical._impl_selection import get_implementation_selector
from stratum.optimizer.physical._representation import FrameLib, frame_lib
from stratum.runtime._scheduler import SequentialScheduler


def execute(build, frames, *, fused=True):
    with skrub.config_context(eager_data_ops=False):
        dag = build(*(skrub.var(name, frame) for name, frame in frames.items()))
    ops, split = optimize(dag, OptConfig(map_fusion=fused,
                                        selector=get_implementation_selector("greedy")),
                          env=frames)
    result = SequentialScheduler(ops, split).evaluate()
    if isinstance(result, (pl.Series, pl.DataFrame)):
        result = result.to_pandas()
    return ops, result


@pytest.mark.parametrize("lengths", [(3, 1), (1, 3), (3, 2), (2, 2), (0, 1)])
@pytest.mark.parametrize("build", [
    lambda x, y: x["a"] + y["a"],
    lambda x, y: x.assign(z=x["a"] + y["a"]),
    lambda x, y: x.assign(z=y["a"]),
])
def test_independent_default_indexes_require_alignment(lengths, build):
    left, right = lengths
    frames = {"x": pd.DataFrame({"a": np.arange(left, dtype=float)}),
              "y": pd.DataFrame({"a": np.arange(right, dtype=float) + 10})}
    ops, result = execute(build, frames)
    expected = build(*frames.values())
    check = pd.testing.assert_frame_equal if isinstance(expected, pd.DataFrame) else pd.testing.assert_series_equal
    check(result, expected)
    aligned = [op for op in ops if isinstance(op, (AssignOp, ExpressionMapOp))]
    assert aligned and all(frame_lib(op) is FrameLib.PANDAS for op in aligned)


@pytest.mark.parametrize("length", [1, 4])
def test_column_concat_aligns_independent_default_indexes(length):
    x = pd.DataFrame({"a": [1., 2., 3.]})
    y = pd.DataFrame({"b": np.arange(length, dtype=float)})
    ops, result = execute(lambda x, y: x.skb.concat([y], axis=1), {"x": x, "y": y})
    [concat] = [op for op in ops if isinstance(op, ConcatOp)]
    assert frame_lib(concat) is FrameLib.PANDAS
    assert has_default_index(concat)
    pd.testing.assert_frame_equal(result, pd.concat([x, y], axis=1))


def test_a_prefix_does_not_have_its_sources_index():
    frame = pd.DataFrame({"a": [1., 2., 3.]})
    build = lambda df: df.assign(z=df.head(1)["a"])
    ops, result = execute(build, {"frame": frame})
    source = next(op for op in ops if isinstance(op, InMemoryFrameOp))
    prefix = next(op for op in ops if type(op).__name__ == "PolarsSliceSelectionOp")
    assert has_default_index(prefix) and has_default_index(source)
    assert index_origin(prefix) != index_origin(source)
    pd.testing.assert_frame_equal(result, build(frame))


@pytest.mark.parametrize("prefix", [False, True])
def test_matching_rows_stay_native_and_fuse(prefix):
    frame = pd.DataFrame({"a": [1., 2., 3.], "b": [4., 5., 6.]})

    def build(df):
        df = df.head(2) if prefix else df
        renamed = df.rename(columns={"b": "other"})
        return df.assign(z=df["a"] + renamed["other"])

    ops, result = execute(build, {"frame": frame})
    assert all(frame_lib(op) is FrameLib.POLARS for op in ops)
    assert isinstance(ops[-1], PolarsFusedAssignOp)
    pd.testing.assert_frame_equal(result, build(frame))


def test_index_renaming_requires_alignment():
    frame = pd.DataFrame({"a": [1., 2., 3.]})
    build = lambda df: df.assign(z=df["a"] + df.rename(index={0: 10})["a"])
    ops, result = execute(build, {"frame": frame})
    rename = next(op for op in ops if type(op).__name__ == "PandasRenameOp")
    assert not has_default_index(rename)
    pd.testing.assert_frame_equal(result, build(frame))


def test_unknown_data_operand_still_requires_alignment():
    frame = pd.DataFrame({"a": [1., 2., 3.]})
    mask = pd.Series([True], index=[0])

    def build(df):
        condition = skrub.deferred(lambda: mask)()
        return df.assign(z=df["a"].where(condition, 0))

    ops, result = execute(build, {"frame": frame})
    assert any(isinstance(op, PandasExpressionMapOp) and isinstance(op.expr, WhereExpr)
               for op in ops)
    pd.testing.assert_frame_equal(result, frame.assign(z=frame["a"].where(mask, 0)))


@pytest.mark.parametrize("fused", [False, True])
def test_where_alignment_preserves_the_receivers_index(fused):
    frames = {"x": pd.DataFrame({"a": [1., 2., 3.]}),
              "y": pd.DataFrame({"mask": [False, True]}, index=[2, 0])}
    build = lambda x, y: x.assign(z=x["a"].where(y["mask"], 0.))
    ops, result = execute(build, frames, fused=fused)
    [where] = [op for op in ops if isinstance(op, PandasExpressionMapOp)
               and isinstance(op.expr, WhereExpr)]
    [assign] = [op for op in ops if isinstance(op, AssignOp)]
    # The condition requires alignment; its labels do not become output labels.
    assert index_origin(where) == index_origin(assign.inputs[0])
    assert frame_lib(assign) is FrameLib.POLARS
    pd.testing.assert_frame_equal(result, build(*frames.values()))


@pytest.mark.parametrize("fused", [False, True])
def test_membership_does_not_require_matching_indexes(fused):
    x = pd.DataFrame({"a": [1., 2., 3.]})
    y = pd.DataFrame({"a": [2.]}, index=[20])
    build = lambda x, y: x.assign(found=x["a"].isin(y["a"]))
    ops, result = execute(build, {"x": x, "y": y}, fused=fused)
    assert all(frame_lib(op) is FrameLib.POLARS for op in ops)
    pd.testing.assert_frame_equal(result, build(x, y))


def test_independent_file_sources_require_alignment(tmp_path):
    x, y = pd.DataFrame({"a": [1., 2., 3.]}), pd.DataFrame({"a": [10.]})
    left, right = tmp_path / "left.parquet", tmp_path / "right.parquet"
    x.to_parquet(left)
    y.to_parquet(right)

    def build():
        first = skrub.deferred(pd.read_parquet)(str(left))
        second = skrub.deferred(pd.read_parquet)(str(right))
        return first.assign(z=first["a"] + second["a"])

    _, result = execute(build, {})
    pd.testing.assert_frame_equal(result, x.assign(z=x["a"] + y["a"]))
