"""Scalar Series reductions under greedy physical selection."""
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import skrub

from stratum.optimizer.physical._aggregation_execs import (
    PandasAggregateOp, PolarsFusedScalarAggregateOp, PolarsScalarAggregateOp)
from stratum.optimizer.physical._fused_map_execs import PolarsFusedMapOp
from stratum.runtime._scheduler import SequentialScheduler
from .test_map_fusion import execute, plan


@pytest.mark.parametrize("dtype,values", [
    ("int32", [2**30, 2**30]),
    ("uint32", [2**31, 2**31]),
    ("UInt64", [2**63, None, 7]),
    ("uint64[pyarrow]", [2**63, None, 7]),
    ("Int16", [None, None]),
    ("int8", []),
])
def test_integer_sum_accumulates_in_64_bits(dtype, values):
    frame = pd.DataFrame({"v": pd.Series(values, dtype=dtype)})
    ops, result = execute(lambda df: df["v"].agg("sum"), frame=frame)
    expected = frame["v"].sum()
    assert isinstance(ops[-1], PolarsScalarAggregateOp)
    assert type(result) is type(expected)
    assert result == expected


@pytest.mark.parametrize("dtype", ["float64", "Float64", "double[pyarrow]"])
@pytest.mark.parametrize("map_fusion,aggregate_fusion",
                         [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("left,right", [
    ([3., np.inf, None], [4., 0., 2.]),
    ([np.inf, None], [0., 2.]),
    ([np.inf, -np.inf], [1., 1.]),
    ([], []),
])
def test_float_product_sum_skips_nulls_and_arithmetic_nan(
        dtype, map_fusion, aggregate_fusion, left, right):
    frame = pd.DataFrame({"a": pd.Series(left, dtype=dtype), "b": pd.Series(right, dtype=dtype)})
    ops, result = execute(lambda df: (df["a"] * df["b"]).sum(), frame=frame,
                          fused=map_fusion, map_aggregate_fusion=aggregate_fusion)
    assert isinstance(ops[-1], PolarsScalarAggregateOp)
    assert isinstance(ops[-1], PolarsFusedScalarAggregateOp) == aggregate_fusion
    with np.errstate(invalid="ignore"):
        expected = (frame["a"] * frame["b"]).sum()
    assert type(result) is type(expected)
    if expected is pd.NA:
        assert result is pd.NA
    else:
        np.testing.assert_equal(result, expected)


@pytest.mark.parametrize("values", [
    np.array([1., np.nan, np.nan]),
    pd.arrays.ArrowExtensionArray(pa.array([1., np.nan, None])),
    pd.arrays.FloatingArray(np.array([1., np.nan, 0.]), np.array([False, False, True])),
], ids=["numpy", "arrow", "nullable"])
def test_direct_sum_preserves_storage_specific_nan_semantics(values):
    frame = pd.DataFrame({"v": values})
    ops, result = execute(lambda df: df["v"].sum(), frame=frame)
    expected = frame["v"].sum()
    assert isinstance(ops[-1], PolarsScalarAggregateOp)
    assert type(result) is type(expected)
    if expected is pd.NA:
        assert result is pd.NA
    else:
        np.testing.assert_equal(result, expected)


@pytest.mark.parametrize("dtype,values", [
    ("int8", [-5, 2]), ("uint32", []), ("Int16", [None, -3]),
    ("Int16", [None]), ("uint64[pyarrow]", [2**63, None]), ("int64[pyarrow]", []),
    ("float64", []), ("float64", [None, None]), ("float64", [3., None]),
    ("double[pyarrow]", [None, None]),
])
def test_min_preserves_scalar_dtype_and_missing_values(dtype, values):
    frame = pd.DataFrame({"v": pd.Series(values, dtype=dtype)})
    ops, result = execute(lambda df: df["v"].min(), frame=frame)
    expected = frame["v"].min()
    assert isinstance(ops[-1], PolarsScalarAggregateOp)
    assert type(result) is type(expected)
    if expected is pd.NA:
        assert result is pd.NA
    else:
        np.testing.assert_equal(result, expected)


@pytest.mark.parametrize("map_fusion,aggregate_fusion",
                         [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("dtype", ["float64", "Float64", "double[pyarrow]"])
@pytest.mark.parametrize("values", [[3., float("inf"), None], [float("inf"), None], []])
def test_computed_min_preserves_scalar_and_fusion(map_fusion, aggregate_fusion, dtype, values):
    frame = pd.DataFrame({"a": pd.Series(values, dtype=dtype), "b": pd.Series(
        [0. if x == float("inf") else 2. for x in values], dtype=dtype)})
    def build(df):
        return (df["a"] * df["b"]).min()
    ops, result = execute(build, frame=frame, fused=map_fusion,
                          map_aggregate_fusion=aggregate_fusion)
    assert isinstance(ops[-1], PolarsScalarAggregateOp)
    assert isinstance(ops[-1], PolarsFusedScalarAggregateOp) == aggregate_fusion
    with np.errstate(invalid="ignore"):
        expected = build(frame)
    assert type(result) is type(expected)
    if expected is pd.NA:
        assert result is pd.NA
    else:
        np.testing.assert_equal(result, expected)


@pytest.mark.parametrize("values", [
    pd.arrays.ArrowExtensionArray(pa.array([1., np.nan, None])),
    pd.arrays.ArrowExtensionArray(pa.array([np.nan, None])),
    pd.arrays.FloatingArray(np.array([1., np.nan, 0.]), np.array([False, False, True])),
])
def test_min_over_valid_nan_respects_storage(values):
    frame = pd.DataFrame({"v": values})
    ops, result = execute(lambda df: df["v"].min(), frame=frame)
    expected = frame["v"].min()
    assert isinstance(ops[-1], PolarsScalarAggregateOp)
    assert type(result) is type(expected)
    if expected is pd.NA:
        assert result is pd.NA
    else:
        np.testing.assert_equal(result, expected)


@pytest.mark.parametrize("reduce,dtype,values", [
    (lambda s: s.sum(skipna=False), "float64", [2., np.nan]),
    (lambda s: s.sum(min_count=3), "int64", [1, 2]),
    (lambda s: s.mean(), "int64", [1, 2]),
    (lambda s: s.sum(), "float32", [2., np.nan]),
    (lambda s: s.sum(), "str", ["a", "b"]),
])
def test_other_reductions_use_pandas(reduce, dtype, values):
    frame = pd.DataFrame({"v": pd.Series(values, dtype=dtype)})
    ops, result = execute(lambda df: reduce(df["v"]), frame=frame)
    assert isinstance(ops[-1], PandasAggregateOp)
    np.testing.assert_equal(result, reduce(frame["v"]))


def test_shared_product_keeps_its_series_name():
    frame = pd.DataFrame({"a": [2., 3.], "b": [4., 5.]})

    def build(df):
        product = df["a"] * df["b"]
        return skrub.deferred(tuple)([product, product.sum()])

    _, (result, total) = execute(build, frame=frame, map_aggregate_fusion=True)
    expected = frame["a"] * frame["b"]
    pd.testing.assert_series_equal(result, expected)
    assert total == expected.sum()


def test_sum_reuses_a_product_shared_with_an_assignment():
    frame = pd.DataFrame({"a": [2., 3.], "b": [4., 5.]})

    def build(df):
        product = df["a"] * df["b"]
        return skrub.deferred(tuple)([df.assign(product=product), product.sum()])

    ops, (assigned, total) = execute(build, frame=frame, map_aggregate_fusion=True)
    (reduction,) = [op for op in ops if isinstance(op, PolarsScalarAggregateOp)]
    (product,) = reduction.inputs
    assert isinstance(product, PolarsFusedMapOp) and len(product.outputs) == 2
    expected = frame["a"] * frame["b"]
    pd.testing.assert_frame_equal(assigned, frame.assign(product=expected))
    assert total == expected.sum()


def test_fused_sum_executes_one_scalar_select(monkeypatch):
    frame = pd.DataFrame({"*": [3., 9.], "^v$": [1., 2.]})
    ops, split = plan(lambda df: (df["*"] - df["^v$"]).sum(), frame=frame,
                      map_aggregate_fusion=True)
    assert len(ops) == 2 and isinstance(ops[-1], PolarsFusedScalarAggregateOp)
    shapes = []
    select = pl.DataFrame.select

    def record(frame, *args, **kwargs):
        result = select(frame, *args, **kwargs)
        shapes.append(result.shape)
        return result

    monkeypatch.setattr(pl.DataFrame, "select", record)
    assert SequentialScheduler(ops, split).evaluate() == (frame["*"] - frame["^v$"]).sum()
    assert shapes == [(1, 1)]


def test_sum_of_unaligned_series_keeps_pandas_label_alignment():
    frame = pd.DataFrame({"a": [2., 3.], "b": [4., 5.]})

    def build(df):
        return (df.head(1)["a"] * df.tail(1)["b"]).sum()

    ops, total = execute(build, frame=frame, map_aggregate_fusion=True)
    assert not any(isinstance(op, PolarsFusedScalarAggregateOp) for op in ops)
    assert total == build(frame) == 0.


def test_product_with_mixed_storage_uses_pandas():
    frame = pd.DataFrame({"a": [2., 3.], "b": pd.Series([4., 5.], dtype="double[pyarrow]")})
    ops, result = execute(lambda df: (df["a"] * df["b"]).sum(), frame=frame)
    assert isinstance(ops[-1], PandasAggregateOp)
    assert result == (frame["a"] * frame["b"]).sum()
