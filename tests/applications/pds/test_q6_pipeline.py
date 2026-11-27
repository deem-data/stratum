"""PDS Q6 filters lineitem and sums price times discount."""
import operator
from datetime import date

import pandas as pd
import polars as pl
import pytest
import stratum as st

from stratum.frontend import _api
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.logical._column_expr import AggExpr, BinOpExpr, Col, Const
from stratum.optimizer.logical._selection_ops import MaskSelectionOp
from stratum.optimizer.physical._aggregation_execs import PolarsFusedScalarAggregateOp, PolarsScalarAggregateOp
from stratum.optimizer.physical._conversion import ConversionOp
from stratum.optimizer.physical._fused_map_execs import PolarsFusedMapOp
from stratum.optimizer.physical._map_execs import PolarsExpressionMapOp
from stratum.optimizer.physical._selection_execs import (
    PandasIndexSelectionOp, PandasQuerySelectionOp, PolarsMaskSelectionOp)
from stratum.runtime._scheduler import SequentialScheduler

# The Q6 bounds: `var1` to `var5` in the upstream query.
START = date(1994, 1, 1)
END = date(1995, 1, 1)
MIN_DISCOUNT = 0.05
MAX_DISCOUNT = 0.07
MAX_QUANTITY = 24


def make_lineitem():
    """A 9-row ``lineitem``, Arrow-backed like the benchmark's parquet read.

    Three lines qualify, sitting on the window start and on both discount
    bounds. Every other line fails exactly one check: a ship date before the
    window or on its end, a discount below or above the range, or a quantity
    on or above its bound.
    """
    df = pd.DataFrame({
        "l_orderkey": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "l_partkey": [1552, 673, 1062, 429, 1901, 88, 1444, 203, 1777],
        "l_suppkey": [93, 28, 75, 41, 12, 67, 39, 81, 54],
        "l_linenumber": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        "l_quantity": [10.0, 23.0, 1.0, 5.0, 5.0, 5.0, 5.0, 24.0, 30.0],
        "l_extendedprice": [1000.0, 2000.0, 3000.0, 9000.0, 9000.0, 9000.0,
                            9000.0, 9000.0, 9000.0],
        "l_discount": [0.06, 0.05, 0.07, 0.06, 0.06, 0.04, 0.08, 0.06, 0.06],
        "l_tax": [0.02, 0.06, 0.00, 0.04, 0.08, 0.01, 0.03, 0.05, 0.07],
        "l_returnflag": ["N", "N", "R", "A", "A", "N", "R", "R", "A"],
        "l_linestatus": ["O", "O", "F", "F", "F", "O", "F", "F", "F"],
        "l_shipdate": [date(1994, 3, 1), date(1994, 1, 1), date(1994, 12, 31),
                       date(1993, 12, 31), date(1995, 1, 1), date(1994, 5, 9),
                       date(1994, 8, 1), date(1994, 10, 21), date(1994, 9, 5)],
    }).convert_dtypes(dtype_backend="pyarrow")
    for column in ["l_quantity", "l_extendedprice", "l_discount", "l_tax"]:
        df[column] = pd.array(df[column], dtype="double[pyarrow]")
    df["l_shipdate"] = pd.array(df["l_shipdate"], dtype="date32[day][pyarrow]")
    return df


def q6_reference(line_item_ds):
    """The upstream pandas query, with the table supplied by the fixture."""
    var1 = START
    var2 = END
    var3 = MIN_DISCOUNT
    var4 = MAX_DISCOUNT
    var5 = MAX_QUANTITY

    filt = line_item_ds[
        (line_item_ds["l_shipdate"] >= var1) & (line_item_ds["l_shipdate"] < var2)
    ]
    filt = filt[(filt["l_discount"] >= var3) & (filt["l_discount"] <= var4)]
    filt = filt[filt["l_quantity"] < var5]
    result_value = (filt["l_extendedprice"] * filt["l_discount"]).sum()
    return pd.DataFrame({"revenue": [result_value]})


def build_q6_prefix(line_item_df):
    """The three filters of Q6 as a skrub DAG.

    ``var1`` to ``var5`` stay as ``skrub.var``s so the bounds reach the fold as
    ``ValueOp``s rather than inline literals.
    """
    line_item_ds = st.var("line_item_ds", line_item_df)
    var1 = st.var("var1", START)
    var2 = st.var("var2", END)
    var3 = st.var("var3", MIN_DISCOUNT)
    var4 = st.var("var4", MAX_DISCOUNT)
    var5 = st.var("var5", MAX_QUANTITY)

    filt = line_item_ds[
        (line_item_ds["l_shipdate"] >= var1) & (line_item_ds["l_shipdate"] < var2)
    ]
    filt = filt[(filt["l_discount"] >= var3) & (filt["l_discount"] <= var4)]
    return filt[filt["l_quantity"] < var5]


def build_q6(line_item_df):
    """Complete Q6. ``pd.DataFrame`` cannot take a DataOp, so the one-row frame
    is built by ``skrub.deferred(pd.DataFrame)`` over the same dict."""
    filt = build_q6_prefix(line_item_df)

    result_value = (filt["l_extendedprice"] * filt["l_discount"]).sum()
    return st.deferred(pd.DataFrame)({"revenue": [result_value]})


@pytest.mark.parametrize("map_fusion", [False, True])
@pytest.mark.parametrize("map_aggregate_fusion", [False, True])
def test_q6_keeps_filters_product_and_sum_on_polars(map_fusion, map_aggregate_fusion):
    frame = make_lineitem()
    dag = build_q6(frame)
    config = OptConfig(map_fusion=map_fusion, map_aggregate_fusion=map_aggregate_fusion)
    with st.config(implementation_selector="greedy"):
        ops, split = optimize(dag, config, env=get_data(dag))
    filters = [op for op in ops if isinstance(op, PolarsMaskSelectionOp)]
    assert len(filters) == 3
    (total,) = [op for op in ops if isinstance(op, PolarsScalarAggregateOp)]
    if map_fusion and map_aggregate_fusion:
        assert isinstance(total, PolarsFusedScalarAggregateOp) and total.inputs == [filters[-1]]
        product = BinOpExpr(operator.mul, Col("l_extendedprice"), Col("l_discount"))
        assert total.aggregations == ((None, AggExpr("sum", product)),)
    else:
        assert type(total) is PolarsScalarAggregateOp
        map_impl = PolarsFusedMapOp if map_fusion else PolarsExpressionMapOp
        assert isinstance(total.inputs[0], map_impl)
    assert not any(isinstance(op, ConversionOp) for op in ops)

    result = SequentialScheduler(ops, split).evaluate()
    pd.testing.assert_frame_equal(result, q6_reference(frame))


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("empty", [False, True])
def test_q6_matches_pandas(selector, empty):
    frame = make_lineitem()
    if empty:
        frame = frame[frame["l_quantity"] >= MAX_QUANTITY]
    with st.config(implementation_selector=selector):
        result = _api.evaluate(build_q6(frame))
    pd.testing.assert_frame_equal(result, q6_reference(frame))


@pytest.mark.parametrize("selector,query,implementation", [
    ("greedy", False, PolarsMaskSelectionOp),
    ("default", False, PandasIndexSelectionOp),
    ("default", True, PandasQuerySelectionOp),
])
def test_q6_filter_bounds_and_backend(selector, query, implementation):
    frame = make_lineitem()
    dag = build_q6_prefix(frame)
    with st.config(implementation_selector=selector):
        ops, split = optimize(dag, OptConfig(pandas_query=query), env=get_data(dag))
    filters = [op for op in ops if isinstance(op, MaskSelectionOp)]
    assert [op.predicate for op in filters] == [
        BinOpExpr(operator.and_,
                  BinOpExpr(operator.ge, Col("l_shipdate"), Const(START)),
                  BinOpExpr(operator.lt, Col("l_shipdate"), Const(END))),
        BinOpExpr(operator.and_,
                  BinOpExpr(operator.ge, Col("l_discount"), Const(MIN_DISCOUNT)),
                  BinOpExpr(operator.le, Col("l_discount"), Const(MAX_DISCOUNT))),
        BinOpExpr(operator.lt, Col("l_quantity"), Const(MAX_QUANTITY)),
    ]
    assert all(isinstance(op, implementation) and len(op.inputs) == 1 for op in filters)
    assert filters[1].inputs == [filters[0]] and filters[2].inputs == [filters[1]]

    result = SequentialScheduler(ops, split).evaluate()
    expected = frame.iloc[:3]
    if selector == "greedy":
        assert isinstance(result, pl.DataFrame) and result.equals(pl.from_pandas(expected))
    else:
        pd.testing.assert_frame_equal(result, expected)
