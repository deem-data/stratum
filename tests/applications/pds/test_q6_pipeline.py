"""PDS Q6 run through Stratum, checked against the query it was ported from.

The source is ``queries/pandas/q6.py`` in polars-benchmark. :func:`q6_reference`
is that query without the benchmark's IO wrapper, and :func:`build_q6` is the same
thing as a skrub DAG. Both run on the same fixture and must agree.

A single-table query: three range filters on ship date, discount and quantity,
then the total of price times discount as a one-row frame. The total folds
into a scalar aggregate, so the whole plan up to the final frame constructor
is frame operators.
"""
import operator
from datetime import date

import pytest
import pandas as pd
import polars as pl

import stratum as st
from stratum.frontend import _api
from stratum.optimizer._optimize import OptConfig, optimize as optimize_
from stratum.optimizer.logical._aggregation_ops import AggregateOp
from stratum.optimizer.logical._column_expr import AggExpr, BinOpExpr, Col, Const
from stratum.optimizer.logical._numeric_ops import NumericOp
from stratum.optimizer.logical._ops import OutputType, ValueOp
from stratum.optimizer.logical._selection_ops import MaskSelectionOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._conversion import ConversionOp
from stratum.optimizer.physical._selection_execs import PolarsMaskSelectionOp
from stratum.optimizer.physical._aggregation_execs import PolarsAggregateOp
from stratum.frontend._skrub_graph import get_data

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


def _plan(dag):
    """The plan ``evaluate`` would run. Without ``env`` the vars stay ``VariableOp``s,
    the frame is never recognised as a source, and nothing folds."""
    return optimize_(dag, OptConfig(dataframe_ops=True), env=get_data(dag))[0]


def _one(ops, cls):
    matches = [o for o in ops if isinstance(o, cls)]
    assert len(matches) == 1, f"expected exactly one {cls.__name__}, got {len(matches)}"
    return matches[0]


@pytest.fixture(params=[False, True], ids=["pandas", "polars"])
def polars(request):
    with st.config(implementation_selector="greedy" if request.param else "default"):
        yield request.param


# --- parity ----------------------------------------------------------------

def test_q6_matches_reference(polars):
    """The scalar revenue matches under both selectors, including pandas fallback.

    The final deferred constructor returns a pandas frame in either case.
    """
    df = make_lineitem()
    expected = q6_reference(df)
    result = _api.evaluate(build_q6(df))

    assert isinstance(result, pd.DataFrame)
    assert list(expected.columns) == list(result.columns)
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True),
        result.reset_index(drop=True),
        check_dtype=False,  # compare values across Arrow-backed input and runtime output
    )


def test_q6_prefix_runs_on_polars():
    df = make_lineitem()
    expected = df[(df["l_shipdate"] >= START) & (df["l_shipdate"] < END)]
    expected = expected[(expected["l_discount"] >= MIN_DISCOUNT)
                        & (expected["l_discount"] <= MAX_DISCOUNT)]
    expected = expected[expected["l_quantity"] < MAX_QUANTITY]
    expected["l_shipdate"] = pd.to_datetime(expected["l_shipdate"])
    with st.config(implementation_selector="greedy"):
        result = _api.evaluate(build_q6_prefix(df))
    assert isinstance(result, pl.DataFrame)
    pd.testing.assert_frame_equal(expected.reset_index(drop=True),
                                  result.to_pandas().reset_index(drop=True),
                                  check_dtype=False)


# --- the folded plan ------------------------------------------------------

def test_q6_filters_inline_the_constants(polars):
    """Each range filter collapses to one mask with its bounds as ``Const``s.

    Every bound is used once, so each fold absorbs it and the frame is the
    selection's only input.
    """
    ops = _plan(build_q6_prefix(make_lineitem()))
    filters = [o for o in ops if isinstance(o, MaskSelectionOp)]

    assert [BinOpExpr(operator.and_,
                      BinOpExpr(operator.ge, Col("l_shipdate"), Const(START)),
                      BinOpExpr(operator.lt, Col("l_shipdate"), Const(END))),
            BinOpExpr(operator.and_,
                      BinOpExpr(operator.ge, Col("l_discount"), Const(MIN_DISCOUNT)),
                      BinOpExpr(operator.le, Col("l_discount"), Const(MAX_DISCOUNT))),
            BinOpExpr(operator.lt, Col("l_quantity"), Const(MAX_QUANTITY))] \
        == [o.predicate for o in filters]
    assert filters[1].inputs[0] is filters[0] and filters[2].inputs[0] is filters[1]
    for sel in filters:
        assert len(sel.inputs) == 1, "the bounds should not be graph inputs"
        assert isinstance(sel, PhysicalOp), \
            f"{type(sel).__name__} was not bound to a physical implementation"
        if polars:
            assert isinstance(sel, PolarsMaskSelectionOp)
    assert [] == [o for o in ops if isinstance(o, ValueOp)]


def test_q6_folds_the_revenue(polars):
    """The revenue total folds into one scalar aggregate over the last filter.

    ``(price * discount).sum()`` used to stay a ``NumericOp`` and an opaque pandas
    ``sum``, which pulled every filter onto pandas under the greedy selector.
    """
    df = make_lineitem()
    ops = _plan(build_q6(df))
    agg = _one(ops, AggregateOp)
    filters = [o for o in ops if isinstance(o, MaskSelectionOp)]

    assert not agg.grouped and OutputType.SCALAR is agg.output_type
    assert ((None, AggExpr("sum", BinOpExpr(operator.mul, Col("l_extendedprice"),
                                            Col("l_discount")))),) == agg.aggregations
    assert len(agg.inputs) == 1 and agg.inputs[0] is filters[-1], \
        "the total should read only the filtered frame"
    assert [] == [o for o in ops if isinstance(o, (NumericOp, ConversionOp))]
    if polars:
        assert isinstance(agg, PolarsAggregateOp)

    result = _api.evaluate(build_q6(df))
    pd.testing.assert_frame_equal(q6_reference(df).reset_index(drop=True),
                                  result.reset_index(drop=True), check_dtype=False)


def test_q6_filter_binds_query_impl_under_flag():
    """``pandas_query`` picks the impl at plan time, both filter correctly.

    All three ranges use the query fast path once their bounds are inlined.
    Pandas-only, so no fixture.
    """
    from stratum.optimizer.physical._selection_execs import (
        PandasIndexSelectionOp, PandasQuerySelectionOp)

    df = make_lineitem()
    expected = df[((df["l_shipdate"] >= START) & (df["l_shipdate"] < END))
                  & ((df["l_discount"] >= MIN_DISCOUNT)
                     & (df["l_discount"] <= MAX_DISCOUNT))
                  & (df["l_quantity"] < MAX_QUANTITY)].reset_index(drop=True)

    for flag, impl in [(False, PandasIndexSelectionOp), (True, PandasQuerySelectionOp)]:
        dag = build_q6(df)
        ops = optimize_(dag, OptConfig(dataframe_ops=True, pandas_query=flag),
                        env=get_data(dag))[0]
        filters = [o for o in ops if isinstance(o, MaskSelectionOp)]
        assert len(filters) == 3
        result = df
        for sel in filters:
            assert isinstance(sel, impl)
            result = sel.process([result])
        pd.testing.assert_frame_equal(
            expected, pd.DataFrame(result).reset_index(drop=True), check_dtype=False)


def test_q6_full_query_keeps_filters_on_polars():
    """The filters bind as they do in the prefix once the total reads them."""
    with st.config(implementation_selector="greedy"):
        ops = _plan(build_q6(make_lineitem()))
    filters = [o for o in ops if isinstance(o, MaskSelectionOp)]

    assert len(filters) == 3
    assert all(isinstance(o, PolarsMaskSelectionOp) for o in filters)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
