"""PDS Q10 run through Stratum, checked against the query it was ported from.

The source is ``queries/pandas/q10.py`` in polars-benchmark. :func:`q10_reference`
is that query without the benchmark's IO wrapper; :func:`build_q10` is the same
thing as a skrub DAG. Both run on the same fixtures and must agree.

The query ranks customers by the revenue of their returned items in one
quarter. Three joins feed a ``pd.Timestamp`` window and a return-flag filter;
revenue is summed per customer over seven grouping columns, then projected,
sorted and limited to 20.
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
from stratum.optimizer.logical._join_ops import JoinOp
from stratum.optimizer.logical._map_ops import AssignMapOp
from stratum.optimizer.logical._ops import ValueOp
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp
from stratum.optimizer.logical._selection_ops import MaskSelectionOp, SliceSelectionOp
from stratum.optimizer.logical._sort_ops import SortOp
from stratum.optimizer.physical._aggregation_execs import PolarsAggregateOp
from stratum.optimizer.physical._join_execs import PolarsJoinOp
from stratum.optimizer.physical._map_execs import PolarsAssignMapOp
from stratum.optimizer.physical._projection_execs import PolarsColumnProjectionOp
from stratum.optimizer.physical._selection_execs import (
    PolarsMaskSelectionOp, PolarsSliceSelectionOp)
from stratum.optimizer.physical._sort_execs import PolarsSortOp
from stratum.frontend._skrub_graph import get_data

# The Q10 window: `var1` and `var2` in the upstream query.
START = pd.Timestamp("1993-10-01")
END = pd.Timestamp("1994-01-01")

GROUP_KEYS = ["c_custkey", "c_name", "c_acctbal", "c_phone", "n_name",
              "c_address", "c_comment"]
SELECTED = ["c_custkey", "c_name", "revenue", "c_acctbal", "n_name",
            "c_address", "c_phone", "c_comment"]


def make_q10_tables():
    """Arrow-backed customer, lineitem, nation and orders.

    Three customers qualify, two summing two lines, with revenues out of key
    order so the sort is observable. Order 701 starts exactly on the window
    start. Every other line fails exactly one check: a return flag other than
    "R", or an order date before the window or on its end.

    Dates are Arrow timestamps rather than date32: the upstream query compares
    them with ``pd.Timestamp``, which pandas 3 refuses against a date32 column
    (pandas GH#62157), while the benchmark's pinned pandas 2.3 accepted it.
    """
    def table(columns, rows):
        return pd.DataFrame(rows, columns=columns).convert_dtypes(dtype_backend="pyarrow")

    customer_ds = table(
        ["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone",
         "c_acctbal", "c_mktsegment", "c_comment"],
        [
            [1, "Customer A", "Address A", 3, "13-111-111", 100.0, "BUILDING", "Canadian customer"],
            [2, "Customer B", "Address B", 7, "17-222-222", 200.0, "MACHINERY", "German customer"],
            [3, "Customer C", "Address C", 19, "29-333-333", 300.0, "AUTOMOBILE", "Romanian customer"],
        ],
    )
    lineitem_ds = table(
        ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity",
         "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus",
         "l_shipdate"],
        [
            [701, 1552, 93, 1, 17.0, 1000.0, 0.10, 0.02, "R", "F", date(1993, 10, 5)],
            [701, 673, 28, 2, 36.0, 500.0, 0.00, 0.06, "R", "F", date(1993, 10, 9)],
            [701, 1062, 75, 3, 8.0, 9000.0, 0.00, 0.00, "N", "O", date(1993, 10, 7)],
            [702, 429, 41, 1, 28.0, 3000.0, 0.00, 0.04, "R", "F", date(1993, 11, 20)],
            [702, 1901, 12, 2, 11.0, 1000.0, 0.50, 0.08, "R", "F", date(1993, 11, 22)],
            [703, 88, 67, 1, 4.0, 2000.0, 0.10, 0.01, "R", "F", date(1994, 1, 3)],
            [703, 1444, 39, 2, 31.0, 9000.0, 0.00, 0.03, "A", "F", date(1994, 1, 4)],
            [704, 203, 81, 1, 6.0, 9000.0, 0.00, 0.05, "R", "F", date(1993, 10, 2)],
            [705, 1777, 54, 1, 19.0, 9000.0, 0.00, 0.07, "R", "F", date(1994, 1, 5)],
        ],
    )
    nation_ds = table(
        ["n_nationkey", "n_name", "n_regionkey", "n_comment"],
        [
            [3, "CANADA", 1, "Canada"],
            [7, "GERMANY", 3, "Germany"],
            [19, "ROMANIA", 3, "Romania"],
        ],
    )
    orders_ds = table(
        ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate",
         "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"],
        [
            [701, 1, "F", 100.0, date(1993, 10, 1), "1-URGENT", "Clerk A", 0, "On the window start"],
            [702, 2, "F", 100.0, date(1993, 11, 15), "2-HIGH", "Clerk B", 0, "Qualifying"],
            [703, 3, "F", 100.0, date(1993, 12, 31), "3-MEDIUM", "Clerk C", 0, "Qualifying"],
            [704, 1, "F", 100.0, date(1993, 9, 30), "1-URGENT", "Clerk A", 0, "Before the window"],
            [705, 2, "F", 100.0, date(1994, 1, 1), "2-HIGH", "Clerk B", 0, "On the window end"],
        ],
    )
    for frame, column in [(customer_ds, "c_acctbal"), (lineitem_ds, "l_quantity"),
                          (lineitem_ds, "l_extendedprice"),
                          (lineitem_ds, "l_discount"), (lineitem_ds, "l_tax"),
                          (orders_ds, "o_totalprice")]:
        frame[column] = pd.array(frame[column], dtype="double[pyarrow]")
    for frame, column in [(lineitem_ds, "l_shipdate"), (orders_ds, "o_orderdate")]:
        frame[column] = frame[column].astype("timestamp[us][pyarrow]")

    return dict(customer_ds=customer_ds, lineitem_ds=lineitem_ds,
                nation_ds=nation_ds, orders_ds=orders_ds)


def q10_reference(customer_ds, lineitem_ds, nation_ds, orders_ds):
    """The upstream pandas query, with the four tables supplied by the fixture."""
    var1 = START
    var2 = END

    jn1 = customer_ds.merge(orders_ds, left_on="c_custkey", right_on="o_custkey")
    jn2 = jn1.merge(lineitem_ds, left_on="o_orderkey", right_on="l_orderkey")
    jn3 = jn2.merge(nation_ds, left_on="c_nationkey", right_on="n_nationkey")

    jn3 = jn3[(jn3["o_orderdate"] >= var1) & (jn3["o_orderdate"] < var2)]
    jn3 = jn3[jn3["l_returnflag"] == "R"]

    jn3["revenue"] = jn3["l_extendedprice"] * (1 - jn3["l_discount"])

    gb = jn3.groupby(
        [
            "c_custkey",
            "c_name",
            "c_acctbal",
            "c_phone",
            "n_name",
            "c_address",
            "c_comment",
        ],
        as_index=False,
    )
    agg = gb.agg(revenue=pd.NamedAgg(column="revenue", aggfunc="sum"))

    sel = agg.loc[
        :,
        [
            "c_custkey",
            "c_name",
            "revenue",
            "c_acctbal",
            "n_name",
            "c_address",
            "c_phone",
            "c_comment",
        ],
    ]

    return sel.sort_values("revenue", ascending=False).head(20)


def build_q10_prefix(customer_ds, lineitem_ds, nation_ds, orders_ds):
    """Q10 up to the aggregation: joins, window and return-flag filters, revenue.

    ``var1`` and ``var2`` stay as ``skrub.var``s so the values reach the fold
    as ``ValueOp``s rather than inline literals; they are only needed inside
    this stage, so they are not returned. DataOps are immutable, so the
    ``jn3[...] = ...`` assignment becomes one ``.assign(...)``.
    """
    var1 = st.var("var1", START)
    var2 = st.var("var2", END)
    customer_ds = st.var("customer_ds", customer_ds)
    lineitem_ds = st.var("lineitem_ds", lineitem_ds)
    nation_ds = st.var("nation_ds", nation_ds)
    orders_ds = st.var("orders_ds", orders_ds)

    jn1 = customer_ds.merge(orders_ds, left_on="c_custkey", right_on="o_custkey")
    jn2 = jn1.merge(lineitem_ds, left_on="o_orderkey", right_on="l_orderkey")
    jn3 = jn2.merge(nation_ds, left_on="c_nationkey", right_on="n_nationkey")

    jn3 = jn3[(jn3["o_orderdate"] >= var1) & (jn3["o_orderdate"] < var2)]
    jn3 = jn3[jn3["l_returnflag"] == "R"]

    return jn3.assign(revenue=jn3["l_extendedprice"] * (1 - jn3["l_discount"]))


def build_q10(customer_ds, lineitem_ds, nation_ds, orders_ds):
    """Complete Q10, column selection uses ``[]`` indexing instead of pandas ``loc``."""
    jn3 = build_q10_prefix(customer_ds, lineitem_ds, nation_ds, orders_ds)

    gb = jn3.groupby(
        [
            "c_custkey",
            "c_name",
            "c_acctbal",
            "c_phone",
            "n_name",
            "c_address",
            "c_comment",
        ],
        as_index=False,
    )
    agg = gb.agg(revenue=pd.NamedAgg(column="revenue", aggfunc="sum"))

    sel = agg[
        [
            "c_custkey",
            "c_name",
            "revenue",
            "c_acctbal",
            "n_name",
            "c_address",
            "c_phone",
            "c_comment",
        ]
    ]

    return sel.sort_values("revenue", ascending=False).head(20)


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

def test_q10_matches_reference(polars):
    """The skrubified query returns what the upstream query returns."""
    tables = make_q10_tables()
    expected = q10_reference(**tables)
    result = _api.evaluate(build_q10(**tables))
    if polars:
        assert isinstance(result, pl.DataFrame)
        result = result.to_pandas()

    assert list(expected.columns) == list(result.columns)
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True),
        result.reset_index(drop=True),
        check_dtype=False,  # compare values across Arrow-backed input and runtime output
    )


def test_q10_prefix_runs_on_polars():
    tables = make_q10_tables()
    expected = (
        tables["customer_ds"].merge(tables["orders_ds"],
                                    left_on="c_custkey", right_on="o_custkey")
        .merge(tables["lineitem_ds"], left_on="o_orderkey", right_on="l_orderkey")
        .merge(tables["nation_ds"], left_on="c_nationkey", right_on="n_nationkey")
    )
    expected = expected[(expected["o_orderdate"] >= START)
                        & (expected["o_orderdate"] < END)]
    expected = expected[expected["l_returnflag"] == "R"]
    expected = expected.assign(
        revenue=expected["l_extendedprice"] * (1 - expected["l_discount"]))
    with st.config(implementation_selector="greedy"):
        result = _api.evaluate(build_q10_prefix(**tables))
    assert isinstance(result, pl.DataFrame)
    # Polars joins may reorder rows; this comparison checks contents, not pandas row order.
    pd.testing.assert_frame_equal(
        expected.sort_values(["o_orderkey", "l_linenumber"]).reset_index(drop=True),
        result.to_pandas().sort_values(["o_orderkey", "l_linenumber"]).reset_index(drop=True),
        check_dtype=False)


# --- the folded plan ------------------------------------------------------

def test_q10_plan_folds_joins(polars):
    """Each merge in the prefix collapses to one join operator."""
    ops = _plan(build_q10_prefix(**make_q10_tables()))
    joins = [o for o in ops if isinstance(o, JoinOp)]

    assert [("c_custkey", "o_custkey"), ("o_orderkey", "l_orderkey"),
            ("c_nationkey", "n_nationkey")] \
        == [(o.left_on, o.right_on) for o in joins]
    if polars:
        assert all(isinstance(o, PolarsJoinOp) for o in joins)


def test_q10_filters_inline_the_constants(polars):
    """The ``pd.Timestamp`` window becomes one combined mask with ``Const``
    bounds; the return flag is a literal upstream and stays one."""
    ops = _plan(build_q10(**make_q10_tables()))
    filters = [o for o in ops if isinstance(o, MaskSelectionOp)]

    assert [BinOpExpr(operator.and_,
                      BinOpExpr(operator.ge, Col("o_orderdate"), Const(START)),
                      BinOpExpr(operator.lt, Col("o_orderdate"), Const(END))),
            BinOpExpr(operator.eq, Col("l_returnflag"), Const("R"))] \
        == [o.predicate for o in filters]
    for sel in filters:
        assert len(sel.inputs) == 1, "the filter constants should not be graph inputs"
    assert [] == [o for o in ops if isinstance(o, ValueOp)]
    if polars:
        assert all(isinstance(o, PolarsMaskSelectionOp) for o in filters)


def test_q10_folds_the_revenue(polars):
    """The revenue column collapses to one assign map."""
    amo = _one(_plan(build_q10_prefix(**make_q10_tables())), AssignMapOp)

    assert {"revenue": BinOpExpr(
        operator.mul, Col("l_extendedprice"),
        BinOpExpr(operator.sub, Const(1), Col("l_discount")))} == amo.entries
    assert len(amo.inputs) == 1, "the map should need only the filtered frame"
    if polars:
        assert isinstance(amo, PolarsAssignMapOp)


def test_q10_folds_the_named_aggregation(polars):
    """The ``NamedAgg`` revenue sum over seven keys folds into one aggregate and
    runs end to end."""
    tables = make_q10_tables()
    agg = _one(_plan(build_q10(**tables)), AggregateOp)

    assert tuple(Col(key) for key in GROUP_KEYS) == agg.grouping
    assert (("revenue", AggExpr("sum", Col("revenue"))),) == agg.aggregations
    if polars:
        assert isinstance(agg, PolarsAggregateOp)

    result = _api.evaluate(build_q10(**tables))
    result = result.to_pandas() if isinstance(result, pl.DataFrame) else result
    pd.testing.assert_frame_equal(q10_reference(**tables).reset_index(drop=True),
                                  result.reset_index(drop=True), check_dtype=False)


def test_q10_folds_the_projection_sort_and_limit(polars):
    """Projection, sorting and ``head`` fold into physical operators.

    More than 20 qualifying customers makes the limit observable after sorting.
    """
    tables = make_q10_tables()
    # Repeat the first customer/order/line rows 22 times with distinct keys and
    # prices: 22 one-line customers with distinct revenues, so ``head(20)``
    # drops the two smallest.
    customers = tables["customer_ds"].iloc[[0] * 22].reset_index(drop=True)
    customers["c_custkey"] = pd.array(range(1000, 1022), dtype="int64[pyarrow]")
    orders = tables["orders_ds"].iloc[[0] * 22].reset_index(drop=True)
    orders["o_orderkey"] = pd.array(range(2000, 2022), dtype="int64[pyarrow]")
    orders["o_custkey"] = customers["c_custkey"]
    lineitem = tables["lineitem_ds"].iloc[[0] * 22].reset_index(drop=True)
    lineitem["l_orderkey"] = orders["o_orderkey"]
    lineitem["l_extendedprice"] = pd.array(
        [100.0 + 10.0 * i for i in range(22)], dtype="double[pyarrow]")
    tables.update(customer_ds=customers, orders_ds=orders, lineitem_ds=lineitem)

    ops = _plan(build_q10(**tables))
    projection = _one(ops, ColumnProjectionOp)
    sort = _one(ops, SortOp)
    head = _one(ops, SliceSelectionOp)

    assert SELECTED == projection.key
    assert projection.inputs[0] is _one(ops, AggregateOp)
    assert (("revenue",), False, 20) == (sort.by, sort.ascending, head.n)
    assert sort.inputs[0] is projection and head.inputs[0] is sort
    assert ops[-1] is head
    if polars:
        assert isinstance(projection, PolarsColumnProjectionOp)
        assert isinstance(sort, PolarsSortOp)
        assert isinstance(head, PolarsSliceSelectionOp)

    result = _api.evaluate(build_q10(**tables))
    result = result.to_pandas() if isinstance(result, pl.DataFrame) else result
    pd.testing.assert_frame_equal(q10_reference(**tables).reset_index(drop=True),
                                  result.reset_index(drop=True), check_dtype=False)


def test_q10_filter_binds_query_impl_under_flag():
    """``pandas_query`` picks the impl at plan time; both filter correctly.

    The window bounds are inlined and the return flag is a literal, so both
    masks can use the query fast path. Pandas-only, so no fixture.
    """
    from stratum.optimizer.physical._selection_execs import (
        PandasIndexSelectionOp, PandasQuerySelectionOp)

    tables = make_q10_tables()
    # The unfiltered join in pandas: input frame for running the selections
    # by hand.
    df = (
        tables["customer_ds"].merge(tables["orders_ds"],
                                    left_on="c_custkey", right_on="o_custkey")
        .merge(tables["lineitem_ds"], left_on="o_orderkey", right_on="l_orderkey")
        .merge(tables["nation_ds"], left_on="c_nationkey", right_on="n_nationkey")
    )
    expected = df[((df["o_orderdate"] >= START) & (df["o_orderdate"] < END))
                  & (df["l_returnflag"] == "R")].reset_index(drop=True)

    def by_predicate(ops, predicate):
        matches = [o for o in ops
                   if isinstance(o, MaskSelectionOp) and o.predicate == predicate]
        assert len(matches) == 1, \
            f"expected exactly one selection for {predicate}, got {len(matches)}"
        return matches[0]

    for flag, impl in [(False, PandasIndexSelectionOp), (True, PandasQuerySelectionOp)]:
        dag = build_q10(**tables)
        ops = optimize_(dag, OptConfig(dataframe_ops=True, pandas_query=flag),
                        env=get_data(dag))[0]
        # Look each filter up by predicate, not by position in the plan, so
        # this test does not care what order the optimizer emits them in.
        # They are still applied in query order below.
        window = by_predicate(
            ops, BinOpExpr(operator.and_,
                           BinOpExpr(operator.ge, Col("o_orderdate"), Const(START)),
                           BinOpExpr(operator.lt, Col("o_orderdate"), Const(END))))
        returned = by_predicate(
            ops, BinOpExpr(operator.eq, Col("l_returnflag"), Const("R")))
        assert isinstance(window, impl) and isinstance(returned, impl)
        result = window.process([df])
        result = returned.process([result])
        pd.testing.assert_frame_equal(
            expected, pd.DataFrame(result).reset_index(drop=True), check_dtype=False)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
