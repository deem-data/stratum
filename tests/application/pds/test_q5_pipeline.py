"""PDS Q5 run through Stratum, checked against the query it was ported from.

The source is ``queries/pandas/q5.py`` in polars-benchmark. :func:`q5_reference`
is that query without the benchmark's IO wrapper; :func:`build_q5` is the same
thing as a skrub DAG. Both run on the same fixtures and must agree.

The query joins six tables, the last on a composite key so that a line only
survives when its supplier is in the customer's own nation. It keeps one region
and a one-year order window, sums a computed revenue per nation and sorts by it.
"""
import operator
from datetime import date

import pytest
import pandas as pd
import polars as pl

import stratum as st
from stratum.optimizer._optimize import OptConfig, optimize as optimize_
from stratum.optimizer.logical._aggregation_ops import AggregateOp
from stratum.optimizer.logical._column_expr import AggExpr, BinOpExpr, Col, Const
from stratum.optimizer.logical._join_ops import JoinOp
from stratum.optimizer.logical._map_ops import AssignMapOp
from stratum.optimizer.logical._ops import ValueOp
from stratum.optimizer.logical._selection_ops import MaskSelectionOp
from stratum.optimizer.logical._sort_ops import SortOp
from stratum.optimizer.physical._aggregation_execs import PolarsAggregateOp
from stratum.optimizer.physical._join_execs import PolarsJoinOp
from stratum.optimizer.physical._map_execs import PolarsAssignMapOp
from stratum.optimizer.physical._selection_execs import PolarsMaskSelectionOp
from stratum.optimizer.physical._sort_execs import PolarsSortOp
from stratum.frontend._skrub_graph import get_data

# The Q5 window: `var2` and `var3` in the upstream query.
START = date(1994, 1, 1)
END = date(1995, 1, 1)


def make_q5_tables():
    """Build six Arrow-backed tables for the Q5 joins, filters and aggregation.

    The query joins regions, nations, customers, orders, line items and
    suppliers. The supplier join matches both supplier and nation keys,
    keeping suppliers from the customer's nation. It selects ASIA orders
    placed in 1994, computes revenue as price * (1 - discount) for each line,
    sums revenue by nation and sorts the totals from highest to lowest.

    Two qualifying lines give INDIA revenue of 2800. Two more give JAPAN
    5000, so the result lists JAPAN before INDIA. Extra rows check the date
    boundaries, the region filter and the supplier/nation match.
    """
    def table(columns, rows):
        return pd.DataFrame(rows, columns=columns).convert_dtypes(dtype_backend="pyarrow")

    region_ds = table(
        ["r_regionkey", "r_name", "r_comment"],
        [
            [2, "ASIA", "Asian region"],
            [3, "EUROPE", "European region"],
        ],
    )
    nation_ds = table(
        ["n_nationkey", "n_name", "n_regionkey", "n_comment"],
        [
            [8, "INDIA", 2, "India"],
            [12, "JAPAN", 2, "Japan"],
            [6, "FRANCE", 3, "France"],
        ],
    )
    customer_ds = table(
        ["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone",
         "c_acctbal", "c_mktsegment", "c_comment"],
        [
            [1, "Customer A", "Address A", 8, "18-111-111", 100.0, "BUILDING", "Indian customer"],
            [2, "Customer B", "Address B", 12, "22-222-222", 200.0, "MACHINERY", "Japanese customer"],
            [3, "Customer C", "Address C", 6, "16-333-333", 300.0, "AUTOMOBILE", "French customer"],
        ],
    )
    line_item_ds = table(
        ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity",
         "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus",
         "l_shipdate"],
        [
            [301, 1552, 101, 1, 17.0, 1000.0, 0.10, 0.02, "N", "O", date(1994, 3, 5)],
            [301, 673, 101, 2, 36.0, 2000.0, 0.05, 0.06, "N", "O", date(1994, 3, 9)],
            [301, 1062, 102, 3, 8.0, 9000.0, 0.00, 0.00, "R", "F", date(1994, 3, 7)],
            [302, 429, 102, 1, 28.0, 3000.0, 0.00, 0.04, "A", "F", date(1994, 1, 3)],
            [303, 1901, 102, 1, 11.0, 2500.0, 0.20, 0.08, "A", "F", date(1995, 1, 4)],
            [304, 88, 103, 1, 4.0, 9000.0, 0.00, 0.01, "N", "O", date(1994, 6, 20)],
            [305, 1444, 101, 1, 31.0, 9000.0, 0.00, 0.03, "R", "F", date(1994, 1, 2)],
            [306, 203, 102, 1, 6.0, 9000.0, 0.00, 0.05, "R", "F", date(1995, 1, 3)],
        ],
    )
    orders_ds = table(
        ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate",
         "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"],
        [
            [301, 1, "F", 100.0, date(1994, 3, 1), "1-URGENT", "Clerk A", 0, "Qualifying, one foreign supplier"],
            [302, 2, "F", 100.0, date(1994, 1, 1), "2-HIGH", "Clerk B", 0, "On the window start"],
            [303, 2, "O", 100.0, date(1994, 12, 31), "3-MEDIUM", "Clerk B", 0, "Qualifying"],
            [304, 3, "F", 100.0, date(1994, 6, 15), "1-URGENT", "Clerk C", 0, "Outside the region"],
            [305, 1, "F", 100.0, date(1993, 12, 31), "2-HIGH", "Clerk A", 0, "Before the window"],
            [306, 2, "O", 100.0, date(1995, 1, 1), "4-NOT SPECIFIED", "Clerk B", 0, "On the window end"],
        ],
    )
    supplier_ds = table(
        ["s_suppkey", "s_name", "s_address", "s_nationkey",
         "s_phone", "s_acctbal", "s_comment"],
        [
            [101, "Supplier A", "Mumbai", 8, "18-101-101", 500.0, "Indian supplier"],
            [102, "Supplier B", "Tokyo", 12, "22-102-102", 600.0, "Japanese supplier"],
            [103, "Supplier C", "Paris", 6, "16-103-103", 700.0, "French supplier"],
        ],
    )
    for frame, column in [(customer_ds, "c_acctbal"), (line_item_ds, "l_quantity"),
                          (line_item_ds, "l_extendedprice"),
                          (line_item_ds, "l_discount"), (line_item_ds, "l_tax"),
                          (orders_ds, "o_totalprice"), (supplier_ds, "s_acctbal")]:
        frame[column] = pd.array(frame[column], dtype="double[pyarrow]")
    for frame, column in [(line_item_ds, "l_shipdate"), (orders_ds, "o_orderdate")]:
        frame[column] = pd.array(frame[column], dtype="date32[day][pyarrow]")

    return dict(region_ds=region_ds, nation_ds=nation_ds, customer_ds=customer_ds,
                line_item_ds=line_item_ds, orders_ds=orders_ds,
                supplier_ds=supplier_ds)


def q5_reference(region_ds, nation_ds, customer_ds, line_item_ds, orders_ds,
                 supplier_ds):
    """The upstream pandas query, with the six tables supplied by the fixture."""
    var1 = "ASIA"
    var2 = START
    var3 = END

    jn1 = region_ds.merge(nation_ds, left_on="r_regionkey", right_on="n_regionkey")
    jn2 = jn1.merge(customer_ds, left_on="n_nationkey", right_on="c_nationkey")
    jn3 = jn2.merge(orders_ds, left_on="c_custkey", right_on="o_custkey")
    jn4 = jn3.merge(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
    jn5 = jn4.merge(
        supplier_ds,
        left_on=["l_suppkey", "n_nationkey"],
        right_on=["s_suppkey", "s_nationkey"],
    )

    jn5 = jn5[jn5["r_name"] == var1]
    jn5 = jn5[(jn5["o_orderdate"] >= var2) & (jn5["o_orderdate"] < var3)]
    jn5["revenue"] = jn5.l_extendedprice * (1.0 - jn5.l_discount)

    gb = jn5.groupby("n_name", as_index=False)["revenue"].sum()
    return gb.sort_values("revenue", ascending=False)


def build_q5_prefix(region_ds, nation_ds, customer_ds, line_item_ds, orders_ds,
                    supplier_ds):
    """Q5 up to the aggregation: five joins, region and date filters, revenue.

    ``var1`` to ``var3`` stay as ``skrub.var``s so the values reach the fold as
    ``ValueOp``s rather than inline literals; they are only needed inside this
    stage, so they are not returned. DataOps are immutable, so the
    ``jn5[...] = ...`` assignment becomes one ``.assign(...)``. The upstream
    attribute column reads become ``jn5["..."]`` indexing.
    """
    var1 = st.var("var1", "ASIA")
    var2 = st.var("var2", START)
    var3 = st.var("var3", END)
    region_ds = st.var("region_ds", region_ds)
    nation_ds = st.var("nation_ds", nation_ds)
    customer_ds = st.var("customer_ds", customer_ds)
    line_item_ds = st.var("line_item_ds", line_item_ds)
    orders_ds = st.var("orders_ds", orders_ds)
    supplier_ds = st.var("supplier_ds", supplier_ds)

    jn1 = region_ds.merge(nation_ds, left_on="r_regionkey", right_on="n_regionkey")
    jn2 = jn1.merge(customer_ds, left_on="n_nationkey", right_on="c_nationkey")
    jn3 = jn2.merge(orders_ds, left_on="c_custkey", right_on="o_custkey")
    jn4 = jn3.merge(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
    jn5 = jn4.merge(
        supplier_ds,
        left_on=["l_suppkey", "n_nationkey"],
        right_on=["s_suppkey", "s_nationkey"],
    )

    jn5 = jn5[jn5["r_name"] == var1]
    jn5 = jn5[(jn5["o_orderdate"] >= var2) & (jn5["o_orderdate"] < var3)]
    return jn5.assign(revenue=jn5["l_extendedprice"] * (1.0 - jn5["l_discount"]))


def build_q5(region_ds, nation_ds, customer_ds, line_item_ds, orders_ds, supplier_ds):
    """Complete Q5."""
    jn5 = build_q5_prefix(region_ds, nation_ds, customer_ds, line_item_ds,
                          orders_ds, supplier_ds)

    gb = jn5.groupby("n_name", as_index=False)["revenue"].sum()
    return gb.sort_values("revenue", ascending=False)


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

def test_q5_matches_reference(polars):
    """The skrubified query returns what the upstream query returns."""
    tables = make_q5_tables()
    expected = q5_reference(**tables)
    result = st._api.evaluate(build_q5(**tables))
    if polars:
        assert isinstance(result, pl.DataFrame)
        result = result.to_pandas()

    assert list(expected.columns) == list(result.columns)
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True),
        result.reset_index(drop=True),
        check_dtype=False,  # compare values across Arrow-backed input and runtime output
    )


def test_q5_prefix_runs_on_polars():
    tables = make_q5_tables()
    expected = (
        tables["region_ds"].merge(tables["nation_ds"],
                                  left_on="r_regionkey", right_on="n_regionkey")
        .merge(tables["customer_ds"], left_on="n_nationkey", right_on="c_nationkey")
        .merge(tables["orders_ds"], left_on="c_custkey", right_on="o_custkey")
        .merge(tables["line_item_ds"], left_on="o_orderkey", right_on="l_orderkey")
        .merge(tables["supplier_ds"], left_on=["l_suppkey", "n_nationkey"],
               right_on=["s_suppkey", "s_nationkey"])
    )
    expected = expected[expected["r_name"] == "ASIA"]
    expected = expected[(expected["o_orderdate"] >= START)
                        & (expected["o_orderdate"] < END)]
    expected = expected.assign(
        revenue=expected["l_extendedprice"] * (1.0 - expected["l_discount"]))
    for column in ["o_orderdate", "l_shipdate"]:
        expected[column] = pd.to_datetime(expected[column])
    with st.config(implementation_selector="greedy"):
        result = st._api.evaluate(build_q5_prefix(**tables))
    assert isinstance(result, pl.DataFrame)
    # Polars joins may reorder rows; this comparison checks contents, not pandas row order.
    pd.testing.assert_frame_equal(
        expected.sort_values(["o_orderkey", "l_linenumber"]).reset_index(drop=True),
        result.to_pandas().sort_values(["o_orderkey", "l_linenumber"]).reset_index(drop=True),
        check_dtype=False)


# --- the folded plan ------------------------------------------------------

def test_q5_plan_folds_joins(polars):
    """Each merge in the prefix collapses to one join operator, the composite
    supplier key included."""
    ops = _plan(build_q5_prefix(**make_q5_tables()))
    joins = [o for o in ops if isinstance(o, JoinOp)]

    assert [("r_regionkey", "n_regionkey"), ("n_nationkey", "c_nationkey"),
            ("c_custkey", "o_custkey"), ("o_orderkey", "l_orderkey"),
            (["l_suppkey", "n_nationkey"], ["s_suppkey", "s_nationkey"])] \
        == [(o.left_on, o.right_on) for o in joins]
    if polars:
        assert all(isinstance(o, PolarsJoinOp) for o in joins)


def test_q5_filters_inline_the_constants(polars):
    """The region becomes a ``Const``, the window becomes one combined mask."""
    ops = _plan(build_q5(**make_q5_tables()))
    filters = [o for o in ops if isinstance(o, MaskSelectionOp)]

    assert [BinOpExpr(operator.eq, Col("r_name"), Const("ASIA")),
            BinOpExpr(operator.and_,
                      BinOpExpr(operator.ge, Col("o_orderdate"), Const(START)),
                      BinOpExpr(operator.lt, Col("o_orderdate"), Const(END)))] \
        == [o.predicate for o in filters]
    for sel in filters:
        assert len(sel.inputs) == 1, "the filter constants should not be graph inputs"
    assert [] == [o for o in ops if isinstance(o, ValueOp)]
    if polars:
        assert all(isinstance(o, PolarsMaskSelectionOp) for o in filters)


def test_q5_folds_the_revenue(polars):
    """The revenue column collapses to one assign map over plain columns."""
    amo = _one(_plan(build_q5_prefix(**make_q5_tables())), AssignMapOp)

    assert {"revenue": BinOpExpr(
        operator.mul, Col("l_extendedprice"),
        BinOpExpr(operator.sub, Const(1.0), Col("l_discount")))} == amo.entries
    assert len(amo.inputs) == 1, "the map should need only the filtered frame"
    if polars:
        assert isinstance(amo, PolarsAssignMapOp)


def test_q5_folds_the_grouped_aggregation(polars):
    """The grouped revenue sum folds into one aggregate and runs end to end."""
    tables = make_q5_tables()
    agg = _one(_plan(build_q5(**tables)), AggregateOp)

    assert (Col("n_name"),) == agg.grouping
    assert ((None, AggExpr("sum", Col("revenue"))),) == agg.aggregations
    if polars:
        assert isinstance(agg, PolarsAggregateOp)

    result = st._api.evaluate(build_q5(**tables))
    result = result.to_pandas() if isinstance(result, pl.DataFrame) else result
    pd.testing.assert_frame_equal(q5_reference(**tables).reset_index(drop=True),
                                  result.reset_index(drop=True), check_dtype=False)


def test_q5_folds_the_sort(polars):
    """The descending revenue sort folds and closes the plan."""
    ops = _plan(build_q5(**make_q5_tables()))
    sort = _one(ops, SortOp)

    assert (("revenue",), False) == (sort.by, sort.ascending)
    assert sort.inputs[0] is _one(ops, AggregateOp) and ops[-1] is sort
    if polars:
        assert isinstance(sort, PolarsSortOp)


def test_q5_filter_binds_query_impl_under_flag():
    """``pandas_query`` picks the impl at plan time, both filter correctly.

    Region and window both use the query fast path once their values are
    inlined. Pandas-only, so no fixture.
    """
    from stratum.optimizer.physical._selection_execs import (
        PandasIndexSelectionOp, PandasQuerySelectionOp)

    tables = make_q5_tables()
    # The unfiltered join in pandas: input frame for running the selections
    # by hand.
    df = (
        tables["region_ds"].merge(tables["nation_ds"],
                                  left_on="r_regionkey", right_on="n_regionkey")
        .merge(tables["customer_ds"], left_on="n_nationkey", right_on="c_nationkey")
        .merge(tables["orders_ds"], left_on="c_custkey", right_on="o_custkey")
        .merge(tables["line_item_ds"], left_on="o_orderkey", right_on="l_orderkey")
        .merge(tables["supplier_ds"], left_on=["l_suppkey", "n_nationkey"],
               right_on=["s_suppkey", "s_nationkey"])
    )
    expected = df[(df["r_name"] == "ASIA")
                  & ((df["o_orderdate"] >= START) & (df["o_orderdate"] < END))
                  ].reset_index(drop=True)

    def by_predicate(ops, predicate):
        matches = [o for o in ops
                   if isinstance(o, MaskSelectionOp) and o.predicate == predicate]
        assert len(matches) == 1, \
            f"expected exactly one selection for {predicate}, got {len(matches)}"
        return matches[0]

    for flag, impl in [(False, PandasIndexSelectionOp), (True, PandasQuerySelectionOp)]:
        dag = build_q5(**tables)
        ops = optimize_(dag, OptConfig(dataframe_ops=True, pandas_query=flag),
                        env=get_data(dag))[0]
        # Look each filter up by predicate, not by position in the plan, so
        # this test does not care what order the optimizer emits them in.
        # They are still applied in query order below.
        region = by_predicate(
            ops, BinOpExpr(operator.eq, Col("r_name"), Const("ASIA")))
        window = by_predicate(
            ops, BinOpExpr(operator.and_,
                           BinOpExpr(operator.ge, Col("o_orderdate"), Const(START)),
                           BinOpExpr(operator.lt, Col("o_orderdate"), Const(END))))
        assert isinstance(region, impl) and isinstance(window, impl)
        result = region.process("fit_transform", [df])
        result = window.process("fit_transform", [result])
        pd.testing.assert_frame_equal(
            expected, pd.DataFrame(result).reset_index(drop=True), check_dtype=False)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
