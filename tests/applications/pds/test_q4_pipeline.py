"""PDS Q4 run through Stratum, checked against the query it was ported from.

The source is ``queries/pandas/q4.py`` in polars-benchmark. :func:`q4_reference`
is that query without the benchmark's IO wrapper; :func:`build_q4` is the same
thing as a skrub DAG. Both run on the same fixtures and must agree.

The query joins line items to orders in a three-month window, keeps lines
committed before they were received, deduplicates on order and priority, and
counts orders per priority.
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
from stratum.optimizer.logical._selection_ops import (
    DropDuplicatesSelectionOp, MaskSelectionOp)
from stratum.optimizer.logical._sort_ops import SortOp
from stratum.optimizer.physical._aggregation_execs import PolarsAggregateOp
from stratum.optimizer.physical._join_execs import PolarsJoinOp
from stratum.optimizer.physical._selection_execs import (
    PolarsDropDuplicatesSelectionOp, PolarsMaskSelectionOp)
from stratum.optimizer.physical._sort_execs import PolarsSortOp
from stratum.frontend._skrub_graph import get_data

# The Q4 window: `var1` and `var2` in the upstream query.
START = date(1993, 7, 1)
END = date(1993, 10, 1)


def make_q4_tables():
    """Arrow-backed orders and lineitem.

    Orders 201, 202, 203 and 207 qualify. Order 201 starts exactly on the
    window start and has two qualifying lines, so its priority would count
    it twice without the deduplication. Every other order fails exactly one
    check: order date before the window, on the end boundary, a line
    committed on or after its receipt, or no matching line or order at all.
    """
    def table(columns, rows):
        return pd.DataFrame(rows, columns=columns).convert_dtypes(dtype_backend="pyarrow")

    orders_ds = table(
        ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate",
         "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"],
        [
            [201, 1, "O", 100.0, date(1993, 7, 1), "1-URGENT", "Clerk A", 0, "On the window start"],
            [202, 1, "O", 100.0, date(1993, 8, 15), "1-URGENT", "Clerk A", 0, "Qualifying"],
            [203, 2, "O", 100.0, date(1993, 9, 30), "2-HIGH", "Clerk B", 0, "Qualifying"],
            [204, 2, "O", 100.0, date(1993, 10, 1), "2-HIGH", "Clerk B", 0, "On the window end"],
            [205, 3, "O", 100.0, date(1993, 6, 30), "3-MEDIUM", "Clerk C", 0, "Before the window"],
            [206, 3, "O", 100.0, date(1993, 8, 1), "3-MEDIUM", "Clerk C", 0, "Committed too late"],
            [207, 1, "O", 100.0, date(1993, 8, 20), "4-NOT SPECIFIED", "Clerk A", 0, "Qualifying"],
            [208, 2, "O", 100.0, date(1993, 8, 20), "5-LOW", "Clerk B", 0, "Committed on receipt"],
            [210, 2, "O", 100.0, date(1993, 8, 20), "2-HIGH", "Clerk B", 0, "No line items"],
        ],
    )
    line_item_ds = table(
        ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity",
         "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus",
         "l_shipdate", "l_commitdate", "l_receiptdate"],
        [
            [201, 1552, 93, 1, 17.0, 100.0, 0.04, 0.02, "N", "O", date(1993, 7, 2), date(1993, 7, 2), date(1993, 7, 5)],
            [201, 673, 28, 2, 36.0, 200.0, 0.09, 0.06, "N", "O", date(1993, 7, 3), date(1993, 7, 3), date(1993, 7, 6)],
            [202, 1062, 75, 1, 8.0, 100.0, 0.02, 0.00, "R", "F", date(1993, 8, 16), date(1993, 8, 16), date(1993, 8, 20)],
            [203, 429, 41, 1, 28.0, 100.0, 0.06, 0.04, "A", "F", date(1993, 10, 1), date(1993, 10, 1), date(1993, 10, 3)],
            [204, 1901, 12, 1, 11.0, 100.0, 0.00, 0.08, "A", "F", date(1993, 10, 2), date(1993, 10, 2), date(1993, 10, 5)],
            [205, 88, 67, 1, 4.0, 100.0, 0.03, 0.01, "N", "O", date(1993, 7, 1), date(1993, 7, 1), date(1993, 7, 4)],
            [206, 1444, 39, 1, 31.0, 100.0, 0.07, 0.03, "R", "F", date(1993, 8, 2), date(1993, 8, 5), date(1993, 8, 5)],
            [207, 203, 81, 1, 6.0, 100.0, 0.01, 0.05, "R", "F", date(1993, 8, 21), date(1993, 8, 21), date(1993, 8, 25)],
            [207, 1777, 54, 2, 19.0, 100.0, 0.05, 0.07, "A", "F", date(1993, 8, 22), date(1993, 8, 28), date(1993, 8, 25)],
            [208, 956, 22, 1, 42.0, 100.0, 0.08, 0.02, "N", "O", date(1993, 8, 21), date(1993, 8, 25), date(1993, 8, 25)],
            [999, 1205, 64, 1, 13.0, 100.0, 0.02, 0.06, "R", "F", date(1993, 8, 21), date(1993, 8, 21), date(1993, 8, 25)],
        ],
    )
    for frame, column in [(orders_ds, "o_totalprice"), (line_item_ds, "l_quantity"),
                          (line_item_ds, "l_extendedprice"),
                          (line_item_ds, "l_discount"), (line_item_ds, "l_tax")]:
        frame[column] = pd.array(frame[column], dtype="double[pyarrow]")
    for frame, column in [(orders_ds, "o_orderdate"), (line_item_ds, "l_shipdate"),
                          (line_item_ds, "l_commitdate"),
                          (line_item_ds, "l_receiptdate")]:
        frame[column] = pd.array(frame[column], dtype="date32[day][pyarrow]")

    return dict(line_item_ds=line_item_ds, orders_ds=orders_ds)


def q4_reference(line_item_ds, orders_ds):
    """The upstream pandas query, with the two tables supplied by the fixture."""
    var1 = START
    var2 = END

    jn = line_item_ds.merge(orders_ds, left_on="l_orderkey", right_on="o_orderkey")

    jn = jn[(jn["o_orderdate"] >= var1) & (jn["o_orderdate"] < var2)]
    jn = jn[jn["l_commitdate"] < jn["l_receiptdate"]]

    jn = jn.drop_duplicates(subset=["o_orderpriority", "l_orderkey"])

    gb = jn.groupby("o_orderpriority", as_index=False)
    agg = gb.agg(order_count=pd.NamedAgg(column="o_orderkey", aggfunc="count"))

    return agg.sort_values(["o_orderpriority"])


def build_q4_prefix(line_item_ds, orders_ds):
    """Q4 up to the aggregation: join, date filters and deduplication.

    ``var1`` and ``var2`` stay as ``skrub.var``s so the values reach the fold
    as ``ValueOp``s rather than inline literals, they are only needed inside
    this stage, so they are not returned.
    """
    var1 = st.var("var1", START)
    var2 = st.var("var2", END)
    line_item_ds = st.var("line_item_ds", line_item_ds)
    orders_ds = st.var("orders_ds", orders_ds)

    jn = line_item_ds.merge(orders_ds, left_on="l_orderkey", right_on="o_orderkey")

    jn = jn[(jn["o_orderdate"] >= var1) & (jn["o_orderdate"] < var2)]
    jn = jn[jn["l_commitdate"] < jn["l_receiptdate"]]

    return jn.drop_duplicates(subset=["o_orderpriority", "l_orderkey"])


def build_q4(line_item_ds, orders_ds):
    """Complete Q4."""
    jn = build_q4_prefix(line_item_ds, orders_ds)

    gb = jn.groupby("o_orderpriority", as_index=False)
    agg = gb.agg(order_count=pd.NamedAgg(column="o_orderkey", aggfunc="count"))

    return agg.sort_values(["o_orderpriority"])


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

def test_q4_matches_reference(polars):
    """The skrubified query returns what the upstream query returns."""
    tables = make_q4_tables()
    expected = q4_reference(**tables)
    result = st._api.evaluate(build_q4(**tables))
    if polars:
        assert isinstance(result, pl.DataFrame)
        result = result.to_pandas()

    assert list(expected.columns) == list(result.columns)
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True),
        result.reset_index(drop=True),
        check_dtype=False,  # compare values across Arrow-backed input and runtime output
    )


def test_q4_prefix_runs_on_polars():
    tables = make_q4_tables()
    expected = tables["line_item_ds"].merge(tables["orders_ds"],
                                            left_on="l_orderkey", right_on="o_orderkey")
    expected = expected[(expected["o_orderdate"] >= START)
                        & (expected["o_orderdate"] < END)]
    expected = expected[expected["l_commitdate"] < expected["l_receiptdate"]]
    expected = expected.drop_duplicates(subset=["o_orderpriority", "l_orderkey"])
    for column in ["o_orderdate", "l_shipdate", "l_commitdate", "l_receiptdate"]:
        expected[column] = pd.to_datetime(expected[column])
    with st.config(implementation_selector="greedy"):
        result = st._api.evaluate(build_q4_prefix(**tables))
    assert isinstance(result, pl.DataFrame)
    # Polars joins may reorder rows, this comparison checks contents, not pandas row order.
    pd.testing.assert_frame_equal(
        expected.sort_values(["l_orderkey", "l_linenumber"]).reset_index(drop=True),
        result.to_pandas().sort_values(["l_orderkey", "l_linenumber"]).reset_index(drop=True),
        check_dtype=False)


# --- the folded plan ------------------------------------------------------

def test_q4_plan_folds_the_join(polars):
    """The merge in the prefix collapses to one join operator."""
    join = _one(_plan(build_q4_prefix(**make_q4_tables())), JoinOp)

    assert ("l_orderkey", "o_orderkey") == (join.left_on, join.right_on)
    if polars:
        assert isinstance(join, PolarsJoinOp)


def test_q4_filters_inline_the_window(polars):
    """The window becomes one combined mask; the late filter compares columns."""
    ops = _plan(build_q4(**make_q4_tables()))
    filters = [o for o in ops if isinstance(o, MaskSelectionOp)]

    assert [BinOpExpr(operator.and_,
                      BinOpExpr(operator.ge, Col("o_orderdate"), Const(START)),
                      BinOpExpr(operator.lt, Col("o_orderdate"), Const(END))),
            BinOpExpr(operator.lt, Col("l_commitdate"), Col("l_receiptdate"))] \
        == [o.predicate for o in filters]
    if polars:
        assert all(isinstance(o, PolarsMaskSelectionOp) for o in filters)


def test_q4_folds_the_deduplication(polars):
    """``drop_duplicates`` on priority and order key folds into one selection."""
    dedup = _one(_plan(build_q4_prefix(**make_q4_tables())),
                 DropDuplicatesSelectionOp)

    assert (["o_orderpriority", "l_orderkey"], "first") == (dedup.subset, dedup.keep)
    if polars:
        assert isinstance(dedup, PolarsDropDuplicatesSelectionOp)


def test_q4_folds_the_named_aggregation(polars):
    """The ``NamedAgg`` count folds into one aggregate and runs end to end."""
    tables = make_q4_tables()
    agg = _one(_plan(build_q4(**tables)), AggregateOp)

    assert (Col("o_orderpriority"),) == agg.grouping
    assert (("order_count", AggExpr("count", Col("o_orderkey"))),) == agg.aggregations
    if polars:
        assert isinstance(agg, PolarsAggregateOp)

    result = st._api.evaluate(build_q4(**tables))
    result = result.to_pandas() if isinstance(result, pl.DataFrame) else result
    pd.testing.assert_frame_equal(q4_reference(**tables).reset_index(drop=True),
                                  result.reset_index(drop=True), check_dtype=False)


def test_q4_folds_the_sort(polars):
    """The final sort by priority folds and closes the plan."""
    ops = _plan(build_q4(**make_q4_tables()))
    sort = _one(ops, SortOp)

    assert (("o_orderpriority",), True) == (sort.by, sort.ascending)
    assert sort.inputs[0] is _one(ops, AggregateOp) and ops[-1] is sort
    if polars:
        assert isinstance(sort, PolarsSortOp)


def test_q4_filter_binds_query_impl_under_flag():
    """``pandas_query`` picks the impl at plan time, both filter correctly.

    Both masks can use the query fast path once the window dates are inlined,
    including the column-versus-column late filter. Pandas-only, so no fixture.
    """
    from stratum.optimizer.physical._selection_execs import (
        PandasIndexSelectionOp, PandasQuerySelectionOp)

    tables = make_q4_tables()
    df = tables["line_item_ds"].merge(tables["orders_ds"],
                                      left_on="l_orderkey", right_on="o_orderkey")
    expected = df[((df["o_orderdate"] >= START) & (df["o_orderdate"] < END))
                  & (df["l_commitdate"] < df["l_receiptdate"])].reset_index(drop=True)

    def by_predicate(ops, predicate):
        matches = [o for o in ops
                   if isinstance(o, MaskSelectionOp) and o.predicate == predicate]
        assert len(matches) == 1, \
            f"expected exactly one selection for {predicate}, got {len(matches)}"
        return matches[0]

    for flag, impl in [(False, PandasIndexSelectionOp), (True, PandasQuerySelectionOp)]:
        dag = build_q4(**tables)
        ops = optimize_(dag, OptConfig(dataframe_ops=True, pandas_query=flag),
                        env=get_data(dag))[0]
        # Look each filter up by predicate, not by position in the plan, so
        # this test does not care what order the optimizer emits them in.
        # They are still applied in query order below.
        window = by_predicate(
            ops, BinOpExpr(operator.and_,
                           BinOpExpr(operator.ge, Col("o_orderdate"), Const(START)),
                           BinOpExpr(operator.lt, Col("o_orderdate"), Const(END))))
        late = by_predicate(
            ops, BinOpExpr(operator.lt, Col("l_commitdate"), Col("l_receiptdate")))
        assert isinstance(window, impl) and isinstance(late, impl)
        result = window.process("fit_transform", [df])
        result = late.process("fit_transform", [result])
        pd.testing.assert_frame_equal(
            expected, pd.DataFrame(result).reset_index(drop=True), check_dtype=False)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
