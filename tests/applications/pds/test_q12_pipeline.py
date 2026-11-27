"""PDS Q12 run through Stratum, checked against the query it was ported from.

The source is ``queries/pandas/q12.py`` in polars-benchmark. :func:`q12_reference`
is that query without the benchmark's IO wrapper; :func:`build_q12` is the same
thing as a skrub DAG. Both run on the same fixtures and must agree.

The query counts late line items by ship mode and order priority. A join feeds
an ``isin`` filter on two ship modes, two date-ordering filters that compare
columns, and a ``pd.Timestamp`` receipt window. Two boolean columns flag high
and low priority, one an ``isin`` and the other its negation, and are summed
per ship mode.
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
from stratum.optimizer.logical._column_expr import (
    AggExpr, BinOpExpr, Col, Const, IsinExpr, UnaryOpExpr)
from stratum.optimizer.logical._join_ops import FilteringJoinOp, JoinOp
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

# The Q12 receipt window: `var3` and `var4` in the upstream query.
START = pd.Timestamp("1994-01-01")
END = pd.Timestamp("1995-01-01")

SHIP_MODES = ["MAIL", "SHIP"]
HIGH_PRIORITIES = ["1-URGENT", "2-HIGH"]


def make_q12_tables():
    """Arrow-backed lineitem and orders.

    MAIL and SHIP each have three late lines, MAIL two of them high priority
    and SHIP one, so both counts are non-trivial in both groups. Receipts on
    the window start and just before its end qualify. Every other line fails
    exactly one check: another ship mode, a commit on the receipt date, a ship
    on the commit date, or a receipt before the window or on its end.

    Dates are Arrow timestamps rather than date32: the upstream query compares
    them with ``pd.Timestamp``, which pandas 3 refuses against a date32 column
    (pandas GH#62157), while the benchmark's pinned pandas 2.3 accepted it.
    """
    def table(columns, rows):
        return pd.DataFrame(rows, columns=columns).convert_dtypes(dtype_backend="pyarrow")

    lineitem_ds = table(
        ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity",
         "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus",
         "l_shipdate", "l_commitdate", "l_receiptdate", "l_shipmode"],
        [
            [801, 1552, 93, 1, 17.0, 100.0, 0.04, 0.02, "N", "O", date(1994, 3, 1), date(1994, 3, 5), date(1994, 3, 9), "MAIL"],
            [802, 673, 28, 1, 36.0, 100.0, 0.09, 0.06, "N", "O", date(1994, 4, 1), date(1994, 4, 5), date(1994, 4, 9), "MAIL"],
            [803, 1062, 75, 1, 8.0, 100.0, 0.02, 0.00, "R", "F", date(1993, 12, 20), date(1993, 12, 25), date(1994, 1, 1), "MAIL"],
            [804, 429, 41, 1, 28.0, 100.0, 0.06, 0.04, "A", "F", date(1994, 5, 1), date(1994, 5, 5), date(1994, 5, 9), "SHIP"],
            [801, 1901, 12, 2, 11.0, 100.0, 0.00, 0.08, "A", "F", date(1994, 12, 20), date(1994, 12, 25), date(1994, 12, 31), "SHIP"],
            [805, 88, 67, 1, 4.0, 100.0, 0.03, 0.01, "N", "O", date(1994, 6, 1), date(1994, 6, 5), date(1994, 6, 9), "SHIP"],
            [801, 1444, 39, 3, 31.0, 100.0, 0.07, 0.03, "R", "F", date(1994, 7, 1), date(1994, 7, 5), date(1994, 7, 9), "AIR"],
            [802, 203, 81, 2, 6.0, 100.0, 0.01, 0.05, "R", "F", date(1994, 8, 1), date(1994, 8, 9), date(1994, 8, 9), "MAIL"],
            [802, 1777, 54, 3, 19.0, 100.0, 0.05, 0.07, "A", "F", date(1994, 9, 5), date(1994, 9, 5), date(1994, 9, 9), "MAIL"],
            [804, 956, 22, 2, 42.0, 100.0, 0.08, 0.02, "N", "O", date(1993, 12, 20), date(1993, 12, 25), date(1993, 12, 31), "SHIP"],
            [805, 1205, 64, 2, 13.0, 100.0, 0.02, 0.06, "R", "F", date(1994, 12, 20), date(1994, 12, 25), date(1995, 1, 1), "SHIP"],
        ],
    )
    orders_ds = table(
        ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate",
         "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"],
        [
            [801, 1, "O", 100.0, date(1994, 2, 1), "1-URGENT", "Clerk A", 0, "High priority"],
            [802, 2, "O", 100.0, date(1994, 3, 1), "3-MEDIUM", "Clerk B", 0, "Low priority"],
            [803, 1, "F", 100.0, date(1993, 12, 1), "2-HIGH", "Clerk A", 0, "High priority"],
            [804, 2, "O", 100.0, date(1994, 4, 1), "5-LOW", "Clerk B", 0, "Low priority"],
            [805, 3, "O", 100.0, date(1994, 5, 1), "4-NOT SPECIFIED", "Clerk C", 0, "Low priority"],
        ],
    )
    for frame, column in [(lineitem_ds, "l_quantity"), (lineitem_ds, "l_extendedprice"),
                          (lineitem_ds, "l_discount"), (lineitem_ds, "l_tax"),
                          (orders_ds, "o_totalprice")]:
        frame[column] = pd.array(frame[column], dtype="double[pyarrow]")
    for frame, column in [(lineitem_ds, "l_shipdate"), (lineitem_ds, "l_commitdate"),
                          (lineitem_ds, "l_receiptdate"), (orders_ds, "o_orderdate")]:
        frame[column] = frame[column].astype("timestamp[us][pyarrow]")

    return dict(lineitem_ds=lineitem_ds, orders_ds=orders_ds)


def q12_reference(lineitem_ds, orders_ds):
    """The upstream pandas query, with the two tables supplied by the fixture."""
    var1 = "MAIL"
    var2 = "SHIP"
    var3 = START
    var4 = END

    jn = orders_ds.merge(lineitem_ds, left_on="o_orderkey", right_on="l_orderkey")

    jn = jn[jn["l_shipmode"].isin([var1, var2])]
    jn = jn[jn["l_commitdate"] < jn["l_receiptdate"]]
    jn = jn[jn["l_shipdate"] < jn["l_commitdate"]]
    jn = jn[(jn["l_receiptdate"] >= var3) & (jn["l_receiptdate"] < var4)]

    jn["high_line_count"] = jn["o_orderpriority"].isin(["1-URGENT", "2-HIGH"])
    jn["low_line_count"] = ~jn["o_orderpriority"].isin(["1-URGENT", "2-HIGH"])

    gb = jn.groupby("l_shipmode", as_index=False)
    agg = gb.agg(
        high_line_count=pd.NamedAgg(column="high_line_count", aggfunc="sum"),
        low_line_count=pd.NamedAgg(column="low_line_count", aggfunc="sum"),
    )

    return agg.sort_values("l_shipmode")


def build_q12_prefix(lineitem_ds, orders_ds):
    """Q12 up to the aggregation: the join, four filters and the two flags.

    ``var1`` to ``var4`` stay as ``skrub.var``s so the values reach the fold as
    ``ValueOp``s rather than inline literals; they are only needed inside this
    stage, so they are not returned. DataOps are immutable, so the two
    ``jn[...] = ...`` assignments become one ``.assign(...)``.
    """
    var1 = st.var("var1", "MAIL")
    var2 = st.var("var2", "SHIP")
    var3 = st.var("var3", START)
    var4 = st.var("var4", END)
    lineitem_ds = st.var("lineitem_ds", lineitem_ds)
    orders_ds = st.var("orders_ds", orders_ds)

    jn = orders_ds.merge(lineitem_ds, left_on="o_orderkey", right_on="l_orderkey")

    jn = jn[jn["l_shipmode"].isin([var1, var2])]
    jn = jn[jn["l_commitdate"] < jn["l_receiptdate"]]
    jn = jn[jn["l_shipdate"] < jn["l_commitdate"]]
    jn = jn[(jn["l_receiptdate"] >= var3) & (jn["l_receiptdate"] < var4)]

    return jn.assign(
        high_line_count=jn["o_orderpriority"].isin(["1-URGENT", "2-HIGH"]),
        low_line_count=~jn["o_orderpriority"].isin(["1-URGENT", "2-HIGH"]),
    )


def build_q12(lineitem_ds, orders_ds):
    """Complete Q12."""
    jn = build_q12_prefix(lineitem_ds, orders_ds)

    gb = jn.groupby("l_shipmode", as_index=False)
    agg = gb.agg(
        high_line_count=pd.NamedAgg(column="high_line_count", aggfunc="sum"),
        low_line_count=pd.NamedAgg(column="low_line_count", aggfunc="sum"),
    )

    return agg.sort_values("l_shipmode")


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

def test_q12_matches_reference(polars):
    """The skrubified query returns what the upstream query returns."""
    tables = make_q12_tables()
    expected = q12_reference(**tables)
    result = _api.evaluate(build_q12(**tables))
    if polars:
        assert isinstance(result, pl.DataFrame)
        result = result.to_pandas()

    assert list(expected.columns) == list(result.columns)
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True),
        result.reset_index(drop=True),
        check_dtype=False,  # compare values across Arrow-backed input and runtime output
    )


def test_q12_prefix_runs_on_polars():
    tables = make_q12_tables()
    expected = tables["orders_ds"].merge(tables["lineitem_ds"],
                                         left_on="o_orderkey", right_on="l_orderkey")
    expected = expected[expected["l_shipmode"].isin(SHIP_MODES)]
    expected = expected[expected["l_commitdate"] < expected["l_receiptdate"]]
    expected = expected[expected["l_shipdate"] < expected["l_commitdate"]]
    expected = expected[(expected["l_receiptdate"] >= START)
                        & (expected["l_receiptdate"] < END)]
    expected = expected.assign(
        high_line_count=expected["o_orderpriority"].isin(HIGH_PRIORITIES),
        low_line_count=~expected["o_orderpriority"].isin(HIGH_PRIORITIES))
    with st.config(implementation_selector="greedy"):
        result = _api.evaluate(build_q12_prefix(**tables))
    assert isinstance(result, pl.DataFrame)
    # Polars joins may reorder rows; this comparison checks contents, not pandas row order.
    pd.testing.assert_frame_equal(
        expected.sort_values(["l_orderkey", "l_linenumber"]).reset_index(drop=True),
        result.to_pandas().sort_values(["l_orderkey", "l_linenumber"]).reset_index(drop=True),
        check_dtype=False)


# --- the folded plan ------------------------------------------------------

def test_q12_plan_folds_the_join(polars):
    """The merge in the prefix collapses to one join operator."""
    join = _one(_plan(build_q12_prefix(**make_q12_tables())), JoinOp)

    assert ("o_orderkey", "l_orderkey") == (join.left_on, join.right_on)
    if polars:
        assert isinstance(join, PolarsJoinOp)


def test_q12_filters_inline_the_constants(polars):
    """The two ship modes are inlined into the ``isin`` list, the receipt window
    into ``Const`` bounds, and the date-ordering filters compare columns.

    A literal ``isin`` list is a predicate, not a relation, so it stays a mask
    rather than becoming a filtering join.
    """
    ops = _plan(build_q12(**make_q12_tables()))
    filters = [o for o in ops if isinstance(o, MaskSelectionOp)]

    assert [IsinExpr(Col("l_shipmode"), SHIP_MODES),
            BinOpExpr(operator.lt, Col("l_commitdate"), Col("l_receiptdate")),
            BinOpExpr(operator.lt, Col("l_shipdate"), Col("l_commitdate")),
            BinOpExpr(operator.and_,
                      BinOpExpr(operator.ge, Col("l_receiptdate"), Const(START)),
                      BinOpExpr(operator.lt, Col("l_receiptdate"), Const(END)))] \
        == [o.predicate for o in filters]
    for sel in filters:
        assert len(sel.inputs) == 1, "the filter constants should not be graph inputs"
    assert [] == [o for o in ops if isinstance(o, (ValueOp, FilteringJoinOp))]
    if polars:
        assert all(isinstance(o, PolarsMaskSelectionOp) for o in filters)


def test_q12_folds_the_priority_flags(polars):
    """Both flags collapse to one assign map: an ``isin`` and its negation."""
    amo = _one(_plan(build_q12_prefix(**make_q12_tables())), AssignMapOp)

    high = IsinExpr(Col("o_orderpriority"), HIGH_PRIORITIES)
    assert {"high_line_count": high,
            "low_line_count": UnaryOpExpr(operator.invert, high)} == amo.entries
    assert len(amo.inputs) == 1, "the map should need only the filtered frame"
    if polars:
        assert isinstance(amo, PolarsAssignMapOp)


def test_q12_folds_the_named_aggregation(polars):
    """Both ``NamedAgg`` flag sums fold into one aggregate, and the query runs
    end to end."""
    tables = make_q12_tables()
    agg = _one(_plan(build_q12(**tables)), AggregateOp)

    assert (Col("l_shipmode"),) == agg.grouping
    assert (("high_line_count", AggExpr("sum", Col("high_line_count"))),
            ("low_line_count", AggExpr("sum", Col("low_line_count")))) \
        == agg.aggregations
    if polars:
        assert isinstance(agg, PolarsAggregateOp)

    result = _api.evaluate(build_q12(**tables))
    result = result.to_pandas() if isinstance(result, pl.DataFrame) else result
    pd.testing.assert_frame_equal(q12_reference(**tables).reset_index(drop=True),
                                  result.reset_index(drop=True), check_dtype=False)


def test_q12_folds_the_sort(polars):
    """The final sort by ship mode folds and closes the plan."""
    ops = _plan(build_q12(**make_q12_tables()))
    sort = _one(ops, SortOp)

    assert (("l_shipmode",), True) == (sort.by, sort.ascending)
    assert sort.inputs[0] is _one(ops, AggregateOp) and ops[-1] is sort
    if polars:
        assert isinstance(sort, PolarsSortOp)


def test_q12_filter_binds_query_impl_under_flag():
    """``pandas_query`` picks the impl at plan time; both filter correctly.

    The date comparisons, column against column or against inlined bounds, can
    use the query fast path. ``isin`` is a column method with no query
    spelling, so the ship-mode filter stays a mask selection. Pandas-only, so
    no fixture.
    """
    from stratum.optimizer.physical._selection_execs import (
        PandasIndexSelectionOp, PandasQuerySelectionOp)

    tables = make_q12_tables()
    df = tables["orders_ds"].merge(tables["lineitem_ds"],
                                   left_on="o_orderkey", right_on="l_orderkey")
    expected = df[df["l_shipmode"].isin(SHIP_MODES)
                  & (df["l_commitdate"] < df["l_receiptdate"])
                  & (df["l_shipdate"] < df["l_commitdate"])
                  & ((df["l_receiptdate"] >= START) & (df["l_receiptdate"] < END))
                  ].reset_index(drop=True)

    def by_predicate(ops, predicate):
        matches = [o for o in ops
                   if isinstance(o, MaskSelectionOp) and o.predicate == predicate]
        assert len(matches) == 1, \
            f"expected exactly one selection for {predicate}, got {len(matches)}"
        return matches[0]

    for flag, impl in [(False, PandasIndexSelectionOp), (True, PandasQuerySelectionOp)]:
        dag = build_q12(**tables)
        ops = optimize_(dag, OptConfig(dataframe_ops=True, pandas_query=flag),
                        env=get_data(dag))[0]
        # Look each filter up by predicate, not by position in the plan, so
        # this test does not care what order the optimizer emits them in.
        # They are still applied in query order below.
        modes = by_predicate(
            ops, IsinExpr(Col("l_shipmode"), SHIP_MODES))
        committed = by_predicate(
            ops, BinOpExpr(operator.lt, Col("l_commitdate"), Col("l_receiptdate")))
        shipped = by_predicate(
            ops, BinOpExpr(operator.lt, Col("l_shipdate"), Col("l_commitdate")))
        window = by_predicate(
            ops, BinOpExpr(operator.and_,
                           BinOpExpr(operator.ge, Col("l_receiptdate"), Const(START)),
                           BinOpExpr(operator.lt, Col("l_receiptdate"), Const(END))))
        assert isinstance(modes, PandasIndexSelectionOp)
        for sel in (committed, shipped, window):
            assert isinstance(sel, impl)
        result = df
        for sel in (modes, committed, shipped, window):
            result = sel.process([result])
        pd.testing.assert_frame_equal(
            expected, pd.DataFrame(result).reset_index(drop=True), check_dtype=False)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
