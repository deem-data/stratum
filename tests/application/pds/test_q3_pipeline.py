"""PDS Q3 run through Stratum, checked against the query it was ported from.

The source is ``queries/pandas/q3.py`` in polars-benchmark. :func:`q3_reference`
is that query without the benchmark's IO wrapper; :func:`build_q3` is the same
thing as a skrub DAG. Both run on the same fixtures and must agree.

The query filters customers by market segment, joins orders and line items,
keeps only orders placed and shipped on opposite sides of a cutoff date,
sums a computed revenue per order, then renames, sorts and limits.
"""
import operator
from datetime import date

import pytest
import pandas as pd
import polars as pl

import stratum as st
from stratum.optimizer._optimize import OptConfig, optimize as optimize_
from stratum.optimizer.logical._aggregation_ops import AggregateOp
from stratum.optimizer.logical._column_expr import (
    AggExpr, BinOpExpr, Col, Const, OperandLeaf)
from stratum.optimizer.logical._join_ops import JoinOp
from stratum.optimizer.logical._map_ops import AssignMapOp
from stratum.optimizer.logical._metadata_ops import RenameOp
from stratum.optimizer.logical._ops import OperandRef
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp
from stratum.optimizer.logical._selection_ops import MaskSelectionOp, SliceSelectionOp
from stratum.optimizer.logical._sort_ops import SortOp
from stratum.optimizer.physical._aggregation_execs import PolarsAggregateOp
from stratum.optimizer.physical._join_execs import PolarsJoinOp
from stratum.optimizer.physical._map_execs import PolarsAssignMapOp
from stratum.optimizer.physical._metadata_execs import PolarsRenameOp
from stratum.optimizer.physical._projection_execs import PolarsColumnProjectionOp
from stratum.optimizer.physical._selection_execs import (
    PolarsMaskSelectionOp, PolarsSliceSelectionOp)
from stratum.optimizer.physical._sort_execs import PolarsSortOp
from stratum.frontend._skrub_graph import get_data

# The Q3 cutoff: `var2` in the upstream query.
CUTOFF = date(1995, 3, 15)


def make_q3_tables():
    """Arrow-backed customer, orders and lineitem.

    Orders 101, 102, 107 and 104 qualify, with 107 and 102 tied on revenue so
    the order date decides their order. The remaining rows each fail exactly
    one check: market segment, the order-date side of the cutoff (including
    the boundary itself), or the ship-date side (including the boundary).
    Orders 101, 104 and 107 each sum two lines, so the aggregation is not
    trivially equal to its inputs.
    """
    def table(columns, rows):
        return pd.DataFrame(rows, columns=columns).convert_dtypes(dtype_backend="pyarrow")

    customer_ds = table(
        ["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone",
         "c_acctbal", "c_mktsegment", "c_comment"],
        [
            [1, "Customer A", "Address A", 10, "111-111", 100.0, "BUILDING", "Building customer"],
            [2, "Customer B", "Address B", 20, "222-222", 200.0, "BUILDING", "Building customer"],
            [3, "Customer C", "Address C", 30, "333-333", 300.0, "AUTOMOBILE", "Wrong segment"],
        ],
    )
    orders_ds = table(
        ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate",
         "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"],
        [
            [101, 1, "O", 290.0, date(1995, 3, 1), "1-URGENT", "Clerk A", 0, "Qualifying"],
            [102, 1, "O", 240.0, date(1995, 2, 10), "2-HIGH", "Clerk A", 1, "Qualifying, tied revenue"],
            [103, 2, "O", 999.0, date(1995, 3, 15), "1-URGENT", "Clerk B", 0, "Order date on the cutoff"],
            [104, 2, "O", 100.0, date(1995, 1, 20), "3-MEDIUM", "Clerk B", 0, "Qualifying"],
            [105, 3, "O", 999.0, date(1995, 1, 15), "1-URGENT", "Clerk C", 0, "Wrong segment"],
            [106, 2, "O", 999.0, date(1995, 4, 1), "1-URGENT", "Clerk B", 0, "Ordered after the cutoff"],
            [107, 1, "O", 240.0, date(1995, 1, 5), "2-HIGH", "Clerk A", 2, "Qualifying, tied revenue"],
        ],
    )
    line_item_ds = table(
        ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity",
         "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus",
         "l_shipdate"],
        [
            [101, 1552, 93, 1, 17.0, 100.0, 0.10, 0.02, "N", "O", date(1995, 3, 20)],
            [101, 673, 28, 2, 36.0, 200.0, 0.00, 0.06, "N", "O", date(1995, 3, 21)],
            [101, 1062, 75, 3, 8.0, 10000.0, 0.00, 0.00, "N", "O", date(1995, 3, 15)],
            [101, 429, 41, 4, 28.0, 10000.0, 0.00, 0.04, "N", "O", date(1995, 3, 1)],
            [102, 1901, 12, 1, 11.0, 300.0, 0.20, 0.08, "A", "F", date(1995, 4, 1)],
            [103, 88, 67, 1, 4.0, 10000.0, 0.00, 0.01, "N", "O", date(1995, 4, 1)],
            [104, 1444, 39, 1, 31.0, 50.0, 0.00, 0.03, "R", "F", date(1995, 3, 20)],
            [104, 203, 81, 2, 6.0, 50.0, 0.00, 0.05, "R", "F", date(1995, 3, 22)],
            [105, 1777, 54, 1, 19.0, 10000.0, 0.00, 0.07, "A", "F", date(1995, 4, 1)],
            [106, 956, 22, 1, 42.0, 10000.0, 0.00, 0.02, "N", "O", date(1995, 4, 1)],
            [107, 1205, 64, 1, 13.0, 120.0, 0.00, 0.06, "R", "F", date(1995, 3, 20)],
            [107, 314, 17, 2, 25.0, 120.0, 0.00, 0.04, "A", "F", date(1995, 3, 25)],
        ],
    )
    for frame, column in [(customer_ds, "c_acctbal"), (orders_ds, "o_totalprice"),
                          (line_item_ds, "l_quantity"),
                          (line_item_ds, "l_extendedprice"),
                          (line_item_ds, "l_discount"), (line_item_ds, "l_tax")]:
        frame[column] = pd.array(frame[column], dtype="double[pyarrow]")
    orders_ds["o_orderdate"] = pd.array(orders_ds["o_orderdate"],
                                         dtype="date32[day][pyarrow]")
    line_item_ds["l_shipdate"] = pd.array(line_item_ds["l_shipdate"],
                                          dtype="date32[day][pyarrow]")

    return dict(customer_ds=customer_ds, orders_ds=orders_ds,
                line_item_ds=line_item_ds)


def q3_reference(customer_ds, orders_ds, line_item_ds):
    """The upstream pandas query, with the three tables supplied by the fixture
    and a ``.copy()`` so the revenue assignment does not touch the filtered frame."""
    var1 = "BUILDING"
    var2 = CUTOFF

    fcustomer = customer_ds[customer_ds["c_mktsegment"] == var1]

    jn1 = fcustomer.merge(orders_ds, left_on="c_custkey", right_on="o_custkey")
    jn2 = jn1.merge(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")

    jn2 = jn2[jn2["o_orderdate"] < var2]
    jn2 = jn2[jn2["l_shipdate"] > var2]
    jn2 = jn2.copy()
    jn2["revenue"] = jn2.l_extendedprice * (1 - jn2.l_discount)

    gb = jn2.groupby(["o_orderkey", "o_orderdate", "o_shippriority"], as_index=False)
    agg = gb["revenue"].sum()

    sel = agg.loc[:, ["o_orderkey", "revenue", "o_orderdate", "o_shippriority"]]
    sel = sel.rename(columns={"o_orderkey": "l_orderkey"})

    sorted = sel.sort_values(by=["revenue", "o_orderdate"], ascending=[False, True])
    return sorted.head(10)


def build_q3_prefix(customer_ds, orders_ds, line_item_ds):
    """Q3 up to the aggregation: segment filter, joins, date filters, revenue.

    ``var1`` and ``var2`` stay as ``skrub.var``s so the values reach the fold
    as ``ValueOp``s rather than inline literals, they are only needed inside
    this stage, so they are not returned. DataOps are immutable, so the
    ``jn2[...] = ...`` assignment becomes one ``.assign(...)``.
    """
    var1 = st.var("var1", "BUILDING")
    var2 = st.var("var2", CUTOFF)
    customer_ds = st.var("customer_ds", customer_ds)
    orders_ds = st.var("orders_ds", orders_ds)
    line_item_ds = st.var("line_item_ds", line_item_ds)

    fcustomer = customer_ds[customer_ds["c_mktsegment"] == var1]

    jn1 = fcustomer.merge(orders_ds, left_on="c_custkey", right_on="o_custkey")
    jn2 = jn1.merge(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")

    jn2 = jn2[jn2["o_orderdate"] < var2]
    jn2 = jn2[jn2["l_shipdate"] > var2]
    return jn2.assign(revenue=jn2["l_extendedprice"] * (1 - jn2["l_discount"]))


def build_q3(customer_ds, orders_ds, line_item_ds):
    """Complete Q3, column selection uses ``[]`` indexing instead of pandas ``loc``."""
    jn2 = build_q3_prefix(customer_ds, orders_ds, line_item_ds)

    gb = jn2.groupby(["o_orderkey", "o_orderdate", "o_shippriority"], as_index=False)
    agg = gb["revenue"].sum()

    sel = agg[["o_orderkey", "revenue", "o_orderdate", "o_shippriority"]]
    sel = sel.rename(columns={"o_orderkey": "l_orderkey"})

    sorted = sel.sort_values(by=["revenue", "o_orderdate"], ascending=[False, True])
    return sorted.head(10)


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

def test_q3_matches_reference(polars):
    """The skrubified query returns what the upstream query returns."""
    tables = make_q3_tables()
    expected = q3_reference(**tables)
    result = st._api.evaluate(build_q3(**tables))
    if polars:
        assert isinstance(result, pl.DataFrame)
        result = result.to_pandas()
        # Polars returns the order date as a datetime; the reference keeps dates.
        expected["o_orderdate"] = pd.to_datetime(expected["o_orderdate"])

    assert list(expected.columns) == list(result.columns)
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True),
        result.reset_index(drop=True),
        check_dtype=False,  # compare values across Arrow-backed input and runtime output
    )


def test_q3_prefix_runs_on_polars():
    tables = make_q3_tables()
    fcustomer = tables["customer_ds"][tables["customer_ds"]["c_mktsegment"] == "BUILDING"]
    expected = fcustomer.merge(tables["orders_ds"],
                               left_on="c_custkey", right_on="o_custkey")
    expected = expected.merge(tables["line_item_ds"],
                              left_on="o_orderkey", right_on="l_orderkey")
    expected = expected[expected["o_orderdate"] < CUTOFF]
    expected = expected[expected["l_shipdate"] > CUTOFF]
    expected = expected.assign(
        revenue=expected["l_extendedprice"] * (1 - expected["l_discount"]))
    expected["o_orderdate"] = pd.to_datetime(expected["o_orderdate"])
    expected["l_shipdate"] = pd.to_datetime(expected["l_shipdate"])
    with st.config(implementation_selector="greedy"):
        result = st._api.evaluate(build_q3_prefix(**tables))
    assert isinstance(result, pl.DataFrame)
    # Polars joins may reorder rows; this comparison checks contents, not pandas row order.
    pd.testing.assert_frame_equal(
        expected.sort_values(["o_orderkey", "l_linenumber"]).reset_index(drop=True),
        result.to_pandas().sort_values(["o_orderkey", "l_linenumber"]).reset_index(drop=True),
        check_dtype=False)


# --- the folded plan ------------------------------------------------------

def test_q3_plan_folds_joins(polars):
    """Each merge in the prefix collapses to one join operator."""
    ops = _plan(build_q3_prefix(**make_q3_tables()))
    joins = [o for o in ops if isinstance(o, JoinOp)]

    assert [("c_custkey", "o_custkey"), ("o_orderkey", "l_orderkey")] \
        == [(o.left_on, o.right_on) for o in joins]
    if polars:
        assert all(isinstance(o, PolarsJoinOp) for o in joins)


def test_q3_filter_inlines_segment_and_shares_the_date(polars):
    """The single-use segment becomes a ``Const``; the shared date does not.

    ``var2`` feeds both date comparisons, so each fold sees an external
    consumer and both predicates reference the one shared date input.
    """
    ops = _plan(build_q3(**make_q3_tables()))
    filters = [o for o in ops if isinstance(o, MaskSelectionOp)]

    assert [BinOpExpr(operator.eq, Col("c_mktsegment"), Const("BUILDING")),
            BinOpExpr(operator.lt, Col("o_orderdate"), OperandLeaf(OperandRef(1))),
            BinOpExpr(operator.gt, Col("l_shipdate"), OperandLeaf(OperandRef(1)))] \
        == [o.predicate for o in filters]
    assert filters[1].inputs[1] is filters[2].inputs[1]
    if polars:
        assert all(isinstance(o, PolarsMaskSelectionOp) for o in filters)


def test_q3_folds_the_revenue(polars):
    """The revenue column collapses to one assign map."""
    amo = _one(_plan(build_q3_prefix(**make_q3_tables())), AssignMapOp)

    assert {"revenue": BinOpExpr(
        operator.mul, Col("l_extendedprice"),
        BinOpExpr(operator.sub, Const(1), Col("l_discount")))} == amo.entries
    if polars:
        assert isinstance(amo, PolarsAssignMapOp)


def test_q3_folds_the_grouped_aggregation(polars):
    """The grouped revenue sum folds into one aggregate and runs end to end.

    The full comparison uses the original fixture, whose orders sum several
    lines and tie on revenue, unlike the one-line groups in the limit test.
    """
    tables = make_q3_tables()
    agg = _one(_plan(build_q3(**tables)), AggregateOp)

    assert (Col("o_orderkey"), Col("o_orderdate"), Col("o_shippriority")) == agg.grouping
    assert ((None, AggExpr("sum", Col("revenue"))),) == agg.aggregations
    if polars:
        assert isinstance(agg, PolarsAggregateOp)

    result = st._api.evaluate(build_q3(**tables))
    result = result.to_pandas() if isinstance(result, pl.DataFrame) else result
    expected = q3_reference(**tables)
    # Polars returns the order date as a datetime; the reference keeps dates.
    expected["o_orderdate"] = pd.to_datetime(expected["o_orderdate"])
    result["o_orderdate"] = pd.to_datetime(result["o_orderdate"])
    pd.testing.assert_frame_equal(expected.reset_index(drop=True),
                                  result.reset_index(drop=True), check_dtype=False)


def test_q3_folds_the_rename_sort_and_limit(polars):
    """Projection, rename, sorting and ``head`` fold into physical operators.

    More than 10 qualifying orders makes the limit observable after sorting.
    """
    tables = make_q3_tables()
    # Repeat the first order/line rows 12 times with distinct keys and
    # prices: 12 one-line groups with distinct revenues, so ``head(10)``
    # drops the two smallest.
    orders = tables["orders_ds"].iloc[[0] * 12].reset_index(drop=True)
    orders["o_orderkey"] = pd.array(range(1000, 1012), dtype="int64[pyarrow]")
    line_items = tables["line_item_ds"].iloc[[0] * 12].reset_index(drop=True)
    line_items["l_orderkey"] = orders["o_orderkey"]
    line_items["l_extendedprice"] = pd.array(
        [100.0 + 10.0 * i for i in range(12)], dtype="double[pyarrow]")
    tables.update(orders_ds=orders, line_item_ds=line_items)

    ops = _plan(build_q3(**tables))
    projection = _one(ops, ColumnProjectionOp)
    rename = _one(ops, RenameOp)
    sort = _one(ops, SortOp)
    head = _one(ops, SliceSelectionOp)

    assert ["o_orderkey", "revenue", "o_orderdate", "o_shippriority"] \
        == projection.key
    assert {"o_orderkey": "l_orderkey"} == rename.columns
    assert (("revenue", "o_orderdate"), (False, True), 10) \
        == (sort.by, sort.ascending, head.n)
    assert sort.inputs[0] is rename and head.inputs[0] is sort
    if polars:
        assert isinstance(projection, PolarsColumnProjectionOp)
        assert isinstance(rename, PolarsRenameOp)
        assert isinstance(sort, PolarsSortOp)
        assert isinstance(head, PolarsSliceSelectionOp)

    result = st._api.evaluate(build_q3(**tables))
    result = result.to_pandas() if isinstance(result, pl.DataFrame) else result
    expected = q3_reference(**tables)
    # Polars returns the order date as a datetime, the reference keeps dates.
    expected["o_orderdate"] = pd.to_datetime(expected["o_orderdate"])
    result["o_orderdate"] = pd.to_datetime(result["o_orderdate"])
    pd.testing.assert_frame_equal(expected.reset_index(drop=True),
                                  result.reset_index(drop=True), check_dtype=False)


def test_q3_filter_binds_query_impl_under_flag():
    """``pandas_query`` picks the impl at plan time; both filter correctly.

    Only the segment can use the query fast path: its value is inlined, while
    the shared date stays an ``OperandLeaf``, which makes ``to_pandas_query``
    return ``None`` for both date filters. Pandas-only, so no fixture.
    """
    from stratum.optimizer.physical._selection_execs import (
        PandasIndexSelectionOp, PandasQuerySelectionOp)

    tables = make_q3_tables()
    # The unfiltered join in pandas: input frame for running the selections
    # by hand. The segment column survives both joins, so the segment
    # selection, which runs on customer alone in the query, also applies here.
    df = (
        tables["customer_ds"].merge(tables["orders_ds"],
                                    left_on="c_custkey", right_on="o_custkey")
        .merge(tables["line_item_ds"], left_on="o_orderkey", right_on="l_orderkey")
    )
    expected = df[(df["c_mktsegment"] == "BUILDING")
                  & (df["o_orderdate"] < CUTOFF)
                  & (df["l_shipdate"] > CUTOFF)].reset_index(drop=True)

    def by_predicate(ops, predicate):
        matches = [o for o in ops
                   if isinstance(o, MaskSelectionOp) and o.predicate == predicate]
        assert len(matches) == 1, \
            f"expected exactly one selection for {predicate}, got {len(matches)}"
        return matches[0]

    for flag, impl in [(False, PandasIndexSelectionOp), (True, PandasQuerySelectionOp)]:
        dag = build_q3(**tables)
        ops = optimize_(dag, OptConfig(dataframe_ops=True, pandas_query=flag),
                        env=get_data(dag))[0]
        # Look each filter up by predicate, not by position in the plan, so
        # this test does not care what order the optimizer emits them in.
        # They are still applied in query order below.
        segment = by_predicate(
            ops, BinOpExpr(operator.eq, Col("c_mktsegment"), Const("BUILDING")))
        ordered = by_predicate(
            ops, BinOpExpr(operator.lt, Col("o_orderdate"),
                           OperandLeaf(OperandRef(1))))
        shipped = by_predicate(
            ops, BinOpExpr(operator.gt, Col("l_shipdate"),
                           OperandLeaf(OperandRef(1))))
        assert isinstance(segment, impl)
        assert isinstance(ordered, PandasIndexSelectionOp)
        assert isinstance(shipped, PandasIndexSelectionOp)
        # The date selections take the shared date as a second input, which
        # the runtime resolves from the ValueOp; pass it directly here.
        result = segment.process("fit_transform", [df])
        for sel in (ordered, shipped):
            result = sel.process("fit_transform", [result, CUTOFF])
        pd.testing.assert_frame_equal(
            expected, pd.DataFrame(result).reset_index(drop=True), check_dtype=False)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
