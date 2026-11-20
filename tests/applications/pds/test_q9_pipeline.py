"""PDS Q9 run through Stratum, checked against the query it was ported from.

The source is ``queries/pandas/q9.py`` in polars-benchmark. :func:`q9_reference`
is that query without the benchmark's IO wrapper; :func:`build_q9` is the same
thing as a skrub DAG. Both run on the same fixtures and must agree.

The query computes profit per supplier nation and year for parts whose name
contains a word. Five joins, one on the composite part and supplier key, feed
a literal substring filter and a profit column; the aggregation runs unsorted,
so the final sort and ``reset_index`` decide the output order.
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
    AggExpr, BinOpExpr, Col, Const, DtExpr, StrExpr)
from stratum.optimizer.logical._join_ops import JoinOp
from stratum.optimizer.logical._map_ops import AssignMapOp
from stratum.optimizer.logical._metadata_ops import RenameOp, ResetIndexOp
from stratum.optimizer.logical._selection_ops import MaskSelectionOp
from stratum.optimizer.logical._sort_ops import SortOp
from stratum.optimizer.physical._aggregation_execs import PolarsAggregateOp
from stratum.optimizer.physical._join_execs import PolarsJoinOp
from stratum.optimizer.physical._map_execs import PolarsAssignMapOp
from stratum.optimizer.physical._metadata_execs import (
    PolarsRenameOp, PolarsResetIndexOp)
from stratum.optimizer.physical._selection_execs import PolarsMaskSelectionOp
from stratum.optimizer.physical._sort_execs import PolarsSortOp
from stratum.frontend._skrub_graph import get_data


def make_q9_tables():
    """Arrow-backed lineitem, nation, orders, part, partsupp and supplier.

    Two of three part names contain "green". BRAZIL profits in three years,
    first seen as 1995, so the unsorted grouping meets the years in the
    opposite of the final descending order; 1995 sums two lines. Every other
    line fails exactly one check: a part without "green", or a part and
    supplier pair that partsupp does not list, which the composite join drops.
    """
    def table(columns, rows):
        return pd.DataFrame(rows, columns=columns).convert_dtypes(dtype_backend="pyarrow")

    lineitem_ds = table(
        ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity",
         "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus",
         "l_shipdate"],
        [
            [601, 1001, 101, 1, 10.0, 1000.0, 0.10, 0.02, "N", "O", date(1995, 3, 5)],
            [601, 1002, 101, 2, 4.0, 500.0, 0.00, 0.06, "N", "O", date(1995, 3, 9)],
            [602, 1001, 102, 1, 5.0, 2000.0, 0.05, 0.00, "R", "F", date(1996, 2, 7)],
            [602, 1001, 101, 2, 1.0, 100.0, 0.00, 0.04, "A", "F", date(1996, 2, 3)],
            [603, 1001, 101, 1, 2.0, 300.0, 0.00, 0.08, "A", "F", date(1997, 1, 4)],
            [603, 1003, 102, 2, 8.0, 9000.0, 0.00, 0.01, "N", "O", date(1997, 1, 6)],
            [604, 1002, 102, 1, 3.0, 9000.0, 0.00, 0.03, "R", "F", date(1996, 6, 2)],
        ],
    )
    nation_ds = table(
        ["n_nationkey", "n_name", "n_regionkey", "n_comment"],
        [
            [2, "BRAZIL", 1, "Brazil"],
            [6, "FRANCE", 3, "France"],
        ],
    )
    orders_ds = table(
        ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate",
         "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"],
        [
            [601, 1, "F", 100.0, date(1995, 3, 1), "1-URGENT", "Clerk A", 0, "Ordered in 1995"],
            [602, 2, "F", 100.0, date(1996, 2, 1), "2-HIGH", "Clerk B", 0, "Ordered in 1996"],
            [603, 1, "O", 100.0, date(1997, 1, 1), "3-MEDIUM", "Clerk A", 0, "Ordered in 1997"],
            [604, 2, "O", 100.0, date(1996, 6, 1), "1-URGENT", "Clerk B", 0, "Unlisted supplier"],
        ],
    )
    part_ds = table(
        ["p_partkey", "p_name", "p_mfgr", "p_brand", "p_type",
         "p_size", "p_container", "p_retailprice", "p_comment"],
        [
            [1001, "forest green metallic", "MFGR1", "BRAND1", "STANDARD BRASS", 15, "BOX", 100.0, "Green part"],
            [1002, "lime green wood", "MFGR2", "BRAND2", "SMALL BRASS", 15, "BOX", 200.0, "Green part"],
            [1003, "blue steel", "MFGR3", "BRAND3", "LARGE BRASS", 20, "BOX", 150.0, "No green"],
        ],
    )
    part_supp_ds = table(
        ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"],
        [
            [1001, 101, 100, 10.0, "Brazilian supply"],
            [1001, 102, 100, 20.0, "French supply"],
            [1002, 101, 100, 5.0, "Brazilian supply"],
            [1003, 102, 100, 1.0, "French supply"],
        ],
    )
    supplier_ds = table(
        ["s_suppkey", "s_name", "s_address", "s_nationkey",
         "s_phone", "s_acctbal", "s_comment"],
        [
            [101, "Supplier A", "Recife", 2, "12-101-101", 500.0, "Brazilian supplier"],
            [102, "Supplier B", "Lyon", 6, "16-102-102", 600.0, "French supplier"],
        ],
    )
    for frame, column in [(lineitem_ds, "l_quantity"), (lineitem_ds, "l_extendedprice"),
                          (lineitem_ds, "l_discount"), (lineitem_ds, "l_tax"),
                          (orders_ds, "o_totalprice"), (part_ds, "p_retailprice"),
                          (part_supp_ds, "ps_supplycost"), (supplier_ds, "s_acctbal")]:
        frame[column] = pd.array(frame[column], dtype="double[pyarrow]")
    for frame, column in [(lineitem_ds, "l_shipdate"), (orders_ds, "o_orderdate")]:
        frame[column] = pd.array(frame[column], dtype="date32[day][pyarrow]")

    return dict(lineitem_ds=lineitem_ds, nation_ds=nation_ds, orders_ds=orders_ds,
                part_ds=part_ds, part_supp_ds=part_supp_ds, supplier_ds=supplier_ds)


def q9_reference(lineitem_ds, nation_ds, orders_ds, part_ds, part_supp_ds,
                 supplier_ds):
    """The upstream pandas query, with the six tables supplied by the fixture."""
    jn1 = part_ds.merge(part_supp_ds, left_on="p_partkey", right_on="ps_partkey")
    jn2 = jn1.merge(supplier_ds, left_on="ps_suppkey", right_on="s_suppkey")
    jn3 = jn2.merge(
        lineitem_ds,
        left_on=["p_partkey", "ps_suppkey"],
        right_on=["l_partkey", "l_suppkey"],
    )
    jn4 = jn3.merge(orders_ds, left_on="l_orderkey", right_on="o_orderkey")
    jn5 = jn4.merge(nation_ds, left_on="s_nationkey", right_on="n_nationkey")

    jn5 = jn5[jn5["p_name"].str.contains("green", regex=False)]

    jn5["o_year"] = jn5["o_orderdate"].dt.year
    jn5["amount"] = jn5["l_extendedprice"] * (1.0 - jn5["l_discount"]) - (
        jn5["ps_supplycost"] * jn5["l_quantity"]
    )
    jn5 = jn5.rename(columns={"n_name": "nation"})

    gb = jn5.groupby(["nation", "o_year"], as_index=False, sort=False)
    agg = gb.agg(sum_profit=pd.NamedAgg(column="amount", aggfunc="sum"))
    sorted_df = agg.sort_values(by=["nation", "o_year"], ascending=[True, False])
    return sorted_df.reset_index(drop=True)


def build_q9_prefix(lineitem_ds, nation_ds, orders_ds, part_ds, part_supp_ds,
                    supplier_ds):
    """Q9 up to the aggregation: joins, the name filter, year, profit and the
    rename.

    The upstream query has no ``var``s; its literals stay literals. DataOps are
    immutable, so the two ``jn5[...] = ...`` assignments become one
    ``.assign(...)``.
    """
    lineitem_ds = st.var("lineitem_ds", lineitem_ds)
    nation_ds = st.var("nation_ds", nation_ds)
    orders_ds = st.var("orders_ds", orders_ds)
    part_ds = st.var("part_ds", part_ds)
    part_supp_ds = st.var("part_supp_ds", part_supp_ds)
    supplier_ds = st.var("supplier_ds", supplier_ds)

    jn1 = part_ds.merge(part_supp_ds, left_on="p_partkey", right_on="ps_partkey")
    jn2 = jn1.merge(supplier_ds, left_on="ps_suppkey", right_on="s_suppkey")
    jn3 = jn2.merge(
        lineitem_ds,
        left_on=["p_partkey", "ps_suppkey"],
        right_on=["l_partkey", "l_suppkey"],
    )
    jn4 = jn3.merge(orders_ds, left_on="l_orderkey", right_on="o_orderkey")
    jn5 = jn4.merge(nation_ds, left_on="s_nationkey", right_on="n_nationkey")

    jn5 = jn5[jn5["p_name"].str.contains("green", regex=False)]

    jn5 = jn5.assign(
        o_year=jn5["o_orderdate"].dt.year,
        amount=jn5["l_extendedprice"] * (1.0 - jn5["l_discount"]) - (
            jn5["ps_supplycost"] * jn5["l_quantity"]
        ),
    )
    return jn5.rename(columns={"n_name": "nation"})


def build_q9(lineitem_ds, nation_ds, orders_ds, part_ds, part_supp_ds, supplier_ds):
    """Complete Q9."""
    jn5 = build_q9_prefix(lineitem_ds, nation_ds, orders_ds, part_ds, part_supp_ds,
                          supplier_ds)

    gb = jn5.groupby(["nation", "o_year"], as_index=False, sort=False)
    agg = gb.agg(sum_profit=pd.NamedAgg(column="amount", aggfunc="sum"))
    sorted_df = agg.sort_values(by=["nation", "o_year"], ascending=[True, False])
    return sorted_df.reset_index(drop=True)


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

def test_q9_matches_reference(polars):
    """The skrubified query returns what the upstream query returns."""
    tables = make_q9_tables()
    expected = q9_reference(**tables)
    result = _api.evaluate(build_q9(**tables))
    if polars:
        assert isinstance(result, pl.DataFrame)
        result = result.to_pandas()

    assert list(expected.columns) == list(result.columns)
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True),
        result.reset_index(drop=True),
        check_dtype=False,  # compare values across Arrow-backed input and runtime output
    )


def test_q9_prefix_runs_on_polars():
    tables = make_q9_tables()
    expected = (
        tables["part_ds"].merge(tables["part_supp_ds"],
                                left_on="p_partkey", right_on="ps_partkey")
        .merge(tables["supplier_ds"], left_on="ps_suppkey", right_on="s_suppkey")
        .merge(tables["lineitem_ds"], left_on=["p_partkey", "ps_suppkey"],
               right_on=["l_partkey", "l_suppkey"])
        .merge(tables["orders_ds"], left_on="l_orderkey", right_on="o_orderkey")
        .merge(tables["nation_ds"], left_on="s_nationkey", right_on="n_nationkey")
    )
    expected = expected[expected["p_name"].str.contains("green", regex=False)]
    expected = expected.assign(
        o_year=expected["o_orderdate"].dt.year,
        amount=expected["l_extendedprice"] * (1.0 - expected["l_discount"])
        - expected["ps_supplycost"] * expected["l_quantity"])
    expected = expected.rename(columns={"n_name": "nation"})
    for column in ["o_orderdate", "l_shipdate"]:
        expected[column] = pd.to_datetime(expected[column])
    with st.config(implementation_selector="greedy"):
        result = _api.evaluate(build_q9_prefix(**tables))
    assert isinstance(result, pl.DataFrame)
    # Polars joins may reorder rows; this comparison checks contents, not pandas row order.
    pd.testing.assert_frame_equal(
        expected.sort_values(["l_orderkey", "l_linenumber"]).reset_index(drop=True),
        result.to_pandas().sort_values(["l_orderkey", "l_linenumber"]).reset_index(drop=True),
        check_dtype=False)


# --- the folded plan ------------------------------------------------------

def test_q9_plan_folds_joins(polars):
    """Each merge in the prefix collapses to one join operator, the composite
    part and supplier key included."""
    ops = _plan(build_q9_prefix(**make_q9_tables()))
    joins = [o for o in ops if isinstance(o, JoinOp)]

    assert [("p_partkey", "ps_partkey"), ("ps_suppkey", "s_suppkey"),
            (["p_partkey", "ps_suppkey"], ["l_partkey", "l_suppkey"]),
            ("l_orderkey", "o_orderkey"), ("s_nationkey", "n_nationkey")] \
        == [(o.left_on, o.right_on) for o in joins]
    if polars:
        assert all(isinstance(o, PolarsJoinOp) for o in joins)


def test_q9_filter_folds_the_substring_match(polars):
    """The literal ``str.contains`` becomes a string expression predicate,
    with its ``regex=False`` option kept."""
    sel = _one(_plan(build_q9(**make_q9_tables())), MaskSelectionOp)

    assert StrExpr(Col("p_name"), "contains", {"pat": "green", "regex": False}) \
        == sel.predicate
    assert len(sel.inputs) == 1, "the substring should not be a graph input"
    if polars:
        assert isinstance(sel, PolarsMaskSelectionOp)


def test_q9_folds_year_profit_and_rename(polars):
    """The year and profit columns collapse to one assign map over plain
    columns, and the nation rename follows it."""
    ops = _plan(build_q9_prefix(**make_q9_tables()))
    amo = _one(ops, AssignMapOp)
    rename = _one(ops, RenameOp)

    revenue = BinOpExpr(operator.mul, Col("l_extendedprice"),
                        BinOpExpr(operator.sub, Const(1.0), Col("l_discount")))
    assert {"o_year": DtExpr(Col("o_orderdate"), "year"),
            "amount": BinOpExpr(operator.sub, revenue,
                                BinOpExpr(operator.mul, Col("ps_supplycost"),
                                          Col("l_quantity")))} == amo.entries
    assert len(amo.inputs) == 1, "the map should need only the filtered frame"
    assert {"n_name": "nation"} == rename.columns and rename.inputs[0] is amo
    if polars:
        assert isinstance(amo, PolarsAssignMapOp)
        assert isinstance(rename, PolarsRenameOp)


def test_q9_folds_the_unsorted_aggregation(polars):
    """The ``NamedAgg`` profit sum folds with ``sort=False`` kept, and the
    query runs end to end."""
    tables = make_q9_tables()
    agg = _one(_plan(build_q9(**tables)), AggregateOp)

    assert (Col("nation"), Col("o_year")) == agg.grouping
    assert (("sum_profit", AggExpr("sum", Col("amount"))),) == agg.aggregations
    assert agg.sort_keys is False
    if polars:
        assert isinstance(agg, PolarsAggregateOp)

    result = _api.evaluate(build_q9(**tables))
    result = result.to_pandas() if isinstance(result, pl.DataFrame) else result
    pd.testing.assert_frame_equal(q9_reference(**tables).reset_index(drop=True),
                                  result.reset_index(drop=True), check_dtype=False)


def test_q9_folds_the_sort_and_reset_index(polars):
    """The mixed-direction sort and the dropping ``reset_index`` fold and close
    the plan."""
    ops = _plan(build_q9(**make_q9_tables()))
    sort = _one(ops, SortOp)
    reset = _one(ops, ResetIndexOp)

    assert (("nation", "o_year"), (True, False)) == (sort.by, sort.ascending)
    assert sort.inputs[0] is _one(ops, AggregateOp)
    assert reset.inputs[0] is sort and ops[-1] is reset
    if polars:
        assert isinstance(sort, PolarsSortOp)
        assert isinstance(reset, PolarsResetIndexOp)


def test_q9_filter_binds_query_impl_under_flag():
    """``pandas_query`` picks the impl at plan time; both filter correctly.

    A string method has no query spelling, so the substring filter stays a
    mask selection under either flag. Pandas-only, so no fixture.
    """
    from stratum.optimizer.physical._selection_execs import PandasIndexSelectionOp

    tables = make_q9_tables()
    df = tables["part_ds"]
    expected = df[df["p_name"].str.contains("green", regex=False)].reset_index(drop=True)

    for flag in (False, True):
        dag = build_q9(**tables)
        ops = optimize_(dag, OptConfig(dataframe_ops=True, pandas_query=flag),
                        env=get_data(dag))[0]
        sel = _one(ops, MaskSelectionOp)
        assert isinstance(sel, PandasIndexSelectionOp)
        # The filter runs on the joined frame in the query; the part name is a
        # part column, so it filters the part table alone the same way.
        result = sel.process([df])
        pd.testing.assert_frame_equal(
            expected, pd.DataFrame(result).reset_index(drop=True), check_dtype=False)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
