"""PDS Q20 run through Stratum, checked against the query it was ported from.

The source is ``queries/pandas/q20.py`` in polars-benchmark. :func:`q20_reference`
is that query without the benchmark's IO wrapper; :func:`build_q20` is the same
thing as a skrub DAG. Both run on the same fixtures and must agree.

The query lists one nation's suppliers holding excess stock of parts whose name
starts with a word. Line quantities in a ``pd.Timestamp`` window are summed per
part and supplier and scaled. A deduplicated part selection joins partsupp and
those sums on a composite key, the stock filter keeps the unique suppliers, and
the supplier and nation joins name them.
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
    AggExpr, BinOpExpr, Col, Const, StrExpr)
from stratum.optimizer.logical._join_ops import JoinOp
from stratum.optimizer.logical._map_ops import AssignMapOp
from stratum.optimizer.logical._ops import ValueOp
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp
from stratum.optimizer.logical._selection_ops import (
    DropDuplicatesSelectionOp, MaskSelectionOp)
from stratum.optimizer.logical._sort_ops import SortOp
from stratum.optimizer.physical._aggregation_execs import PolarsAggregateOp
from stratum.optimizer.physical._join_execs import PolarsJoinOp
from stratum.optimizer.physical._map_execs import PolarsAssignMapOp
from stratum.optimizer.physical._projection_execs import PolarsColumnProjectionOp
from stratum.optimizer.physical._selection_execs import (
    PolarsDropDuplicatesSelectionOp, PolarsMaskSelectionOp)
from stratum.optimizer.physical._sort_execs import PolarsSortOp
from stratum.frontend._skrub_graph import get_data

# The Q20 window: `var1` and `var2` in the upstream query.
START = pd.Timestamp("1994-01-01")
END = pd.Timestamp("1995-01-01")


def make_q20_tables():
    """Arrow-backed lineitem, nation, part, partsupp and supplier.

    Suppliers 101 and 103 qualify, listed out of name order so the sort is
    observable, and 103 twice so the unique-supplier step is too. Part 2 appears
    twice so the part deduplication matters. Every other supplier fails exactly
    one check: 102 holds no more than half its shipped quantity, 104 is outside
    CANADA, and 105 only stocks a part whose name does not start with "forest".
    The lines before the window and on its end would fail suppliers 101 and 103
    on stock if they were counted.

    Dates are Arrow timestamps rather than date32: the upstream query compares
    them with ``pd.Timestamp``, which pandas 3 refuses against a date32 column
    (pandas GH#62157), while the benchmark's pinned pandas 2.3 accepted it.
    """
    def table(columns, rows):
        return pd.DataFrame(rows, columns=columns).convert_dtypes(dtype_backend="pyarrow")

    lineitem_ds = table(
        ["l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity",
         "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus",
         "l_shipdate"],
        [
            [1, 1, 101, 1, 100.0, 100.0, 0.04, 0.02, "N", "O", date(1994, 1, 1)],
            [1, 1, 103, 2, 10.0, 100.0, 0.09, 0.06, "N", "O", date(1994, 6, 9)],
            [2, 2, 103, 1, 20.0, 100.0, 0.02, 0.00, "R", "F", date(1994, 12, 31)],
            [2, 2, 102, 2, 40.0, 100.0, 0.06, 0.04, "A", "F", date(1994, 7, 3)],
            [3, 1, 104, 1, 10.0, 100.0, 0.00, 0.08, "A", "F", date(1994, 3, 4)],
            [3, 3, 105, 2, 10.0, 100.0, 0.03, 0.01, "N", "O", date(1994, 3, 6)],
            [4, 1, 101, 1, 900.0, 100.0, 0.07, 0.03, "R", "F", date(1993, 12, 31)],
            [4, 2, 103, 2, 900.0, 100.0, 0.01, 0.05, "R", "F", date(1995, 1, 1)],
        ],
    )
    nation_ds = table(
        ["n_nationkey", "n_name", "n_regionkey", "n_comment"],
        [
            [3, "CANADA", 1, "Canada"],
            [6, "FRANCE", 3, "France"],
        ],
    )
    part_ds = table(
        ["p_partkey", "p_name", "p_mfgr", "p_brand", "p_type",
         "p_size", "p_container", "p_retailprice", "p_comment"],
        [
            [1, "forest green", "MFGR1", "BRAND1", "STANDARD TIN", 15, "BOX", 100.0, "Qualifying"],
            [2, "forest lime", "MFGR2", "BRAND2", "SMALL TIN", 15, "BOX", 100.0, "Qualifying"],
            [2, "forest lime", "MFGR2", "BRAND2", "SMALL TIN", 15, "BOX", 100.0, "Repeated part key"],
            [3, "green forest", "MFGR3", "BRAND3", "LARGE TIN", 15, "BOX", 100.0, "Word not first"],
        ],
    )
    part_supp_ds = table(
        ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"],
        [
            [1, 101, 100, 10.0, "Above half of 100"],
            [1, 103, 100, 10.0, "Above half of 10"],
            [2, 103, 11, 10.0, "Just above half of 20"],
            [2, 102, 20, 10.0, "Exactly half of 40"],
            [1, 104, 100, 10.0, "French supplier"],
            [3, 105, 100, 10.0, "Part name does not qualify"],
        ],
    )
    supplier_ds = table(
        ["s_suppkey", "s_name", "s_address", "s_nationkey",
         "s_phone", "s_acctbal", "s_comment"],
        [
            [101, "Supplier Z", "Toronto", 3, "13-101-101", 500.0, "Qualifying"],
            [102, "Supplier Y", "Ottawa", 3, "13-102-102", 600.0, "Too little stock"],
            [103, "Supplier A", "Calgary", 3, "13-103-103", 700.0, "Qualifying twice"],
            [104, "Supplier B", "Lyon", 6, "16-104-104", 800.0, "French"],
            [105, "Supplier C", "Halifax", 3, "13-105-105", 900.0, "Wrong part name"],
        ],
    )
    for frame, column in [(lineitem_ds, "l_quantity"), (lineitem_ds, "l_extendedprice"),
                          (lineitem_ds, "l_discount"), (lineitem_ds, "l_tax"),
                          (part_ds, "p_retailprice"), (part_supp_ds, "ps_supplycost"),
                          (supplier_ds, "s_acctbal")]:
        frame[column] = pd.array(frame[column], dtype="double[pyarrow]")
    lineitem_ds["l_shipdate"] = lineitem_ds["l_shipdate"].astype("timestamp[us][pyarrow]")

    return dict(lineitem_ds=lineitem_ds, nation_ds=nation_ds, part_ds=part_ds,
                part_supp_ds=part_supp_ds, supplier_ds=supplier_ds)


def q20_reference(lineitem_ds, nation_ds, part_ds, part_supp_ds, supplier_ds):
    """The upstream pandas query, with the five tables supplied by the fixture."""
    var1 = START
    var2 = END
    var3 = "CANADA"
    var4 = "forest"

    # Aggregate lineitem by partkey and suppkey
    filtered_lineitem = lineitem_ds[
        (lineitem_ds["l_shipdate"] >= var1) & (lineitem_ds["l_shipdate"] < var2)
    ]
    qty_agg = filtered_lineitem.groupby(
        ["l_partkey", "l_suppkey"], as_index=False
    ).agg(sum_quantity=pd.NamedAgg(column="l_quantity", aggfunc="sum"))
    qty_agg["sum_quantity"] = qty_agg["sum_quantity"] * 0.5

    # Filter nation
    filtered_nation = nation_ds[nation_ds["n_name"] == var3]

    # Filter parts starting with "forest"
    filtered_part = part_ds[part_ds["p_name"].str.startswith(var4)][
        ["p_partkey"]
    ].drop_duplicates()

    # Join partsupp with filtered parts
    jn1 = filtered_part.merge(
        part_supp_ds, left_on="p_partkey", right_on="ps_partkey"
    )

    # Join with quantity aggregation
    jn2 = jn1.merge(
        qty_agg,
        left_on=["ps_suppkey", "p_partkey"],
        right_on=["l_suppkey", "l_partkey"],
    )

    # Filter by availqty > sum_quantity
    jn2 = jn2[jn2["ps_availqty"] > jn2["sum_quantity"]]

    # Get unique suppliers
    unique_suppliers = jn2[["ps_suppkey"]].drop_duplicates()

    # Join with supplier and nation
    jn3 = unique_suppliers.merge(
        supplier_ds, left_on="ps_suppkey", right_on="s_suppkey"
    )
    jn4 = jn3.merge(filtered_nation, left_on="s_nationkey", right_on="n_nationkey")

    result = jn4.loc[:, ["s_name", "s_address"]]
    return result.sort_values("s_name")


def build_q20_prefix(lineitem_ds, nation_ds, part_ds, part_supp_ds, supplier_ds):
    """Q20 up to the final projection: every filter, both deduplications and all
    four joins.

    ``var1`` to ``var4`` stay as ``skrub.var``s so the values reach the fold as
    ``ValueOp``s rather than inline literals; they are only needed inside this
    stage, so they are not returned. DataOps are immutable, so the
    ``qty_agg[...] = ...`` assignment becomes one ``.assign(...)``.
    """
    var1 = st.var("var1", START)
    var2 = st.var("var2", END)
    var3 = st.var("var3", "CANADA")
    var4 = st.var("var4", "forest")
    lineitem_ds = st.var("lineitem_ds", lineitem_ds)
    nation_ds = st.var("nation_ds", nation_ds)
    part_ds = st.var("part_ds", part_ds)
    part_supp_ds = st.var("part_supp_ds", part_supp_ds)
    supplier_ds = st.var("supplier_ds", supplier_ds)

    # Aggregate lineitem by partkey and suppkey
    filtered_lineitem = lineitem_ds[
        (lineitem_ds["l_shipdate"] >= var1) & (lineitem_ds["l_shipdate"] < var2)
    ]
    qty_agg = filtered_lineitem.groupby(
        ["l_partkey", "l_suppkey"], as_index=False
    ).agg(sum_quantity=pd.NamedAgg(column="l_quantity", aggfunc="sum"))
    qty_agg = qty_agg.assign(sum_quantity=qty_agg["sum_quantity"] * 0.5)

    # Filter nation
    filtered_nation = nation_ds[nation_ds["n_name"] == var3]

    # Filter parts starting with "forest"
    filtered_part = part_ds[part_ds["p_name"].str.startswith(var4)][
        ["p_partkey"]
    ].drop_duplicates()

    # Join partsupp with filtered parts
    jn1 = filtered_part.merge(
        part_supp_ds, left_on="p_partkey", right_on="ps_partkey"
    )

    # Join with quantity aggregation
    jn2 = jn1.merge(
        qty_agg,
        left_on=["ps_suppkey", "p_partkey"],
        right_on=["l_suppkey", "l_partkey"],
    )

    # Filter by availqty > sum_quantity
    jn2 = jn2[jn2["ps_availqty"] > jn2["sum_quantity"]]

    # Get unique suppliers
    unique_suppliers = jn2[["ps_suppkey"]].drop_duplicates()

    # Join with supplier and nation
    jn3 = unique_suppliers.merge(
        supplier_ds, left_on="ps_suppkey", right_on="s_suppkey"
    )
    return jn3.merge(filtered_nation, left_on="s_nationkey", right_on="n_nationkey")


def build_q20(lineitem_ds, nation_ds, part_ds, part_supp_ds, supplier_ds):
    """Complete Q20, column selection uses ``[]`` indexing instead of pandas ``loc``."""
    jn4 = build_q20_prefix(lineitem_ds, nation_ds, part_ds, part_supp_ds, supplier_ds)

    result = jn4[["s_name", "s_address"]]
    return result.sort_values("s_name")


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

def test_q20_matches_reference(polars):
    """The skrubified query returns what the upstream query returns."""
    tables = make_q20_tables()
    expected = q20_reference(**tables)
    result = _api.evaluate(build_q20(**tables))
    if polars:
        assert isinstance(result, pl.DataFrame)
        result = result.to_pandas()

    assert list(expected.columns) == list(result.columns)
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True),
        result.reset_index(drop=True),
        check_dtype=False,  # compare values across Arrow-backed input and runtime output
    )


def test_q20_prefix_runs_on_polars():
    tables = make_q20_tables()
    line_item = tables["lineitem_ds"]
    line_item = line_item[(line_item["l_shipdate"] >= START)
                          & (line_item["l_shipdate"] < END)]
    qty_agg = line_item.groupby(["l_partkey", "l_suppkey"], as_index=False).agg(
        sum_quantity=pd.NamedAgg(column="l_quantity", aggfunc="sum"))
    qty_agg = qty_agg.assign(sum_quantity=qty_agg["sum_quantity"] * 0.5)
    nation = tables["nation_ds"]
    part = tables["part_ds"]
    filtered_part = part[part["p_name"].str.startswith("forest")][["p_partkey"]]
    jn = (
        filtered_part.drop_duplicates()
        .merge(tables["part_supp_ds"], left_on="p_partkey", right_on="ps_partkey")
        .merge(qty_agg, left_on=["ps_suppkey", "p_partkey"],
               right_on=["l_suppkey", "l_partkey"])
    )
    jn = jn[jn["ps_availqty"] > jn["sum_quantity"]]
    expected = (
        jn[["ps_suppkey"]].drop_duplicates()
        .merge(tables["supplier_ds"], left_on="ps_suppkey", right_on="s_suppkey")
        .merge(nation[nation["n_name"] == "CANADA"],
               left_on="s_nationkey", right_on="n_nationkey")
    )
    with st.config(implementation_selector="greedy"):
        result = _api.evaluate(build_q20_prefix(**tables))
    assert isinstance(result, pl.DataFrame)
    # Polars joins may reorder rows; this comparison checks contents, not pandas row order.
    pd.testing.assert_frame_equal(
        expected.sort_values("s_suppkey").reset_index(drop=True),
        result.to_pandas().sort_values("s_suppkey").reset_index(drop=True),
        check_dtype=False)


# --- the folded plan ------------------------------------------------------

def test_q20_plan_folds_joins(polars):
    """Each merge in the prefix collapses to one join operator, the composite
    supplier and part key included."""
    ops = _plan(build_q20_prefix(**make_q20_tables()))
    joins = [o for o in ops if isinstance(o, JoinOp)]

    assert [("p_partkey", "ps_partkey"),
            (["ps_suppkey", "p_partkey"], ["l_suppkey", "l_partkey"]),
            ("ps_suppkey", "s_suppkey"), ("s_nationkey", "n_nationkey")] \
        == [(o.left_on, o.right_on) for o in joins]
    if polars:
        assert all(isinstance(o, PolarsJoinOp) for o in joins)


def test_q20_filters_inline_the_constants(polars):
    """Window, nation and name prefix are absorbed as constants; the stock filter
    compares two columns."""
    ops = _plan(build_q20_prefix(**make_q20_tables()))
    filters = [o for o in ops if isinstance(o, MaskSelectionOp)]

    assert sorted([BinOpExpr(operator.and_,
                             BinOpExpr(operator.ge, Col("l_shipdate"), Const(START)),
                             BinOpExpr(operator.lt, Col("l_shipdate"), Const(END))),
                   BinOpExpr(operator.eq, Col("n_name"), Const("CANADA")),
                   StrExpr(Col("p_name"), "startswith", {"pat": "forest"}),
                   BinOpExpr(operator.gt, Col("ps_availqty"), Col("sum_quantity"))],
                  key=repr) \
        == sorted((o.predicate for o in filters), key=repr)
    for sel in filters:
        assert len(sel.inputs) == 1, "the filter constants should not be graph inputs"
    assert [] == [o for o in ops if isinstance(o, ValueOp)]
    if polars:
        assert all(isinstance(o, PolarsMaskSelectionOp) for o in filters)


def test_q20_folds_the_quantity_sum_and_its_scaling(polars):
    """The ``NamedAgg`` sum folds into one aggregate over the composite key, its
    scaling into one assign map that overwrites the column."""
    ops = _plan(build_q20_prefix(**make_q20_tables()))
    agg = _one(ops, AggregateOp)
    amo = _one(ops, AssignMapOp)

    assert (Col("l_partkey"), Col("l_suppkey")) == agg.grouping
    assert (("sum_quantity", AggExpr("sum", Col("l_quantity"))),) == agg.aggregations
    assert {"sum_quantity": BinOpExpr(operator.mul, Col("sum_quantity"), Const(0.5))} \
        == amo.entries
    assert amo.inputs == [agg]
    if polars:
        assert isinstance(agg, PolarsAggregateOp)
        assert isinstance(amo, PolarsAssignMapOp)


def test_q20_folds_the_deduplications(polars):
    """Both ``drop_duplicates`` calls fold, each over a one-column projection."""
    ops = _plan(build_q20_prefix(**make_q20_tables()))
    dedups = [o for o in ops if isinstance(o, DropDuplicatesSelectionOp)]

    assert sorted(["p_partkey", "ps_suppkey"]) \
        == sorted(o.inputs[0].key[0] for o in dedups)
    for op in dedups:
        assert isinstance(op.inputs[0], ColumnProjectionOp)
        assert (None, "first") == (op.subset, op.keep)
    if polars:
        assert all(isinstance(o, PolarsDropDuplicatesSelectionOp) for o in dedups)


def test_q20_folds_the_projection_and_sort(polars):
    """The final projection and sort fold and close the plan."""
    ops = _plan(build_q20(**make_q20_tables()))
    sort = _one(ops, SortOp)
    projection = sort.inputs[0]

    assert isinstance(projection, ColumnProjectionOp)
    assert ["s_name", "s_address"] == projection.key
    assert (("s_name",), True) == (sort.by, sort.ascending)
    assert ops[-1] is sort
    if polars:
        assert isinstance(projection, PolarsColumnProjectionOp)
        assert isinstance(sort, PolarsSortOp)


def test_q20_filter_binds_query_impl_under_flag():
    """``pandas_query`` picks the impl at plan time; both filter correctly.

    The window, the nation and the column comparison can use the query fast path.
    A string method has no query spelling, so the name filter stays a mask
    selection. Pandas-only, so no fixture.
    """
    from stratum.optimizer.physical._selection_execs import (
        PandasIndexSelectionOp, PandasQuerySelectionOp)

    tables = make_q20_tables()
    nation = tables["nation_ds"]
    expected = nation[nation["n_name"] == "CANADA"].reset_index(drop=True)
    name = StrExpr(Col("p_name"), "startswith", {"pat": "forest"})
    canada = BinOpExpr(operator.eq, Col("n_name"), Const("CANADA"))

    for flag, impl in [(False, PandasIndexSelectionOp), (True, PandasQuerySelectionOp)]:
        dag = build_q20(**tables)
        ops = optimize_(dag, OptConfig(dataframe_ops=True, pandas_query=flag),
                        env=get_data(dag))[0]
        filters = [o for o in ops if isinstance(o, MaskSelectionOp)]
        assert len(filters) == 4
        for sel in filters:
            assert isinstance(sel, PandasIndexSelectionOp if sel.predicate == name else impl)
        sel = next(o for o in filters if o.predicate == canada)
        pd.testing.assert_frame_equal(
            expected,
            pd.DataFrame(sel.process([nation])).reset_index(drop=True),
            check_dtype=False)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
