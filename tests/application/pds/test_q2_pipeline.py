"""PDS Q2 run through Stratum, checked against the query it was ported from.

The source is ``queries/pandas/q2.py`` in polars-benchmark. :func:`q2_reference`
is that query without the benchmark's IO wrapper; :func:`build_q2` is the skrub
DAG from ``canvas.ipynb``. Both run on the same fixtures and must agree.

The query joins five tables, filters parts and suppliers, finds the minimum
supply cost per part, and joins back to retain every tied supplier. Projection,
mixed-direction sorting and ``head(100)`` finish the pipeline.
"""
import operator

import pytest
import pandas as pd
import polars as pl

import stratum as st
from stratum.optimizer._optimize import OptConfig, optimize as optimize_
from stratum.optimizer.logical._aggregation_ops import AggregateOp
from stratum.optimizer.logical._column_expr import AggExpr, BinOpExpr, Col, Const, StrExpr
from stratum.optimizer.logical._join_ops import JoinOp
from stratum.optimizer.logical._ops import MethodCallOp, ValueOp
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp
from stratum.optimizer.logical._selection_ops import MaskSelectionOp, SliceSelectionOp
from stratum.optimizer.logical._sort_ops import SortOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._conversion import ConversionOp
from stratum.optimizer.physical._aggregation_execs import PolarsAggregateOp
from stratum.optimizer.physical._join_execs import PolarsJoinOp
from stratum.optimizer.physical._projection_execs import PolarsColumnProjectionOp
from stratum.optimizer.physical._selection_execs import (
    PolarsMaskSelectionOp, PolarsSliceSelectionOp)
from stratum.optimizer.physical._sort_execs import PolarsSortOp
from stratum.frontend._skrub_graph import get_data


def make_q2_tables():
    """The notebook's Arrow-backed region, nation, supplier, part and partsupp.

    Part 1001 has tied European minima and a cheaper Asian supplier. Wrong-size,
    wrong-type and Asia-only parts exercise each filter independently.
    """
    def table(columns, rows):
        return pd.DataFrame(rows, columns=columns).convert_dtypes(dtype_backend="pyarrow")

    region_ds = table(
        ["r_regionkey", "r_name", "r_comment"],
        [
            [1, "EUROPE", "European region"],
            [2, "ASIA", "Asian region"],
        ],
    )
    nation_ds = table(
        ["n_nationkey", "n_name", "n_regionkey", "n_comment"],
        [
            [10, "GERMANY", 1, "Germany"],
            [20, "FRANCE", 1, "France"],
            [30, "CHINA", 2, "China"],
        ],
    )
    supplier_ds = table(
        ["s_suppkey", "s_name", "s_address", "s_nationkey",
         "s_phone", "s_acctbal", "s_comment"],
        [
            [101, "Supplier A", "Berlin", 10, "111-111", 120.0, "German supplier"],
            [102, "Supplier B", "Paris", 20, "222-222", 200.0, "French supplier"],
            [103, "Supplier C", "Munich", 10, "333-333", 90.0, "German supplier"],
            [104, "Supplier D", "Beijing", 30, "444-444", 300.0, "Chinese supplier"],
        ],
    )
    part_ds = table(
        ["p_partkey", "p_name", "p_mfgr", "p_brand", "p_type",
         "p_size", "p_container", "p_retailprice", "p_comment"],
        [
            [1001, "Part A", "MFGR1", "BRAND1", "STANDARD BRASS", 15, "BOX", 100.0, "Valid part"],
            [1002, "Part B", "MFGR2", "BRAND2", "SMALL BRASS", 15, "BOX", 200.0, "Valid part"],
            [1003, "Part C", "MFGR3", "BRAND3", "STANDARD BRASS", 20, "BOX", 150.0, "Wrong size"],
            [1004, "Part D", "MFGR4", "BRAND4", "POLISHED STEEL", 15, "BOX", 180.0, "Wrong type"],
            [1005, "Part E", "MFGR5", "BRAND5", "LARGE BRASS", 15, "BOX", 250.0, "Asia only"],
        ],
    )
    part_supp_ds = table(
        ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"],
        [
            [1001, 101, 100, 9.0, "European minimum"],
            [1001, 102, 200, 9.0, "Tied European minimum"],
            [1001, 103, 150, 12.0, "More expensive"],
            [1001, 104, 100, 2.0, "Cheaper, but outside Europe"],
            [1002, 101, 100, 20.0, "More expensive"],
            [1002, 102, 100, 13.0, "European minimum"],
            [1003, 101, 100, 1.0, "Wrong part size"],
            [1004, 102, 100, 1.0, "Wrong part type"],
            [1005, 104, 100, 1.0, "Supplier outside Europe"],
        ],
    )
    for frame, column in [(supplier_ds, "s_acctbal"), (part_ds, "p_retailprice"),
                          (part_supp_ds, "ps_supplycost")]:
        frame[column] = pd.array(frame[column], dtype="double[pyarrow]")

    return dict(region_ds=region_ds, nation_ds=nation_ds, supplier_ds=supplier_ds,
                part_ds=part_ds, part_supp_ds=part_supp_ds)


def q2_reference(region_ds, nation_ds, supplier_ds, part_ds, part_supp_ds):
    """The upstream pandas query, with the five tables supplied by the fixture."""
    var1 = 15
    var2 = "BRASS"
    var3 = "EUROPE"

    jn = (
        part_ds.merge(part_supp_ds, left_on="p_partkey", right_on="ps_partkey")
        .merge(supplier_ds, left_on="ps_suppkey", right_on="s_suppkey")
        .merge(nation_ds, left_on="s_nationkey", right_on="n_nationkey")
        .merge(region_ds, left_on="n_regionkey", right_on="r_regionkey")
    )
    jn = jn[jn["p_size"] == var1]
    jn = jn[jn["p_type"].str.endswith(var2)]
    jn = jn[jn["r_name"] == var3]

    gb = jn.groupby("p_partkey", as_index=False)
    agg = gb["ps_supplycost"].min()
    jn2 = agg.merge(jn, on=["p_partkey", "ps_supplycost"])

    sel = jn2.loc[:, ["s_acctbal", "s_name", "n_name", "p_partkey", "p_mfgr",
                     "s_address", "s_phone", "s_comment"]]
    sort = sel.sort_values(
        by=["s_acctbal", "n_name", "s_name", "p_partkey"],
        ascending=[False, True, True, True],
    )
    return sort.head(100)


def build_q2_prefix(region_ds, nation_ds, supplier_ds, part_ds, part_supp_ds):
    """Q2 up to the aggregation: join the five tables, then apply the filters.

    Like ``build_q1_prefix``, this is the pre-aggregation stage as a single
    DataOp, so tests can evaluate and inspect it without the rest of the query.
    ``var1``, ``var2`` and ``var3`` stay as ``skrub.var``s so the values reach
    the fold as ``ValueOp``s rather than inline literals; they are only needed
    inside this stage, so they are not returned.
    """
    var1 = st.var("var1", 15)
    var2 = st.var("var2", "BRASS")
    var3 = st.var("var3", "EUROPE")
    region_ds = st.var("region_ds", region_ds)
    nation_ds = st.var("nation_ds", nation_ds)
    supplier_ds = st.var("supplier_ds", supplier_ds)
    part_ds = st.var("part_ds", part_ds)
    part_supp_ds = st.var("part_supp_ds", part_supp_ds)

    jn = (
        part_ds.merge(part_supp_ds, left_on="p_partkey", right_on="ps_partkey")
        .merge(supplier_ds, left_on="ps_suppkey", right_on="s_suppkey")
        .merge(nation_ds, left_on="s_nationkey", right_on="n_nationkey")
        .merge(region_ds, left_on="n_regionkey", right_on="r_regionkey")
    )
    jn = jn[jn["p_size"] == var1]
    jn = jn[jn["p_type"].str.endswith(var2)]
    jn = jn[jn["r_name"] == var3]
    return jn


def build_q2(region_ds, nation_ds, supplier_ds, part_ds, part_supp_ds):
    """Complete Q2, column selection uses ``[]`` indexing instead of pandas ``loc``."""
    jn = build_q2_prefix(
        region_ds, nation_ds, supplier_ds, part_ds, part_supp_ds)

    gb = jn.groupby("p_partkey", as_index=False)
    agg = gb["ps_supplycost"].min()
    jn2 = agg.merge(jn, on=["p_partkey", "ps_supplycost"])

    sel = jn2[["s_acctbal", "s_name", "n_name", "p_partkey", "p_mfgr",
               "s_address", "s_phone", "s_comment"]]
    sort = sel.sort_values(
        by=["s_acctbal", "n_name", "s_name", "p_partkey"],
        ascending=[False, True, True, True],
    )
    return sort.head(100)


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

def test_q2_matches_reference(polars):
    """The skrubified query returns what the upstream query returns."""
    tables = make_q2_tables()
    expected = q2_reference(**tables)
    result = st._api.evaluate(build_q2(**tables))
    if polars:
        assert isinstance(result, pl.DataFrame)
        result = result.to_pandas()

    assert list(expected.columns) == list(result.columns)
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True),
        result.reset_index(drop=True),
        check_dtype=False,  # compare values across Arrow-backed input and runtime output
    )


def test_q2_prefix_runs_on_polars():
    tables = make_q2_tables()
    expected = (
        tables["part_ds"].merge(tables["part_supp_ds"],
                                left_on="p_partkey", right_on="ps_partkey")
        .merge(tables["supplier_ds"], left_on="ps_suppkey", right_on="s_suppkey")
        .merge(tables["nation_ds"], left_on="s_nationkey", right_on="n_nationkey")
        .merge(tables["region_ds"], left_on="n_regionkey", right_on="r_regionkey")
    )
    expected = expected[expected["p_size"] == 15]
    expected = expected[expected["p_type"].str.endswith("BRASS")]
    expected = expected[expected["r_name"] == "EUROPE"]
    with st.config(implementation_selector="greedy"):
        result = st._api.evaluate(build_q2_prefix(**tables))
    assert isinstance(result, pl.DataFrame)
    # Polars joins may reorder rows, this comparison checks contents, not pandas row order.
    pd.testing.assert_frame_equal(
        expected.sort_values(["p_partkey", "ps_suppkey"]).reset_index(drop=True),
        result.to_pandas().sort_values(["p_partkey", "ps_suppkey"]).reset_index(drop=True),
        check_dtype=False)


# --- the folded plan ------------------------------------------------------

def test_q2_plan_folds_joins(polars):
    """Each merge in the prefix collapses to one physical join operator."""
    ops = _plan(build_q2_prefix(**make_q2_tables()))
    joins = [o for o in ops if isinstance(o, JoinOp)]

    assert [("p_partkey", "ps_partkey"), ("ps_suppkey", "s_suppkey"),
            ("s_nationkey", "n_nationkey"), ("n_regionkey", "r_regionkey")] \
        == [(o.left_on, o.right_on) for o in joins]
    for op in joins:
        assert len(op.inputs) == 2, "each join should need only its two frames"
        assert isinstance(op, PhysicalOp), \
            f"{type(op).__name__} was not bound to a physical implementation"
    if polars:
        assert all(isinstance(o, PolarsJoinOp) for o in joins)


def test_q2_filter_inlines_the_constants(polars):
    """The three filter variables are absorbed into the predicates.

    The size and region become ``Const`` expressions, the suffix becomes the
    string expression's argument. Each filter needs only its frame input.
    """
    ops = _plan(build_q2(**make_q2_tables()))
    filters = [o for o in ops if isinstance(o, MaskSelectionOp)]

    assert [BinOpExpr(operator.eq, Col("p_size"), Const(15)),
            StrExpr(Col("p_type"), "endswith", {"pat": "BRASS"}),
            BinOpExpr(operator.eq, Col("r_name"), Const("EUROPE"))] \
        == [o.predicate for o in filters]
    for sel in filters:
        assert len(sel.inputs) == 1, "the filter constants should not be graph inputs"
        assert isinstance(sel, PhysicalOp), \
            f"{type(sel).__name__} was not bound to a physical implementation"
        if polars:
            assert isinstance(sel, PolarsMaskSelectionOp)
    assert [] == [o for o in ops if isinstance(o, ValueOp)]


def test_q2_folds_the_grouped_aggregation(polars):
    """The grouped minimum folds into one aggregate and shares its input with
    the join-back. The whole query runs on either backend, with no conversion.
    """
    tables = make_q2_tables()
    ops = _plan(build_q2(**tables))
    agg = _one(ops, AggregateOp)
    joins = [o for o in ops if isinstance(o, JoinOp)]
    join_back = joins[-1]

    assert agg.grouped
    assert (Col("p_partkey"),) == agg.grouping
    assert ((None, AggExpr("min", Col("ps_supplycost"))),) == agg.aggregations
    assert agg.keys_as_columns
    assert ["p_partkey", "ps_supplycost"] == join_back.left_on == join_back.right_on
    assert join_back.inputs[0] is agg
    assert len(agg.inputs) == 1
    assert join_back.inputs[1] is agg.inputs[0], \
        "the minimum and join-back should share the filtered frame"
    assert [] == [o for o in ops if isinstance(o, (MethodCallOp, ConversionOp))]
    if polars:
        assert isinstance(agg, PolarsAggregateOp)

    result = st._api.evaluate(build_q2(**tables))
    result = result.to_pandas() if isinstance(result, pl.DataFrame) else result
    pd.testing.assert_frame_equal(q2_reference(**tables).reset_index(drop=True),
                                  result.reset_index(drop=True), check_dtype=False)


def test_q2_folds_the_sort_and_limit(polars):
    """Projection, sorting and ``head`` fold into physical frame operators.

    More than 100 tied suppliers makes the limit observable after sorting.
    """
    tables = make_q2_tables()
    # Repeat the first supplier/partsupp rows 105 times, then give them
    # distinct keys and balances. Every copy supplies part 1001 (valid size,
    # type and region) at the same 9.0 cost, so all 105 tie for the European
    # minimum and the join-back keeps all of them. The distinct balances make
    # the sort order deterministic, so ``head(100)`` must drop exactly 5 rows.
    suppliers = tables["supplier_ds"].iloc[[0] * 105].reset_index(drop=True)
    suppliers["s_suppkey"] = pd.array(range(1000, 1105), dtype="int64[pyarrow]")
    suppliers["s_acctbal"] = pd.array(range(105), dtype="double[pyarrow]")
    partsupp = tables["part_supp_ds"].iloc[[0] * 105].reset_index(drop=True)
    partsupp["ps_suppkey"] = suppliers["s_suppkey"]
    tables.update(supplier_ds=suppliers, part_supp_ds=partsupp)

    ops = _plan(build_q2(**tables))
    projection = _one(ops, ColumnProjectionOp)
    sort = _one(ops, SortOp)
    head = _one(ops, SliceSelectionOp)

    assert ["s_acctbal", "s_name", "n_name", "p_partkey", "p_mfgr",
            "s_address", "s_phone", "s_comment"] == projection.key
    assert ("s_acctbal", "n_name", "s_name", "p_partkey") == sort.by
    assert (False, True, True, True) == sort.ascending
    assert sort.inputs[0] is projection
    assert head.n == 100 and not head.from_end
    assert head.inputs[0] is sort
    assert ops[-1] is head
    for op in (projection, sort, head):
        assert isinstance(op, PhysicalOp), \
            f"{type(op).__name__} was not bound to a physical implementation"
    if polars:
        assert isinstance(projection, PolarsColumnProjectionOp)
        assert isinstance(sort, PolarsSortOp)
        assert isinstance(head, PolarsSliceSelectionOp)

    result = st._api.evaluate(build_q2(**tables))
    result = result.to_pandas() if isinstance(result, pl.DataFrame) else result
    pd.testing.assert_frame_equal(q2_reference(**tables).reset_index(drop=True),
                                  result.reset_index(drop=True), check_dtype=False)


def test_q2_filter_binds_query_impl_under_flag():
    """``pandas_query`` picks the impl at plan time; both filter correctly.

    Size and region can use the query fast path once their values are inlined;
    the string suffix remains a mask selection. Pandas-only, so no fixture.
    """
    from stratum.optimizer.physical._selection_execs import (
        PandasIndexSelectionOp, PandasQuerySelectionOp)

    tables = make_q2_tables()
    # The unfiltered join in pandas: input frame for running the selections
    # by hand. Building it here keeps this test off the prefix, which is
    # already filtered.
    df = (
        tables["part_ds"].merge(tables["part_supp_ds"],
                                left_on="p_partkey", right_on="ps_partkey")
        .merge(tables["supplier_ds"], left_on="ps_suppkey", right_on="s_suppkey")
        .merge(tables["nation_ds"], left_on="s_nationkey", right_on="n_nationkey")
        .merge(tables["region_ds"], left_on="n_regionkey", right_on="r_regionkey")
    )
    expected = df[(df["p_size"] == 15) & df["p_type"].str.endswith("BRASS")
                  & (df["r_name"] == "EUROPE")].reset_index(drop=True)

    def by_predicate(ops, predicate):
        matches = [o for o in ops
                   if isinstance(o, MaskSelectionOp) and o.predicate == predicate]
        assert len(matches) == 1, \
            f"expected exactly one selection for {predicate}, got {len(matches)}"
        return matches[0]

    for flag, impl in [(False, PandasIndexSelectionOp), (True, PandasQuerySelectionOp)]:
        dag = build_q2(**tables)
        ops = optimize_(dag, OptConfig(dataframe_ops=True, pandas_query=flag),
                        env=get_data(dag))[0]
        # Look each filter up by predicate, not by position in the plan, so
        # this test does not care what order the optimizer emits them in.
        # They are still applied in query order below.
        size = by_predicate(ops, BinOpExpr(operator.eq, Col("p_size"), Const(15)))
        suffix = by_predicate(ops, StrExpr(Col("p_type"), "endswith", {"pat": "BRASS"}))
        region = by_predicate(ops, BinOpExpr(operator.eq, Col("r_name"),
                                             Const("EUROPE")))
        assert isinstance(size, impl)
        assert isinstance(suffix, PandasIndexSelectionOp)
        assert isinstance(region, impl)
        result = df
        for sel in (size, suffix, region):
            result = sel.process("fit_transform", [result])
        pd.testing.assert_frame_equal(
            expected, pd.DataFrame(result).reset_index(drop=True), check_dtype=False)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
