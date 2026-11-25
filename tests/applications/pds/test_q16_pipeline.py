"""PDS Q16 run through Stratum, checked against the query it was ported from.

The source is https://github.com/pola-rs/polars-benchmark/blob/main/queries/pandas/q16.py.
:func:`q16_reference`
is that query without the benchmark's IO wrapper; :func:`build_q16` is the same
thing as a skrub DAG. Both run on the same fixtures and must agree.

The query counts the distinct suppliers per brand, type and size. A regex match
on the supplier comment finds the suppliers with complaints; a left join followed
by ``isna`` removes them. Filters on brand, type prefix and size come before it,
and the distinct count and a mixed-direction sort after it.
"""
import operator

import pytest
import pandas as pd
import polars as pl

import stratum as st
from stratum.frontend import _api
from stratum.optimizer._optimize import OptConfig, optimize as optimize_
from stratum.optimizer.logical._aggregation_ops import AggregateOp
from stratum.optimizer.logical._column_expr import (
    AggExpr, BinOpExpr, Col, Const, IsinExpr, IsnaExpr, StrExpr, UnaryOpExpr)
from stratum.optimizer.logical._join_ops import JoinOp
from stratum.optimizer.logical._ops import ValueOp
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp
from stratum.optimizer.logical._selection_ops import MaskSelectionOp
from stratum.optimizer.logical._sort_ops import SortOp
from stratum.optimizer.physical._aggregation_execs import PolarsAggregateOp
from stratum.optimizer.physical._join_execs import PolarsJoinOp
from stratum.optimizer.physical._projection_execs import PolarsColumnProjectionOp
from stratum.optimizer.physical._selection_execs import PolarsMaskSelectionOp
from stratum.optimizer.physical._sort_execs import PolarsSortOp
from stratum.frontend._skrub_graph import get_data

# The upstream query's comment pattern and size list.
PATTERN = ".*Customer.*Complaints.*"
SIZES = [49, 14, 23, 45, 19, 3, 36, 9]

GROUP_KEYS = ["p_brand", "p_type", "p_size"]


def make_q16_tables():
    """Arrow-backed part, partsupp and supplier.

    Parts 1 and 2 share a group with suppliers 101 and 102, 102 twice, so the
    distinct count is 2 where a plain count would be 3. Parts 3 and 8 form two
    groups of one, in brand order, so the sort needs its second key. Every other
    part fails exactly one check: the excluded brand, the excluded type prefix,
    a size outside the list, or only a supplier with complaints.

    Supplier 105 matches the pattern and is removed. Supplier 102's comment has
    the two words reversed, and supplier 103 has no comment, which ``na=False``
    treats as no match, so both stay.
    """
    def table(columns, rows):
        return pd.DataFrame(rows, columns=columns).convert_dtypes(dtype_backend="pyarrow")

    part_ds = table(
        ["p_partkey", "p_name", "p_mfgr", "p_brand", "p_type",
         "p_size", "p_container", "p_retailprice", "p_comment"],
        [
            [1, "Part A", "MFGR1", "Brand#12", "STANDARD POLISHED TIN", 49, "BOX", 100.0, "Qualifying"],
            [2, "Part B", "MFGR1", "Brand#12", "STANDARD POLISHED TIN", 49, "BOX", 100.0, "Same group"],
            [3, "Part C", "MFGR2", "Brand#23", "SMALL PLATED COPPER", 14, "BOX", 100.0, "Qualifying"],
            [4, "Part D", "MFGR3", "Brand#45", "SMALL PLATED COPPER", 14, "BOX", 100.0, "Excluded brand"],
            [5, "Part E", "MFGR2", "Brand#23", "MEDIUM POLISHED BRASS", 14, "BOX", 100.0, "Excluded type"],
            [6, "Part F", "MFGR2", "Brand#23", "SMALL PLATED COPPER", 7, "BOX", 100.0, "Excluded size"],
            [7, "Part G", "MFGR4", "Brand#34", "ECONOMY BRUSHED STEEL", 3, "BOX", 100.0, "Complaints only"],
            [8, "Part H", "MFGR1", "Brand#12", "LARGE ANODIZED NICKEL", 9, "BOX", 100.0, "Qualifying"],
        ],
    )
    part_supp_ds = table(
        ["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost", "ps_comment"],
        [
            [1, 101, 100, 10.0, "First supplier"],
            [1, 102, 100, 10.0, "Second supplier"],
            [2, 102, 100, 10.0, "Second supplier again"],
            [3, 103, 100, 10.0, "No supplier comment"],
            [3, 105, 100, 10.0, "Complaints"],
            [4, 101, 100, 10.0, "Excluded brand"],
            [5, 101, 100, 10.0, "Excluded type"],
            [6, 101, 100, 10.0, "Excluded size"],
            [7, 105, 100, 10.0, "Complaints"],
            [8, 101, 100, 10.0, "First supplier"],
        ],
    )
    supplier_ds = table(
        ["s_suppkey", "s_name", "s_address", "s_nationkey",
         "s_phone", "s_acctbal", "s_comment"],
        [
            [101, "Supplier A", "Recife", 2, "12-101-101", 500.0, "Reliable supplier"],
            [102, "Supplier B", "Munich", 7, "17-102-102", 600.0, "Complaints from a Customer"],
            [103, "Supplier C", "Lyon", 6, "16-103-103", 700.0, None],
            [105, "Supplier E", "Denver", 24, "34-105-105", 800.0, "Customer filed Complaints"],
        ],
    )
    for frame, column in [(part_ds, "p_retailprice"), (part_supp_ds, "ps_supplycost"),
                          (supplier_ds, "s_acctbal")]:
        frame[column] = pd.array(frame[column], dtype="double[pyarrow]")

    return dict(part_ds=part_ds, part_supp_ds=part_supp_ds, supplier_ds=supplier_ds)


def q16_reference(part_ds, part_supp_ds, supplier_ds):
    """The upstream pandas query, with the three tables supplied by the fixture."""
    var1 = "Brand#45"

    # Filter suppliers with complaints
    filtered_supplier = supplier_ds[
        supplier_ds["s_comment"].str.contains(
            ".*Customer.*Complaints.*", regex=True, na=False
        )
    ][["s_suppkey"]]

    jn = part_ds.merge(part_supp_ds, left_on="p_partkey", right_on="ps_partkey")

    jn = jn[jn["p_brand"] != var1]
    jn = jn[~jn["p_type"].str.startswith("MEDIUM POLISHED")]
    jn = jn[jn["p_size"].isin([49, 14, 23, 45, 19, 3, 36, 9])]

    # Left join to exclude suppliers with complaints
    jn2 = jn.merge(
        filtered_supplier,
        left_on="ps_suppkey",
        right_on="s_suppkey",
        how="left",
    )
    jn2 = jn2[jn2["s_suppkey"].isna()]

    gb = jn2.groupby(["p_brand", "p_type", "p_size"], as_index=False)
    agg = gb.agg(supplier_cnt=pd.NamedAgg(column="ps_suppkey", aggfunc="nunique"))

    return agg.sort_values(
        by=["supplier_cnt", "p_brand", "p_type", "p_size"],
        ascending=[False, True, True, True],
    )


def build_q16_prefix(part_ds, part_supp_ds, supplier_ds):
    """Q16 up to the aggregation: the supplier selection, the part filters and
    the left join with its ``isna`` filter.

    ``var1`` stays a ``skrub.var`` so the brand reaches the fold as a ``ValueOp``
    rather than an inline literal. The upstream pattern, prefix and sizes are
    literals and stay literals.
    """
    var1 = st.var("var1", "Brand#45")
    part_ds = st.var("part_ds", part_ds)
    part_supp_ds = st.var("part_supp_ds", part_supp_ds)
    supplier_ds = st.var("supplier_ds", supplier_ds)

    # Filter suppliers with complaints
    filtered_supplier = supplier_ds[
        supplier_ds["s_comment"].str.contains(
            ".*Customer.*Complaints.*", regex=True, na=False
        )
    ][["s_suppkey"]]

    jn = part_ds.merge(part_supp_ds, left_on="p_partkey", right_on="ps_partkey")

    jn = jn[jn["p_brand"] != var1]
    jn = jn[~jn["p_type"].str.startswith("MEDIUM POLISHED")]
    jn = jn[jn["p_size"].isin([49, 14, 23, 45, 19, 3, 36, 9])]

    # Left join to exclude suppliers with complaints
    jn2 = jn.merge(
        filtered_supplier,
        left_on="ps_suppkey",
        right_on="s_suppkey",
        how="left",
    )
    return jn2[jn2["s_suppkey"].isna()]


def build_q16(part_ds, part_supp_ds, supplier_ds):
    """Complete Q16."""
    jn2 = build_q16_prefix(part_ds, part_supp_ds, supplier_ds)

    gb = jn2.groupby(["p_brand", "p_type", "p_size"], as_index=False)
    agg = gb.agg(supplier_cnt=pd.NamedAgg(column="ps_suppkey", aggfunc="nunique"))

    return agg.sort_values(
        by=["supplier_cnt", "p_brand", "p_type", "p_size"],
        ascending=[False, True, True, True],
    )


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

def test_q16_matches_reference(polars):
    """The skrubified query returns what the upstream query returns."""
    tables = make_q16_tables()
    expected = q16_reference(**tables)
    assert list(expected.itertuples(index=False, name=None)) == [
        ("Brand#12", "STANDARD POLISHED TIN", 49, 2),
        ("Brand#12", "LARGE ANODIZED NICKEL", 9, 1),
        ("Brand#23", "SMALL PLATED COPPER", 14, 1),
    ]
    result = _api.evaluate(build_q16(**tables))
    if polars:
        assert isinstance(result, pl.DataFrame)
        result = result.to_pandas()

    assert list(expected.columns) == list(result.columns)
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True),
        result.reset_index(drop=True),
        check_dtype=False,  # compare values across Arrow-backed input and runtime output
    )


def test_q16_prefix_runs_on_polars():
    tables = make_q16_tables()
    supplier = tables["supplier_ds"]
    complaints = supplier[supplier["s_comment"].str.contains(PATTERN, regex=True, na=False)
                          ][["s_suppkey"]]
    expected = tables["part_ds"].merge(tables["part_supp_ds"],
                                       left_on="p_partkey", right_on="ps_partkey")
    expected = expected[expected["p_brand"] != "Brand#45"]
    expected = expected[~expected["p_type"].str.startswith("MEDIUM POLISHED")]
    expected = expected[expected["p_size"].isin(SIZES)]
    expected = expected.merge(complaints, left_on="ps_suppkey", right_on="s_suppkey",
                              how="left")
    expected = expected[expected["s_suppkey"].isna()]
    with st.config(implementation_selector="greedy"):
        result = _api.evaluate(build_q16_prefix(**tables))
    assert isinstance(result, pl.DataFrame)
    # Keep Arrow nulls for unmatched suppliers, and normalize only the join's row order.
    pd.testing.assert_frame_equal(
        expected.sort_values(["p_partkey", "ps_suppkey"]).reset_index(drop=True),
        result.to_pandas(use_pyarrow_extension_array=True)
        .sort_values(["p_partkey", "ps_suppkey"]).reset_index(drop=True),
        check_dtype=False)


# --- the folded plan ------------------------------------------------------

def test_q16_plan_folds_the_joins(polars):
    """Both merges collapse to one join operator each, the second a left join
    with the projected suppliers."""
    ops = _plan(build_q16_prefix(**make_q16_tables()))
    inner, left = [o for o in ops if isinstance(o, JoinOp)]

    assert ("inner", "p_partkey", "ps_partkey") == (inner.how, inner.left_on, inner.right_on)
    assert ("left", "ps_suppkey", "s_suppkey") == (left.how, left.left_on, left.right_on)
    assert ["s_suppkey"] == _one(ops, ColumnProjectionOp).key
    if polars:
        assert isinstance(inner, PolarsJoinOp)
        assert isinstance(left, PolarsJoinOp)
        assert isinstance(_one(ops, ColumnProjectionOp), PolarsColumnProjectionOp)


def test_q16_filters_fold(polars):
    """Each filter collapses to one mask, the brand inlined as a ``Const``."""
    ops = _plan(build_q16_prefix(**make_q16_tables()))
    filters = [o for o in ops if isinstance(o, MaskSelectionOp)]

    predicates = [o.predicate for o in filters]

    # The selectors emit these in different orders, so check each one by membership.
    assert len(predicates) == 5
    assert BinOpExpr(operator.ne, Col("p_brand"), Const("Brand#45")) in predicates
    assert UnaryOpExpr(operator.invert, StrExpr(Col("p_type"), "startswith",
                                                {"pat": "MEDIUM POLISHED"})) in predicates
    assert IsinExpr(Col("p_size"), SIZES) in predicates
    assert StrExpr(Col("s_comment"), "contains",
                   {"pat": PATTERN, "regex": True, "na": False}) in predicates
    assert IsnaExpr(Col("s_suppkey")) in predicates
    for sel in filters:
        assert len(sel.inputs) == 1, "the filter constants should not be graph inputs"
    assert [] == [o for o in ops if isinstance(o, ValueOp)]
    if polars:
        assert all(isinstance(o, PolarsMaskSelectionOp) for o in filters)


def test_q16_folds_the_distinct_count(polars):
    """The ``nunique`` over three keys folds into one aggregate and runs end to end."""
    tables = make_q16_tables()
    agg = _one(_plan(build_q16(**tables)), AggregateOp)

    assert tuple(Col(key) for key in GROUP_KEYS) == agg.grouping
    assert (("supplier_cnt", AggExpr("nunique", Col("ps_suppkey"))),) == agg.aggregations
    if polars:
        assert isinstance(agg, PolarsAggregateOp)

    result = _api.evaluate(build_q16(**tables))
    result = result.to_pandas() if isinstance(result, pl.DataFrame) else result
    pd.testing.assert_frame_equal(q16_reference(**tables).reset_index(drop=True),
                                  result.reset_index(drop=True), check_dtype=False)


def test_q16_folds_the_sort(polars):
    """The mixed-direction sort folds and closes the plan."""
    ops = _plan(build_q16(**make_q16_tables()))
    sort = _one(ops, SortOp)

    assert (("supplier_cnt", "p_brand", "p_type", "p_size"), (False, True, True, True)) \
        == (sort.by, sort.ascending)
    assert sort.inputs[0] is _one(ops, AggregateOp) and ops[-1] is sort
    if polars:
        assert isinstance(sort, PolarsSortOp)


def test_q16_filter_binds_query_impl_under_flag():
    """``pandas_query`` picks the impl at plan time; both filter correctly.

    Only the brand comparison can use the query fast path: ``str``, ``isin`` and
    ``isna`` are column methods with no query spelling, so those filters stay
    mask selections. Pandas-only, so no fixture.
    """
    from stratum.optimizer.physical._selection_execs import (
        PandasIndexSelectionOp, PandasQuerySelectionOp)

    tables = make_q16_tables()
    df = tables["part_ds"]
    expected = df[df["p_brand"] != "Brand#45"].reset_index(drop=True)

    for flag, impl in [(False, PandasIndexSelectionOp), (True, PandasQuerySelectionOp)]:
        dag = build_q16(**tables)
        ops = optimize_(dag, OptConfig(dataframe_ops=True, pandas_query=flag),
                        env=get_data(dag))[0]
        brand = BinOpExpr(operator.ne, Col("p_brand"), Const("Brand#45"))
        filters = [o for o in ops if isinstance(o, MaskSelectionOp)]
        for sel in filters:
            assert isinstance(sel, impl if sel.predicate == brand
                              else PandasIndexSelectionOp)
        # The brand filter runs on the joined frame in the query; the brand is a
        # part column, so it filters the part table alone the same way.
        sel = next(o for o in filters if o.predicate == brand)
        pd.testing.assert_frame_equal(
            expected,
            pd.DataFrame(sel.process([df])).reset_index(drop=True),
            check_dtype=False)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
