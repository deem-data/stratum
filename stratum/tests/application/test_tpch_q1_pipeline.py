"""TPC-H Q1 run through Stratum, checked against the query it was ported from.

The source is ``queries/modin/q1.py`` in pola-rs/polars-benchmark. :func:`q1_reference`
is that query; :func:`build_q1` is the same thing as a skrub DAG. Both run on the same
fixture and must agree, so the port is verified rather than assumed.

A pure analytics pipeline, so the frame abstractions are exercised on their own,
unlike ``test_frame_ops_pipeline``. That also makes this a place to read the
compiled plan and look for operators that could still fold; anything found that
way gets its own assertion here once it is fixed.
"""
import operator
from datetime import date

import pytest
import pandas as pd

import stratum as st
from stratum.optimizer._optimize import OptConfig, optimize as optimize_
from stratum.optimizer.ir._column_expr import BinOpExpr, Col, Const
from stratum.optimizer.ir._map_ops import AssignMapOp
from stratum.optimizer.ir._ops import ValueOp
from stratum.optimizer.ir._selection_ops import SelectionKind, SelectionOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.tests.logical_optimizer.test_dataframe_ops import force_polars
from stratum.utils._skrub_graph import get_data

# The Q1 cutoff: `var1` in the upstream query.
SHIPDATE_CUTOFF = date(1998, 9, 2)


def make_lineitem():
    """A 12-row ``lineitem``, Arrow-backed like the benchmark's parquet read.

    Every (l_returnflag, l_linestatus) group keeps > 1 rows through the filter, so
    the mean aggregations are not trivially equal to their inputs.
    """
    df = pd.DataFrame({
        "l_orderkey": [1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9],
        "l_partkey": [1552, 673, 1062, 429, 1901, 88, 1444, 203, 1777, 956, 1205, 314],
        "l_suppkey": [93, 28, 75, 41, 12, 67, 39, 81, 54, 22, 64, 17],
        "l_linenumber": [1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1],
        "l_quantity": [17.0, 36.0, 8.0, 28.0, 11.0, 4.0, 31.0, 6.0, 19.0, 42.0, 13.0, 25.0],
        "l_extendedprice": [21168.23, 45983.16, 10432.80, 37984.52, 14872.11,
                            5320.44, 41870.09, 8123.76, 25674.31, 56892.18,
                            17654.72, 33741.50],
        "l_discount": [0.04, 0.09, 0.02, 0.06, 0.00, 0.03, 0.07, 0.01, 0.05, 0.08, 0.02, 0.06],
        "l_tax": [0.02, 0.06, 0.00, 0.04, 0.08, 0.01, 0.03, 0.05, 0.07, 0.02, 0.06, 0.04],
        "l_returnflag": ["N", "N", "R", "A", "A", "N", "R", "R", "A", "N", "R", "A"],
        "l_linestatus": ["O", "O", "F", "F", "F", "O", "F", "F", "F", "O", "F", "F"],
        # Straddles the cutoff: rows 3 and 8 fall outside it.
        "l_shipdate": [date(1998, 8, 30), date(1998, 9, 2), date(1998, 7, 14),
                       date(1998, 9, 3), date(1997, 12, 18), date(1996, 5, 9),
                       date(1998, 8, 1), date(1995, 10, 21), date(1998, 9, 5),
                       date(1994, 3, 11), date(1998, 6, 23), date(1997, 1, 7)],
    }).convert_dtypes(dtype_backend="pyarrow")
    for column in ["l_quantity", "l_extendedprice", "l_discount", "l_tax"]:
        df[column] = pd.array(df[column], dtype="double[pyarrow]")
    df["l_shipdate"] = pd.array(df["l_shipdate"], dtype="date32[day][pyarrow]")
    return df


def q1_reference(line_item_ds):
    """Q1 verbatim, with pandas standing in for ``modin.pandas`` and a ``.copy()``
    so the in-place assignments do not touch the fixture."""
    var1 = SHIPDATE_CUTOFF

    filt = line_item_ds[line_item_ds["l_shipdate"] <= var1].copy()

    filt["disc_price"] = filt.l_extendedprice * (1.0 - filt.l_discount)
    filt["charge"] = (
        filt.l_extendedprice * (1.0 - filt.l_discount) * (1.0 + filt.l_tax)
    )

    gb = filt.groupby(["l_returnflag", "l_linestatus"], as_index=False)
    agg = gb.agg(
        sum_qty=pd.NamedAgg(column="l_quantity", aggfunc="sum"),
        sum_base_price=pd.NamedAgg(column="l_extendedprice", aggfunc="sum"),
        sum_disc_price=pd.NamedAgg(column="disc_price", aggfunc="sum"),
        sum_charge=pd.NamedAgg(column="charge", aggfunc="sum"),
        avg_qty=pd.NamedAgg(column="l_quantity", aggfunc="mean"),
        avg_price=pd.NamedAgg(column="l_extendedprice", aggfunc="mean"),
        avg_disc=pd.NamedAgg(column="l_discount", aggfunc="mean"),
        count_order=pd.NamedAgg(column="l_orderkey", aggfunc="size"),
    )

    return agg.sort_values(["l_returnflag", "l_linestatus"])


def build_q1(line_item_df):
    """The same query as a skrub DAG.

    DataOps are immutable, so the two ``filt[...] = ...`` assignments become one
    ``.assign(...)`` and ``filt.l_extendedprice`` becomes ``filt["..."]`` (attribute
    access on a DataOp is a ``GetAttr`` step). ``var1`` stays a ``skrub.var`` so the
    cutoff reaches the fold as a ``ValueOp`` rather than an inline literal.
    """
    line_item = st.var("line_item", line_item_df)
    var1 = st.var("var1", SHIPDATE_CUTOFF)

    filt = line_item[line_item["l_shipdate"] <= var1]
    filt = filt.assign(
        disc_price=filt["l_extendedprice"] * (1.0 - filt["l_discount"]),
        charge=(filt["l_extendedprice"] * (1.0 - filt["l_discount"])
                * (1.0 + filt["l_tax"])),
    )

    gb = filt.groupby(["l_returnflag", "l_linestatus"], as_index=False)
    agg = gb.agg(
        sum_qty=pd.NamedAgg(column="l_quantity", aggfunc="sum"),
        sum_base_price=pd.NamedAgg(column="l_extendedprice", aggfunc="sum"),
        sum_disc_price=pd.NamedAgg(column="disc_price", aggfunc="sum"),
        sum_charge=pd.NamedAgg(column="charge", aggfunc="sum"),
        avg_qty=pd.NamedAgg(column="l_quantity", aggfunc="mean"),
        avg_price=pd.NamedAgg(column="l_extendedprice", aggfunc="mean"),
        avg_disc=pd.NamedAgg(column="l_discount", aggfunc="mean"),
        count_order=pd.NamedAgg(column="l_orderkey", aggfunc="size"),
    )

    return agg.sort_values(["l_returnflag", "l_linestatus"])


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
    with force_polars(request.param):
        yield request.param


# --- parity ----------------------------------------------------------------

def test_q1_matches_reference(polars):
    """The skrubified query returns what the upstream query returns."""
    df = make_lineitem()
    expected = q1_reference(df)
    result = st._api.evaluate(build_q1(df))

    assert list(expected.columns) == list(result.columns)
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True),
        pd.DataFrame(result).reset_index(drop=True),
        check_dtype=False,  # the polars round-trip renormalises the Arrow dtypes
    )


# --- the folded plan ------------------------------------------------------

def test_q1_plan_folds_filter_and_map(polars):
    """The filter and the two derived columns each collapse to one physical op.

    Both derived columns are pure column arithmetic, so neither needs an
    ``OperandLeaf`` and the map takes only the filtered frame.
    """
    ops = _plan(build_q1(make_lineitem()))
    sel = _one(ops, SelectionOp)
    amo = _one(ops, AssignMapOp)

    assert sel.kind is SelectionKind.MASK

    disc_price = BinOpExpr(operator.mul,
                           Col("l_extendedprice"),
                           BinOpExpr(operator.sub, Const(1.0), Col("l_discount")))
    assert ["disc_price", "charge"] == list(amo.entries)
    assert disc_price == amo.entries["disc_price"]
    assert BinOpExpr(operator.mul, disc_price,
                     BinOpExpr(operator.add, Const(1.0), Col("l_tax"))) \
        == amo.entries["charge"]
    assert len(amo.inputs) == 1, "the map should need only the filtered frame"

    for op in (sel, amo):
        assert isinstance(op, PhysicalOp), \
            f"{type(op).__name__} was not bound to a physical implementation"


def test_q1_filter_inlines_the_date_constant(polars):
    """``var1`` reaches the fold as a ``ValueOp`` input, not an inline literal.

    A ``date`` is a scalar, so the predicate absorbs it as a ``Const`` and the frame
    is the selection's only input.
    """
    ops = _plan(build_q1(make_lineitem()))
    sel = _one(ops, SelectionOp)

    assert BinOpExpr(operator.le,
                     Col("l_shipdate"), Const(SHIPDATE_CUTOFF)) == sel.predicate
    assert len(sel.inputs) == 1, "the date threshold should not be a graph input"
    assert [] == [o for o in ops if isinstance(o, ValueOp)]


def test_q1_filter_binds_query_impl_under_flag():
    """``pandas_query`` picks the impl at plan time, and both compute Q1.

    The fast path is reachable only because the date is inlined: an ``OperandLeaf``
    makes ``to_pandas_query`` return ``None``. Pandas-only, so no fixture.
    """
    from stratum.optimizer.physical._selection_execs import (
        PandasIndexSelectionOp, PandasQuerySelectionOp)

    df = make_lineitem()
    expected = q1_reference(df).reset_index(drop=True)

    for flag, impl in [(False, PandasIndexSelectionOp), (True, PandasQuerySelectionOp)]:
        with st.config(pandas_query=flag):
            assert isinstance(_one(_plan(build_q1(df)), SelectionOp), impl)
            result = st._api.evaluate(build_q1(df))
        pd.testing.assert_frame_equal(
            expected, pd.DataFrame(result).reset_index(drop=True), check_dtype=False)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
