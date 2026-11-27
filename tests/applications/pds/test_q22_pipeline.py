"""PDS Q22 run through Stratum, checked against the query it was ported from.

The source is ``queries/pandas/q22.py`` in polars-benchmark. :func:`q22_reference`
is that query without the benchmark's IO wrapper; :func:`build_q22` is the same
thing as a skrub DAG. Both run on the same fixtures and must agree.

The query measures customers with above-average balances and no orders, per
country code. A code column cut from the phone number feeds a regex match that
keeps seven codes. The mean of the positive balances is a scalar step that reaches
a later filter as a graph input; a left join with ``isna`` removes the customers
that have orders.
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
    AggExpr, BinOpExpr, Col, IsnaExpr, OperandLeaf, StrExpr, Const)
from stratum.optimizer.logical._combine_ops import AssignOp
from stratum.optimizer.logical._join_ops import JoinOp
from stratum.optimizer.logical._map_ops import ExpressionMapOp
from stratum.optimizer.logical._ops import OperandRef
from stratum.optimizer.logical._selection_ops import (
    DropDuplicatesSelectionOp, MaskSelectionOp)
from stratum.optimizer.logical._sort_ops import SortOp
from stratum.optimizer.physical._aggregation_execs import PolarsAggregateOp
from stratum.optimizer.physical._fused_map_execs import PolarsFusedAssignOp
from stratum.optimizer.physical._join_execs import PolarsJoinOp
from stratum.optimizer.physical._selection_execs import (
    PolarsDropDuplicatesSelectionOp, PolarsMaskSelectionOp)
from stratum.optimizer.physical._sort_execs import PolarsSortOp
from stratum.frontend._skrub_graph import get_data

# The regex over country codes in the upstream query.
CODES = "13|31|23|29|30|18|17"


def make_q22_tables():
    """Arrow-backed customer and orders.

    The positive balances among the selected codes average 475. Customers 2 and
    9 clear it in code 13 and customer 3 in code 31, so the two groups have
    different counts and totals. Customer 4 clears it but has two orders, which
    the deduplication folds into one match. Customer 5 sits below the average,
    and customer 6's negative balance stays out of it. Customer 7's code is not
    selected and customer 8 has no phone, which ``na=False`` treats as no match;
    their balances would raise the average if they were counted.
    """
    def table(columns, rows):
        return pd.DataFrame(rows, columns=columns).convert_dtypes(dtype_backend="pyarrow")

    customer_ds = table(
        ["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone",
         "c_acctbal", "c_mktsegment", "c_comment"],
        [
            [1, "Customer A", "Address A", 3, "13-111-111", 100.0, "BUILDING", "Below the average"],
            [2, "Customer B", "Address B", 3, "13-222-222", 500.0, "MACHINERY", "Qualifying"],
            [3, "Customer C", "Address C", 21, "31-333-333", 700.0, "AUTOMOBILE", "Qualifying"],
            [4, "Customer D", "Address D", 21, "31-444-444", 900.0, "BUILDING", "Has orders"],
            [5, "Customer E", "Address E", 7, "17-555-555", 50.0, "HOUSEHOLD", "Below the average"],
            [6, "Customer F", "Address F", 7, "17-666-666", -100.0, "FURNITURE", "Negative balance"],
            [7, "Customer G", "Address G", 9, "99-777-777", 9000.0, "BUILDING", "Other code"],
            [8, "Customer H", "Address H", 9, None, 9000.0, "BUILDING", "No phone"],
            [9, "Customer I", "Address I", 3, "13-999-999", 600.0, "MACHINERY", "Qualifying"],
        ],
    )
    orders_ds = table(
        ["o_orderkey", "o_custkey", "o_orderstatus", "o_totalprice", "o_orderdate",
         "o_orderpriority", "o_clerk", "o_shippriority", "o_comment"],
        [
            [1501, 4, "O", 100.0, date(1995, 1, 1), "1-URGENT", "Clerk A", 0, "First order"],
            [1502, 4, "F", 100.0, date(1995, 2, 1), "2-HIGH", "Clerk B", 0, "Second order"],
            [1503, 7, "F", 100.0, date(1995, 3, 1), "3-MEDIUM", "Clerk C", 0, "Other code"],
        ],
    )
    for frame, column in [(customer_ds, "c_acctbal"), (orders_ds, "o_totalprice")]:
        frame[column] = pd.array(frame[column], dtype="double[pyarrow]")
    orders_ds["o_orderdate"] = pd.array(orders_ds["o_orderdate"],
                                        dtype="date32[day][pyarrow]")

    return dict(customer_ds=customer_ds, orders_ds=orders_ds)


def q22_reference(customer_ds, orders_ds):
    """The upstream pandas query, with the two tables supplied by the fixture."""
    # Extract country code (first 2 chars of phone)
    customer_with_cntry = customer_ds.copy()
    customer_with_cntry["cntrycode"] = customer_with_cntry["c_phone"].str.slice(0, 2)

    # Filter by country codes
    filtered_customer = customer_with_cntry[
        customer_with_cntry["cntrycode"].str.match("13|31|23|29|30|18|17", na=False)
    ][["c_acctbal", "c_custkey", "cntrycode"]]

    # Calculate average account balance for positive balances
    avg_acctbal = filtered_customer[filtered_customer["c_acctbal"] > 0.0][
        "c_acctbal"
    ].mean()

    # Get unique customer keys from orders
    customers_with_orders = orders_ds[["o_custkey"]].drop_duplicates()

    # Left join to find customers without orders
    jn = filtered_customer.merge(
        customers_with_orders,
        left_on="c_custkey",
        right_on="o_custkey",
        how="left",
    )
    jn = jn[jn["o_custkey"].isna()]

    # Filter by account balance > average
    jn = jn[jn["c_acctbal"] > avg_acctbal]

    # Group by country code
    gb = jn.groupby("cntrycode", as_index=False)
    agg = gb.agg(
        numcust=pd.NamedAgg(column="c_acctbal", aggfunc="count"),
        totacctbal=pd.NamedAgg(column="c_acctbal", aggfunc="sum"),
    )

    return agg.sort_values("cntrycode")


def build_q22_filtered_customer(customer_ds):
    """The selected customers. DataOps are immutable, so the upstream ``.copy()``
    and the ``cntrycode`` assignment become one ``.assign(...)``."""
    customer_with_cntry = customer_ds.assign(
        cntrycode=customer_ds["c_phone"].str.slice(0, 2)
    )
    return customer_with_cntry[
        customer_with_cntry["cntrycode"].str.match("13|31|23|29|30|18|17", na=False)
    ][["c_acctbal", "c_custkey", "cntrycode"]]


def build_q22_prefix(customer_ds, orders_ds):
    """Q22 up to the average: the selected customers, the left join with the
    deduplicated order customers, and its ``isna`` filter. The upstream query has
    no ``var``s; its literals stay literals."""
    customer_ds = st.var("customer_ds", customer_ds)
    orders_ds = st.var("orders_ds", orders_ds)

    filtered_customer = build_q22_filtered_customer(customer_ds)

    # Get unique customer keys from orders
    customers_with_orders = orders_ds[["o_custkey"]].drop_duplicates()

    # Left join to find customers without orders
    jn = filtered_customer.merge(
        customers_with_orders,
        left_on="c_custkey",
        right_on="o_custkey",
        how="left",
    )
    return jn[jn["o_custkey"].isna()]


def build_q22(customer_ds, orders_ds):
    """Complete Q22."""
    customer_ds = st.var("customer_ds", customer_ds)
    orders_ds = st.var("orders_ds", orders_ds)

    filtered_customer = build_q22_filtered_customer(customer_ds)

    # Calculate average account balance for positive balances
    avg_acctbal = filtered_customer[filtered_customer["c_acctbal"] > 0.0][
        "c_acctbal"
    ].mean()

    # Get unique customer keys from orders
    customers_with_orders = orders_ds[["o_custkey"]].drop_duplicates()

    # Left join to find customers without orders
    jn = filtered_customer.merge(
        customers_with_orders,
        left_on="c_custkey",
        right_on="o_custkey",
        how="left",
    )
    jn = jn[jn["o_custkey"].isna()]

    # Filter by account balance > average
    jn = jn[jn["c_acctbal"] > avg_acctbal]

    # Group by country code
    gb = jn.groupby("cntrycode", as_index=False)
    agg = gb.agg(
        numcust=pd.NamedAgg(column="c_acctbal", aggfunc="count"),
        totacctbal=pd.NamedAgg(column="c_acctbal", aggfunc="sum"),
    )

    return agg.sort_values("cntrycode")


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

def test_q22_matches_reference(polars):
    """The skrubified query returns what the upstream query returns."""
    tables = make_q22_tables()
    expected = q22_reference(**tables)
    result = _api.evaluate(build_q22(**tables))
    if polars:
        assert isinstance(result, pl.DataFrame)
        result = result.to_pandas()

    assert list(expected.columns) == list(result.columns)
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True),
        result.reset_index(drop=True),
        check_dtype=False,  # compare values across Arrow-backed input and runtime output
    )


def test_q22_prefix_runs_on_polars():
    tables = make_q22_tables()
    customer = tables["customer_ds"]
    customer = customer.assign(cntrycode=customer["c_phone"].str.slice(0, 2))
    filtered = customer[customer["cntrycode"].str.match(CODES, na=False)
                        ][["c_acctbal", "c_custkey", "cntrycode"]]
    expected = filtered.merge(tables["orders_ds"][["o_custkey"]].drop_duplicates(),
                              left_on="c_custkey", right_on="o_custkey", how="left")
    expected = expected[expected["o_custkey"].isna()]
    with st.config(implementation_selector="greedy"):
        result = _api.evaluate(build_q22_prefix(**tables))
    assert isinstance(result, pl.DataFrame)
    # Keep Arrow nulls for unmatched customers, and normalize only the join's row order.
    pd.testing.assert_frame_equal(
        expected.sort_values("c_custkey").reset_index(drop=True),
        result.to_pandas(use_pyarrow_extension_array=True)
        .sort_values("c_custkey").reset_index(drop=True),
        check_dtype=False)


# --- the folded plan ------------------------------------------------------

def test_q22_folds_the_country_code(polars):
    """The code column collapses to one assign over the phone number."""
    ops = _plan(build_q22_prefix(**make_q22_tables()))
    amo = _one(ops, PolarsFusedAssignOp if polars else AssignOp)

    if polars:
        assert {"cntrycode": StrExpr(Col("c_phone"), "slice", {"start": 0, "stop": 2})} \
            == amo.entries
        assert len(amo.inputs) == 1, "the map should need only the customer frame"
    else:
        assert ["cntrycode"] == list(amo.kwargs)
        sliced = amo.inputs[amo.kwargs["cntrycode"].k]
        assert isinstance(sliced, ExpressionMapOp)
        assert sliced.inputs[0].key == "c_phone"
        assert isinstance(sliced.expr, StrExpr) and sliced.expr.method == "slice"
        assert sliced.expr.params == {"start": 0, "stop": 2}
        assert len(amo.inputs) == 2


def test_q22_plan_folds_the_left_join(polars):
    """The merge collapses to one left join with the deduplicated order customers."""
    ops = _plan(build_q22_prefix(**make_q22_tables()))
    join = _one(ops, JoinOp)
    dedup = _one(ops, DropDuplicatesSelectionOp)

    assert ("left", "c_custkey", "o_custkey") == (join.how, join.left_on, join.right_on)
    assert join.inputs[1] is dedup
    assert ["o_custkey"] == dedup.inputs[0].key
    if polars:
        assert isinstance(join, PolarsJoinOp)
        assert isinstance(dedup, PolarsDropDuplicatesSelectionOp)


def test_q22_filters_fold(polars):
    """The code match and the ``isna`` each collapse to one mask.

    The positive-balance filter feeds the pandas ``mean`` with the prefix, so
    only the prefix filters can stay on polars; the ``> 0`` constant is covered
    by the query-impl test and the average filter by the scalar test below.
    """
    ops = _plan(build_q22_prefix(**make_q22_tables()))
    filters = [o for o in ops if isinstance(o, MaskSelectionOp)]
    predicates = [o.predicate for o in filters]

    # The selectors emit these in different orders, so check each one by membership.
    assert len(predicates) == 2
    assert StrExpr(Col("cntrycode"), "match", {"pat": CODES, "na": False}) in predicates
    assert IsnaExpr(Col("o_custkey")) in predicates
    for sel in filters:
        assert len(sel.inputs) == 1, "the filter constants should not be graph inputs"
    if polars:
        assert all(isinstance(o, PolarsMaskSelectionOp) for o in filters)


def test_q22_average_filter_reads_the_scalar(polars):
    """The filter on the balance compares with the average as its one extra input."""
    ops = _plan(build_q22(**make_q22_tables()))
    average = [o for o in ops if isinstance(o, MaskSelectionOp)
               and o.predicate == BinOpExpr(operator.gt, Col("c_acctbal"),
                                            OperandLeaf(OperandRef(1)))]

    assert len(average) == 1
    assert len(average[0].inputs) == 2, "the filter should read the join and the average"
    if polars:
        assert isinstance(average[0], PolarsMaskSelectionOp)


def test_q22_folds_the_named_aggregation_and_sort(polars):
    """Both ``NamedAgg``s fold into one grouped aggregate, the sort closes the
    plan, and the query runs end to end. The scalar ``mean`` stays a plain call
    feeding the average filter."""
    tables = make_q22_tables()
    ops = _plan(build_q22(**tables))
    agg = _one(ops, AggregateOp)
    sort = _one(ops, SortOp)

    assert (Col("cntrycode"),) == agg.grouping
    assert (("numcust", AggExpr("count", Col("c_acctbal"))),
            ("totacctbal", AggExpr("sum", Col("c_acctbal")))) == agg.aggregations
    assert (("cntrycode",), True) == (sort.by, sort.ascending)
    assert sort.inputs[0] is agg and ops[-1] is sort
    if polars:
        assert isinstance(agg, PolarsAggregateOp)
        assert isinstance(sort, PolarsSortOp)

    result = _api.evaluate(build_q22(**tables))
    result = result.to_pandas() if isinstance(result, pl.DataFrame) else result
    pd.testing.assert_frame_equal(q22_reference(**tables).reset_index(drop=True),
                                  result.reset_index(drop=True), check_dtype=False)


def test_q22_filter_binds_query_impl_under_flag():
    """``pandas_query`` picks the impl at plan time; both filter correctly.

    Only the positive balance can use the query fast path: a string method and
    ``isna`` have no query spelling, and the average stays an ``OperandLeaf``,
    which makes ``to_pandas_query`` return ``None``. Pandas-only, so no fixture.
    """
    from stratum.optimizer.physical._selection_execs import (
        PandasIndexSelectionOp, PandasQuerySelectionOp)

    tables = make_q22_tables()
    customer = tables["customer_ds"]
    expected = customer[customer["c_acctbal"] > 0.0].reset_index(drop=True)
    positive = BinOpExpr(operator.gt, Col("c_acctbal"), Const(0.0))

    for flag, impl in [(False, PandasIndexSelectionOp), (True, PandasQuerySelectionOp)]:
        dag = build_q22(**tables)
        ops = optimize_(dag, OptConfig(dataframe_ops=True, pandas_query=flag),
                        env=get_data(dag))[0]
        filters = [o for o in ops if isinstance(o, MaskSelectionOp)]
        assert len(filters) == 4
        for sel in filters:
            assert isinstance(sel, impl if sel.predicate == positive
                              else PandasIndexSelectionOp)
        sel = next(o for o in filters if o.predicate == positive)
        pd.testing.assert_frame_equal(
            expected,
            pd.DataFrame(sel.process([customer])).reset_index(drop=True),
            check_dtype=False)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
