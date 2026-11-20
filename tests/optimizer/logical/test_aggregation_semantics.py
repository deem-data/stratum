"""Semantic regressions for expression aggregation and filtering joins."""
import operator

import numpy as np
import pandas as pd
import polars as pl
import pytest
import stratum as st

from stratum.frontend import _api
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import OptConfig, optimize as optimize_
from stratum.optimizer.logical._aggregation_ops import AggregateOp
from stratum.optimizer.logical._column_expr import AggExpr, BinOpExpr, Col
from stratum.optimizer.logical._ops import MethodCallOp, OutputType
from stratum.optimizer.physical._aggregation_execs import PolarsAggregateOp
from stratum.optimizer.physical._join_execs import (
    PandasIsInSemiJoinOp, PandasMergeSemiJoinOp, PolarsFilteringJoinOp)
from .test_dataframe_ops import optimize, run_op, force_polars


def execute(expression):
    ops = optimize(expression, OptConfig(dataframe_ops=True))
    values = {}
    for op in ops:
        values[id(op)] = op.process([values[id(i)] for i in op.inputs])
    return values[id(ops[-1])], ops


def test_cse_keeps_series_and_frame_aggregations_distinct():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1., 2., 3.]})
    def build(d):
        return st.deferred(lambda x, y: (x, y))(
            d.groupby("g")["v"].sum(), d.groupby("g").agg({"v": "sum"}))
    result, ops = execute(build(st.as_data_op(df)))
    pd.testing.assert_series_equal(result[0], df.groupby("g")["v"].sum())
    pd.testing.assert_frame_equal(result[1], df.groupby("g").agg({"v": "sum"}))
    aggs = [op for op in ops if isinstance(op, AggregateOp)]
    assert len(aggs) == 2
    for agg in aggs:
        logical = AggregateOp(**{field: getattr(agg, field) for field in AggregateOp.fields})
        assert logical.clone().output_type is agg.output_type


@pytest.mark.parametrize("selected", [False, True])
def test_as_index_false(selected):
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    def build(d):
        grouped = d.groupby("g", as_index=False)
        return grouped["v"].sum() if selected else grouped.agg({"v": "sum"})
    result, ops = execute(build(st.as_data_op(df)))
    pd.testing.assert_frame_equal(result, build(df))
    assert next(op for op in ops if isinstance(op, AggregateOp)).output_type is OutputType.FRAME


@pytest.mark.parametrize("method,arg", [("std", 0), ("sum", True)])
def test_direct_positional_arguments_remain_semantic(method, arg):
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1., 2., 3.]})
    def build(d):
        return getattr(d.groupby("g"), method)(arg)
    result, _ = execute(build(st.as_data_op(df)))
    pd.testing.assert_frame_equal(result, build(df))


def test_unnamed_computed_frame_output_has_one_logical_name():
    data = {"g": ["a", "a"], "a": [2, 3], "b": [4, 5]}
    def make():
        return AggregateOp(grouped=True, grouping=(Col("g"),), aggregations=(
            (None, AggExpr("sum", BinOpExpr(operator.mul, Col("a"), Col("b")))),))
    pandas_result = run_op(make(), pd.DataFrame(data))
    with force_polars():
        polars_result = run_op(make(), pl.DataFrame(data))
    assert pandas_result.columns.tolist() == ["_agg0"]
    assert polars_result["_agg0"].to_list() == pandas_result["_agg0"].tolist()


@pytest.mark.parametrize("dropna", [True, False])
@pytest.mark.parametrize("func", ["sum", "first", "last"])
def test_grouped_null_semantics(dropna, func):
    data = {"g": ["a", "a", "b", None], "v": [None, 2., None, 4.]}
    def make():
        return AggregateOp(grouped=True, grouping=(Col("g"),),
                           aggregations=(("v", AggExpr(func, Col("v"))),),
                           drop_null_keys=dropna)
    expected = run_op(make(), pd.DataFrame(data)).reset_index().sort_values("g", na_position="last")
    with force_polars():
        result = run_op(make(), pl.DataFrame(data))
    pd.testing.assert_frame_equal(result.to_pandas().reset_index(drop=True),
                                  expected.reset_index(drop=True), check_dtype=False)


@pytest.mark.parametrize("func,params", [("idxmin", {}), ("idxmax", {}),
                                         ("sem", {}), ("sum", {"min_count": 2})])
def test_polars_rejects_unimplemented_reduction_before_execution(func, params):
    op = AggregateOp(grouped=True, grouping=(Col("g"),),
                     aggregations=(("v", AggExpr(func, Col("v"), params)),))
    assert not PolarsAggregateOp.supports(op, None)


def test_dataframe_value_counts_falls_back():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [2, 2, 3]})
    result, _ = execute(st.as_data_op(df).value_counts())
    pd.testing.assert_series_equal(result, df.value_counts())


@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("values", [["b", "b", "a"], ["b", "a"]])
def test_categorical_counts_include_unused_categories(sort, values):
    df = pd.DataFrame({"c": pd.Categorical(values, categories=["a", "b", "c"])})
    result, _ = execute(st.as_data_op(df)["c"].value_counts(sort=sort))
    pd.testing.assert_series_equal(result, df["c"].value_counts(sort=sort))


@pytest.mark.parametrize("how", ["semi", "anti"])
def test_filtering_join_nulls_order_duplicates_and_index(how):
    left = pd.DataFrame({"c": ["b", None, "a", "b"], "_merge": [1, 2, 3, 4]}, index=[9, 7, 7, 2])
    build = pd.Series([None, "b", "b"], dtype="str")
    mask = left["c"].isin(build)
    expected = left[mask if how == "semi" else ~mask]
    for cls in [PandasIsInSemiJoinOp, PandasMergeSemiJoinOp]:
        op = cls(anti=how == "anti", left_on="c")
        assert cls.supports(op, None)
        pd.testing.assert_frame_equal(op.process([left, build]), expected)
    op = PolarsFilteringJoinOp(anti=how == "anti", left_on="c")
    result = op.process([pl.from_pandas(left), pl.from_pandas(build)])
    assert result["_merge"].to_list() == expected["_merge"].tolist()


@pytest.mark.parametrize("build_values", [[None], [float("nan")], [pd.NA],
                                         [None, float("nan"), pd.NA]])
@pytest.mark.parametrize("how", ["semi", "anti"])
def test_merge_strategy_preserves_object_membership_sentinels(build_values, how):
    left = pd.DataFrame({"k": pd.Series([None, float("nan"), pd.NA, "x"], dtype=object)})
    build = pd.Series(build_values, dtype=object)
    mask = left["k"].isin(build)
    expected = left[mask if how == "semi" else ~mask]
    result = PandasMergeSemiJoinOp(anti=how == "anti", left_on="k").process([left, build])
    pd.testing.assert_frame_equal(result, expected)


def test_computed_reduction_keeps_external_consumers():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3], "w": [4, 5, 6]})
    d = st.as_data_op(df)
    product = d["v"] * d["w"]
    expression = st.deferred(lambda x, y: (x, y))(
        product.groupby(d["g"]).sum(), product)
    result, _ = execute(expression)
    expected = df["v"] * df["w"]
    pd.testing.assert_series_equal(result[0], expected.groupby(df["g"]).sum())
    pd.testing.assert_series_equal(result[1], expected)



def test_as_index_false_aggregation_may_reuse_a_group_key_name():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    def build(d):
        return d.groupby("g", as_index=False).agg({"g": "first", "v": "sum"})
    result, _ = execute(build(st.as_data_op(df)))
    pd.testing.assert_frame_equal(result, build(df))


# Pandas parity for aggregation fusion.

_NUMERIC = pd.DataFrame({"v": [1., None, 3.], "w": [4., 5., 6.]})
_MIXED = pd.DataFrame({"g": ["a", "a", "b"], "v": [1., 2., 3.], "w": [4., 5., 6.]})

# Fuse into one ungrouped AggregateOp over _NUMERIC. A projection stays in the plan.
_UNGROUPED = {
    "sum()": lambda d: d.sum(),
    "agg('sum')": lambda d: d.agg("sum"),
    "agg(func='mean')": lambda d: d.agg(func="mean"),
    "mean(skipna=False)": lambda d: d.mean(skipna=False),
    "sum(min_count=3)": lambda d: d.sum(min_count=3),
    "std(ddof=0)": lambda d: d.std(ddof=0),
    "agg('std', ddof=0)": lambda d: d.agg("std", ddof=0),
    "count()": lambda d: d.count(),
    "nunique()": lambda d: d.nunique(),
    "agg(dict)": lambda d: d.agg({"v": "sum", "w": "mean"}),
    "agg(dict) of mixed reductions": lambda d: d.agg({"v": "count", "w": "max"}),
    "agg(dict) of one column": lambda d: d.agg({"w": "max"}),
    "frame selection": lambda d: d[["w", "v"]].sum(),
    "frame selection + dict": lambda d: d[["v", "w"]].agg({"v": "sum"}),
    "masked frame": lambda d: d[d["w"] > 4].sum(),
    "column": lambda d: d["v"].sum(),
    "column agg('mean')": lambda d: d["v"].agg("mean"),
    "column std(ddof=0)": lambda d: d["v"].std(ddof=0),
}

# Fold the groupby and its projection into one grouped AggregateOp over _MIXED.
_GROUPED_PROJECTIONS = {
    "['v'].sum()": lambda d: d.groupby("g")["v"].sum(),
    "[['w', 'v']].sum()": lambda d: d.groupby("g")[["w", "v"]].sum(),
    "[['v']].size()": lambda d: d.groupby("g")[["v"]].size(),
    "[['v', 'w']].agg(dict)": lambda d: d.groupby("g")[["v", "w"]].agg({"v": "sum"}),
    "[['v', 'w']].agg(named)": lambda d: d.groupby("g")[["v", "w"]].agg(
        total=("v", "sum"), top=("w", "max")),
    "as_index=False ['v'].sum()": lambda d: d.groupby("g", as_index=False)["v"].sum(),
    "series.groupby": lambda d: d["v"].groupby(d["g"]).sum(),
}

# Stay unfused, so pandas runs them: (frame, build).
_REFUSED = {
    "df.agg(list)": (_NUMERIC, lambda d: d.agg(["sum", "mean"])),
    "df.agg('size')": (_NUMERIC, lambda d: d.agg("size")),
    "df.sum(axis=1)": (_NUMERIC, lambda d: d.sum(axis=1)),
    "df.agg(named)": (_NUMERIC, lambda d: d.agg(total=("v", "sum"))),
    "groupby.agg(list)": (_MIXED, lambda d: d.groupby("g").agg(["sum", "mean"])),
    "groupby['v'].agg(list)": (_MIXED, lambda d: d.groupby("g")["v"].agg(["sum", "mean"])),
}

# Rejected by pandas: (build, error, how many aggregations fuse anyway).
_PANDAS_ERRORS = {
    "dict key outside a frame selection": (
        lambda d: d[["v"]].agg({"w": "sum"}), KeyError, 1),
    "dict key outside a grouped selection": (
        lambda d: d.groupby("g")[["v"]].agg({"w": "sum"}), KeyError, 0),
    "named column outside a grouped selection": (
        lambda d: d.groupby("g")[["v"]].agg(t=("w", "sum")), KeyError, 0),
    "dict on a SeriesGroupBy": (
        lambda d: d.groupby("g")["v"].agg({"t": "sum"}), pd.errors.SpecificationError, 0),
    "labels on a DataFrame": (lambda d: d.agg(t="sum"), TypeError, 0),
}


def _assert_same(result, expected, **kwargs):
    """Compare a plan's value with pandas', whatever its kind."""
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(result, expected, **kwargs)
    elif isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(result, expected, **kwargs)
    else:
        assert (pd.isna(result) and pd.isna(expected)) or result == expected


def _run(ops):
    """`execute` for a plan optimized elsewhere."""
    values = {}
    for op in ops:
        values[id(op)] = op.process([values[id(i)] for i in op.inputs])
    return values[id(ops[-1])]


def _plan_and_evaluate(build, frame, selector):
    """The plan ``build`` compiles to under ``selector``, and its result."""
    with st.config(implementation_selector=selector):
        dag = build(st.var("df", frame))
        ops = optimize_(dag, OptConfig(dataframe_ops=True), env=get_data(dag))[0]
        result = _api.evaluate(build(st.as_data_op(frame)))
    return ops, result


def _assert_matches_pandas(result, expected):
    """Compare with pandas. A grouped frame comes back from polars with its keys
    as columns, and with polars' dtypes."""
    if isinstance(result, pl.DataFrame):
        result = result.to_pandas()
        if "g" not in expected.columns:
            expected = expected.reset_index()
        _assert_same(result, expected, check_dtype=False)
    else:
        _assert_same(result, expected)


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("build", list(_UNGROUPED.values()), ids=list(_UNGROUPED))
def test_ungrouped_aggregation_fuses_and_matches_pandas(build, selector):
    ops, result = _plan_and_evaluate(build, _NUMERIC, selector)
    expected = build(_NUMERIC)
    _assert_matches_pandas(result, expected)
    (agg,) = [op for op in ops if isinstance(op, AggregateOp)]
    assert not agg.grouped
    # A frame reduces to a series, a column to a scalar.
    kind = OutputType.SERIES if isinstance(expected, pd.Series) else OutputType.SCALAR
    assert agg.output_type is kind


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("build", list(_GROUPED_PROJECTIONS.values()),
                         ids=list(_GROUPED_PROJECTIONS))
def test_grouped_projection_folds_and_matches_pandas(build, selector):
    ops, result = _plan_and_evaluate(build, _MIXED, selector)
    _assert_matches_pandas(result, build(_MIXED))
    (agg,) = [op for op in ops if isinstance(op, AggregateOp)]
    assert agg.grouped
    # The projection is folded with the groupby, so no groupby call survives.
    assert not any(isinstance(op, MethodCallOp) and op.method_name == "groupby"
                   for op in ops)


@pytest.mark.parametrize("frame,build", list(_REFUSED.values()), ids=list(_REFUSED))
def test_refused_spellings_stay_unfused_and_match_pandas(frame, build):
    result, ops = execute(build(st.as_data_op(frame)))
    _assert_same(result, build(frame))
    assert not any(isinstance(op, AggregateOp) for op in ops)


@pytest.mark.parametrize("build,error,fused", list(_PANDAS_ERRORS.values()),
                         ids=list(_PANDAS_ERRORS))
def test_pandas_errors_surface_unchanged(build, error, fused):
    with pytest.raises(error):
        build(_MIXED)  # the premise: pandas itself rejects this spelling
    # Built over a variable: over a literal frame the expression fails while it
    # is built, before the optimizer sees it.
    ops = optimize(build(st.var("df")), OptConfig(dataframe_ops=True), env={"df": _MIXED})
    assert sum(isinstance(op, AggregateOp) for op in ops) == fused
    with pytest.raises(error):
        _run(ops)


@pytest.mark.parametrize("func,on_polars", [("sum", False), ("max", True)])
def test_grouped_string_reduction_binds_polars_only_where_it_agrees(func, on_polars):
    # pandas concatenates strings on `sum` where polars raises, while both agree
    # on `max`. The greedy plan used to bind polars for both and fail on `sum`.
    frame = pd.DataFrame({"g": ["a", "a", "b"], "s": ["x", "y", "z"]})
    def build(d):
        return getattr(d.groupby("g", as_index=False)["s"], func)()
    ops, result = _plan_and_evaluate(build, frame, "greedy")
    agg = next(op for op in ops if isinstance(op, AggregateOp))
    assert isinstance(agg, PolarsAggregateOp) is on_polars
    _assert_matches_pandas(result, build(frame))


# A dict or keyword labels on a Series or SeriesGroupBy, each fused into one
# labelled AggregateOp. The frame has a missing value and a non-default index.
_LABELLED_FRAME = pd.DataFrame(
    {"g": ["a", "b", "a", "b"], "v": [1., None, 3., 4.], "w": [2., 5., 6., 1.]},
    index=[7, 3, 9, 1])
_LABELLED = {
    "Series dict": lambda d: d["v"].agg({"total": "sum"}),
    "Series dict of mixed results": lambda d: d["v"].agg(
        {"total": "sum", "n": "count", "avg": "mean"}),
    "Series labels": lambda d: d["v"].agg(total="sum"),
    "Series labels, two": lambda d: d["v"].agg(total="sum", avg="mean"),
    "label named like a parameter": lambda d: d["v"].agg(ddof="std"),
    "computed Series": lambda d: (d["v"] * d["w"]).agg(total="sum"),
    "computed Series keeping its name": lambda d: (d["v"] * d["v"]).agg(total="sum"),
    "SeriesGroupBy labels": lambda d: d.groupby("g")["v"].agg(total="sum"),
    "SeriesGroupBy labels, two": lambda d: d.groupby("g")["v"].agg(
        total="sum", avg="mean"),
    "SeriesGroupBy labels, as_index=False": lambda d: d.groupby(
        "g", as_index=False)["v"].agg(total="sum"),
    "series.groupby labels": lambda d: d["v"].groupby(d["g"]).agg(total="sum"),
    "computed series.groupby labels": lambda d: (d["v"] * d["w"]).groupby(
        d["g"]).agg(total="sum"),
}


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("rows", ["data", "empty"])
@pytest.mark.parametrize("build", list(_LABELLED.values()), ids=list(_LABELLED))
def test_labels_on_a_series_fuse_and_match_pandas(build, rows, selector):
    frame = _LABELLED_FRAME if rows == "data" else _LABELLED_FRAME.iloc[:0]
    ops, result = _plan_and_evaluate(build, frame, selector)
    (agg,) = [op for op in ops if isinstance(op, AggregateOp)]
    assert agg.labelled()
    # The labels widen the result: a series without a groupby, a frame with one.
    assert agg.output_type is (OutputType.FRAME if agg.grouped else OutputType.SERIES)
    _assert_matches_pandas(result, build(frame))


# pandas drops the keywords beside a dict spec, so ddof=0 must not reach the std.
_DICT_WITH_KEYWORDS = {
    "DataFrame": lambda d: d.agg({"v": "std", "w": "std"}, ddof=0),
    "Series": lambda d: d["v"].agg({"spread": "std"}, ddof=0),
    "DataFrameGroupBy": lambda d: d.groupby("g").agg({"v": "std"}, ddof=0),
    "grouped selection": lambda d: d.groupby("g")[["v", "w"]].agg({"w": "std"}, ddof=0),
}


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("build", list(_DICT_WITH_KEYWORDS.values()),
                         ids=list(_DICT_WITH_KEYWORDS))
def test_keywords_beside_a_dict_spec_are_dropped_as_in_pandas(build, selector):
    ops, result = _plan_and_evaluate(build, _MIXED, selector)
    (agg,) = [op for op in ops if isinstance(op, AggregateOp)]
    assert all(not expr.params for _, expr in agg.aggregations)
    _assert_matches_pandas(result, build(_MIXED))


# Grouped reductions where Polars disagrees with pandas: (build, pandas' error or
# None when pandas computes). On bdfcff1 the greedy selector bound Polars for each,
# which raised where pandas computes, or returned a value where pandas raises.
_POLARS_DISAGREES_FRAME = pd.DataFrame({
    "g": ["a", "a", "b", "b"],
    "v": [0., 2., 3., 4.],
    "s": ["x", "y", "z", "q"],
    "c": pd.Categorical(["p", "q", "p", "q"]),
    "dt": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
})
_POLARS_DISAGREES = {
    "std(ddof=-1)": (lambda g: g["v"].std(ddof=-1), None),
    "std(ddof=1.5)": (lambda g: g["v"].std(ddof=1.5), None),
    "string mean": (lambda g: g["s"].mean(), TypeError),
    "unordered categorical min": (lambda g: g["c"].min(), TypeError),
    "datetime sum": (lambda g: g["dt"].sum(), TypeError),
}


@pytest.mark.parametrize("reduce,error", list(_POLARS_DISAGREES.values()),
                         ids=list(_POLARS_DISAGREES))
def test_grouped_reductions_polars_disagrees_on_run_on_pandas(reduce, error):
    def build(d):
        return reduce(d.groupby("g", as_index=False))
    frame = _POLARS_DISAGREES_FRAME
    with st.config(implementation_selector="greedy"):
        # Built over a variable: over a literal frame a pandas error is raised
        # while the expression is built, before the optimizer sees it.
        ops = optimize_(build(st.var("df")), OptConfig(dataframe_ops=True),
                        env={"df": frame})[0]
    (agg,) = [op for op in ops if isinstance(op, AggregateOp)]
    assert not isinstance(agg, PolarsAggregateOp)
    if error is None:
        pd.testing.assert_frame_equal(_run(ops), build(frame))
        return
    with pytest.raises(error):
        build(frame)  # the premise: pandas itself rejects this reduction
    with pytest.raises(error):
        _run(ops)


def test_numeric_only_survives_a_whole_frame_reduction():
    result, _ = execute(st.as_data_op(_MIXED).mean(numeric_only=True))
    pd.testing.assert_series_equal(result, _MIXED.mean(numeric_only=True))


# Computed scalars over a frame with a missing value and a 0 / 0.
_COMPUTED_SCALARS_FRAME = pd.DataFrame({"v": [1.0, 0.0, 3.0, np.nan],
                                        "w": [2.0, 0.0, 5.0, 4.0]})
_COMPUTED_SCALARS = {
    "v * w": lambda d: (d["v"] * d["w"]).sum(),
    "v * (1 - w), as in Q6": lambda d: (d["v"] * (1 - d["w"])).sum(),
    "-v": lambda d: (-d["v"]).mean(),
    "v / w, where 0 / 0 is NaN": lambda d: (d["v"] / d["w"]).sum(),
}


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("build", list(_COMPUTED_SCALARS.values()), ids=list(_COMPUTED_SCALARS))
def test_ungrouped_computed_scalars_match_pandas_including_internal_nan(build, selector):
    with st.config(implementation_selector=selector):
        result = _api.evaluate(build(st.as_data_op(_COMPUTED_SCALARS_FRAME)))
    assert result == pytest.approx(build(_COMPUTED_SCALARS_FRAME))


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("dtype", ["float64", "int64", "bool"])
@pytest.mark.parametrize("func", [
    "sum", "prod", "mean", "median", "min", "max", "std", "var", "count",
    "nunique", "all", "any",
])
def test_empty_series_scalar_reductions_match_pandas(selector, dtype, func):
    frame = pd.DataFrame({"v": pd.Series([], dtype=dtype)})
    expected = getattr(frame["v"], func)()
    with st.config(implementation_selector=selector):
        got = _api.evaluate(getattr(st.as_data_op(frame)["v"], func)())
    _assert_same(got, expected)
    if pd.isna(expected):
        # A missing numeric result is NaN, as in pandas, not None.
        assert isinstance(got, float) and np.isnan(got)


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("values,dtype", [
    ([np.nan, np.nan], "float64"),
    ([None, None], "Float64"),
    ([None, None], "Int64"),
    ([None, None], "double[pyarrow]"),
])
@pytest.mark.parametrize("func", ["sum", "prod", "mean", "min", "max", "std", "count", "nunique"])
def test_all_null_scalar_reductions_match_pandas(selector, values, dtype, func):
    frame = pd.DataFrame({"v": pd.Series(values, dtype=dtype)})
    expected = getattr(frame["v"], func)()
    with st.config(implementation_selector=selector):
        got = _api.evaluate(getattr(st.as_data_op(frame)["v"], func)())
    _assert_same(got, expected)


# Scalar reductions whose parameters or dtypes decide the backend.
_SCALAR_FRAME = pd.DataFrame({
    "v": [1.0, np.nan, 3.0, 5.0],
    "one": [4.0, np.nan, np.nan, np.nan],
    "s": pd.Series(["a", "b", "c", "d"], dtype="string"),
    "b": [True, False, True, False],
    "dt": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
})
_SCALARS = {
    "std(ddof=0)": lambda d: d["v"].std(ddof=0),
    "var(ddof=2)": lambda d: d["v"].var(ddof=2),
    "std(ddof=1) of one value": lambda d: d["one"].std(ddof=1),
    "std(ddof=-1)": lambda d: d["v"].std(ddof=-1),
    "sum(min_count=4)": lambda d: d["v"].sum(min_count=4),
    "sum(skipna=False)": lambda d: d["v"].sum(skipna=False),
    "sem()": lambda d: d["v"].sem(),
    "idxmax()": lambda d: d["v"].idxmax(),
    "string sum": lambda d: d["s"].sum(),
    "string min": lambda d: d["s"].min(),
    "datetime min": lambda d: d["dt"].min(),
    "datetime std": lambda d: d["dt"].std(),
    "boolean sum": lambda d: d["b"].sum(),
    "boolean mean": lambda d: d["b"].mean(),
    "quantile(0.25, linear)": lambda d: d["v"].quantile(0.25, interpolation="linear"),
}


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("build", list(_SCALARS.values()), ids=list(_SCALARS))
def test_scalar_ddof_and_fallback_semantics_match_pandas(build, selector):
    with st.config(implementation_selector=selector):
        result = _api.evaluate(build(st.as_data_op(_SCALAR_FRAME)))
    _assert_same(result, build(_SCALAR_FRAME))


# Scalar reductions pandas rejects: Stratum must raise the same error.
_SCALAR_ERRORS = {
    "unordered categorical min": lambda d: d["cat"].min(),
    "boolean quantile": lambda d: d["b"].quantile(),
}


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("build", list(_SCALAR_ERRORS.values()), ids=list(_SCALAR_ERRORS))
def test_unsupported_scalar_errors_match_pandas(build, selector):
    frame = pd.DataFrame({"cat": pd.Categorical(["b", "a"]), "b": [True, False]})
    with pytest.raises(TypeError):
        build(frame)  # the premise: pandas itself rejects this reduction
    # Built over a variable: over a literal frame the error is raised while the
    # expression is built, before the optimizer sees it.
    with st.config(implementation_selector=selector):
        ops = optimize_(build(st.var("df")), OptConfig(dataframe_ops=True),
                        env={"df": frame})[0]
    with pytest.raises(TypeError):
        _run(ops)
