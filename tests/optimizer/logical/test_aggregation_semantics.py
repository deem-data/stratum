"""Semantic regressions for expression aggregation and filtering joins."""
import operator

import pandas as pd
import polars as pl
import pytest
import stratum as st

from stratum.optimizer._optimize import OptConfig, optimize as optimize_plan
from stratum.optimizer.logical._aggregation_ops import AggregateOp
from stratum.optimizer.logical._column_expr import AggExpr, AllCols, BinOpExpr, Col
from stratum.optimizer.logical._map_ops import ExpressionMapOp
from stratum.optimizer.logical._numeric_ops import NumericOp
from stratum.optimizer.logical._ops import OutputType
from stratum.optimizer.physical._aggregation_execs import PolarsAggregateOp
from stratum.optimizer.physical._join_execs import (
    PandasIsInSemiJoinOp, PandasMergeSemiJoinOp, PolarsFilteringJoinOp)
from stratum.runtime._scheduler import SequentialScheduler
from .test_dataframe_ops import optimize, run_op, force_polars


def execute(expression, env=None):
    ops = optimize(expression, OptConfig(dataframe_ops=True), env=env)
    values = {}
    for op in ops:
        values[id(op)] = op.process([values[id(i)] for i in op.inputs])
    return values[id(ops[-1])], ops


def plan_and_evaluate(build, frame, selector):
    dag = build(st.var("frame"))
    with st.config(implementation_selector=selector):
        ops, split = optimize_plan(dag, OptConfig(), env={"frame": frame})
    result = SequentialScheduler(ops, split).evaluate()
    if isinstance(result, (pl.DataFrame, pl.Series)):
        result = result.to_pandas()
    return result, ops


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("build", [
    lambda d: d.sum(), lambda d: d.agg(func="mean"),
    lambda d: d.mean(skipna=False), lambda d: d.sum(min_count=3),
    lambda d: d.agg("std", ddof=0), lambda d: d.agg({"w": "mean", "v": "max"}),
    lambda d: d[["w", "v"]].sum(), lambda d: d[["v", "w"]].agg({"v": "std"}, ddof=0),
    lambda d: d[d["w"] > 4].sum(),
])
def test_ungrouped_frame_aggregations(build, selector):
    frame = pd.DataFrame({"v": [1., None, 3.], "w": [4., 5., 6.]})
    result, ops = plan_and_evaluate(build, frame, selector)
    assert isinstance(ops[-1], AggregateOp) and ops[-1].output_type is OutputType.SERIES
    pd.testing.assert_series_equal(result, build(frame))


def test_frame_reduction_preserves_numeric_only_and_dict_column_labels():
    frame = pd.DataFrame({"v": [1., 3.], "s": ["a", "b"]})
    result, ops = execute(st.var("frame").mean(numeric_only=True), {"frame": frame})
    assert isinstance(ops[-1], AggregateOp)
    pd.testing.assert_series_equal(result, frame.mean(numeric_only=True))
    frame.columns = pd.Index(["v", "v"], name="feature")
    result, ops = execute(st.var("frame").agg({"v": "max"}), {"frame": frame})
    pd.testing.assert_series_equal(result, frame.agg({"v": "max"}))


@pytest.mark.parametrize("reduce", [
    lambda s: s.std(ddof=0), lambda s: s.agg("std", ddof=0),
    lambda s: s.aggregate(func="std", ddof=0),
])
def test_scalar_reduction_of_a_bound_series(reduce):
    series = pd.Series([2., None, 6.], index=[4, 1, 7])
    result, ops = execute(reduce(st.var("s")), env={"s": series})
    agg = ops[-1]
    assert isinstance(agg, AggregateOp) and not agg.grouped
    assert agg.output_type is OutputType.SCALAR
    assert agg.aggregations == ((None, AggExpr("std", AllCols(), {"ddof": 0})),)
    assert result == reduce(series)


def test_scalar_reduction_keeps_the_computed_series_as_its_input():
    frame = pd.DataFrame({"price": [2., None, 4.], "quantity": [3, 4, 5]})
    d = st.as_data_op(frame)
    product = d["price"] * d["quantity"]
    result, ops = execute(st.deferred(tuple)([product, product.sum(), product.agg("sum")]))
    (agg,) = [op for op in ops if isinstance(op, AggregateOp)]
    assert isinstance(agg.inputs[0], ExpressionMapOp)
    expected = frame["price"] * frame["quantity"]
    pd.testing.assert_series_equal(result[0], expected)
    assert result[1:] == (expected.sum(), expected.sum())


def test_scalar_reduction_preserves_numeric_only():
    series = pd.Series([1, 2], dtype=object)
    with pytest.raises(TypeError):
        series.sum(numeric_only=True)
    with pytest.raises(TypeError):
        execute(st.var("s").sum(numeric_only=True), env={"s": series})


def test_labelled_reduction_of_a_bound_series():
    series = pd.Series([1, 2], name="v")
    result, ops = execute(st.var("s").agg({"total": "sum"}), env={"s": series})
    assert isinstance(ops[-1], AggregateOp) and ops[-1].labelled()
    pd.testing.assert_series_equal(result, series.agg({"total": "sum"}))


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("build", [
    lambda d: d["v"].agg(ddof="std"),
    lambda d: (d["v"] * d["w"]).agg(total="sum", avg="mean"),
    lambda d: d["v"].agg({"spread": "std"}, ddof=0),
    lambda d: d.groupby("g")["v"].agg(total="sum"),
    lambda d: d.groupby("g", as_index=False)["v"].agg(total="sum", avg="mean"),
    lambda d: (d["v"] * d["w"]).groupby(d["g"]).agg(total="sum"),
])
def test_labelled_series_reductions_preserve_result_shape(build, empty):
    frame = pd.DataFrame({"g": ["a", "b", "a"], "v": [1., None, 3.], "w": [2., 5., 6.]},
                         index=[7, 3, 9])
    if empty:
        frame = frame.iloc[:0]
    result, ops = plan_and_evaluate(build, frame, "default")
    expected = build(frame)
    assert isinstance(ops[-1], AggregateOp) and ops[-1].labelled()
    if isinstance(expected, pd.Series):
        assert ops[-1].output_type is OutputType.SERIES
        pd.testing.assert_series_equal(result, expected)
    else:
        assert ops[-1].output_type is OutputType.FRAME
        pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("reduce,params", [
    (lambda g: g.agg({"v": "std"}, ddof=0), {}),
    (lambda g: g.aggregate(func={"v": "std"}, ddof=0), {}),
    (lambda g: g.agg("std", ddof=0), {"ddof": 0}),
], ids=["dict", "keyword-dict", "function"])
def test_grouped_aggregation_binds_parameters_by_spec(selector, reduce, params):
    frame = pd.DataFrame({"g": ["a", "a", "b"], "v": [0., 2., 3.]})
    dag = reduce(st.var("frame").groupby("g", as_index=False))
    with st.config(implementation_selector=selector):
        ops, split = optimize_plan(dag, OptConfig(), env={"frame": frame})
    (agg,) = [op for op in ops if isinstance(op, AggregateOp)]
    assert agg.aggregations[0][1].params == params
    result = SequentialScheduler(ops, split).evaluate()
    if isinstance(result, pl.DataFrame):
        result = result.to_pandas()
    pd.testing.assert_frame_equal(result, reduce(frame.groupby("g", as_index=False)))


@pytest.mark.parametrize("build,error", [
    (lambda d: d["v"].groupby(d["g"]).agg({"v": "std"}, ddof=0), pd.errors.SpecificationError),
    (lambda d: d.groupby("g").agg({}, ddof=0), ValueError),
    (lambda d: d.groupby("g").agg({"v": "std"}, axis="invalid"), ValueError),
    (lambda d: d.groupby("g").agg({"v": "std"}, func="std"), TypeError),
], ids=["series-groupby", "empty-dict", "invalid-axis", "duplicate-func"])
def test_invalid_grouped_dict_aggregations_preserve_pandas_errors(build, error):
    frame = pd.DataFrame({"g": ["a", "a"], "v": [0., 2.]})
    with pytest.raises(error):
        build(frame)
    with pytest.raises(error):
        execute(build(st.var("frame")), env={"frame": frame})


def test_grouped_dict_aggregation_evaluates_dynamic_keyword_operand():
    frame = pd.DataFrame({"g": ["a", "a"], "v": [0., 2.]})
    dag = st.var("frame").groupby("g").agg({"v": "std"}, ddof=st.var("params")["ddof"])
    with pytest.raises(KeyError, match="ddof"):
        execute(dag, env={"frame": frame, "params": {}})


def test_shared_grouped_series_keeps_one_result_per_group():
    frame = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 5]})
    grouped = st.as_data_op(frame).groupby("g")["v"]
    result, ops = execute(st.deferred(tuple)([grouped.sum(), grouped.max()]))
    assert not any(isinstance(op, AggregateOp) and not op.grouped for op in ops)
    expected = frame.groupby("g")["v"]
    pd.testing.assert_series_equal(result[0], expected.sum())
    pd.testing.assert_series_equal(result[1], expected.max())


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("reduce", [
    lambda g: g.agg({"w": "std"}, ddof=0),
    lambda g: g.agg(top=("w", "max"), total=("v", "sum")),
    lambda g: g.agg(spread=pd.NamedAgg("w", "std")),
])
def test_grouped_projection_binds_input_columns_and_output_labels(selector, reduce):
    frame = pd.DataFrame({"g": ["a", "a", "b"], "v": [0., 2., 3.], "w": [1., 5., 7.]})
    def build(d):
        return reduce(d.groupby("g", as_index=False)[["v", "w"]])
    result, ops = plan_and_evaluate(build, frame, selector)
    assert isinstance(ops[-1], AggregateOp)
    pd.testing.assert_frame_equal(result, build(frame))


@pytest.mark.parametrize("columns,reduction", [
    (["v"], {"w": "sum"}), (["v", "missing"], {"v": "sum"}),
])
def test_grouped_projection_preserves_column_validation(columns, reduction):
    frame = pd.DataFrame({"g": ["a"], "v": [1], "w": [2]})
    dag = st.var("frame").groupby("g")[columns].agg(reduction)
    with pytest.raises(KeyError):
        execute(dag, {"frame": frame})


def test_parameterized_named_agg_preserves_pandas_callable_validation():
    frame = pd.DataFrame({"g": ["a", "a"], "v": [1., 2.]})
    def build(d):
        return d.groupby("g")[["v"]].agg(spread=pd.NamedAgg("v", "std", ddof=0))
    with pytest.raises(TypeError):
        build(frame)
    with pytest.raises(TypeError):
        execute(build(st.var("frame")), {"frame": frame})


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


def test_real_computed_reduction_folds_and_preserves_series_name():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3], "w": [4, 5, 6]})
    def build(d):
        return (d["v"] * d["w"]).groupby(d["g"]).sum()
    result, ops = execute(build(st.as_data_op(df)))
    pd.testing.assert_series_equal(result, build(df))
    assert not any(isinstance(op, NumericOp) for op in ops)
    agg = next(op for op in ops if isinstance(op, AggregateOp))
    assert isinstance(agg.aggregations[0][1].child, BinOpExpr)


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
