"""Semantic regressions for expression aggregation and filtering joins."""
import operator

import pandas as pd
import polars as pl
import pytest
import stratum as st

from stratum.optimizer._optimize import OptConfig
from stratum.optimizer.logical._aggregation_ops import AggregateOp
from stratum.optimizer.logical._column_expr import AggExpr, BinOpExpr, Col
from stratum.optimizer.logical._numeric_ops import NumericOp
from stratum.optimizer.logical._ops import OutputType
from stratum.optimizer.physical._aggregation_execs import PolarsAggregateOp
from stratum.optimizer.physical._join_execs import (
    PandasIsInSemiJoinOp, PandasMergeSemiJoinOp, PolarsFilteringJoinOp)
from .test_dataframe_ops import optimize, run_op, force_polars


def execute(expression):
    ops = optimize(expression, OptConfig(dataframe_ops=True))
    values = {}
    for op in ops:
        values[id(op)] = op.process("fit_transform", [values[id(i)] for i in op.inputs])
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
                           options={"dropna": dropna})
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
        op = cls(how=how, left_on="c")
        assert cls.supports(op, None)
        pd.testing.assert_frame_equal(op.process("fit_transform", [left, build]), expected)
    op = PolarsFilteringJoinOp(how=how, left_on="c")
    result = op.process("fit_transform", [pl.from_pandas(left), pl.from_pandas(build)])
    assert result["_merge"].to_list() == expected["_merge"].tolist()


@pytest.mark.parametrize("build_values", [[None], [float("nan")], [pd.NA],
                                         [None, float("nan"), pd.NA]])
@pytest.mark.parametrize("how", ["semi", "anti"])
def test_merge_strategy_preserves_object_membership_sentinels(build_values, how):
    left = pd.DataFrame({"k": pd.Series([None, float("nan"), pd.NA, "x"], dtype=object)})
    build = pd.Series(build_values, dtype=object)
    mask = left["k"].isin(build)
    expected = left[mask if how == "semi" else ~mask]
    result = PandasMergeSemiJoinOp(how=how, left_on="k").process("fit_transform", [left, build])
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
