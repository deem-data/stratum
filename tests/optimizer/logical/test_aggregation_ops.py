import operator
import unittest

import pandas as pd
import polars as pl
import stratum as st
from stratum.frontend import _api
from stratum.optimizer._optimize import OptConfig, optimize as optimize_
from stratum.optimizer.logical._aggregation_ops import (
    AggregateOp, _AggCall, _AggItem, _DICT, _FUNC, _LABELS, _NAMED, _aggregation_entries,
    _aggregation_params, _get_computed_sources, _extract_aggregations, _extract_grouping,
    _get_sources, _is_aggregation, _is_groupby_op, _is_value_counts,
    _labelled_aggregation_entries, _projected_aggregation_entries, make_aggregate_op,
    make_value_counts_ops)
from stratum.optimizer.logical._column_expr import (
    AggExpr, AllCols, BinOpExpr, Col, Const, DtExpr, OperandLeaf, UnaryOpExpr)
from stratum.optimizer.logical._metadata_ops import MetadataOp
from stratum.optimizer.logical._numeric_ops import NumericOp
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp
from stratum.optimizer.logical._index_ops import GroupKeysOp, IndexAccessOp
from stratum.optimizer.logical._selection_ops import MaskSelectionOp, SelectionOp
from stratum.optimizer.logical._sort_ops import SortOp
from stratum.optimizer.logical._source_ops import InMemoryFrameOp
from stratum.optimizer.logical._ops import (
    BinOp, GetItemOp, MethodCallOp, Op, OperandRef, OutputType)
from stratum.runtime._buffer_pool import BufferPool
from .test_dataframe_ops import (
    force_polars, optimize, run_op)


def _groupby_agg_pair(group_args=("g",), group_kwargs=None,
                      agg_method="sum", agg_args=(), agg_kwargs=None):
    """Build a `groupby(...)` MethodCallOp feeding an aggregation MethodCallOp."""
    groupby = MethodCallOp("groupby", args=group_args, kwargs=group_kwargs or {})
    agg = MethodCallOp(agg_method, args=agg_args, kwargs=agg_kwargs or {})
    agg.inputs = [groupby]
    groupby.outputs = [agg]
    return groupby, agg


def _src(output_type):
    """A stand-in producer carrying just an output kind."""
    op = Op()
    op.output_type = output_type
    return op


def _wildcard(func="sum", **params):
    """The `.agg(func)` shape: one unnamed entry over every column."""
    return ((None, AggExpr(func, AllCols(), params or None)),)


def _groupby(*args, **kwargs):
    """A `groupby(...)` call with no producers wired yet."""
    return MethodCallOp("groupby", args=args, kwargs=kwargs)


def _method(name, *args, **kwargs):
    """A bare method call, for tests that only parse it."""
    return MethodCallOp(name, args=args, kwargs=kwargs)


def _call_over(*producers, method="sum", args=(), kwargs=None):
    """Chain ``producers`` (source first) and hang an aggregation call off the end.

    Each op is the single consumer of the one before it, which is how the
    extraction pass sees a plain ``df.groupby(...)[...].agg(...)`` chain.
    """
    call = MethodCallOp(method, args=args, kwargs=kwargs or {})
    chain = [*producers, call]
    for producer, consumer in zip(chain, chain[1:]):
        consumer.inputs = [producer]
        producer.outputs = [consumer]
    return call


class TestAggregateOp(unittest.TestCase):
    """`AggregateOp` execution on both backends."""

    def setUp(self):
        self.data = {"g": ["a", "a", "b"], "v": [1, 2, 3]}
        self.df = pd.DataFrame(self.data)

    def test_pandas_wildcard_reduction(self):
        op = AggregateOp(grouped=True, grouping=(Col("g"),),
                         aggregations=_wildcard("sum"))
        pd.testing.assert_frame_equal(run_op(op, self.df),
                                      self.df.groupby("g").agg("sum"))

    def test_pandas_single_column_entry(self):
        df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3], "w": [4, 5, 6]})
        op = AggregateOp(grouped=True, grouping=(Col("g"),),
                         aggregations=((None, AggExpr("sum", Col("v"))),))
        result = run_op(op, df)
        expected = df.groupby("g").agg({"v": "sum"})
        pd.testing.assert_frame_equal(result, expected.rename(
            columns={"v": result.columns[0]}))

    def test_pandas_computed_child_needs_no_map_op(self):
        # SUM(v * w) is one op: the multiply lives inside the AggExpr.
        df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3], "w": [4, 5, 6]})
        op = AggregateOp(
            grouped=True, grouping=(Col("g"),),
            aggregations=(("vw", AggExpr("sum", BinOpExpr(operator.mul,
                                                          Col("v"), Col("w")))),))
        result = run_op(op, df)
        expected = (df.assign(vw=df["v"] * df["w"])
                    .groupby("g")[["vw"]].sum())
        pd.testing.assert_frame_equal(result, expected)

    def test_pandas_multiple_named_entries(self):
        df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3], "w": [4, 5, 6]})
        op = AggregateOp(grouped=True, grouping=(Col("g"),),
                         aggregations=(("v_sum", AggExpr("sum", Col("v"))),
                                       ("w_max", AggExpr("max", Col("w")))))
        result = run_op(op, df)
        expected = df.groupby("g").agg(v_sum=("v", "sum"), w_max=("w", "max"))
        pd.testing.assert_frame_equal(result, expected)

    def test_pandas_whole_object_reduction(self):
        op = AggregateOp(grouped=False, aggregations=_wildcard("sum"),
                         inputs=[_src(OutputType.FRAME)])
        self.assertEqual(OutputType.SERIES, op.output_type)
        numeric = self.df[["v"]]
        pd.testing.assert_series_equal(run_op(op, numeric), numeric.sum())

    def test_grouping_placeholder_resolved_from_inputs(self):
        op = AggregateOp(grouped=True, grouping=(OperandLeaf(OperandRef(1)),),
                         aggregations=_wildcard("sum"))
        pd.testing.assert_frame_equal(run_op(op, self.df, "g"),
                                      self.df.groupby("g").agg("sum"))

    def test_options_reach_the_groupby(self):
        df = pd.DataFrame({"g": ["b", "a", "b"], "v": [1, 2, 3]})
        op = AggregateOp(grouped=True, grouping=(Col("g"),),
                         aggregations=_wildcard("sum"), sort_keys=False)
        pd.testing.assert_frame_equal(run_op(op, df),
                                      df.groupby("g", sort=False).agg("sum"))

    def test_str_renders_grouping_and_reductions(self):
        op = AggregateOp(grouped=True, grouping=(Col("g"),),
                         aggregations=_wildcard("sum"))
        text = str(op)
        self.assertIn("Aggregation", text)
        self.assertIn("g", text)
        self.assertIn("sum", text)

    def test_polars_grouped_matches_native(self):
        with force_polars():
            op = AggregateOp(grouped=True, grouping=(Col("g"),),
                             aggregations=(("total", AggExpr("sum", Col("v"))),))
            pldf = pl.DataFrame(self.data)
            result = run_op(op, pldf)
            expected = (pldf.group_by("g")
                        .agg(pl.col("v").sum().alias("total")).sort("g"))
            self.assertEqual(expected.to_dicts(), result.to_dicts())

    def test_polars_computed_child_stays_one_kernel(self):
        with force_polars():
            data = {"g": ["a", "a", "b"], "v": [1, 2, 3], "w": [4, 5, 6]}
            op = AggregateOp(
                grouped=True, grouping=(Col("g"),),
                aggregations=(("vw", AggExpr("sum", BinOpExpr(operator.mul,
                                                              Col("v"), Col("w")))),))
            pldf = pl.DataFrame(data)
            result = run_op(op, pldf)
            expected = (pldf.group_by("g")
                        .agg((pl.col("v") * pl.col("w")).sum().alias("vw"))
                        .sort("g"))
            self.assertEqual(expected.to_dicts(), result.to_dicts())

    def test_polars_sort_option_controls_row_order(self):
        with force_polars():
            pldf = pl.DataFrame({"g": ["b", "a", "b"], "v": [1, 2, 3]})
            sorted_op = AggregateOp(grouped=True, grouping=(Col("g"),),
                                    aggregations=(("s", AggExpr("sum", Col("v"))),))
            self.assertEqual(["a", "b"], run_op(sorted_op, pldf)["g"].to_list())
            unsorted = AggregateOp(grouped=True, grouping=(Col("g"),),
                                   aggregations=(("s", AggExpr("sum", Col("v"))),),
                                   sort_keys=False)
            self.assertEqual(["b", "a"], run_op(unsorted, pldf)["g"].to_list())

    def test_backends_agree_on_grouped_sums(self):
        data = {"g": ["b", "a", "b", "a"], "v": [1, 2, 3, 4]}
        op = AggregateOp(grouped=True, grouping=(Col("g"),),
                         aggregations=(("total", AggExpr("sum", Col("v"))),))
        pandas_out = run_op(op, pd.DataFrame(data))
        with force_polars():
            op2 = AggregateOp(grouped=True, grouping=(Col("g"),),
                              aggregations=(("total", AggExpr("sum", Col("v"))),))
            polars_out = run_op(op2, pl.DataFrame(data))
        self.assertEqual(pandas_out["total"].tolist(),
                         polars_out["total"].to_list())
        self.assertEqual(pandas_out.index.tolist(), polars_out["g"].to_list())


class TestAggregateOutputType(unittest.TestCase):
    """Output kind: frame-world or not, and FRAME vs SERIES within it."""

    def test_grouped_frame_stays_a_frame(self):
        op = AggregateOp(grouped=True, grouping=(Col("g"),),
                         aggregations=_wildcard(), inputs=[_src(OutputType.FRAME)])
        self.assertEqual(OutputType.FRAME, op.output_type)

    def test_grouped_series_stays_a_series(self):
        op = AggregateOp(grouped=True, grouping=(Col("g"),),
                         aggregations=((None, AggExpr("sum", Col("v"))),),
                         inputs=[_src(OutputType.SERIES)])
        self.assertEqual(OutputType.SERIES, op.output_type)

    def test_grouped_series_widens_to_a_frame_with_several_entries(self):
        op = AggregateOp(grouped=True, grouping=(Col("g"),),
                         aggregations=(("a", AggExpr("sum", Col("v"))),
                                       ("b", AggExpr("max", Col("v")))),
                         inputs=[_src(OutputType.SERIES)])
        self.assertEqual(OutputType.FRAME, op.output_type)

    def test_frame_reduction_collapses_to_a_series(self):
        op = AggregateOp(grouped=False, aggregations=_wildcard(),
                         inputs=[_src(OutputType.FRAME)])
        self.assertEqual(OutputType.SERIES, op.output_type)

    def test_series_reduction_leaves_frame_world(self):
        # One reduction of a series is a scalar, which is not frame-world data.
        op = AggregateOp(grouped=False, aggregations=_wildcard(),
                         inputs=[_src(OutputType.SERIES)])
        self.assertEqual(OutputType.SCALAR, op.output_type)

    def test_defaults_to_frame_when_the_source_kind_is_unknown(self):
        self.assertEqual(OutputType.FRAME, AggregateOp().output_type)
        op = AggregateOp(grouped=True, inputs=[_src(OutputType.UNKNOWN)])
        self.assertEqual(OutputType.FRAME, op.output_type)

    def test_aggregation_stays_frame_like_for_downstream_extraction(self):
        # The #192 regression: an UNKNOWN aggregate sent every downstream op into
        # extract_dataframe_op's read/source branch.
        df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
        data = st.as_data_op(df).groupby("g").agg("sum")["v"]
        ops = optimize(data, OptConfig(dataframe_ops=True))
        agg = next(o for o in ops if isinstance(o, AggregateOp))
        self.assertIn(agg.output_type, (OutputType.FRAME, OutputType.SERIES))


_GROUP_SIZE_SPELLINGS = {
    "size()": lambda d: d.groupby("k").size(),
    "agg('size')": lambda d: d.groupby("k").agg("size"),
    "frame selection": lambda d: d.groupby("k")[["v", "w"]].size(),
    "column selection": lambda d: d.groupby("k")["v"].size(),
    "as_index=False": lambda d: d.groupby("k", as_index=False).size(),
    "per-column size": lambda d: d.groupby("k").agg({"v": "size"}),
    "named size": lambda d: d.groupby("k").agg(n=("v", "size")),
}


class TestGroupedSize(unittest.TestCase):
    """A grouped `size()` counts rows per group: a series, unlike other reductions."""

    def setUp(self):
        self.df = pd.DataFrame({"k": [1, 1, 2], "v": [1, 2, None], "w": [4, 5, 6]})

    def test_the_static_kind_matches_pandas(self):
        for name, build in _GROUP_SIZE_SPELLINGS.items():
            with self.subTest(name):
                ops = optimize(build(st.as_data_op(self.df)), OptConfig(dataframe_ops=True))
                agg = next(o for o in ops if isinstance(o, AggregateOp))
                expected = build(self.df)
                kind = (OutputType.SERIES if isinstance(expected, pd.Series)
                        else OutputType.FRAME)
                self.assertIs(kind, agg.output_type)

    def test_both_backends_match_pandas(self):
        # A keyed series exists only on pandas, so polars plans fall back there.
        from stratum._config import config
        for name, build in _GROUP_SIZE_SPELLINGS.items():
            expected = build(self.df)
            for selector in ("default", "greedy"):
                with self.subTest(name, selector=selector), \
                        config(implementation_selector=selector):
                    got = _api.evaluate(build(st.as_data_op(self.df)))
                if isinstance(expected, pd.Series):
                    pd.testing.assert_series_equal(got, expected)
                    continue
                got = got.to_pandas() if isinstance(got, pl.DataFrame) else got
                if "k" not in got.columns:
                    got = got.reset_index()
                if "k" not in expected.columns:
                    expected = expected.reset_index()
                pd.testing.assert_frame_equal(got.reset_index(drop=True),
                                              expected.reset_index(drop=True),
                                              check_dtype=False)

    def test_counting_group_rows_is_a_single_unnamed_wildcard_size(self):
        def agg(*entries):
            return AggregateOp(grouped=True, grouping=(Col("k"),), aggregations=entries)
        self.assertTrue(agg((None, AggExpr("size", AllCols()))).counts_group_rows())
        self.assertFalse(agg((None, AggExpr("size", Col("v")))).counts_group_rows())
        self.assertFalse(agg(("n", AggExpr("size", AllCols()))).counts_group_rows())
        self.assertFalse(agg((None, AggExpr("count", AllCols()))).counts_group_rows())


class TestAggregateHelpers(unittest.TestCase):
    """Unit tests for the groupby/aggregation fusion predicates and extractors."""

    def test_is_groupby_op(self):
        self.assertTrue(_is_groupby_op(MethodCallOp("groupby", args=("g",), kwargs={})))
        self.assertFalse(_is_groupby_op(MethodCallOp("sum", args=(), kwargs={})))
        self.assertFalse(_is_groupby_op(Op()))

    def test_is_aggregation_direct_method(self):
        _, agg = _groupby_agg_pair(agg_method="mean")
        self.assertTrue(_is_aggregation(agg))

    def test_is_aggregation_agg_with_spec(self):
        _, agg = _groupby_agg_pair(agg_method="agg", agg_args=("sum",))
        self.assertTrue(_is_aggregation(agg))

    def test_is_aggregation_reads_only_the_method_name(self):
        # Whether a call can fuse is make_aggregate_op's decision alone, so the
        # gate gives the same answer whatever feeds the call.
        groupby, shared = _groupby_agg_pair()
        groupby.outputs.append(_method("count"))
        calls = {
            "no input": _method("sum"),
            "a frame input": _call_over(_src(OutputType.FRAME)),
            "a shared groupby": shared,
            "agg without a spec": _method("agg"),
        }
        for name, call in calls.items():
            with self.subTest(name):
                self.assertTrue(_is_aggregation(call))
        self.assertFalse(_is_aggregation(_method("head")))

    def test_is_aggregation_agg_with_spec_as_a_kwarg(self):
        # `.agg(func="sum")` is the same call as `.agg("sum")`.
        _, keyword = _groupby_agg_pair(agg_method="agg", agg_kwargs={"func": "sum"})
        _, positional = _groupby_agg_pair(agg_method="agg", agg_args=("sum",))
        self.assertTrue(_is_aggregation(keyword))
        self.assertEqual(
            _AggCall(form=_FUNC, multi=False, items=(
                _AggItem(key=None, column=None, func="sum", params={}),)),
            _extract_aggregations(keyword))
        self.assertEqual(_extract_aggregations(positional),
                         _extract_aggregations(keyword))

    def test_named_aggregation_requires_column_reduction_pairs(self):
        _, named = _groupby_agg_pair(
            agg_method="agg", agg_kwargs={"total": ("v", "sum")})
        self.assertEqual(
            _AggCall(form=_NAMED, multi=False, items=(
                _AggItem(key="total", column="v", func="sum", params={}),)),
            _extract_aggregations(named))
        # A list where the pair belongs is neither a pair nor a label.
        groupby, multi_output = _groupby_agg_pair(
            agg_method="agg", agg_kwargs={"total": ["sum", "mean"]})
        groupby.inputs = [_src(OutputType.FRAME)]
        self.assertIsNone(_extract_aggregations(multi_output))
        self.assertIsNone(make_aggregate_op(multi_output))

    def test_named_aggregation_accepts_named_agg(self):
        # pandas 3 made NamedAgg a dataclass, no longer a (column, aggfunc) tuple.
        call = _extract_aggregations(_method(
            "agg", total=pd.NamedAgg(column="v", aggfunc="sum"),
            spread=pd.NamedAgg("v", "std", ddof=0)))
        self.assertEqual(_NAMED, call.form)
        self.assertEqual((("total", AggExpr("sum", Col("v"))),
                          ("spread", AggExpr("std", Col("v"), {"ddof": 0}))),
                         _projected_aggregation_entries(call, None))

    def test_named_agg_with_positional_reduction_arguments_is_not_fused(self):
        # AggExpr has no slot for positional reduction arguments.
        groupby, agg = _groupby_agg_pair(agg_method="agg", agg_kwargs={
            "q": pd.NamedAgg("v", "quantile", 0.5)})
        groupby.inputs = [_src(OutputType.FRAME)]
        self.assertIsNone(_extract_aggregations(agg))
        self.assertIsNone(make_aggregate_op(agg))

    def test_is_aggregation_unknown_method_is_false(self):
        _, agg = _groupby_agg_pair(agg_method="head")
        self.assertFalse(_is_aggregation(agg))

    def test_extract_grouping_from_args(self):
        gb = MethodCallOp("groupby", args=("g",), kwargs={})
        self.assertEqual("g", _extract_grouping(gb))

    def test_extract_grouping_from_kwarg(self):
        gb = MethodCallOp("groupby", args=(), kwargs={"by": "g"})
        self.assertEqual("g", _extract_grouping(gb))

    def test_extract_grouping_none(self):
        gb = MethodCallOp("groupby", args=(), kwargs={})
        self.assertIsNone(_extract_grouping(gb))

    def test_extract_aggregations_from_agg_spec(self):
        self.assertEqual(
            _AggCall(form=_FUNC, multi=False, items=(
                _AggItem(key=None, column=None, func="mean", params={}),)),
            _extract_aggregations(_method("agg", "mean")))

    def test_extract_aggregations_from_direct_method(self):
        self.assertEqual(
            _AggCall(form=_FUNC, multi=False, items=(
                _AggItem(key=None, column=None, func="sum", params={}),)),
            _extract_aggregations(_method("sum")))

    def test_make_aggregate_op_normalizes_direct_method(self):
        df = InMemoryFrameOp(data=pd.DataFrame({"g": ["a"], "v": [1]}))
        groupby = MethodCallOp("groupby", args=("g",), kwargs={})
        groupby.inputs = [df]
        df.outputs = [groupby]
        agg = MethodCallOp("sum", args=(), kwargs={})
        agg.inputs = [groupby]
        groupby.outputs = [agg]

        new_op = make_aggregate_op(agg)
        self.assertIsInstance(new_op, AggregateOp)
        self.assertTrue(new_op.grouped)
        self.assertEqual((Col("g"),), new_op.grouping)
        self.assertEqual(_wildcard("sum"), new_op.aggregations)
        # The groupby op is bypassed: the frame now feeds the AggregateOp.
        self.assertIs(df, new_op.inputs[0])
        self.assertIn(new_op, df.outputs)

    def test_aggregation_entries_refuse_a_parameter_the_reduction_lacks(self):
        # ddof belongs to std/var/sem, and sum has no spelling for it. The entry
        # builder raises, and make_aggregate_op turns that into no fusion.
        def call(func):
            return _AggCall(form=_FUNC, multi=False, items=(
                _AggItem(key=None, column=None, func=func, params={"ddof": 1}),))
        with self.assertRaises(NotImplementedError):
            _aggregation_entries(call("sum"), None, False)
        self.assertIsNotNone(_aggregation_entries(call("std"), None, False))
        groupby, agg = _groupby_agg_pair(agg_kwargs={"ddof": 1})
        groupby.inputs = [_src(OutputType.FRAME)]
        self.assertIsNone(make_aggregate_op(agg))

    def test_aggregation_params_discard_hints_and_refuse_graph_fed_values(self):
        direct = MethodCallOp("sum", args=(), kwargs={"engine": "cython",
                                                      "min_count": 2})
        self.assertEqual({"min_count": 2}, _aggregation_params(direct))
        graph_fed = MethodCallOp("std", args=(), kwargs={"ddof": OperandRef(1)})
        self.assertIsNone(_aggregation_params(graph_fed))
        extra_positional = MethodCallOp("agg", args=("sum", 3), kwargs={})
        self.assertIsNone(_aggregation_params(extra_positional))
        # `func` names the reduction, it is not a parameter of it; leaving it in
        # would push an unknown parameter into AggExpr and refuse the fusion.
        spec_kwarg = MethodCallOp("agg", args=(), kwargs={"func": "std",
                                                          "ddof": 0})
        self.assertEqual({"ddof": 0}, _aggregation_params(spec_kwarg))

    def test_make_aggregate_op_refuses_an_unrepresentable_spec(self):
        # A list spec produces MultiIndex-keyed columns in pandas; the expression
        # grammar has no spelling for it, so the chain is left unfused.
        _, agg = _groupby_agg_pair(agg_method="agg", agg_args=(["sum", "mean"],))
        agg.inputs[0].inputs = [InMemoryFrameOp(data=pd.DataFrame({"g": ["a"]}))]
        self.assertIsNone(make_aggregate_op(agg))


# Aggregation fusion, one class per stage.
class TestAggregationSources(unittest.TestCase):
    """`_get_sources`: what feeds an aggregation call, and which of it is folded."""

    def _assert_sources(self, sources, *, groupby_op=None, projection_op=None,
                        frame, columns=None, series=False):
        self.assertIs(groupby_op, sources.groupby_op)
        self.assertIs(projection_op, sources.projection_op)
        self.assertIs(frame, sources.src)
        self.assertEqual(columns, sources.columns)
        self.assertIs(series, sources.is_series)
        self.assertIs(groupby_op is not None, sources.grouped)

    def test_a_bare_groupby_is_folded_and_nothing_is_projected(self):
        frame, groupby = _src(OutputType.FRAME), _groupby("g")
        self._assert_sources(_get_sources(_call_over(frame, groupby)),
                             groupby_op=groupby, frame=frame)

    def test_a_groupby_over_a_series_aggregates_a_series(self):
        # series.groupby(...) is a SeriesGroupBy with no projection after it.
        series, groupby = _src(OutputType.SERIES), _groupby("g")
        self._assert_sources(_get_sources(_call_over(series, groupby)),
                             groupby_op=groupby, frame=series, series=True)

    def test_a_grouped_projection_is_folded_with_its_columns_in_selection_order(self):
        # "v" and ["v"] select the same column, but the first aggregates a Series
        # and the second a DataFrame.
        for key, columns, series in (("v", ("v",), True),
                                     (["v"], ("v",), False),
                                     (["w", "v"], ("w", "v"), False)):
            with self.subTest(key=key):
                frame, groupby = _src(OutputType.FRAME), _groupby("g")
                projection = ColumnProjectionOp(key=key)
                sources = _get_sources(_call_over(frame, groupby, projection))
                self._assert_sources(sources, groupby_op=groupby,
                                     projection_op=projection, frame=frame,
                                     columns=columns, series=series)

    def test_a_shared_groupby_or_grouped_projection_is_refused(self):
        # Both get folded, so a second reader of either would lose its value.
        for shared in ("groupby", "projection"):
            with self.subTest(shared):
                groupby, projection = _groupby("g"), ColumnProjectionOp(key="v")
                call = _call_over(_src(OutputType.FRAME), groupby, projection)
                reader = groupby if shared == "groupby" else projection
                reader.outputs.append(_method("max"))
                self.assertIsNone(_get_sources(call))

    def test_a_grouped_getitem_without_literal_columns_is_refused(self):
        getitem = GetItemOp(key=OperandRef(1))
        call = _call_over(_src(OutputType.FRAME), _groupby("g"), getitem)
        self.assertIsNone(_get_sources(call))

    def test_an_ungrouped_projection_stays_as_the_input_and_names_no_columns(self):
        # Nothing is folded, so the entries read the projection's whole output.
        for key, series in (("v", True), (["w", "v"], False)):
            with self.subTest(key=key):
                projection = ColumnProjectionOp(key=key)
                call = _call_over(_src(OutputType.FRAME), projection)
                self._assert_sources(_get_sources(call), frame=projection, series=series)

    def test_an_ungrouped_projection_may_have_other_readers(self):
        # Nothing is folded, so a second reader of the projection loses nothing.
        projection = ColumnProjectionOp(key=["v", "w"])
        call = _call_over(_src(OutputType.FRAME), projection)
        projection.outputs.append(_method("max"))
        self.assertIs(projection, _get_sources(call).src)

    def test_a_frame_without_a_projection_is_aggregated_whole(self):
        frame = _src(OutputType.FRAME)
        self._assert_sources(_get_sources(_call_over(frame)), frame=frame)

    def test_a_row_selecting_getitem_is_a_plain_source(self):
        # df[mask] keeps every column: a frame to reduce whole, not a projection.
        mask = GetItemOp(key=OperandRef(1), is_filter=True)
        mask.output_type = OutputType.FRAME
        self._assert_sources(_get_sources(_call_over(_src(OutputType.FRAME), mask)),
                             frame=mask)

    def test_an_object_that_is_not_a_frame_is_refused_without_a_projection(self):
        # An UNKNOWN producer may be a window such as df.rolling(3). Reading it
        # as a frame would turn a rolling sum into a whole-frame sum.
        for kind in (OutputType.SERIES, OutputType.UNKNOWN):
            with self.subTest(kind=kind.name):
                self.assertIsNone(_get_sources(_call_over(_src(kind))))

    @staticmethod
    def _product_of_columns():
        """``df["a"] * df["b"]`` wired over one frame: ``(frame, product)``."""
        frame = _src(OutputType.FRAME)
        a, b = ColumnProjectionOp(key="a"), ColumnProjectionOp(key="b")
        product = BinOp(operator.mul, OperandRef(0), OperandRef(1))
        product.output_type = OutputType.SERIES
        a.inputs, b.inputs, product.inputs = [frame], [frame], [a, b]
        frame.outputs, a.outputs, b.outputs = [a, b], [product], [product]
        return frame, product

    def test_an_ungrouped_computed_series_is_found_by_the_computed_stage(self):
        # (df["a"] * df["b"]).sum() aggregates a computed Series, which
        # _get_sources refuses and the computed stage finds.
        frame, product = self._product_of_columns()
        call = _call_over(product)
        self.assertIsNone(_get_sources(call))
        sources = _get_computed_sources(call, None)
        self._assert_sources(sources, frame=frame, series=True)
        self.assertIs(product, sources.producer)

    def test_a_grouped_computed_series_is_re_read_by_the_computed_stage(self):
        # (df["a"] * df["b"]).groupby(df["g"]).sum(): _get_sources sees a
        # SeriesGroupBy over the product, and the computed stage re-reads it as
        # the computed Series.
        frame, product = self._product_of_columns()
        key = ColumnProjectionOp(key="g")
        key.inputs, frame.outputs = [frame], [*frame.outputs, key]
        groupby = _groupby(OperandRef(1))
        call = _call_over(product, groupby)
        groupby.inputs = [product, key]
        key.outputs = [groupby]
        plain = _get_sources(call)
        self._assert_sources(plain, groupby_op=groupby, frame=product, series=True)
        sources = _get_computed_sources(call, plain)
        self._assert_sources(sources, groupby_op=groupby, frame=frame, series=True)
        self.assertIs(product, sources.producer)

    def test_the_computed_stage_skips_objects_that_are_not_computed(self):
        # A grouped projection, a plain frame, and arithmetic over two frames.
        frame = _src(OutputType.FRAME)
        projected = _call_over(frame, _groupby("g"), ColumnProjectionOp(key="v"))
        self.assertIsNone(_get_computed_sources(projected, _get_sources(projected)))
        whole = _call_over(_src(OutputType.FRAME))
        self.assertIsNone(_get_computed_sources(whole, _get_sources(whole)))
        left, right = _src(OutputType.FRAME), _src(OutputType.FRAME)
        a, b = ColumnProjectionOp(key="a"), ColumnProjectionOp(key="b")
        cross = BinOp(operator.mul, OperandRef(0), OperandRef(1))
        cross.output_type = OutputType.SERIES
        a.inputs, b.inputs, cross.inputs = [left], [right], [a, b]
        self.assertIsNone(_get_computed_sources(_call_over(cross), None))


class TestAggregationCallParsing(unittest.TestCase):
    """`_extract_aggregations`: the syntax only. Its meaning depends on the aggregated object."""

    def test_keywords_are_parameters_beside_a_spec_and_labels_without_one(self):
        # The same keyword means two things, and only the parser can tell them
        # apart. A direct method's keywords are parameters too.
        std_ddof0 = _AggCall(form=_FUNC, multi=False, items=(
            _AggItem(key=None, column=None, func="std", params={"ddof": 0}),))
        self.assertEqual(std_ddof0, _extract_aggregations(_method("std", ddof=0)))
        self.assertEqual(std_ddof0, _extract_aggregations(_method("agg", "std", ddof=0)))
        self.assertEqual(
            _AggCall(form=_LABELS, multi=False, items=(
                _AggItem(key="ddof", column=None, func="std", params={}),)),
            _extract_aggregations(_method("agg", ddof="std")))

    def test_a_dict_is_parsed_by_key_and_drops_the_other_keywords(self):
        self.assertEqual(
            _AggCall(form=_DICT, multi=False, items=(
                _AggItem(key="v", column=None, func="sum", params={}),
                _AggItem(key="w", column=None, func="mean", params={}))),
            _extract_aggregations(_method("agg", {"v": "sum", "w": "mean"})))
        # pandas drops the keywords beside a dict spec: agg({"v": "std"}, ddof=0)
        # is a sample std, as if ddof were never passed.
        self.assertEqual(
            _AggCall(form=_DICT, multi=False, items=(
                _AggItem(key="v", column=None, func="std", params={}),)),
            _extract_aggregations(_method("agg", {"v": "std"}, ddof=0)))
        # axis keeps its meaning and has no spelling here.
        self.assertIsNone(_extract_aggregations(_method("agg", {"v": "sum"}, axis=0)))

    def test_named_pairs_carry_label_column_and_their_own_parameters(self):
        call = _method("agg", total=("v", "sum"),
                       spread=pd.NamedAgg("v", "std", ddof=0))
        self.assertEqual(
            _AggCall(form=_NAMED, multi=False, items=(
                _AggItem(key="total", column="v", func="sum", params={}),
                _AggItem(key="spread", column="v", func="std", params={"ddof": 0}))),
            _extract_aggregations(call))

    def test_a_list_of_functions_is_flagged_multi(self):
        for name, call, form in (
                ("agg(list)", _method("agg", ["sum", "mean"]), _FUNC),
                ("agg(dict of lists)", _method("agg", {"v": ["sum", "mean"]}), _DICT)):
            with self.subTest(name):
                parsed = _extract_aggregations(call)
                self.assertEqual(form, parsed.form)
                self.assertTrue(parsed.multi)
        self.assertFalse(_extract_aggregations(_method("agg", {"v": "sum"})).multi)

    def test_execution_hints_are_neither_parameters_nor_labels(self):
        for name, call, items in (
                ("direct", _method("sum", engine="cython"),
                 (_AggItem(key=None, column=None, func="sum", params={}),)),
                ("agg spec", _method("agg", "sum", engine="numba"),
                 (_AggItem(key=None, column=None, func="sum", params={}),)),
                ("named", _method("agg", total=("v", "sum"), engine="numba"),
                 (_AggItem(key="total", column="v", func="sum", params={}),))):
            with self.subTest(name):
                self.assertEqual(items, _extract_aggregations(call).items)

    def test_unrepresentable_calls_parse_to_none(self):
        calls = {
            "no spec": _method("agg"),
            "graph-fed spec": _method("agg", OperandRef(1)),
            "callable spec": _method("agg", sum),
            "empty dict": _method("agg", {}),
            "extra positional argument": _method("agg", "sum", 3),
            "positional reduction argument": _method("sum", 0),
            "graph-fed parameter": _method("std", ddof=OperandRef(1)),
            "list as a label": _method("agg", total=["sum", "mean"]),
            "labels mixed with pairs": _method("agg", total="sum", avg=("v", "mean")),
            "NamedAgg with positional arguments": _method(
                "agg", q=pd.NamedAgg("v", "quantile", 0.5)),
            "not an aggregation": _method("head"),
        }
        for name, call in calls.items():
            with self.subTest(name):
                self.assertIsNone(_extract_aggregations(call))


class TestAggregationEntries(unittest.TestCase):
    """The entry builders: one function over the aggregated object, a spec that picks
    columns, or a dict or labels on a Series."""

    def test_a_function_reduces_everything_unless_a_projection_was_folded(self):
        call = _AggCall(form=_FUNC, multi=False, items=(
            _AggItem(key=None, column=None, func="sum", params={}),))
        self.assertEqual(_wildcard("sum"), _aggregation_entries(call, None, False))
        self.assertEqual(((None, AggExpr("sum", Col("w"))),
                          (None, AggExpr("sum", Col("v")))),
                         _aggregation_entries(call, ("w", "v"), False))

    def test_size_ignores_a_frame_shaped_projection_but_keeps_a_column(self):
        # groupby(k)[["v"]].size() is groupby(k).size(). groupby(k)["v"].size()
        # keeps "v" because it names the resulting series.
        size = _AggCall(form=_FUNC, multi=False, items=(
            _AggItem(key=None, column=None, func="size", params={}),))
        self.assertEqual(_wildcard("size"), _aggregation_entries(size, ("v",), False))
        self.assertEqual(((None, AggExpr("size", Col("v"))),),
                         _aggregation_entries(size, ("v",), True))

    def test_dict_keys_are_input_columns_and_named_entries_keep_their_labels(self):
        as_dict = _AggCall(form=_DICT, multi=False, items=(
            _AggItem(key="v", column=None, func="sum", params={}),
            _AggItem(key="w", column=None, func="max", params={})))
        self.assertEqual(((None, AggExpr("sum", Col("v"))),
                          (None, AggExpr("max", Col("w")))),
                         _projected_aggregation_entries(as_dict, None))
        named = _AggCall(form=_NAMED, multi=False, items=(
            _AggItem(key="total", column="v", func="sum", params={}),))
        self.assertEqual((("total", AggExpr("sum", Col("v"))),),
                         _projected_aggregation_entries(named, None))

    def test_a_spec_column_outside_the_folded_projection_is_refused(self):
        # Only a folded (grouped) projection is checked here. An ungrouped one
        # stays in the plan, and pandas raises the KeyError when it runs.
        call = _AggCall(form=_DICT, multi=False, items=(
            _AggItem(key="w", column=None, func="sum", params={}),))
        self.assertIsNone(_projected_aggregation_entries(call, ("v",)))
        self.assertIsNotNone(_projected_aggregation_entries(call, ("v", "w")))
        self.assertIsNotNone(_projected_aggregation_entries(call, None))

    def test_labels_on_a_series_reduce_the_folded_column_or_the_series_itself(self):
        labels = _AggCall(form=_LABELS, multi=False, items=(
            _AggItem(key="total", column=None, func="sum", params={}),
            _AggItem(key="avg", column=None, func="mean", params={})))
        # df.groupby("g")["v"].agg(...): the folded selection names the column.
        self.assertEqual((("total", AggExpr("sum", Col("v"))),
                          ("avg", AggExpr("mean", Col("v")))),
                         _labelled_aggregation_entries(labels, ("v",)))
        # df["v"].agg(...) and df["v"].groupby(...).agg(...): the Series is the input.
        self.assertEqual((("total", AggExpr("sum", AllCols())),
                          ("avg", AggExpr("mean", AllCols()))),
                         _labelled_aggregation_entries(labels, None))
        # On a Series a dict key is a label too, not a column.
        as_dict = _AggCall(form=_DICT, multi=False, items=(
            _AggItem(key="total", column=None, func="sum", params={}),))
        self.assertEqual((("total", AggExpr("sum", AllCols())),),
                         _labelled_aggregation_entries(as_dict, None))

    def test_a_key_or_function_that_is_not_a_name_is_refused(self):
        def call(key, func):
            return _AggCall(form=_DICT, multi=False, items=(
                _AggItem(key=key, column=None, func=func, params={}),))
        for builder in (_projected_aggregation_entries, _labelled_aggregation_entries):
            for key, func in ((1, "sum"), ("v", sum)):
                with self.subTest(builder=builder.__name__, key=key, func=func):
                    self.assertIsNone(builder(call(key, func), None))


class TestAggregateFusion(unittest.TestCase):
    """`make_aggregate_op`: what is folded, what stays, and which pairs are refused."""

    def test_a_whole_frame_reduction_becomes_an_ungrouped_aggregate(self):
        frame = _src(OutputType.FRAME)
        call = _call_over(frame)
        new_op = make_aggregate_op(call)
        self.assertFalse(new_op.grouped)
        self.assertEqual((), new_op.grouping)
        self.assertEqual(_wildcard("sum"), new_op.aggregations)
        self.assertEqual([frame], new_op.inputs)
        self.assertEqual([new_op], frame.outputs)
        self.assertEqual([], call.inputs)
        self.assertIs(OutputType.SERIES, new_op.output_type)

    def test_an_ungrouped_projection_stays_as_the_input(self):
        for key in ("v", ["v", "w"]):
            with self.subTest(key=key):
                frame, projection = _src(OutputType.FRAME), ColumnProjectionOp(key=key)
                new_op = make_aggregate_op(_call_over(frame, projection))
                self.assertEqual([projection], new_op.inputs)
                self.assertEqual([frame], projection.inputs)
                self.assertEqual([new_op], projection.outputs)
                self.assertEqual(_wildcard("sum"), new_op.aggregations)

    def test_a_grouped_projection_is_folded_away(self):
        frame, groupby = _src(OutputType.FRAME), _groupby("g")
        projection = ColumnProjectionOp(key=["w", "v"])
        new_op = make_aggregate_op(_call_over(frame, groupby, projection))
        self.assertEqual([frame], new_op.inputs)
        self.assertEqual([new_op], frame.outputs)
        self.assertEqual([], groupby.inputs)
        self.assertEqual([], projection.inputs)
        self.assertEqual((Col("g"),), new_op.grouping)
        self.assertEqual(((None, AggExpr("sum", Col("w"))),
                          (None, AggExpr("sum", Col("v")))),
                         new_op.aggregations)

    def test_the_aggregated_object_and_its_labels_decide_the_result_kind(self):
        # A label widens a Series result by one level: a series instead of a
        # scalar without a groupby, a frame instead of a series with one.
        def frame(*selection):
            return [_src(OutputType.FRAME), *selection]

        def grouped(*selection, **options):
            return frame(_groupby("g", **options), *selection)
        labels = dict(method="agg", kwargs={"total": "sum"})
        cases = {
            "df.sum()": (frame(), {}, OutputType.SERIES),
            "df['v'].sum()": (frame(ColumnProjectionOp(key="v")), {}, OutputType.SCALAR),
            "df[['v']].sum()": (frame(ColumnProjectionOp(key=["v"])), {}, OutputType.SERIES),
            "df['v'].agg(total='sum')": (frame(ColumnProjectionOp(key="v")), labels,
                                         OutputType.SERIES),
            "df['v'].agg({'total': 'sum'})": (
                frame(ColumnProjectionOp(key="v")),
                dict(method="agg", args=({"total": "sum"},)), OutputType.SERIES),
            "df.groupby('g').sum()": (grouped(), {}, OutputType.FRAME),
            "df.groupby('g').size()": (grouped(), dict(method="size"), OutputType.SERIES),
            "df.groupby('g')['v'].sum()": (grouped(ColumnProjectionOp(key="v")), {},
                                           OutputType.SERIES),
            "df.groupby('g')['v'].agg(total='sum')": (
                grouped(ColumnProjectionOp(key="v")), labels, OutputType.FRAME),
            "df.groupby('g')[['v']].sum()": (grouped(ColumnProjectionOp(key=["v"])), {},
                                             OutputType.FRAME),
            "df.groupby('g', as_index=False)['v'].sum()": (
                grouped(ColumnProjectionOp(key="v"), as_index=False), {}, OutputType.FRAME),
            "series.groupby(key).sum()": ([_src(OutputType.SERIES), _groupby("g")], {},
                                          OutputType.SERIES),
        }
        for name, (producers, call, kind) in cases.items():
            with self.subTest(name):
                new_op = make_aggregate_op(_call_over(*producers, **call))
                self.assertIs(kind, new_op.output_type)

    def test_a_shared_groupby_blocks_fusion_and_leaves_the_graph_alone(self):
        # Folding a groupby that another call also reads would drop its input.
        frame, groupby = _src(OutputType.FRAME), _groupby("g")
        call = _call_over(frame, groupby)
        other = _method("count")
        groupby.outputs.append(other)
        self.assertIsNone(make_aggregate_op(call))
        self.assertEqual([call, other], groupby.outputs)
        self.assertEqual([groupby], call.inputs)

    def test_refused_calls_leave_the_chain_as_it_was(self):
        # A refused call stays a MethodCallOp, and pandas runs it.
        def frame(*selection):
            return [_src(OutputType.FRAME), *selection]

        def grouped(*selection):
            return frame(_groupby("g"), *selection)

        def agg(*args, **kwargs):
            return dict(method="agg", args=args, kwargs=kwargs)
        column, columns = (lambda: ColumnProjectionOp(key="v"),
                           lambda: ColumnProjectionOp(key=["v"]))
        cases = {
            # Nothing to aggregate.
            "no input": ([], {}),
            "agg() without a spec": (grouped(), agg()),
            # A spelling the aggregated object does not take (_SUPPORTED_FORMS).
            "SeriesGroupBy + dict": (grouped(column()), agg({"t": "sum"})),
            "SeriesGroupBy + named": (grouped(column()), agg(t=("v", "sum"))),
            "Series + named": (frame(column()), agg(t=("v", "sum"))),
            "DataFrameGroupBy + labels": (grouped(), agg(t="sum")),
            "DataFrame + labels": (frame(), agg(t="sum")),
            "DataFrame + named": (frame(), agg(t=("v", "sum"))),
            # Several outputs per input: a MultiIndex layout.
            "df.agg(list)": (frame(), agg(["sum", "mean"])),
            "groupby.agg(dict of lists)": (grouped(), agg({"v": ["sum", "mean"]})),
            "groupby['v'].agg(list)": (grouped(column()), agg(["sum", "mean"])),
            # size, first and last mean something else without a groupby.
            "df.size()": (frame(), dict(method="size")),
            "df.first()": (frame(), dict(method="first")),
            "df.agg('last')": (frame(), agg("last")),
            "df.agg({'v': 'size'})": (frame(), agg({"v": "size"})),
            # A grouped selection that dropped a spec column: pandas raises KeyError.
            "groupby[['v']].agg({'w': 'sum'})": (grouped(columns()), agg({"w": "sum"})),
            "groupby[['v']].agg(t=('w', 'sum'))": (grouped(columns()), agg(t=("w", "sum"))),
            # A groupby without keys, or a parameter AggExpr cannot represent.
            "groupby(level=0)": ([_src(OutputType.FRAME), _groupby(level=0)], {}),
            "df.sum(axis=1)": (frame(), dict(kwargs={"axis": 1})),
            "df.sum(ddof=1)": (frame(), dict(kwargs={"ddof": 1})),
        }
        for name, (producers, call) in cases.items():
            with self.subTest(name):
                op = _call_over(*producers, **call)
                self.assertIsNone(make_aggregate_op(op))
                self.assertEqual(producers[-1:], op.inputs)
                for producer, consumer in zip(producers, [*producers[1:], op]):
                    self.assertEqual([consumer], producer.outputs)


def _run_plan(ops):
    """Execute an extracted plan in order and return the last op's value."""
    pool = BufferPool()
    for op in ops:
        inputs = [pool.pin(key) for key in op.inputs]
        pool.put(op, op.process(inputs))
    return pool.pin(ops[-1])


def _assert_same_value(result, expected):
    """Compare a plan's value with pandas', whatever its kind."""
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(result, expected)
    elif isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(result, expected)
    else:
        assert result == expected


class TestAggregateRewrites(unittest.TestCase):
    """End-to-end: skrub `groupby(...).agg(...)` expressions fuse into AggregateOp."""

    _run_plan = staticmethod(_run_plan)

    def setUp(self):
        self.df = pd.DataFrame({
            "g": ["a", "a", "b"],
            "h": ["x", "y", "x"],
            "v": [1, 2, 3],
            "w": [4, 5, 6],
        })

    def _agg_ops(self, data, literal_inlining=False, **kwargs):
        ops = optimize(data, OptConfig(dataframe_ops=True, literal_inlining=literal_inlining), **kwargs)
        return ops, [o for o in ops if isinstance(o, AggregateOp)]

    def test_agg_with_spec_fuses_and_executes(self):
        data = st.as_data_op(self.df).groupby("g").agg("sum")
        ops, agg_ops = self._agg_ops(data)
        self.assertEqual(1, len(agg_ops))
        self.assertEqual((Col("g"),), agg_ops[0].grouping)
        self.assertEqual(_wildcard("sum"), agg_ops[0].aggregations)
        pd.testing.assert_frame_equal(
            self._run_plan(ops), self.df.groupby("g").agg("sum"))

    def test_direct_method_and_agg_spec_share_a_structure_key(self):
        # The CSE win the expression representation exists for.
        _, direct = self._agg_ops(st.as_data_op(self.df).groupby("g").sum())
        _, spec = self._agg_ops(st.as_data_op(self.df).groupby("g").agg("sum"))
        self.assertEqual(direct[0].aggregations, spec[0].aggregations)
        self.assertEqual(direct[0].structure_key()[2],
                         spec[0].structure_key()[2])

    def test_agg_spec_as_a_kwarg_fuses_to_the_same_op(self):
        ops, kwarg = self._agg_ops(
            st.as_data_op(self.df).groupby("g").agg(func="sum"))
        _, positional = self._agg_ops(
            st.as_data_op(self.df).groupby("g").agg("sum"))
        self.assertEqual(1, len(kwarg))
        self.assertEqual(positional[0].structure_key()[2],
                         kwarg[0].structure_key()[2])
        pd.testing.assert_frame_equal(
            self._run_plan(ops), self.df.groupby("g").agg(func="sum"))

    def test_named_aggregation_fuses_and_preserves_output_names(self):
        data = st.as_data_op(self.df).groupby("g").agg(
            total=("v", "sum"), minimum=("v", "min"))
        ops, agg_ops = self._agg_ops(data)
        self.assertEqual(1, len(agg_ops))
        self.assertEqual(
            (("total", AggExpr("sum", Col("v"))),
             ("minimum", AggExpr("min", Col("v")))),
            agg_ops[0].aggregations)
        pd.testing.assert_frame_equal(
            self._run_plan(ops),
            self.df.groupby("g").agg(total=("v", "sum"), minimum=("v", "min")))

    def test_direct_method_fuses_and_executes(self):
        data = st.as_data_op(self.df).groupby("g").mean(numeric_only=True)
        ops, agg_ops = self._agg_ops(data)
        self.assertEqual(1, len(agg_ops))
        self.assertEqual("mean", agg_ops[0].aggregations[0][1].func)

    def test_multikey_dict_spec_fuses(self):
        data = st.as_data_op(self.df).groupby(["g", "h"]).agg({"v": "sum"})
        ops, agg_ops = self._agg_ops(data)
        self.assertEqual(1, len(agg_ops))
        self.assertEqual((Col("g"), Col("h")), agg_ops[0].grouping)
        self.assertEqual(((None, AggExpr("sum", Col("v"))),),
                         agg_ops[0].aggregations)

    def test_by_kwarg_fuses(self):
        data = st.as_data_op(self.df).groupby(by="g").agg("sum")
        ops, agg_ops = self._agg_ops(data)
        self.assertEqual(1, len(agg_ops))
        self.assertEqual((Col("g"),), agg_ops[0].grouping)

    def test_variable_grouping_key_selects_column(self):
        data = st.as_data_op(self.df).groupby(st.var("key")).agg("sum")
        ops, agg_ops = self._agg_ops(data, env={"key": "g"})
        self.assertEqual(1, len(agg_ops))
        self.assertEqual((Col("g"),), agg_ops[0].grouping)
        pd.testing.assert_frame_equal(self._run_plan(ops),
                                      self.df.groupby("g").agg("sum"))

    def test_variable_grouping_key_selects_column_on_polars(self):
        with force_polars():
            data = st.as_data_op(self.df).groupby(st.var("key")).agg({"v": "sum"})
            ops, agg_ops = self._agg_ops(data, env={"key": "g"})
            self.assertEqual((Col("g"),), agg_ops[0].grouping)
            result = self._run_plan(ops)
            self.assertEqual([{"g": "a", "v": 3}, {"g": "b", "v": 3}],
                             result.to_dicts())

    def test_variable_aggregation_spec_does_not_fuse(self):
        # A graph-fed spec has no canonical reduction name at plan time, so there
        # is no AggExpr to build; the chain stays unfused rather than mis-modelled.
        data = st.as_data_op(self.df).groupby("g").agg(st.var("spec"))
        ops, agg_ops = self._agg_ops(data, env={"spec": "sum"})
        self.assertEqual(0, len(agg_ops))
        pd.testing.assert_frame_equal(self._run_plan(ops),
                                      self.df.groupby("g").agg("sum"))

    def test_reduction_parameters_are_carried_and_applied(self):
        # Dropping these silently changes the numbers: ddof=0 computed as ddof=1.
        floats = pd.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 2.0, 3.0]})
        for expr, expected, params in [
            (st.as_data_op(floats).groupby("g").std(ddof=0),
             floats.groupby("g").std(ddof=0), {"ddof": 0}),
            (st.as_data_op(floats).groupby("g").sum(min_count=5),
             floats.groupby("g").sum(min_count=5), {"min_count": 5}),
        ]:
            with self.subTest(params=params):
                ops, agg_ops = self._agg_ops(expr)
                self.assertEqual(1, len(agg_ops))
                self.assertEqual(params, agg_ops[0].aggregations[0][1].params)
                pd.testing.assert_frame_equal(self._run_plan(ops), expected)

    def test_agg_spec_forwards_its_kwargs_to_the_reduction(self):
        floats = pd.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 2.0, 3.0]})
        data = st.as_data_op(floats).groupby("g").agg("std", ddof=0)
        ops, agg_ops = self._agg_ops(data)
        self.assertEqual(1, len(agg_ops))
        self.assertEqual({"ddof": 0}, agg_ops[0].aggregations[0][1].params)
        pd.testing.assert_frame_equal(self._run_plan(ops),
                                      floats.groupby("g").agg("std", ddof=0))

    def test_execution_hints_are_discarded_not_refused(self):
        # engine says how to run, never what the result is, so it must neither
        # reach the logical op nor block the rewrite.
        data = st.as_data_op(self.df).groupby("g").sum(engine="cython")
        ops, agg_ops = self._agg_ops(data)
        self.assertEqual(1, len(agg_ops))
        self.assertEqual({}, agg_ops[0].aggregations[0][1].params)
        pd.testing.assert_frame_equal(self._run_plan(ops),
                                      self.df.groupby("g").sum())

    def test_graph_fed_reduction_parameter_does_not_fuse(self):
        data = st.as_data_op(self.df).groupby("g").std(ddof=st.var("d"))
        _, agg_ops = self._agg_ops(data, env={"d": 0})
        self.assertEqual(0, len(agg_ops))

    def test_extra_positional_agg_args_do_not_fuse(self):
        # `.agg(func, *args)` forwards the extras to func; zipping them away
        # would silently change the result.
        def scaled(group, factor):
            return group.sum() * factor

        data = st.as_data_op(self.df).groupby("g").agg(scaled, 2)
        _, agg_ops = self._agg_ops(data)
        self.assertEqual(0, len(agg_ops))

    def test_groupby_options_preserved_after_fusion(self):
        data = st.as_data_op(self.df).groupby("g", sort=False).agg("sum")
        ops, agg_ops = self._agg_ops(data)
        self.assertEqual(1, len(agg_ops))
        self.assertIs(False, agg_ops[0].sort_keys)
        self.assertIs(True, agg_ops[0].drop_null_keys)
        pd.testing.assert_frame_equal(
            self._run_plan(ops), self.df.groupby("g", sort=False).agg("sum"))

    def test_groupby_options_become_fields(self):
        data = st.as_data_op(self.df).groupby(
            "g", sort=False, dropna=False, observed=False, as_index=False).agg("sum")
        _, (agg,) = self._agg_ops(data)
        self.assertEqual((False, False, False, True, None),
                         (agg.sort_keys, agg.drop_null_keys, agg.observed,
                          agg.keys_as_columns, agg.level))

    def test_groupby_option_defaults_are_pandas(self):
        _, (agg,) = self._agg_ops(st.as_data_op(self.df).groupby("g").agg("sum"))
        self.assertEqual((True, True, True, False, None, False),
                         (agg.sort_keys, agg.drop_null_keys, agg.observed,
                          agg.keys_as_columns, agg.level, agg.sort_categories))

    def test_a_graph_fed_groupby_option_does_not_fuse(self):
        sort = st.as_data_op(False).skb.apply_func(lambda flag: flag)
        data = st.as_data_op(self.df).groupby("g", sort=sort).agg("sum")
        ops, agg_ops = self._agg_ops(data)
        self.assertEqual(0, len(agg_ops))
        pd.testing.assert_frame_equal(
            self._run_plan(ops), self.df.groupby("g", sort=False).agg("sum"))

    def test_level_based_groupby_does_not_fuse(self):
        # groupby(level=...) has no 'by' argument; fusion must be skipped to
        # avoid passing groupby(None) at runtime.
        idx = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)], names=["g", "h"])
        df = pd.DataFrame({"v": [1, 2, 3]}, index=idx)
        data = st.as_data_op(df).groupby(level=0).sum()
        _, agg_ops = self._agg_ops(data)
        self.assertEqual(0, len(agg_ops))

    def test_single_column_selection_is_absorbed_and_yields_a_series(self):
        # groupby('g')['v'].sum() puts a column projection between the two; the
        # selection is absorbed into the entry rather than blocking the fusion.
        data = st.as_data_op(self.df).groupby("g")["v"].sum()
        ops, agg_ops = self._agg_ops(data)
        self.assertEqual(1, len(agg_ops))
        self.assertEqual(((None, AggExpr("sum", Col("v"))),),
                         agg_ops[0].aggregations)
        self.assertEqual(OutputType.SERIES, agg_ops[0].output_type)
        pd.testing.assert_series_equal(self._run_plan(ops),
                                       self.df.groupby("g")["v"].sum())

    def test_multi_column_selection_is_absorbed_and_stays_a_frame(self):
        data = st.as_data_op(self.df).groupby("g")[["v", "w"]].sum()
        ops, agg_ops = self._agg_ops(data)
        self.assertEqual(1, len(agg_ops))
        self.assertEqual(((None, AggExpr("sum", Col("v"))),
                          (None, AggExpr("sum", Col("w")))),
                         agg_ops[0].aggregations)
        self.assertEqual(OutputType.FRAME, agg_ops[0].output_type)
        pd.testing.assert_frame_equal(self._run_plan(ops),
                                      self.df.groupby("g")[["v", "w"]].sum())

    def test_multi_consumer_selection_blocks_fusion(self):
        # Absorbing a selection that someone else also reads would drop a value.
        src = st.as_data_op(self.df)
        selected = src.groupby("g")["v"]
        _, shared = self._agg_ops(selected.sum() + selected.max())
        self.assertEqual(0, len(shared))
        # Control: the same shape with one consumer must fuse, otherwise the
        # assertion above would pass for the wrong reason.
        single = st.as_data_op(self.df)
        _, once = self._agg_ops(single.groupby("g")["v"].sum() + 1)
        self.assertEqual(1, len(once))

    def test_graph_fed_column_key_folds_to_the_same_shape_as_a_literal(self):
        # groupby(df["g"]) and groupby("g") are the same computation; folding the
        # key makes them share a grouping value, hence a CSE key.
        src = st.as_data_op(self.df)
        _, folded = self._agg_ops(src.groupby(src["g"]).agg("sum"))
        _, literal = self._agg_ops(st.as_data_op(self.df).groupby("g").agg("sum"))
        self.assertEqual((Col("g"),), folded[0].grouping)
        self.assertEqual(literal[0].grouping, folded[0].grouping)
        self.assertEqual(literal[0].structure_key()[2],
                         folded[0].structure_key()[2])

    def test_expression_grouping_key_folds_into_the_grouping(self):
        df = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", "2021-02-03",
                                                "2021-05-06"]),
                           "v": [1, 2, 3]})
        src = st.as_data_op(df)
        ops, agg_ops = self._agg_ops(src.groupby(src["d"].dt.year).agg({"v": "sum"}))
        self.assertEqual(1, len(agg_ops))
        self.assertEqual((DtExpr(Col("d"), "year"),), agg_ops[0].grouping)
        result = self._run_plan(ops)
        expected = df.groupby(df["d"].dt.year).agg({"v": "sum"})
        pd.testing.assert_frame_equal(result, expected)

    # End to end through the extraction pass: plan shapes and rule interactions
    # the parity tables in test_aggregation_semantics.py do not check.
    def test_direct_method_and_agg_spec_share_a_structure_key_without_a_groupby(self):
        numeric = self.df[["v", "w"]]
        _, direct = self._agg_ops(st.as_data_op(numeric).sum())
        _, spec = self._agg_ops(st.as_data_op(numeric).agg("sum"))
        self.assertEqual(direct[0].aggregations, spec[0].aggregations)
        self.assertEqual(direct[0].structure_key()[2], spec[0].structure_key()[2])

    def test_a_shared_ungrouped_projection_feeds_every_aggregate(self):
        # Unlike a grouped selection, an ungrouped one is never folded, so a
        # second reader blocks neither aggregation.
        selected = st.as_data_op(self.df)[["v", "w"]]
        ops, agg_ops = self._agg_ops(selected.sum() + selected.max())
        self.assertEqual(2, len(agg_ops))
        self.assertEqual(1, sum(isinstance(o, ColumnProjectionOp) for o in ops))
        numeric = self.df[["v", "w"]]
        pd.testing.assert_series_equal(self._run_plan(ops),
                                       numeric.sum() + numeric.max())

    def test_a_computed_series_folds_into_the_entries(self):
        # The arithmetic becomes each entry's child, so no column projection or
        # NumericOp is left and the frame is the only input.
        mul = BinOpExpr(operator.mul, Col("v"), Col("w"))
        q6 = BinOpExpr(operator.mul, Col("v"), BinOpExpr(operator.sub, Const(1), Col("w")))
        cases = {
            "(v * w).sum()": (lambda d: (d["v"] * d["w"]).sum(),
                              ((None, AggExpr("sum", mul)),), OutputType.SCALAR),
            "(v * (1 - w)).sum()": (lambda d: (d["v"] * (1 - d["w"])).sum(),
                                    ((None, AggExpr("sum", q6)),), OutputType.SCALAR),
            "(-v).sum()": (lambda d: (-d["v"]).sum(),
                           ((None, AggExpr("sum", UnaryOpExpr(operator.neg, Col("v")))),),
                           OutputType.SCALAR),
            "(v * w).agg({'total': 'sum'})": (
                lambda d: (d["v"] * d["w"]).agg({"total": "sum"}),
                (("total", AggExpr("sum", mul)),), OutputType.SERIES),
            "(v * w).agg(total='sum', avg='mean')": (
                lambda d: (d["v"] * d["w"]).agg(total="sum", avg="mean"),
                (("total", AggExpr("sum", mul)), ("avg", AggExpr("mean", mul))),
                OutputType.SERIES),
            "(v * w).groupby(g).sum()": (
                lambda d: (d["v"] * d["w"]).groupby(d["g"]).sum(),
                ((None, AggExpr("sum", mul)),), OutputType.SERIES),
            "(v * w).groupby(g).agg(total='sum')": (
                lambda d: (d["v"] * d["w"]).groupby(d["g"]).agg(total="sum"),
                (("total", AggExpr("sum", mul)),), OutputType.FRAME),
        }
        for name, (build, aggregations, kind) in cases.items():
            with self.subTest(name):
                ops, (agg,) = self._agg_ops(build(st.as_data_op(self.df)))
                self.assertEqual(aggregations, agg.aggregations)
                self.assertEqual((Col("g"),) if agg.grouped else (), agg.grouping)
                self.assertIs(kind, agg.output_type)
                self.assertEqual(1, len(agg.inputs))
                self.assertEqual(
                    [], [o for o in ops if isinstance(o, (ColumnProjectionOp, NumericOp))])
                _assert_same_value(self._run_plan(ops), build(self.df))

    def test_ungrouped_computed_reduction_after_filter_and_join(self):
        right = pd.DataFrame({"g": ["a", "b"], "u": [0.25, 0.5]})

        def after_filter(d):
            flt = d[d["g"] == "a"]
            return (flt["v"] * (1 - flt["w"])).sum()

        def after_join(d, r):
            jn = d.merge(r, on="g")
            return (jn["v"] * (1 - jn["u"])).sum()

        for name, build, env_args in [
            ("after filter", after_filter, (self.df,)),
            ("after join", after_join, (self.df, right)),
        ]:
            with self.subTest(name):
                st_args = [st.as_data_op(arg) for arg in env_args]
                ops, (agg,) = self._agg_ops(build(*st_args))
                self.assertFalse(agg.grouped)
                self.assertIs(OutputType.SCALAR, agg.output_type)
                self.assertEqual(1, len(agg.inputs))
                self.assertEqual(self._run_plan(ops), build(*env_args))

    def test_a_computed_series_another_op_reads_is_not_absorbed(self):
        src = st.as_data_op(self.df)
        # A shared column stays a leaf input next to the frame.
        col = src["v"]
        ops, aggs = self._agg_ops((col * src["w"]).sum() + col.mean())
        (computed,) = [a for a in aggs if isinstance(a.aggregations[0][1].child, BinOpExpr)]
        self.assertEqual(2, len(computed.inputs))
        self.assertEqual(self._run_plan(ops),
                         (self.df["v"] * self.df["w"]).sum() + self.df["v"].mean())
        # A shared product replaces the frame as the only input, and every entry
        # reduces it directly.
        prod = src["v"] * src["w"]
        ops, aggs = self._agg_ops(prod.sum() + prod.mean())
        self.assertEqual(2, len(aggs))
        for agg in aggs:
            self.assertEqual(1, len(agg.inputs))
            self.assertIs(OutputType.SCALAR, agg.output_type)
            self.assertEqual(AllCols(), agg.aggregations[0][1].child)
        expected = self.df["v"] * self.df["w"]
        self.assertEqual(self._run_plan(ops), expected.sum() + expected.mean())


def _value_counts(**kwargs) -> MethodCallOp:
    """A `value_counts(**kwargs)` call over a series source."""
    call = MethodCallOp("value_counts", args=(), kwargs=kwargs)
    call.inputs = [_src(OutputType.SERIES)]
    call.inputs[0].outputs = [call]
    return call


class TestPolarsRefusesASeriesSource(unittest.TestCase):
    """Polars accepts grouped frames and ungrouped scalars, not keyed or frame Series."""

    def _agg(self, src_kind):
        return AggregateOp(grouped=True, inputs=[_src(src_kind)])

    def test_a_series_source_is_refused_and_a_frame_source_is_not(self):
        from stratum.optimizer.physical._aggregation_execs import PolarsAggregateOp
        self.assertFalse(PolarsAggregateOp.supports(
            self._agg(OutputType.SERIES), None))
        self.assertTrue(PolarsAggregateOp.supports(
            self._agg(OutputType.FRAME), None))

    def test_polars_support_contract_matches_executable_shapes(self):
        from stratum.optimizer.physical._aggregation_execs import PolarsAggregateOp
        typed = _src(OutputType.FRAME)
        typed.output_schema = pl.Schema({"s": pl.String, "b": pl.Boolean,
                                         "cat": pl.Categorical()})

        def reduce(child, src, func="sum", grouped=False, output_type=OutputType.SCALAR):
            return AggregateOp(grouped=grouped, grouping=(Col("g"),) if grouped else (),
                               aggregations=((None, AggExpr(func, child)),),
                               inputs=[src], output_type=output_type)
        mul = BinOpExpr(operator.mul, Col("a"), Col("b"))
        cases = {
            # One scalar reduction of a series, or of an expression over a frame.
            "series scalar": (reduce(AllCols(), _src(OutputType.SERIES)), True),
            "computed scalar": (reduce(mul, _src(OutputType.FRAME)), True),
            # A result this path cannot build: a series keyed by columns or groups.
            "frame reduced to a series": (
                reduce(AllCols(), _src(OutputType.FRAME), output_type=OutputType.SERIES), False),
            "grouped series": (reduce(Col("v"), _src(OutputType.FRAME), grouped=True,
                                      output_type=OutputType.SERIES), False),
            "frame wildcard as a scalar": (reduce(AllCols(), _src(OutputType.FRAME)), False),
            # An input dtype where Polars disagrees with pandas.
            "string sum": (reduce(Col("s"), typed), False),
            "boolean quantile": (reduce(Col("b"), typed, "quantile"), False),
            "categorical min": (reduce(Col("cat"), typed, "min"), False),
        }
        for name, (op, supported) in cases.items():
            with self.subTest(name):
                self.assertEqual(supported, PolarsAggregateOp.supports(op, None))

    def test_polars_executes_series_and_computed_scalar_reductions(self):
        from stratum._config import config
        from stratum.optimizer.physical._aggregation_execs import PolarsAggregateOp

        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.5, 1.5, 2.5]})
        for name, build in [
            ("series sum", lambda d: d["a"].sum()),
            ("computed mul sum", lambda d: (d["a"] * d["b"]).sum()),
            ("computed expr mean", lambda d: (d["a"] * (1 - d["b"])).mean()),
        ]:
            with self.subTest(name):
                with force_polars():
                    ops = optimize(build(st.as_data_op(df)), OptConfig(dataframe_ops=True))
                aggs = [o for o in ops if isinstance(o, AggregateOp)]
                self.assertEqual(1, len(aggs))
                self.assertIsInstance(aggs[0], PolarsAggregateOp)
                self.assertIs(OutputType.SCALAR, aggs[0].output_type)
                with config(implementation_selector="greedy"):
                    self.assertAlmostEqual(
                        build(df), _api.evaluate(build(st.as_data_op(df))))

    def test_the_pandas_impl_takes_a_series_source(self):
        from stratum.optimizer.physical._aggregation_execs import PandasAggregateOp
        self.assertTrue(PandasAggregateOp.supports(
            self._agg(OutputType.SERIES), None))


class TestValueCountsExtraction(unittest.TestCase):
    """`value_counts()` becomes a grouped count keyed on the values, plus a sort."""

    def test_the_grouping_key_is_the_source_itself(self):
        sort = make_value_counts_ops(_value_counts())
        agg = sort.inputs[0]
        values = OperandLeaf(OperandRef(0))
        self.assertEqual((values,), agg.grouping)
        self.assertEqual((("count", AggExpr("size", values)),), agg.aggregations)

    def test_the_groupby_runs_unsorted_so_the_sort_can_break_ties(self):
        # A key-sorted groupby would replace pandas' first-appearance tie order.
        agg = make_value_counts_ops(_value_counts()).inputs[0]
        self.assertIs(False, agg.sort_keys)

    def test_series_counts_are_a_series_and_frame_counts_stay_unfused(self):
        call = _value_counts()
        sort = make_value_counts_ops(call)
        self.assertIs(OutputType.SERIES, sort.output_type)
        self.assertIs(OutputType.SERIES, sort.inputs[0].output_type)
        call = _value_counts()
        call.inputs[0].output_type = OutputType.FRAME
        self.assertIsNone(make_value_counts_ops(call))

    def test_sorting_is_a_separate_op_and_defaults_to_descending(self):
        sort = make_value_counts_ops(_value_counts())
        self.assertIsInstance(sort, SortOp)
        self.assertEqual((), sort.by)          # a series orders by its own values
        self.assertIs(False, sort.ascending)
        self.assertIs(True, make_value_counts_ops(
            _value_counts(ascending=True)).ascending)

    def test_sort_false_yields_the_aggregation_alone(self):
        op = make_value_counts_ops(_value_counts(sort=False))
        self.assertIsInstance(op, AggregateOp)

    def test_dropna_is_carried_to_the_groupby(self):
        for dropna in (True, False):
            with self.subTest(dropna=dropna):
                agg = make_value_counts_ops(_value_counts(dropna=dropna)).inputs[0]
                self.assertIs(dropna, agg.drop_null_keys)

    def test_options_without_a_spelling_here_are_refused(self):
        # normalize is a ratio not a reduction, bins buckets the values first,
        # and subset groups by some columns instead of all of them.
        for kwargs in ({"normalize": True}, {"bins": 3}, {"subset": ["a"]}):
            with self.subTest(**kwargs):
                self.assertFalse(_is_value_counts(_value_counts(**kwargs)))
                self.assertIsNone(make_value_counts_ops(_value_counts(**kwargs)))

    def test_graph_fed_and_positional_calls_are_refused(self):
        self.assertFalse(_is_value_counts(_value_counts(dropna=OperandRef(1))))
        positional = MethodCallOp("value_counts", args=(True,), kwargs={})
        positional.inputs = [_src(OutputType.SERIES)]
        self.assertFalse(_is_value_counts(positional))

    def test_a_call_with_no_input_is_not_a_value_counts(self):
        self.assertFalse(_is_value_counts(MethodCallOp("value_counts", args=(),
                                                       kwargs={})))

    def test_a_grouped_value_counts_is_refused(self):
        # A second grouping level counted and ordered within each group is a
        # different computation; the plain groupby path already runs it.
        for selection in (None, ColumnProjectionOp(key="v")):
            with self.subTest(selection=selection is not None):
                groupby = MethodCallOp("groupby", args=("g",), kwargs={})
                call = MethodCallOp("value_counts", args=(), kwargs={})
                chain = [groupby] if selection is None else [groupby, selection]
                for producer, consumer in zip(chain, chain[1:] + [call]):
                    consumer.inputs = [producer]
                    producer.outputs = [consumer]
                self.assertFalse(_is_value_counts(call))
                self.assertIsNone(make_value_counts_ops(call))


class TestValueCountsPipeline(unittest.TestCase):
    """End to end: the `value_counts` filter pipeline that motivates #195."""

    _run_plan = staticmethod(_run_plan)

    def setUp(self):
        # `c` and `e` tie on 3, and `c` appears first, so the tie order pandas
        # produces differs from the one a key-sorted groupby would.
        self.df = pd.DataFrame({
            "t": ["c"] * 2 + ["a"] * 5 + ["b"] * 4 + ["c"] + ["d"] + ["e"] * 3,
            "x": range(16),
        })

    def test_value_counts_matches_pandas_including_tie_order(self):
        ops = optimize(st.as_data_op(self.df)["t"].value_counts(),
                       OptConfig(dataframe_ops=True))
        self.assertEqual(1, len([o for o in ops if isinstance(o, AggregateOp)]))
        self.assertEqual(1, len([o for o in ops if isinstance(o, SortOp)]))
        pd.testing.assert_series_equal(self._run_plan(ops),
                                       self.df["t"].value_counts())

    def test_a_grouped_value_counts_stays_unfused_and_still_runs(self):
        data = st.as_data_op(self.df).groupby("t")["x"].value_counts()
        ops = optimize(data, OptConfig(dataframe_ops=True))
        self.assertEqual([], [o for o in ops if isinstance(o, AggregateOp)])
        pd.testing.assert_series_equal(
            self._run_plan(ops), self.df.groupby("t")["x"].value_counts())

    def test_value_counts_without_sorting_skips_the_sort_op(self):
        ops = optimize(st.as_data_op(self.df)["t"].value_counts(sort=False),
                       OptConfig(dataframe_ops=True))
        self.assertEqual([], [o for o in ops if isinstance(o, SortOp)])
        pd.testing.assert_series_equal(self._run_plan(ops),
                                       self.df["t"].value_counts(sort=False))

    def _plan(self, **config):
        def build(frame):
            target = frame["t"]
            counts = target.value_counts()
            eligible = counts[counts >= 3].index
            return frame[target.isin(eligible)].reset_index(drop=True)
        return (optimize(build(st.as_data_op(self.df)),
                         OptConfig(dataframe_ops=True, **config)),
                build(self.df))

    def test_the_driving_pipeline_extracts_and_matches_pandas(self):
        ops, expected = self._plan()
        pd.testing.assert_frame_equal(self._run_plan(ops), expected)

    def test_value_counts_index_stays_an_index_when_order_is_observed(self):
        counts = st.as_data_op(self.df)["t"].value_counts()
        result = counts[counts >= 3].index
        ops = optimize(result, OptConfig(dataframe_ops=True))
        self.assertTrue(any(isinstance(op, IndexAccessOp) for op in ops))
        self.assertFalse(any(isinstance(op, GroupKeysOp) for op in ops))
        expected_counts = self.df["t"].value_counts()
        pd.testing.assert_index_equal(
            self._run_plan(ops), expected_counts[expected_counts >= 3].index)

    def test_null_including_counts_keep_the_pandas_index_path(self):
        df = pd.DataFrame({"t": ["a", None, "a", None], "x": range(4)})
        source = st.as_data_op(df)
        values = source["t"]
        counts = values.value_counts(dropna=False)
        result = source[values.isin(counts[counts >= 2].index)]
        ops = optimize(result, OptConfig(dataframe_ops=True))
        self.assertTrue(any(isinstance(op, IndexAccessOp) for op in ops))
        self.assertFalse(any(isinstance(op, GroupKeysOp) for op in ops))
        expected_counts = df["t"].value_counts(dropna=False)
        expected = df[df["t"].isin(expected_counts[expected_counts >= 2].index)]
        pd.testing.assert_frame_equal(self._run_plan(ops), expected)

    def test_group_key_named_count_does_not_collide_with_count_result(self):
        df = self.df.rename(columns={"t": "count"})
        source = st.as_data_op(df)
        values = source["count"]
        counts = values.value_counts()
        result = source[values.isin(counts[counts >= 3].index)].reset_index(drop=True)
        expected_counts = df["count"].value_counts()
        expected = df[df["count"].isin(
            expected_counts[expected_counts >= 3].index)].reset_index(drop=True)
        for selector in ("default", "greedy"):
            with self.subTest(selector=selector), st.config(implementation_selector=selector):
                ops, *_ = optimize_(result, OptConfig(dataframe_ops=True))
                actual = self._run_plan(ops)
                if isinstance(actual, pl.DataFrame):
                    actual = actual.to_pandas()
                pd.testing.assert_frame_equal(actual, expected)

    def test_the_driving_pipeline_runs_with_polars(self):
        source = st.as_data_op(self.df)
        target = source["t"]
        counts = target.value_counts()
        eligible = counts[counts >= 3].index
        filtered = source[target.isin(eligible)].reset_index(drop=True)
        expected = self.df[self.df["t"].isin(
            self.df["t"].value_counts().loc[lambda values: values >= 3].index
        )].reset_index(drop=True)
        with st.config(implementation_selector="greedy"):
            ops, *_ = optimize_(filtered, OptConfig(dataframe_ops=True))
            actual = self._run_plan(ops)
        from stratum.optimizer.physical._aggregation_execs import PolarsAggregateOp
        from stratum.optimizer.physical._index_execs import PolarsGroupKeysOp
        self.assertTrue(any(isinstance(op, PolarsAggregateOp) for op in ops))
        self.assertTrue(any(isinstance(op, PolarsGroupKeysOp) for op in ops))
        self.assertFalse(any(isinstance(op, SortOp) for op in ops))
        self.assertIsInstance(actual, pl.DataFrame)
        pd.testing.assert_frame_equal(actual.to_pandas(), expected)

    def test_every_gap_in_the_pipeline_is_extracted(self):
        ops, _ = self._plan()
        kinds = [type(o) for o in ops]
        self.assertEqual(1, sum(issubclass(k, AggregateOp) for k in kinds))
        self.assertEqual(0, sum(issubclass(k, SortOp) for k in kinds))
        self.assertEqual(1, sum(issubclass(k, GroupKeysOp) for k in kinds))
        self.assertEqual(0, sum(issubclass(k, IndexAccessOp) for k in kinds))
        self.assertEqual(1, sum(issubclass(k, MetadataOp) for k in kinds))
        # Two mask selections before the semi-join promotion runs: one over the
        # counts, a mask on a *series*, which used to be refused, and one over
        # the rows of the frame.
        unpromoted, _ = self._plan(semi_join_rewrite=False)
        masks = [o for o in unpromoted if isinstance(o, MaskSelectionOp)]
        self.assertEqual(2, len(masks))
        self.assertEqual({OutputType.SERIES, OutputType.FRAME},
                         {m.output_type for m in masks})
        # The frame one is what the promotion consumes, leaving the series one.
        self.assertEqual(1, sum(isinstance(o, SelectionOp) for o in ops))
        self.assertEqual(1, sum(isinstance(o, IndexAccessOp) for o in unpromoted))

    def test_no_raw_method_calls_survive(self):
        ops, _ = self._plan()
        self.assertEqual([], [o.method_name for o in ops
                              if type(o) is MethodCallOp])


if __name__ == "__main__":
    unittest.main()
