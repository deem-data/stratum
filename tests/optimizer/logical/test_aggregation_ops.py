import operator
import unittest

import pandas as pd
import polars as pl
import stratum as st
from stratum.optimizer._optimize import OptConfig
from stratum.optimizer.logical._aggregation_ops import (
    AggregateOp, _aggregation_entries, _aggregation_params,
    _extract_aggregations, _extract_grouping, _is_aggregation, _is_groupby_op,
    _is_value_counts, make_aggregate_op, make_value_counts_ops)
from stratum.optimizer.logical._column_expr import (
    AggExpr, AllCols, BinOpExpr, Col, DtExpr, OperandLeaf)
from stratum.optimizer.logical._projection_ops import (
    ColumnProjectionOp, GetAttrProjectionOp, MetadataOp)
from stratum.optimizer.logical._selection_ops import SelectionKind, SelectionOp
from stratum.optimizer.logical._sort_ops import SortOp
from stratum.optimizer.logical._source_ops import DataSourceOp
from stratum.optimizer.logical._ops import MethodCallOp, Op, OperandRef, OutputType
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
                         aggregations=_wildcard("sum"), options={"sort": False})
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
                                   options={"sort": False})
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
        # A scalar is not frame-world data; the lattice spells that UNKNOWN.
        op = AggregateOp(grouped=False, aggregations=_wildcard(),
                         inputs=[_src(OutputType.SERIES)])
        self.assertEqual(OutputType.UNKNOWN, op.output_type)

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

    def test_is_aggregation_agg_with_spec_as_a_kwarg(self):
        # `.agg(func="sum")` is the same call as `.agg("sum")`.
        _, agg = _groupby_agg_pair(agg_method="agg", agg_kwargs={"func": "sum"})
        self.assertTrue(_is_aggregation(agg))
        self.assertEqual("sum", _extract_aggregations(agg))

    def test_named_aggregation_requires_column_reduction_pairs(self):
        _, named = _groupby_agg_pair(
            agg_method="agg", agg_kwargs={"total": ("v", "sum")})
        self.assertTrue(_is_aggregation(named))
        _, multi_output = _groupby_agg_pair(
            agg_method="agg", agg_kwargs={"total": ["sum", "mean"]})
        self.assertFalse(_is_aggregation(multi_output))

    def test_is_aggregation_agg_without_spec_is_false(self):
        _, agg = _groupby_agg_pair(agg_method="agg", agg_args=())
        self.assertFalse(_is_aggregation(agg))

    def test_is_aggregation_no_inputs_is_false(self):
        self.assertFalse(_is_aggregation(MethodCallOp("sum", args=(), kwargs={})))

    def test_is_aggregation_non_groupby_input_is_false(self):
        agg = MethodCallOp("sum", args=(), kwargs={})
        agg.inputs = [DataSourceOp(data=pd.DataFrame({"a": [1]}))]
        self.assertFalse(_is_aggregation(agg))

    def test_is_aggregation_multi_consumer_groupby_is_false(self):
        groupby, agg = _groupby_agg_pair()
        # A second consumer of the groupby blocks fusion.
        groupby.outputs.append(MethodCallOp("count", args=(), kwargs={}))
        self.assertFalse(_is_aggregation(agg))

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
        agg = MethodCallOp("agg", args=("mean",), kwargs={})
        self.assertEqual("mean", _extract_aggregations(agg))

    def test_extract_aggregations_from_direct_method(self):
        agg = MethodCallOp("sum", args=(), kwargs={})
        self.assertEqual("sum", _extract_aggregations(agg))

    def test_make_aggregate_op_normalizes_direct_method(self):
        df = DataSourceOp(data=pd.DataFrame({"g": ["a"], "v": [1]}))
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
        # ddof belongs to std/var/sem; sum has no spelling for it, so the entry
        # cannot be built and the caller must leave the chain unfused.
        self.assertIsNone(_aggregation_entries("sum", {"ddof": 1}))
        self.assertIsNotNone(_aggregation_entries("std", {"ddof": 1}))

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
        agg.inputs[0].inputs = [DataSourceOp(data=pd.DataFrame({"g": ["a"]}))]
        self.assertIsNone(make_aggregate_op(agg))


def _run_plan(ops):
    """Execute an extracted plan in order and return the last op's value."""
    pool = BufferPool()
    for op in ops:
        inputs = [pool.pin(key) for key in op.inputs]
        pool.put(op, op.process("fit_transform", inputs))
    return pool.pin(ops[-1])


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

    def _agg_ops(self, data, **kwargs):
        ops = optimize(data, OptConfig(dataframe_ops=True), **kwargs)
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

    def test_variable_grouping_key_uses_a_placeholder_leaf(self):
        data = st.as_data_op(self.df).groupby(st.var("key")).agg("sum")
        ops, agg_ops = self._agg_ops(data, env={"key": "g"})
        self.assertEqual(1, len(agg_ops))
        self.assertEqual((OperandLeaf(OperandRef(1)),), agg_ops[0].grouping)
        pd.testing.assert_frame_equal(self._run_plan(ops),
                                      self.df.groupby("g").agg("sum"))

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
        self.assertEqual({"sort": False}, agg_ops[0].options)
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


def _value_counts(**kwargs) -> MethodCallOp:
    """A `value_counts(**kwargs)` call over a series source."""
    call = MethodCallOp("value_counts", args=(), kwargs=kwargs)
    call.inputs = [_src(OutputType.SERIES)]
    call.inputs[0].outputs = [call]
    return call


class TestPolarsRefusesASeriesSource(unittest.TestCase):
    """polars has no expression context for a series, so it opts out at plan time."""

    def _agg(self, src_kind):
        return AggregateOp(grouped=True, inputs=[_src(src_kind)])

    def test_a_series_source_is_refused_and_a_frame_source_is_not(self):
        from stratum.optimizer.physical._aggregation_execs import PolarsAggregateOp
        self.assertFalse(PolarsAggregateOp.supports(
            self._agg(OutputType.SERIES), None))
        self.assertTrue(PolarsAggregateOp.supports(
            self._agg(OutputType.FRAME), None))

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
        self.assertIs(False, agg.options["sort"])

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
                self.assertIs(dropna, agg.options["dropna"])

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

    def test_every_gap_in_the_pipeline_is_extracted(self):
        ops, _ = self._plan()
        kinds = [type(o) for o in ops]
        self.assertEqual(1, sum(issubclass(k, AggregateOp) for k in kinds))
        self.assertEqual(1, sum(issubclass(k, SortOp) for k in kinds))
        self.assertEqual(1, sum(issubclass(k, GetAttrProjectionOp) for k in kinds))
        self.assertEqual(1, sum(issubclass(k, MetadataOp) for k in kinds))
        # Two mask selections before the semi-join promotion runs: one over the
        # counts, a mask on a *series*, which used to be refused, and one over
        # the rows of the frame.
        unpromoted, _ = self._plan(semi_join_rewrite=False)
        masks = [o for o in unpromoted if isinstance(o, SelectionOp)
                 and o.kind is SelectionKind.MASK]
        self.assertEqual(2, len(masks))
        self.assertEqual({OutputType.SERIES, OutputType.FRAME},
                         {m.output_type for m in masks})
        # The frame one is what the promotion consumes, leaving the series one.
        self.assertEqual(1, sum(isinstance(o, SelectionOp) for o in ops))

    def test_no_raw_method_calls_survive(self):
        ops, _ = self._plan()
        self.assertEqual([], [o.method_name for o in ops
                              if type(o) is MethodCallOp])


if __name__ == "__main__":
    unittest.main()
