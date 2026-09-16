import unittest

import pandas as pd
import polars as pl
import stratum as st
from stratum.optimizer._optimize import OptConfig
from stratum.optimizer.logical._aggregation_ops import (
    AggregateOp, _extract_aggregations, _extract_grouping, _is_aggregation,
    _is_groupby_op, make_aggregate_op)
from stratum.optimizer.logical._source_ops import DataSourceOp
from stratum.optimizer.logical._ops import MethodCallOp, Op, OperandRef
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


class TestAggregateOp(unittest.TestCase):
    """`AggregateOp.process` execution on both backends."""

    def test_pandas_direct_spec(self):
        df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
        op = AggregateOp(grouping_attributes="g", aggregations="sum")
        result = run_op(op, df)
        pd.testing.assert_frame_equal(result, df.groupby("g").agg("sum"))

    def test_pandas_dict_spec(self):
        df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3], "w": [4, 5, 6]})
        op = AggregateOp(grouping_attributes="g", aggregations={"v": "sum"})
        result = run_op(op, df)
        pd.testing.assert_frame_equal(result, df.groupby("g").agg({"v": "sum"}))

    def test_grouping_placeholder_resolved_from_inputs(self):
        df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
        op = AggregateOp(grouping_attributes=OperandRef(1), aggregations="sum")
        result = run_op(op, df, "g")
        pd.testing.assert_frame_equal(result, df.groupby("g").agg("sum"))

    def test_aggregation_placeholder_resolved_from_inputs(self):
        df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
        op = AggregateOp(grouping_attributes="g", aggregations=OperandRef(1))
        result = run_op(op, df, "mean")
        pd.testing.assert_frame_equal(result, df.groupby("g").agg("mean"))

    def test_str(self):
        op = AggregateOp(grouping_attributes="g", aggregations="sum")
        self.assertIn("Aggregation", str(op))
        self.assertIn("g", str(op))

    def test_polars_not_implemented(self):
        with force_polars():
            op = AggregateOp(grouping_attributes="g", aggregations="sum")
            with self.assertRaises(NotImplementedError):
                run_op(op, pl.DataFrame({"g": ["a"], "v": [1]}))


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
        self.assertEqual("g", new_op.grouping_attributes)
        self.assertEqual("sum", new_op.aggregations)
        # The groupby op is bypassed: the frame now feeds the AggregateOp.
        self.assertIs(df, new_op.inputs[0])
        self.assertIn(new_op, df.outputs)


class TestAggregateRewrites(unittest.TestCase):
    """End-to-end: skrub `groupby(...).agg(...)` expressions fuse into AggregateOp."""

    def _run_plan(self, ops):
        pool = BufferPool()
        for op in ops:
            inputs = [pool.pin(key) for key in op.inputs]
            pool.put(op, op.process("fit_transform", inputs))
        return pool.pin(ops[-1])

    def setUp(self):
        self.df = pd.DataFrame({
            "g": ["a", "a", "b"],
            "h": ["x", "y", "x"],
            "v": [1, 2, 3],
            "w": [4, 5, 6],
        })

    def test_agg_with_spec_fuses_and_executes(self):
        data = st.as_data_op(self.df).groupby("g").agg("sum")
        ops = optimize(data, OptConfig(dataframe_ops=True))
        agg_ops = [o for o in ops if isinstance(o, AggregateOp)]
        self.assertEqual(1, len(agg_ops))
        self.assertEqual("g", agg_ops[0].grouping_attributes)
        self.assertEqual("sum", agg_ops[0].aggregations)
        pd.testing.assert_frame_equal(
            self._run_plan(ops), self.df.groupby("g").agg("sum"))

    def test_direct_method_fuses_and_executes(self):
        data = st.as_data_op(self.df).groupby("g").mean(numeric_only=True)
        ops = optimize(data, OptConfig(dataframe_ops=True))
        agg_ops = [o for o in ops if isinstance(o, AggregateOp)]
        self.assertEqual(1, len(agg_ops))
        self.assertEqual("mean", agg_ops[0].aggregations)

    def test_multikey_dict_spec_fuses(self):
        data = st.as_data_op(self.df).groupby(["g", "h"]).agg({"v": "sum"})
        ops = optimize(data, OptConfig(dataframe_ops=True))
        agg_ops = [o for o in ops if isinstance(o, AggregateOp)]
        self.assertEqual(1, len(agg_ops))
        self.assertEqual(["g", "h"], agg_ops[0].grouping_attributes)
        self.assertEqual({"v": "sum"}, agg_ops[0].aggregations)

    def test_by_kwarg_fuses(self):
        data = st.as_data_op(self.df).groupby(by="g").agg("sum")
        ops = optimize(data, OptConfig(dataframe_ops=True))
        agg_ops = [o for o in ops if isinstance(o, AggregateOp)]
        self.assertEqual(1, len(agg_ops))
        self.assertEqual("g", agg_ops[0].grouping_attributes)

    def test_variable_grouping_key_uses_placeholder(self):
        data = st.as_data_op(self.df).groupby(st.var("key")).agg("sum")
        ops = optimize(data, OptConfig(dataframe_ops=True), env={"key": "g"})
        agg_ops = [o for o in ops if isinstance(o, AggregateOp)]
        self.assertEqual(1, len(agg_ops))
        self.assertEqual(OperandRef(1), agg_ops[0].grouping_attributes)
        result = self._run_plan(ops)
        pd.testing.assert_frame_equal(result, self.df.groupby("g").agg("sum"))

    def test_variable_aggregation_spec_uses_placeholder(self):
        data = st.as_data_op(self.df).groupby("g").agg(st.var("spec"))
        ops = optimize(data, OptConfig(dataframe_ops=True), env={"spec": "sum"})
        agg_ops = [o for o in ops if isinstance(o, AggregateOp)]
        self.assertEqual(1, len(agg_ops))
        self.assertEqual(OperandRef(1), agg_ops[0].aggregations)
        result = self._run_plan(ops)
        pd.testing.assert_frame_equal(result, self.df.groupby("g").agg("sum"))

    def test_both_grouping_key_and_agg_spec_are_variables(self):
        # Both operands are graph-fed; the aggregation OperandRef must be shifted
        # by the number of extra groupby inputs to avoid aliasing the key slot.
        data = st.as_data_op(self.df).groupby(st.var("key")).agg(st.var("spec"))
        ops = optimize(data, OptConfig(dataframe_ops=True),
                       env={"key": "g", "spec": "sum"})
        agg_ops = [o for o in ops if isinstance(o, AggregateOp)]
        self.assertEqual(1, len(agg_ops))
        self.assertEqual(OperandRef(1), agg_ops[0].grouping_attributes)
        self.assertEqual(OperandRef(2), agg_ops[0].aggregations)
        result = self._run_plan(ops)
        pd.testing.assert_frame_equal(result, self.df.groupby("g").agg("sum"))

    def test_groupby_kwargs_preserved_after_fusion(self):
        data = st.as_data_op(self.df).groupby("g", sort=False).agg("sum")
        ops = optimize(data, OptConfig(dataframe_ops=True))
        agg_ops = [o for o in ops if isinstance(o, AggregateOp)]
        self.assertEqual(1, len(agg_ops))
        self.assertEqual({"sort": False}, agg_ops[0].groupby_kwargs)
        result = self._run_plan(ops)
        pd.testing.assert_frame_equal(
            result, self.df.groupby("g", sort=False).agg("sum"))

    def test_level_based_groupby_does_not_fuse(self):
        # groupby(level=...) has no 'by' argument; fusion must be skipped to
        # avoid passing groupby(None) at runtime.
        idx = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)], names=["g", "h"])
        df = pd.DataFrame({"v": [1, 2, 3]}, index=idx)
        data = st.as_data_op(df).groupby(level=0).sum()
        ops = optimize(data, OptConfig(dataframe_ops=True))
        self.assertEqual(0, len([o for o in ops if isinstance(o, AggregateOp)]))

    def test_column_selection_between_groupby_and_agg_does_not_fuse(self):
        # groupby('g')['v'].sum() inserts a GetItemOp between the two, so the
        # aggregation no longer consumes the groupby directly -> no fusion.
        data = st.as_data_op(self.df).groupby("g")["v"].sum()
        ops = optimize(data, OptConfig(dataframe_ops=True))
        self.assertEqual(0, len([o for o in ops if isinstance(o, AggregateOp)]))


if __name__ == "__main__":
    unittest.main()
