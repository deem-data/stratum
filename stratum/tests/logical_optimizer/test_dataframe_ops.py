import os
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import polars as pl
import stratum as st
from skrub._data_ops._data_ops import DataOp
from stratum._config import FLAGS
from stratum.optimizer._optimize import OptConfig, optimize as optimize_
from stratum.optimizer.ir._dataframe_ops import (
    AggregateOp, ApplyUDFOp, AssignOp, ConcatOp, DataSourceOp,
    DatetimeConversionOp, DropOp, GetAttrProjectionOp, JoinOp, MetadataOp,
    ProjectionOp, SplitOp, _extract_aggregations, _extract_grouping,
    _is_aggregation, _is_groupby_op, make_aggregate_op,
    make_datetime_conversion_op, make_read_op)
from stratum.optimizer.ir._ops import (CallOp, OperandRef, GetItemOp,
                                       MethodCallOp, Op, ValueOp)
from stratum.runtime._buffer_pool import BufferPool


def optimize(dag, conf=None):
    linearized_dag, *_ = optimize_(dag, conf)
    return linearized_dag


def _inp(val):
    op = Op()
    op.intermediate = val
    op.is_dataframe_op = True
    return op


def _inputs_for(op):
    return [in_op.intermediate for in_op in op.inputs]


def run_op(op, *values, mode="fit_transform", environment=None):
    """Wire `values` as op.inputs (wrapped via `_inp`) and run `op.process`."""
    op.inputs = [_inp(v) for v in values]
    return op.process(mode, environment or {}, _inputs_for(op))


@contextmanager
def force_polars(enabled=True):
    """Temporarily set `FLAGS.force_polars`."""
    orig = FLAGS.force_polars
    FLAGS.force_polars = enabled
    try:
        yield
    finally:
        FLAGS.force_polars = orig


@contextmanager
def csv_file(df, **to_csv_kwargs):
    """Write `df` to a temp .csv file and yield its path; cleaned up on exit."""
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp, index=False, **to_csv_kwargs)
    tmp.close()
    try:
        yield tmp.name
    finally:
        os.remove(tmp.name)


@contextmanager
def npy_file(arr):
    """Write `arr` to a temp .npy file and yield its path; cleaned up on exit."""
    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False, mode="wb")
    np.save(tmp, arr)
    tmp.close()
    try:
        yield tmp.name
    finally:
        os.remove(tmp.name)


class PolarsTestCase(unittest.TestCase):
    """Base class that pins `FLAGS.force_polars=True` for every test."""

    def setUp(self):
        super().setUp()
        self._orig_force_polars = FLAGS.force_polars
        FLAGS.force_polars = True

    def tearDown(self):
        FLAGS.force_polars = self._orig_force_polars
        super().tearDown()


class TestRewrites(unittest.TestCase):
    """End-to-end rewrites produced by `optimize` on skrub DAGs."""

    def setUp(self):
        self.df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "datetime": ["2025-11-01 10:00:00",
                         "2025-11-02 15:30:00",
                         "2025-11-03 09:45:00"],
        })

    def test_data_source_from_dataframe(self):
        ops = optimize(st.as_data_op(self.df))
        self.assertEqual(1, len(ops))
        self.assertIsInstance(ops[0], DataSourceOp)

    def test_data_source_from_read_csv(self):
        with csv_file(self.df) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv)
            ops = optimize(data, OptConfig(dataframe_ops=True))
        self.assertEqual(1, len(ops))
        self.assertIsInstance(ops[0], DataSourceOp)

    def test_data_source_from_np_load(self):
        with npy_file(np.array([1, 2, 3])) as path:
            data = st.as_data_op(path).skb.apply_func(np.load)
            ops = optimize(data, OptConfig(dataframe_ops=True))
        self.assertTrue(any(isinstance(op, DataSourceOp) and op.format == "npy"
                            for op in ops))

    def test_projection_drop(self):
        ops = optimize(st.as_data_op(self.df).drop("y", axis=1))
        self.assertEqual(2, len(ops))
        self.assertIsInstance(ops[1], ProjectionOp)

    @unittest.skip("Skipping this test for now")
    def test_projection_fused_get_item(self):
        data = st.as_data_op(self.df)["x"].apply(lambda x: x + 1)
        ops = optimize(data)
        self.assertEqual(2, len(ops))
        self.assertIsInstance(ops[1], ProjectionOp)

    def test_projection_fused_get_item_with_choice(self):
        data = st.as_data_op(self.df)["x"]
        sub_dag1 = data.apply(lambda x, a: x + a, a=st.as_data_op(1))
        sub_dag2 = data
        root = st.choose_from([sub_dag1, sub_dag2]).as_data_op()
        ops = optimize(root)
        self.assertEqual(5, len(ops))
        self.assertIsInstance(ops[1], GetItemOp)
        self.assertIsInstance(ops[3], ProjectionOp)

    def test_fused_get_attr(self):
        data = st.as_data_op(self.df)[["datetime"]].apply(
            pd.to_datetime, format='%Y-%m-%d %H:%M:%S')
        data = data.assign(year=data["datetime"].dt.year,
                           month=data["datetime"].dt.month)
        data = data.copy()
        ops = optimize(data)
        self.assertEqual(8, len(ops))
        op_iter = iter(ops[3:])
        next(op_iter)
        self.assertIsInstance(next(op_iter), GetAttrProjectionOp)
        self.assertIsInstance(next(op_iter), GetAttrProjectionOp)
        self.assertIsInstance(next(op_iter), AssignOp)
        self.assertIsInstance(next(op_iter), MethodCallOp)


class TestDataSourceOp(unittest.TestCase):
    def test_unsupported_format_raises(self):
        op = DataSourceOp(file_path="nofile", _format="parquet",
                          read_args=(), read_kwargs={})
        with self.assertRaises(ValueError):
            op.process("fit_transform", {}, [])

    def test_numpy_read(self):
        with npy_file(np.array([1, 2, 3])) as path:
            op = DataSourceOp(file_path=path, _format="npy",
                              read_args=(), read_kwargs={})
            result = op.process("fit_transform", {}, [])
            np.testing.assert_array_equal(result, [1, 2, 3])

    def test_polars_from_dataframe(self):
        with force_polars():
            op = DataSourceOp(data=pd.DataFrame({"a": [1, 2]}))
            self.assertIsInstance(op.process("fit_transform", {}, []), pl.DataFrame)

    def test_polars_from_read_csv(self):
        with csv_file(pd.DataFrame({"a": [1, 2]})) as path, force_polars():
            op = DataSourceOp(file_path=path, _format="csv",
                              read_args=(), read_kwargs={})
            self.assertIsInstance(op.process("fit_transform", {}, []), pl.DataFrame)


class TestMetadataOp(unittest.TestCase):
    def test_kwargs_none_skips_check(self):
        self.assertIsNone(MetadataOp(func="rename").kwargs)

    def test_rename_polars_with_columns_kwarg(self):
        with force_polars():
            op = MetadataOp(func="rename", args=(), kwargs={"columns": {"a": "x"}})
            result = run_op(op, pl.DataFrame({"a": [1, 2], "b": [3, 4]}))
            self.assertIn("x", result.columns)

    def test_rename_polars_without_columns_kwarg(self):
        with force_polars():
            op = MetadataOp(func="rename", args=({"a": "x"},), kwargs={})
            result = run_op(op, pl.DataFrame({"a": [1], "b": [2]}))
            self.assertIn("x", result.columns)


class TestProjectionOp(unittest.TestCase):
    def test_func_and_method_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            ProjectionOp(func=lambda x: x, method="drop", args=(), kwargs={})

    def test_no_func_no_method_raises(self):
        with self.assertRaises(TypeError):
            run_op(ProjectionOp(args=(), kwargs={}), pd.DataFrame({"a": [1]}))

    def test_func_path(self):
        op = ProjectionOp(func=lambda df, v: df * v,
                          args=(OperandRef(0), 2), kwargs={})
        result = run_op(op, pd.DataFrame({"a": [1, 2]}))
        self.assertEqual([2, 4], result["a"].tolist())

    def test_method_pandas_path(self):
        op = ProjectionOp(method="drop", args=("y",), kwargs={"axis": 1})
        result = run_op(op, pd.DataFrame({"x": [1, 2], "y": [3, 4]}))
        self.assertNotIn("y", result.columns)

    def test_method_polars_raises(self):
        with force_polars():
            op = ProjectionOp(method="drop", args=(), kwargs={})
            with self.assertRaises(ValueError):
                run_op(op, pl.DataFrame({"a": [1]}))


class TestDropOpPolars(PolarsTestCase):
    def test_drop_with_columns_kwarg(self):
        op = DropOp(args=(), kwargs={"columns": ["b"]})
        result = run_op(op, pl.DataFrame({"a": [1], "b": [2], "c": [3]}))
        self.assertNotIn("b", result.columns)

    def test_ignore_errors_kwarg_branch(self):
        # NOTE: current code path appends a bool to polars' positional args, which
        # polars rejects. Test pins this (buggy) behaviour for coverage.
        op = DropOp(args=(), kwargs={"columns": ["a"], "ignore_errors": "raise"})
        with self.assertRaises(TypeError):
            run_op(op, pl.DataFrame({"a": [1], "b": [2]}))


class TestApplyUDFOp(unittest.TestCase):
    def test_pandas_single_column_str(self):
        op = ApplyUDFOp(args=(lambda x: x * 10,), kwargs={}, columns="a")
        result = run_op(op, pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        self.assertEqual([10, 20], result.tolist())

    def test_pandas_multi_column(self):
        op = ApplyUDFOp(args=(lambda x: x * 2,), kwargs={}, columns=["a", "b"])
        result = run_op(op, pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        self.assertEqual([2, 4], result["a"].tolist())

    def test_polars_sin_rewrite(self):
        with force_polars():
            op = ApplyUDFOp(args=(np.sin,), kwargs={})
            result = run_op(op, pl.Series("a", [0.0, np.pi / 2]))
            self.assertAlmostEqual(1.0, result[1], places=5)

    def test_polars_cos_rewrite(self):
        with force_polars():
            op = ApplyUDFOp(args=(np.cos,), kwargs={})
            result = run_op(op, pl.Series("a", [0.0]))
            self.assertAlmostEqual(1.0, result[0], places=5)

    def test_polars_single_col_general_func(self):
        with force_polars():
            op = ApplyUDFOp(args=(lambda x: x + 1,), kwargs={})
            result = run_op(op, pl.Series("a", [1, 2, 3]))
            self.assertEqual([2, 3, 4], result.to_list())

    def test_polars_multi_col_map_rows(self):
        with force_polars():
            op = ApplyUDFOp(args=(lambda row: (row[0] + row[1],),),
                            kwargs={}, columns=["a", "b"])
            result = run_op(op, pl.DataFrame({"a": [1, 2], "b": [3, 4]}))
            self.assertIsNotNone(result)


class TestAssignOpPolars(PolarsTestCase):
    def test_polars_series(self):
        op = AssignOp(args=(), kwargs={"b": pl.Series([10, 20])})
        result = run_op(op, pl.DataFrame({"a": [1, 2]}))
        self.assertIn("b", result.columns)

    def test_pandas_series_converted_to_polars(self):
        op = AssignOp(args=(), kwargs={"b": pd.Series([10, 20])})
        result = run_op(op, pl.DataFrame({"a": [1, 2]}))
        self.assertIn("b", result.columns)

    def test_placeholder_raises(self):
        # An OperandRef surviving into a polars assign kwarg is unsupported.
        op = AssignOp(args=(), kwargs={"b": OperandRef(1)})
        with self.assertRaises(NotImplementedError):
            run_op(op, pl.DataFrame({"a": [1, 2]}), OperandRef(1))


class TestDatetimeConversionOp(unittest.TestCase):
    def test_polars_path(self):
        with force_polars():
            op = DatetimeConversionOp(args=(), kwargs={})
            result = run_op(op, pl.Series("dt", ["2025-01-01", "2025-06-15"]))
            self.assertEqual(pl.Datetime, result.dtype)


class TestGetAttrProjectionOp(unittest.TestCase):
    def test_init_with_none(self):
        self.assertEqual([], GetAttrProjectionOp(attr_name=None).attr_name)

    def test_init_with_str(self):
        self.assertEqual(["dt"], GetAttrProjectionOp(attr_name="dt").attr_name)

    def _run_polars(self, dt_values, attr_name):
        with force_polars():
            s = pl.Series("dt", pd.to_datetime(dt_values))
            op = GetAttrProjectionOp(attr_name=attr_name, inputs=[_inp(s)], outputs=[])
            return op.process("fit_transform", {}, _inputs_for(op))

    def test_polars_year(self):
        result = self._run_polars(["2025-01-15", "2025-06-20"], ["dt", "year"])
        self.assertEqual([2025, 2025], result.to_list())

    def test_polars_dayofweek(self):
        # polars: Monday=1 (pandas: Monday=0)
        result = self._run_polars(["2025-01-06"], ["dt", "dayofweek"])
        self.assertEqual([1], result.to_list())

    def test_polars_is_month_end(self):
        result = self._run_polars(["2025-01-31", "2025-01-15"],
                                  ["dt", "is_month_end"])
        self.assertEqual([True, False], result.to_list())

class TestConcatOpPolars(PolarsTestCase):
    def test_polars_concat(self):
        op = ConcatOp(first=OperandRef(0), others=[OperandRef(1)], axis=0)
        result = run_op(op, pl.DataFrame({"a": [1, 2]}), pl.DataFrame({"a": [3, 4]}))
        self.assertEqual(4, len(result))


class TestSplitOp(unittest.TestCase):
    def _make(self, x, y, indices):
        op = SplitOp(inputs=[_inp(x), _inp(y)])
        op.indices = indices
        return op

    def test_polars(self):
        op = self._make(pl.DataFrame({"a": [10, 20, 30]}),
                        pl.DataFrame({"b": [1, 2, 3]}), [0, 2])
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertEqual(2, len(result[0]))

    def test_numpy(self):
        op = self._make(np.array([10, 20, 30, 40]), np.array([1, 2, 3, 4]), [1, 3])
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertEqual([20, 40], result[0].tolist())
        self.assertEqual([2, 4], result[1].tolist())

    def test_unsupported_type_raises(self):
        op = self._make("not_a_df", "not_a_df", [0])
        with self.assertRaises(ValueError):
            op.process("fit_transform", {}, _inputs_for(op))


class TestMakeReadOp(unittest.TestCase):
    """`make_read_op` and its end-to-end usage via the optimizer."""

    def _optimize_read(self, data):
        with st.config(fast_dataops_convert=True):
            return optimize(data, OptConfig(dataframe_ops=True))

    def test_with_variable_input(self):
        with csv_file(pd.DataFrame({"col": [1, 2]})) as path:
            data = st.var("path").skb.apply_func(pd.read_csv)
            ops = self._optimize_read(data)
            self.assertIsInstance(ops[-1], DataSourceOp)

            # Verify the resulting plan actually runs.
            pool = BufferPool()
            inputs0 = [pool.pin(key) for key in ops[0].inputs]
            result0 = ops[0].process("fit_transform", {"path": path}, inputs0)
            pool.put(ops[0], result0)
            inputs1 = [pool.pin(key) for key in ops[1].inputs]
            result1 = ops[1].process("fit_transform", {}, inputs1)
            self.assertIsInstance(result1, pd.DataFrame)

    def test_with_variable_kwarg(self):
        with csv_file(pd.DataFrame({"col": [1, 2]})) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv, sep=st.var("path"))
            ops = self._optimize_read(data)
            self.assertIsInstance(ops[-1], DataSourceOp)

    def test_with_plain_kwarg(self):
        with csv_file(pd.DataFrame({"a": [1, 2]}), sep=";") as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv, sep=";")
            ops = self._optimize_read(data)
            self.assertIsInstance(ops[-1], DataSourceOp)
            self.assertEqual(";", ops[-1].read_kwargs.get("sep"))

    def test_with_dataop_kwarg(self):
        with csv_file(pd.DataFrame({"a": [1, 2]}), sep=";") as path:
            data = st.as_data_op(path).skb.apply_func(
                pd.read_csv, sep=st.as_data_op(";"))
            ops = self._optimize_read(data)
            self.assertIsInstance(ops[-1], DataSourceOp)
            self.assertEqual(";", ops[-1].read_kwargs.get("sep"))

    def test_with_plain_positional_arg(self):
        call_op = CallOp(func=pd.read_csv,
                         args=(OperandRef(0), ","), kwargs={})
        call_op.inputs = [ValueOp("dummy.csv")]
        new_op = make_read_op(call_op)
        self.assertIsInstance(new_op, DataSourceOp)
        self.assertEqual((",",), tuple(new_op.read_args))


class TestMakeDatetimeConversionOp(unittest.TestCase):
    def test_extra_positional_args(self):
        op = CallOp(func=pd.to_datetime,
                    args=(OperandRef(0), "ISO8601"), kwargs={})
        new_op = make_datetime_conversion_op(op)
        self.assertEqual(("ISO8601",), tuple(new_op.args))


class TestJoinOpPandas(unittest.TestCase):
    """`JoinOp.process` on the pandas backend."""

    def test_merge_on_key(self):
        left = pd.DataFrame({"k": [1, 2, 3], "a": [10, 20, 30]})
        right = pd.DataFrame({"k": [2, 3, 4], "b": [200, 300, 400]})
        op = JoinOp(how="inner", left_on="k", right_on="k")
        result = run_op(op, left, right)
        self.assertEqual([2, 3], result["k"].tolist())
        self.assertEqual([20, 30], result["a"].tolist())
        self.assertEqual([200, 300], result["b"].tolist())

    def test_merge_left_on_right_on_distinct(self):
        left = pd.DataFrame({"lk": [1, 2], "a": [10, 20]})
        right = pd.DataFrame({"rk": [2, 3], "b": [200, 300]})
        op = JoinOp(how="inner", left_on="lk", right_on="rk")
        result = run_op(op, left, right)
        self.assertEqual([2], result["lk"].tolist())
        self.assertEqual([2], result["rk"].tolist())

    def test_join_index_based_with_suffixes(self):
        left = pd.DataFrame({"x": [1, 2, 3]}, index=["a", "b", "c"])
        right = pd.DataFrame({"x": [10, 20, 30]}, index=["b", "c", "d"])
        op = JoinOp(how="left", left_index=True, right_index=True,
                    suffixes=("_L", "_R"))
        result = run_op(op, left, right)
        self.assertEqual(["a", "b", "c"], result.index.tolist())
        self.assertIn("x_L", result.columns)
        self.assertIn("x_R", result.columns)

    def test_outer_join(self):
        left = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
        right = pd.DataFrame({"b": [10, 20]}, index=["y", "z"])
        op = JoinOp(how="outer", left_index=True, right_index=True)
        result = run_op(op, left, right)
        self.assertEqual({"x", "y", "z"}, set(result.index.tolist()))

    def test_wrong_input_count_raises(self):
        op = JoinOp(how="inner", left_on="k", right_on="k")
        with self.assertRaises(ValueError):
            run_op(op, pd.DataFrame({"k": [1]}))

    def test_polars_not_implemented(self):
        with force_polars():
            op = JoinOp(how="inner", left_on="k", right_on="k")
            with self.assertRaises(NotImplementedError):
                run_op(op, pl.DataFrame({"k": [1]}), pl.DataFrame({"k": [1]}))


class TestJoinRewrites(unittest.TestCase):
    """End-to-end: skrub DataOp expressions get rewritten to JoinOp(s)."""

    def _run_plan(self, ops):
        """Execute a linearized DAG and return the last op's output."""
        pool = BufferPool()
        for op in ops:
            inputs = [pool.pin(key) for key in op.inputs]
            pool.put(op, op.process("fit_transform", {}, inputs))
        return pool.pin(ops[-1])

    def test_merge_on_key_rewrites_and_executes(self):
        df1 = pd.DataFrame({"k": [1, 2, 3], "a": [10, 20, 30]})
        df2 = pd.DataFrame({"k": [2, 3, 4], "b": [200, 300, 400]})
        data = st.as_data_op(df1).merge(st.as_data_op(df2), on="k")
        ops = optimize(data, OptConfig(dataframe_ops=True))

        join_ops = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(1, len(join_ops))
        self.assertEqual("k", join_ops[0].left_on)
        self.assertEqual("k", join_ops[0].right_on)
        self.assertEqual("inner", join_ops[0].how)

        result = self._run_plan(ops)
        expected = df1.merge(df2, on="k")
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True))

    def test_merge_left_on_right_on_preserved(self):
        df1 = pd.DataFrame({"lk": [1, 2], "a": [10, 20]})
        df2 = pd.DataFrame({"rk": [2, 3], "b": [200, 300]})
        data = st.as_data_op(df1).merge(
            st.as_data_op(df2), left_on="lk", right_on="rk", how="outer")
        ops = optimize(data, OptConfig(dataframe_ops=True))

        join_ops = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(1, len(join_ops))
        self.assertEqual("lk", join_ops[0].left_on)
        self.assertEqual("rk", join_ops[0].right_on)
        self.assertEqual("outer", join_ops[0].how)

    def test_merge_sort_true_raises(self):
        df1 = pd.DataFrame({"k": [1, 2]})
        df2 = pd.DataFrame({"k": [1, 2]})
        data = st.as_data_op(df1).merge(
            st.as_data_op(df2), on="k", sort=True)
        with self.assertRaises(NotImplementedError):
            optimize(data, OptConfig(dataframe_ops=True))

    def test_merge_sort_false_is_accepted(self):
        df1 = pd.DataFrame({"k": [1, 2], "a": [10, 20]})
        df2 = pd.DataFrame({"k": [1, 2], "b": [100, 200]})
        data = st.as_data_op(df1).merge(
            st.as_data_op(df2), on="k", sort=False)
        ops = optimize(data, OptConfig(dataframe_ops=True))
        join_ops = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(1, len(join_ops))

    def test_join_no_args_defaults_to_index_based_left(self):
        df1 = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
        df2 = pd.DataFrame({"b": [10, 20]}, index=["x", "y"])
        data = st.as_data_op(df1).join(st.as_data_op(df2))
        ops = optimize(data, OptConfig(dataframe_ops=True))

        join_ops = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(1, len(join_ops))
        self.assertEqual("left", join_ops[0].how)
        self.assertTrue(join_ops[0].left_index)
        self.assertTrue(join_ops[0].right_index)

    def test_join_with_on_uses_left_on_and_right_index(self):
        df1 = pd.DataFrame({"k": ["x", "y"], "a": [1, 2]})
        df2 = pd.DataFrame({"b": [10, 20]}, index=["x", "y"])
        data = st.as_data_op(df1).join(st.as_data_op(df2), on="k")
        ops = optimize(data, OptConfig(dataframe_ops=True))

        join_ops = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(1, len(join_ops))
        self.assertEqual("k", join_ops[0].left_on)
        self.assertFalse(join_ops[0].left_index)
        self.assertTrue(join_ops[0].right_index)

    def test_join_with_suffixes_rewrites_and_executes(self):
        df1 = pd.DataFrame({"x": [1, 2, 3]}, index=["a", "b", "c"])
        df2 = pd.DataFrame({"x": [10, 20, 30]}, index=["b", "c", "d"])
        data = st.as_data_op(df1).join(
            st.as_data_op(df2), lsuffix="_L", rsuffix="_R")
        ops = optimize(data, OptConfig(dataframe_ops=True))

        join_ops = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(1, len(join_ops))
        self.assertTrue(join_ops[0].left_index)
        self.assertTrue(join_ops[0].right_index)
        self.assertEqual(("_L", "_R"), join_ops[0].suffixes)

        result = self._run_plan(ops)
        expected = df1.join(df2, lsuffix="_L", rsuffix="_R")
        pd.testing.assert_frame_equal(result, expected)

    def test_merge_overlapping_non_key_columns_uses_pandas_default_suffixes(self):
        df1 = pd.DataFrame({"k": [1, 2], "v": [10, 20]})
        df2 = pd.DataFrame({"k": [1, 2], "v": [100, 200]})
        data = st.as_data_op(df1).merge(st.as_data_op(df2), on="k")
        ops = optimize(data, OptConfig(dataframe_ops=True))

        join_ops = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(1, len(join_ops))
        self.assertEqual(("_x", "_y"), join_ops[0].suffixes)

        result = self._run_plan(ops)
        expected = df1.merge(df2, on="k")
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True))
        self.assertIn("v_x", result.columns)
        self.assertIn("v_y", result.columns)

    def test_join_overlapping_columns_without_suffixes_raises(self):
        # Pandas .join() defaults both lsuffix and rsuffix to "", so overlapping
        # columns raise ValueError. JoinOp must reproduce that — not silently
        # invent suffixes like "_left"/"_right".
        df1 = pd.DataFrame({"x": [1, 2]}, index=["a", "b"])
        df2 = pd.DataFrame({"x": [10, 20]}, index=["a", "b"])
        with self.assertRaisesRegex(Exception, "columns overlap"):
            data = st.as_data_op(df1).join(st.as_data_op(df2))
            optimize(data, OptConfig(dataframe_ops=True))

    def test_join_overlapping_columns_with_suffixes_succeeds(self):
        # Sibling to the above: with suffixes provided, the same join works.
        df1 = pd.DataFrame({"x": [1, 2]}, index=["a", "b"])
        df2 = pd.DataFrame({"x": [10, 20]}, index=["a", "b"])
        data = st.as_data_op(df1).join(
            st.as_data_op(df2), lsuffix="_L", rsuffix="_R")
        ops = optimize(data, OptConfig(dataframe_ops=True))

        join_ops = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(("_L", "_R"), join_ops[0].suffixes)

        result = self._run_plan(ops)
        expected = df1.join(df2, lsuffix="_L", rsuffix="_R")
        pd.testing.assert_frame_equal(result, expected)

    def test_chained_join_decomposes_into_binary_chain(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])
        df2 = pd.DataFrame({"b": [10, 20, 30]}, index=["x", "y", "z"])
        df3 = pd.DataFrame({"c": [100, 200, 300]}, index=["x", "y", "z"])
        data = st.as_data_op(df1).join(
            [st.as_data_op(df2), st.as_data_op(df3)])
        ops = optimize(data, OptConfig(dataframe_ops=True))

        join_ops = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(2, len(join_ops))
        # Both chain links are index-based.
        for j in join_ops:
            self.assertTrue(j.left_index)
            self.assertTrue(j.right_index)
        # Second JoinOp's left input is the first JoinOp.
        self.assertIs(join_ops[0], join_ops[1].inputs[0])

        result = self._run_plan(ops)
        expected = df1.join([df2, df3])
        pd.testing.assert_frame_equal(result, expected)

    def test_chained_join_with_duplicate_inputs_raises_error(self):
        df1 = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])
        df2 = pd.DataFrame(index=["x", "y", "z"])

        df2_op = st.as_data_op(df2)
        data = st.as_data_op(df1).join([df2_op, df2_op])
        with self.assertRaisesRegex(ValueError, "Duplicate right-hand frames in chained joins are not supported"):
            optimize(data, OptConfig(dataframe_ops=True))

    def test_join_with_other_kwarg(self):
        df1 = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
        df2 = pd.DataFrame({"b": [10, 20]}, index=["x", "y"])
        data = st.as_data_op(df1).join(other=st.as_data_op(df2))
        ops = optimize(data, OptConfig(dataframe_ops=True))

        join_ops = [o for o in ops if isinstance(o, JoinOp)]
        self.assertEqual(1, len(join_ops))

        result = self._run_plan(ops)
        expected = df1.join(df2)
        pd.testing.assert_frame_equal(result, expected)

    def test_merge_unsupported_arguments_raises(self):
        df1 = pd.DataFrame({"k": [1, 2]})
        df2 = pd.DataFrame({"k": [1, 2]})
        data = st.as_data_op(df1).merge(
            st.as_data_op(df2), on="k", indicator=True
        )
        with self.assertRaisesRegex(NotImplementedError, "Unsupported arguments for merge"):
            optimize(data, OptConfig(dataframe_ops=True))

    def test_join_unsupported_arguments_raises(self):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [1, 2]})
        data = st.as_data_op(df1).join(
            st.as_data_op(df2), validate="one_to_one"
        )
        with self.assertRaisesRegex(NotImplementedError, "Unsupported arguments for join"):
            optimize(data, OptConfig(dataframe_ops=True))


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
        self.assertIn("AggregateOp", str(op))
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

    def _run_plan(self, ops, env=None):
        pool = BufferPool()
        for op in ops:
            inputs = [pool.pin(key) for key in op.inputs]
            pool.put(op, op.process("fit_transform", env or {}, inputs))
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
        ops = optimize(data, OptConfig(dataframe_ops=True))
        agg_ops = [o for o in ops if isinstance(o, AggregateOp)]
        self.assertEqual(1, len(agg_ops))
        self.assertEqual(OperandRef(1), agg_ops[0].grouping_attributes)
        result = self._run_plan(ops, env={"key": "g"})
        pd.testing.assert_frame_equal(result, self.df.groupby("g").agg("sum"))

    def test_variable_aggregation_spec_uses_placeholder(self):
        data = st.as_data_op(self.df).groupby("g").agg(st.var("spec"))
        ops = optimize(data, OptConfig(dataframe_ops=True))
        agg_ops = [o for o in ops if isinstance(o, AggregateOp)]
        self.assertEqual(1, len(agg_ops))
        self.assertEqual(OperandRef(1), agg_ops[0].aggregations)
        result = self._run_plan(ops, env={"spec": "sum"})
        pd.testing.assert_frame_equal(result, self.df.groupby("g").agg("sum"))

    def test_both_grouping_key_and_agg_spec_are_variables(self):
        # Both operands are graph-fed; the aggregation OperandRef must be shifted
        # by the number of extra groupby inputs to avoid aliasing the key slot.
        data = st.as_data_op(self.df).groupby(st.var("key")).agg(st.var("spec"))
        ops = optimize(data, OptConfig(dataframe_ops=True))
        agg_ops = [o for o in ops if isinstance(o, AggregateOp)]
        self.assertEqual(1, len(agg_ops))
        self.assertEqual(OperandRef(1), agg_ops[0].grouping_attributes)
        self.assertEqual(OperandRef(2), agg_ops[0].aggregations)
        result = self._run_plan(ops, env={"key": "g", "spec": "sum"})
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


