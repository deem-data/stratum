import os
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import polars as pl
import stratum as st
from skrub._data_ops._data_ops import DataOp
from stratum._config import FLAGS
from stratum.optimizer.ir._dataframe_ops import (
    ApplyUDFOp, AssignOp, ConcatOp, DataSourceOp, DatetimeConversionOp,
    DropOp, GetAttrProjectionOp, GroupedDataframeOp, MetadataOp, ProjectionOp,
    SplitOp)
from stratum.optimizer.ir._ops import DATA_OP_PLACEHOLDER, GetItemOp, MethodCallOp, Op
from stratum.optimizer._optimize import OptConfig, optimize as optimize_
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
    """Extract intermediate values from op.inputs."""
    return [in_op.intermediate for in_op in op.inputs]


class TestDataframeOps(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "datetime": [
                "2025-11-01 10:00:00",
                "2025-11-02 15:30:00",
                "2025-11-03 09:45:00"
            ]
        })

    def test_data_source_rewrite_df(self):
        data = st.as_data_op(self.df)
        ops = optimize(data)
        assert len(ops) == 1
        assert isinstance(ops[0], DataSourceOp)

    def test_data_source_rewrite_read(self):
        tmp_file = tempfile.mktemp(suffix=".csv")
        self.df.to_csv(tmp_file, index=False)
        data = st.as_data_op(tmp_file).skb.apply_func(pd.read_csv)
        ops = optimize(data, OptConfig(dataframe_ops=True))
        assert len(ops) == 1
        assert isinstance(ops[0], DataSourceOp)
        os.remove(tmp_file)

    def test_projection_rewrite_df(self):
        data = st.as_data_op(self.df).drop("y", axis=1)
        ops = optimize(data)
        assert len(ops) == 2
        assert isinstance(ops[1], ProjectionOp)

    @unittest.skip("Skipping this test for now")
    def test_projection_fused_get_item_rewrite_df1(self):
        data = st.as_data_op(self.df)["x"].apply(lambda x: x + 1)
        ops = optimize(data)
        assert len(ops) == 2
        assert isinstance(ops[1], ProjectionOp)

    def test_projection_fused_get_item_rewrite_df2(self):
        data = st.as_data_op(self.df)["x"]
        sub_dag1 = data.apply(lambda x, a: x + a, a=st.as_data_op(1))
        sub_dag2 = data
        root = st.choose_from([sub_dag1, sub_dag2]).as_data_op()
        ops = optimize(root)
        self.assertEqual(5, len(ops))
        self.assertTrue(isinstance(ops[1], GetItemOp))
        self.assertTrue(isinstance(ops[3], ProjectionOp))

    def test_fused_get_attr_rewrite_df(self):
        data = st.as_data_op(self.df)[["datetime"]].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S')
        data = data.assign(year=data["datetime"].dt.year, month= data["datetime"].dt.month)
        data = data.copy()
        ops = optimize(data)
        self.assertEqual(8,len(ops))
        op_iter = iter(ops[3:])
        next(op_iter)
        self.assertTrue(isinstance(next(op_iter), GetAttrProjectionOp))
        self.assertTrue(isinstance(next(op_iter), GetAttrProjectionOp))
        self.assertTrue(isinstance(next(op_iter), AssignOp))
        self.assertTrue(isinstance(next(op_iter), MethodCallOp))


class TestDataSourceOpPolars(unittest.TestCase):
    def setUp(self):
        self.orig = FLAGS.force_polars
        FLAGS.force_polars = True

    def tearDown(self):
        FLAGS.force_polars = self.orig

    def test_process_data_polars(self):
        df = pd.DataFrame({"a": [1, 2]})
        op = DataSourceOp(data=df)
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertIsInstance(result, pl.DataFrame)

    def test_process_read_csv_polars(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        pd.DataFrame({"a": [1, 2]}).to_csv(tmp, index=False)
        tmp.close()
        try:
            op = DataSourceOp(file_path=tmp.name, _format="csv", read_args=(), read_kwargs={})
            result = op.process("fit_transform", {}, _inputs_for(op))
            self.assertIsInstance(result, pl.DataFrame)
        finally:
            os.remove(tmp.name)


class TestMetadataOpPolars(unittest.TestCase):
    def setUp(self):
        self.orig = FLAGS.force_polars
        FLAGS.force_polars = True

    def tearDown(self):
        FLAGS.force_polars = self.orig

    def test_process_rename_polars(self):
        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        op = MetadataOp(func="rename", args=(), kwargs={"columns": {"a": "x"}})
        op.inputs = [_inp(df)]
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertIn("x", result.columns)


class TestProjectionOp(unittest.TestCase):
    def test_process_non_method(self):
        op = ProjectionOp(func=lambda df, v: df * v, args=(DATA_OP_PLACEHOLDER, 2), kwargs={})
        op.inputs = [_inp(pd.DataFrame({"a": [1, 2]}))]
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertEqual(result["a"].tolist(), [2, 4])

    def test_process_polars_method_raises(self):
        orig = FLAGS.force_polars
        FLAGS.force_polars = True
        try:
            op = ProjectionOp(method="drop", args=(), kwargs={})
            op.inputs = [_inp(pl.DataFrame({"a": [1]}))]
            with self.assertRaises(ValueError):
                op.process("fit_transform", {}, _inputs_for(op))
        finally:
            FLAGS.force_polars = orig


class TestDropOpPolars(unittest.TestCase):
    def setUp(self):
        self.orig = FLAGS.force_polars
        FLAGS.force_polars = True

    def tearDown(self):
        FLAGS.force_polars = self.orig

    def test_drop_with_columns_kwarg(self):
        df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
        op = DropOp(args=(), kwargs={"columns": ["b"]})
        op.inputs = [_inp(df)]
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertNotIn("b", result.columns)


class TestApplyUDFOp(unittest.TestCase):
    def test_single_column_str(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        op = ApplyUDFOp(args=(lambda x: x * 10,), kwargs={}, columns="a")
        op.inputs = [_inp(df)]
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertEqual(result.tolist(), [10, 20])

    def test_multi_column(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        op = ApplyUDFOp(args=(lambda x: x * 2,), kwargs={}, columns=["a", "b"])
        op.inputs = [_inp(df)]
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertEqual(result["a"].tolist(), [2, 4])

    def test_polars_sin_rewrite(self):
        orig = FLAGS.force_polars
        FLAGS.force_polars = True
        try:
            series = pl.Series("a", [0.0, np.pi / 2])
            op = ApplyUDFOp(args=(np.sin,), kwargs={})
            op.inputs = [_inp(series)]
            result = op.process("fit_transform", {}, _inputs_for(op))
            self.assertAlmostEqual(result[1], 1.0, places=5)
        finally:
            FLAGS.force_polars = orig

    def test_polars_cos_rewrite(self):
        orig = FLAGS.force_polars
        FLAGS.force_polars = True
        try:
            series = pl.Series("a", [0.0])
            op = ApplyUDFOp(args=(np.cos,), kwargs={})
            op.inputs = [_inp(series)]
            result = op.process("fit_transform", {}, _inputs_for(op))
            self.assertAlmostEqual(result[0], 1.0, places=5)
        finally:
            FLAGS.force_polars = orig

    def test_polars_multi_col_map_rows(self):
        orig = FLAGS.force_polars
        FLAGS.force_polars = True
        try:
            df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
            op = ApplyUDFOp(args=(lambda row: (row[0] + row[1],),), kwargs={}, columns=["a", "b"])
            op.inputs = [_inp(df)]
            result = op.process("fit_transform", {}, _inputs_for(op))
            self.assertIsNotNone(result)
        finally:
            FLAGS.force_polars = orig


class TestAssignOpPolars(unittest.TestCase):
    def setUp(self):
        self.orig = FLAGS.force_polars
        FLAGS.force_polars = True

    def tearDown(self):
        FLAGS.force_polars = self.orig

    def test_assign_polars(self):
        df = pl.DataFrame({"a": [1, 2]})
        op = AssignOp(args=(), kwargs={"b": pl.Series([10, 20])})
        op.inputs = [_inp(df)]
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertIn("b", result.columns)

    def test_assign_polars_pandas_conversion(self):
        df = pl.DataFrame({"a": [1, 2]})
        op = AssignOp(args=(), kwargs={"b": pd.Series([10, 20])})
        op.inputs = [_inp(df)]
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertIn("b", result.columns)

    def test_assign_polars_placeholder_raises(self):
        df = pl.DataFrame({"a": [1, 2]})
        op = AssignOp(args=(), kwargs={"b": DATA_OP_PLACEHOLDER})
        op.inputs = [_inp(df), _inp(DATA_OP_PLACEHOLDER)]
        with self.assertRaises(NotImplementedError):
            op.process("fit_transform", {}, _inputs_for(op))


class TestDatetimeConversionOpPolars(unittest.TestCase):
    def test_polars_path(self):
        orig = FLAGS.force_polars
        FLAGS.force_polars = True
        try:
            s = pl.Series("dt", ["2025-01-01", "2025-06-15"])
            op = DatetimeConversionOp(args=(), kwargs={})
            op.inputs = [_inp(s)]
            result = op.process("fit_transform", {}, _inputs_for(op))
            self.assertEqual(result.dtype, pl.Datetime)
        finally:
            FLAGS.force_polars = orig


class TestGetAttrProjectionOp(unittest.TestCase):
    def test_init_none(self):
        op = GetAttrProjectionOp(attr_name=None)
        self.assertEqual(op.attr_name, [])

    def test_init_str(self):
        op = GetAttrProjectionOp(attr_name="dt")
        self.assertEqual(op.attr_name, ["dt"])

    def test_polars_process(self):
        orig = FLAGS.force_polars
        FLAGS.force_polars = True
        try:
            s = pl.Series("dt", pd.to_datetime(["2025-01-15", "2025-06-20"]))
            op = GetAttrProjectionOp(attr_name=["dt", "year"], inputs=[_inp(s)], outputs=[])
            result = op.process("fit_transform", {}, _inputs_for(op))
            self.assertEqual(result.to_list(), [2025, 2025])
        finally:
            FLAGS.force_polars = orig

    def test_polars_dayofweek(self):
        orig = FLAGS.force_polars
        FLAGS.force_polars = True
        try:
            s = pl.Series("dt", pd.to_datetime(["2025-01-06"]))  # Monday
            op = GetAttrProjectionOp(attr_name=["dt", "dayofweek"], inputs=[_inp(s)], outputs=[])
            result = op.process("fit_transform", {}, _inputs_for(op))
            self.assertEqual(result.to_list(), [1])  # polars: Monday=1
        finally:
            FLAGS.force_polars = orig

    def test_polars_is_month_end(self):
        orig = FLAGS.force_polars
        FLAGS.force_polars = True
        try:
            s = pl.Series("dt", pd.to_datetime(["2025-01-31", "2025-01-15"]))
            op = GetAttrProjectionOp(attr_name=["dt", "is_month_end"], inputs=[_inp(s)], outputs=[])
            result = op.process("fit_transform", {}, _inputs_for(op))
            self.assertEqual(result.to_list(), [True, False])
        finally:
            FLAGS.force_polars = orig


class TestGroupedDataframeOp(unittest.TestCase):
    def test_process(self):
        op = GroupedDataframeOp(ops=[Op(), Op()])
        with self.assertRaises(NotImplementedError):
            op.process("fit_transform", {}, _inputs_for(op))


class TestConcatOpPolars(unittest.TestCase):
    def test_polars_concat(self):
        orig = FLAGS.force_polars
        FLAGS.force_polars = True
        try:
            df1 = pl.DataFrame({"a": [1, 2]})
            df2 = pl.DataFrame({"a": [3, 4]})
            mock_dataop1 = MagicMock(spec=DataOp)
            mock_dataop2 = MagicMock(spec=DataOp)
            op = ConcatOp(first=mock_dataop1, others=[mock_dataop2], axis=0)
            op.inputs = [_inp(df1), _inp(df2)]
            result = op.process("fit_transform", {}, _inputs_for(op))
            self.assertEqual(len(result), 4)
        finally:
            FLAGS.force_polars = orig


class TestSplitOp(unittest.TestCase):
    def test_polars(self):
        x = pl.DataFrame({"a": [10, 20, 30]})
        y = pl.DataFrame({"b": [1, 2, 3]})
        op = SplitOp(inputs=[_inp(x), _inp(y)])
        op.indices = [0, 2]
        result = op.process("fit_transform", {}, _inputs_for(op))
        self.assertEqual(len(result[0]), 2)

    def test_unsupported_type(self):
        op = SplitOp(inputs=[_inp("not_a_df"), _inp("not_a_df")])
        op.indices = [0]
        with self.assertRaises(ValueError):
            op.process("fit_transform", {}, _inputs_for(op))



class TestMakeReadOpWithVariable(unittest.TestCase):
    def test_read_op_with_variable_input(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        pd.DataFrame({"col": [1, 2]}).to_csv(tmp, index=False)
        tmp.close()
        try:
            var = st.var("path")
            data = var.skb.apply_func(pd.read_csv)
            with st.config(fast_dataops_convert=True):
                ops = optimize(data, OptConfig(dataframe_ops=True))
            self.assertIsInstance(ops[-1], DataSourceOp)
            # Verify it can actually process using resolve_inputs
            pool = BufferPool()
            inputs0 = [pool.pin(key) for key in ops[0].inputs]
            result0 = ops[0].process("fit_transform", {"path": tmp.name}, inputs0)
            pool.put(ops[0], result0)
            inputs1 = [pool.pin(key) for key in ops[1].inputs]
            result1 = ops[1].process("fit_transform", {}, inputs1)
            self.assertIsInstance(result1, pd.DataFrame)
        finally:
            os.remove(tmp.name)

    def test_read_op_with_variable_kwarg(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        pd.DataFrame({"col": [1, 2]}).to_csv(tmp, index=False)
        tmp.close()
        try:
            path = st.var("path")
            data = st.as_data_op(tmp.name).skb.apply_func(pd.read_csv, sep=path)
            with st.config(fast_dataops_convert=True):
                ops = optimize(data, OptConfig(dataframe_ops=True))
            self.assertIsInstance(ops[-1], DataSourceOp)
        finally:
            os.remove(tmp.name)
