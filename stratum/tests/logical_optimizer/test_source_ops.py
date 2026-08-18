import unittest

import numpy as np
import pandas as pd
import stratum as st
from stratum.optimizer._optimize import OptConfig
from stratum.optimizer.ir._source_ops import DataSourceOp, make_read_op
from stratum.optimizer.ir._ops import CallOp, OperandRef, ValueOp, VariableOp
from stratum.optimizer.physical._source_execs import (
    InMemoryFrame, NumpyLoad, ReadCSV, ReadParquet)
from stratum.runtime._buffer_pool import BufferPool
from stratum.tests._helpers import csv_file, npy_file, parquet_file
from stratum.tests.logical_optimizer.test_dataframe_ops import optimize


class TestDataSourceRewrites(unittest.TestCase):
    """`optimize` lowers a directly-passed frame / a read call into a physical source op.

    The logical DataSourceOp produced by extraction is lowered (and its impl
    selected) so the plan carries a concrete backend-specific source, not a
    DataSourceOp. Backend impls all subclass the abstract source op, so the
    assertions below are backend-agnostic (they check the abstract base type)."""

    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    def test_data_source_from_dataframe(self):
        ops = optimize(st.as_data_op(self.df))
        self.assertEqual(1, len(ops))
        self.assertIsInstance(ops[0], InMemoryFrame)

    def test_data_source_from_read_csv(self):
        with csv_file(self.df) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv)
            ops = optimize(data, OptConfig(dataframe_ops=True))
        self.assertEqual(1, len(ops))
        self.assertIsInstance(ops[0], ReadCSV)

    def test_data_source_from_np_load(self):
        with npy_file(np.array([1, 2, 3])) as path:
            data = st.as_data_op(path).skb.apply_func(np.load)
            ops = optimize(data, OptConfig(dataframe_ops=True))
        self.assertTrue(any(isinstance(op, NumpyLoad) for op in ops))

    def test_data_source_from_read_parquet(self):
        with parquet_file(self.df) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_parquet)
            ops = optimize(data, OptConfig(dataframe_ops=True))
        self.assertTrue(any(isinstance(op, ReadParquet) for op in ops))


class TestMakeReadOp(unittest.TestCase):
    """`make_read_op` and its end-to-end usage via the optimizer."""

    def _optimize_read(self, data, env=None):
        return optimize(data, OptConfig(dataframe_ops=True), env=env)

    def test_with_variable_input(self):
        with csv_file(pd.DataFrame({"col": [1, 2]})) as path:
            data = st.var("path").skb.apply_func(pd.read_csv)
            # Resolve the path variable at compile time so the plan runs with no env.
            ops = self._optimize_read(data, env={"path": path})
            self.assertIsInstance(ops[-1], ReadCSV)

            # Verify the resulting plan actually runs without a runtime environment.
            pool = BufferPool()
            for op in ops:
                inputs = [pool.pin(key) for key in op.inputs]
                pool.put(op, op.process("fit_transform", inputs))
            self.assertIsInstance(pool.pin(ops[-1]), pd.DataFrame)

    def test_with_variable_kwarg(self):
        with csv_file(pd.DataFrame({"col": [1, 2]})) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv, sep=st.var("path"))
            ops = self._optimize_read(data)
            self.assertIsInstance(ops[-1], ReadCSV)

    def test_with_plain_kwarg(self):
        with csv_file(pd.DataFrame({"a": [1, 2]}), sep=";") as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv, sep=";")
            ops = self._optimize_read(data)
            self.assertIsInstance(ops[-1], ReadCSV)
            self.assertEqual(";", ops[-1].read_kwargs.get("sep"))

    def test_with_dataop_kwarg(self):
        with csv_file(pd.DataFrame({"a": [1, 2]}), sep=";") as path:
            data = st.as_data_op(path).skb.apply_func(
                pd.read_csv, sep=st.as_data_op(";"))
            ops = self._optimize_read(data)
            self.assertIsInstance(ops[-1], ReadCSV)
            self.assertEqual(";", ops[-1].read_kwargs.get("sep"))

    def test_with_plain_positional_arg(self):
        call_op = CallOp(func=pd.read_csv,
                         args=(OperandRef(0), ","), kwargs={})
        call_op.inputs = [ValueOp("dummy.csv")]
        new_op = make_read_op(call_op)
        self.assertIsInstance(new_op, DataSourceOp)
        self.assertEqual((",",), tuple(new_op.read_args))


class TestDeferredReadRewrites(unittest.TestCase):
    """`skrub.deferred(pd.read_csv)(path)` is recognised as a read source.

    ``X.skb.apply_func(f)`` is just ``skrub.deferred(f)(X)``, so the deferred
    spelling produces the same ``Call``. The difference is the path: written this
    way it is usually a plain literal rather than a DataOp, so the call reaches
    extraction with *no* operands at all -- which used to leave it a ``CallOp``.
    """

    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    def _optimize(self, data, env=None):
        return optimize(data, OptConfig(dataframe_ops=True), env=env)

    def _run(self, ops, env=None):
        """Execute a linearized plan, feeding any VariableOp from `env`."""
        pool = BufferPool()
        for op in ops:
            inputs = [pool.pin(key) for key in op.inputs]
            value = (env[op.name] if isinstance(op, VariableOp)
                     else op.process("fit_transform", inputs))
            pool.put(op, value)
        return pool.pin(ops[-1])

    def test_deferred_read_csv_with_literal_path(self):
        with csv_file(self.df) as path:
            ops = self._optimize(st.deferred(pd.read_csv)(path))
            self.assertEqual(1, len(ops))
            self.assertIsInstance(ops[0], ReadCSV)
            self.assertEqual(path, ops[0].file_path)
            # The rewritten source is self-contained: no operands to feed.
            self.assertEqual([], ops[0].inputs)
            pd.testing.assert_frame_equal(self.df, self._run(ops))

    def test_deferred_read_parquet_with_literal_path(self):
        with parquet_file(self.df) as path:
            ops = self._optimize(st.deferred(pd.read_parquet)(path))
            self.assertTrue(any(isinstance(op, ReadParquet) for op in ops))

    def test_deferred_np_load_with_literal_path(self):
        with npy_file(np.array([1, 2, 3])) as path:
            ops = self._optimize(st.deferred(np.load)(path))
            self.assertTrue(any(isinstance(op, NumpyLoad) for op in ops))
            np.testing.assert_array_equal(np.array([1, 2, 3]), self._run(ops))

    def test_deferred_read_csv_with_extra_kwarg(self):
        with csv_file(self.df, sep=";") as path:
            ops = self._optimize(st.deferred(pd.read_csv)(path, sep=";"))
            self.assertIsInstance(ops[-1], ReadCSV)
            self.assertEqual(";", ops[-1].read_kwargs.get("sep"))
            pd.testing.assert_frame_equal(self.df, self._run(ops))

    def test_deferred_read_csv_with_data_op_path(self):
        # The path as a DataOp: an operand that inlines to a constant, i.e. the
        # exact shape `apply_func` produces.
        with csv_file(self.df) as path:
            ops = self._optimize(st.deferred(pd.read_csv)(st.as_data_op(path)))
            self.assertEqual(1, len(ops))
            self.assertIsInstance(ops[0], ReadCSV)

    def test_deferred_read_csv_with_variable_path(self):
        # A graph-fed path stays an operand, so the plan keeps the VariableOp and
        # the read resolves it at runtime.
        with csv_file(self.df) as path:
            ops = self._optimize(st.deferred(pd.read_csv)(st.var("p")))
            self.assertIsInstance(ops[-1], ReadCSV)
            self.assertIsInstance(ops[-1].file_path, OperandRef)
            pd.testing.assert_frame_equal(self.df, self._run(ops, env={"p": path}))

    def test_deferred_read_feeds_downstream_ops(self):
        # The rewrite has to leave the source wired to its consumers.
        with csv_file(self.df) as path:
            ops = self._optimize(st.deferred(pd.read_csv)(path)[["x"]])
            self.assertIsInstance(ops[0], ReadCSV)
            self.assertEqual(2, len(ops))
            pd.testing.assert_frame_equal(self.df[["x"]], self._run(ops))

    def test_deferred_read_csv_with_keyword_path_stays_call(self):
        # The path keyword differs per reader (filepath_or_buffer / path / file),
        # so this form is not rewritten -- it must still work as a plain CallOp.
        with csv_file(self.df) as path:
            ops = self._optimize(st.deferred(pd.read_csv)(filepath_or_buffer=path))
            self.assertFalse(any(isinstance(op, ReadCSV) for op in ops))
            pd.testing.assert_frame_equal(self.df, self._run(ops))

    def test_deferred_non_read_func_is_not_a_source(self):
        ops = self._optimize(st.deferred(len)([1, 2, 3]))
        self.assertFalse(any(isinstance(op, (ReadCSV, ReadParquet, NumpyLoad))
                             for op in ops))


if __name__ == "__main__":
    unittest.main()
