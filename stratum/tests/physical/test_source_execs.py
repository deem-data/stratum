"""Physical source operators: lowering, implementation selection, and the
central guarantee that execution carries no operator-selection control flow.

The branch-free tests build a plan under one backend, then flip the global
``force_polars`` flag before executing. Because the backend was chosen at plan
time (baked into the concrete op class), the flip must have *no* effect on the
result -- if it did, some ``process`` would still be reading the flag at runtime.
"""
import unittest

import numpy as np
import pandas as pd
import polars as pl

import stratum as st
from stratum._config import config
from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.physical._impl_selection import FlagBasedSelector
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.physical._registry import get_default_physical_registry
from stratum.optimizer.ir._source_ops import DataSourceOp
from stratum.optimizer.physical._source_execs import (
    InMemoryFrame, NumpyLoad, PandasInMemoryFrame, PandasReadCSV,
    PandasReadParquet, PolarsInMemoryFrame, PolarsReadCSV, PolarsReadParquet,
    ReadCSV, ReadParquet, lower_data_source)
from stratum.runtime._buffer_pool import BufferPool
from stratum.tests._helpers import csv_file, npy_file, parquet_file
from stratum.tests.logical_optimizer.test_dataframe_ops import force_polars


def run_plan(ops, mode="fit_transform"):
    """Execute a linearized plan through a fresh BufferPool; return the sink value."""
    pool = BufferPool()
    for op in ops:
        inputs = [pool.pin(key) for key in op.inputs]
        pool.put(op, op.process(mode, inputs))
    return pool.pin(ops[-1])


class TestPlanContext(unittest.TestCase):
    def test_backend_from_flags(self):
        with force_polars(False):
            self.assertEqual("pandas", PlanContext.from_flags().backend)
        with force_polars(True):
            self.assertEqual("polars", PlanContext.from_flags().backend)


class TestSourceImplSelection(unittest.TestCase):
    """Lowering + selection pick the backend-specific concrete source class."""

    def setUp(self):
        self.df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    def _read_csv_ops(self):
        return csv_file(self.df)

    def test_default_selector_prefers_pandas_in_memory_frame(self):
        ops, *_ = optimize(st.as_data_op(self.df))
        self.assertIsInstance(ops[0], PandasInMemoryFrame)
        with force_polars(True):
            ops, *_ = optimize(st.as_data_op(self.df))
        self.assertIsInstance(ops[0], PandasInMemoryFrame)

    def test_default_selector_prefers_pandas_read_csv(self):
        with self._read_csv_ops() as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv)
            ops, *_ = optimize(data, OptConfig(dataframe_ops=True))
            self.assertIsInstance(ops[-1], PandasReadCSV)
            with force_polars(True):
                ops, *_ = optimize(data, OptConfig(dataframe_ops=True))
            self.assertIsInstance(ops[-1], PandasReadCSV)

    def test_default_selector_prefers_pandas_read_parquet(self):
        with parquet_file(self.df) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_parquet)
            ops, *_ = optimize(data, OptConfig(dataframe_ops=True))
            self.assertIsInstance(ops[-1], PandasReadParquet)
            with force_polars(True):
                ops, *_ = optimize(data, OptConfig(dataframe_ops=True))
            self.assertIsInstance(ops[-1], PandasReadParquet)

    def test_npy_is_single_impl(self):
        # np.load yields an ndarray; a single concrete impl serves both backends.
        with npy_file(np.array([1, 2, 3])) as path:
            data = st.as_data_op(path).skb.apply_func(np.load)
            ops, *_ = optimize(data, OptConfig(dataframe_ops=True))
            self.assertIsInstance(ops[-1], NumpyLoad)
            with force_polars(True):
                ops, *_ = optimize(data, OptConfig(dataframe_ops=True))
            self.assertIsInstance(ops[-1], NumpyLoad)

    def test_abstract_bases_are_abstract(self):
        for cls in (InMemoryFrame, ReadCSV, ReadParquet):
            self.assertTrue(cls.is_abstract, f"{cls.__name__} should be abstract")
        for cls in (PandasInMemoryFrame, PolarsInMemoryFrame, PandasReadCSV,
                    PolarsReadCSV, NumpyLoad):
            self.assertFalse(cls.is_abstract, f"{cls.__name__} should be concrete")


class TestSourcesInRegistry(unittest.TestCase):
    """The source impls are registered in the default PhysicalRegistry and
    selection resolves through it (no per-class backend maps)."""

    def setUp(self):
        self.registry = get_default_physical_registry()

    def test_source_candidates_are_registered(self):
        for abstract, impls in ((ReadCSV, {PandasReadCSV, PolarsReadCSV}),
                                (ReadParquet, {PandasReadParquet, PolarsReadParquet}),
                                (InMemoryFrame, {PandasInMemoryFrame, PolarsInMemoryFrame})):
            candidates = self.registry.candidates_for(abstract)
            self.assertEqual(impls, {c.impl_class for c in candidates})
            self.assertEqual({"pandas", "polars"},
                             {c.backend_name for c in candidates})

    def test_numpy_load_is_registered(self):
        candidates = self.registry.candidates_for(NumpyLoad)
        self.assertEqual(("numpy",), tuple(c.backend_name for c in candidates))

    def test_flag_selector_picks_backend_match(self):
        candidates = list(self.registry.candidates_for(ReadCSV))
        selector = FlagBasedSelector()
        op = ReadCSV(file_path="x.csv")
        with force_polars(False):
            chosen = selector.choose(op, candidates, PlanContext.from_flags())
        self.assertIs(PandasReadCSV, chosen.impl_class)
        with force_polars(True):
            chosen = selector.choose(op, candidates, PlanContext.from_flags())
        self.assertIs(PolarsReadCSV, chosen.impl_class)

    def test_flag_selector_falls_back_to_backend_agnostic(self):
        # NumpyLoad's only candidate is backend "numpy"; it is chosen under
        # either frame backend.
        candidates = list(self.registry.candidates_for(NumpyLoad))
        selector = FlagBasedSelector()
        op = NumpyLoad(file_path="x.npy")
        for polars in (False, True):
            with force_polars(polars):
                chosen = selector.choose(op, candidates, PlanContext.from_flags())
            self.assertIs(NumpyLoad, chosen.impl_class)


class TestConcreteSourceProcess(unittest.TestCase):
    """Direct process() coverage per concrete source op (moved here from the
    logical TestDataSourceOp when DataSourceOp lost its process method)."""

    def test_numpy_read(self):
        with npy_file(np.array([1, 2, 3])) as path:
            op = NumpyLoad(file_path=path, read_args=(), read_kwargs={})
            result = op.process("fit_transform", [])
            np.testing.assert_array_equal(result, [1, 2, 3])

    def test_pandas_from_dataframe(self):
        df = pd.DataFrame({"a": [1, 2]})
        op = PandasInMemoryFrame(data=df)
        self.assertIs(df, op.process("fit_transform", []))

    def test_polars_from_dataframe(self):
        op = PolarsInMemoryFrame(data=pd.DataFrame({"a": [1, 2]}))
        self.assertIsInstance(op.process("fit_transform", []), pl.DataFrame)

    def test_polars_read_csv(self):
        with csv_file(pd.DataFrame({"a": [1, 2]})) as path:
            op = PolarsReadCSV(file_path=path, read_args=(), read_kwargs={})
            self.assertIsInstance(op.process("fit_transform", []), pl.DataFrame)

    def test_pandas_read_parquet(self):
        with parquet_file(pd.DataFrame({"a": [1, 2], "b": [3, 4]})) as path:
            op = PandasReadParquet(file_path=path, read_args=(), read_kwargs={})
            result = op.process("fit_transform", [])
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual([1, 2], result["a"].tolist())

    def test_polars_read_parquet(self):
        with parquet_file(pd.DataFrame({"a": [1, 2]})) as path:
            op = PolarsReadParquet(file_path=path, read_args=(), read_kwargs={})
            self.assertIsInstance(op.process("fit_transform", []), pl.DataFrame)

    def test_unsupported_format_fails_at_lowering(self):
        # A DataSourceOp has no process; an unknown format is rejected when the
        # lowering rule tries to pick its physical source op.
        op = DataSourceOp(file_path="nofile", _format="orc",
                          read_args=(), read_kwargs={})
        with self.assertRaises(ValueError):
            lower_data_source(op, PlanContext.from_flags())

    def test_logical_data_source_has_no_process(self):
        # The logical op is plan-time data only; executing it is a bug.
        op = DataSourceOp(data=pd.DataFrame({"a": [1]}))
        with self.assertRaises(NotImplementedError):
            op.process("fit_transform", [])


class TestExecutionIsBranchFree(unittest.TestCase):
    """The backend is fixed at plan time: flipping the flag at run time is a no-op."""

    def setUp(self):
        self.df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    def test_in_memory_pandas_plan_ignores_runtime_flag(self):
        ops, *_ = optimize(st.as_data_op(self.df))  # planned as pandas
        with force_polars(True):                     # flag flipped for execution
            result = run_plan(ops)
        self.assertIsInstance(result, pd.DataFrame)

    def test_default_in_memory_plan_ignores_backend_flag(self):
        with force_polars(True):
            ops, *_ = optimize(st.as_data_op(self.df))  # default still plans pandas
        with force_polars(False):
            result = run_plan(ops)
        self.assertIsInstance(result, pd.DataFrame)

    def test_read_csv_pandas_plan_ignores_runtime_flag(self):
        with csv_file(self.df) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv)
            ops, *_ = optimize(data, OptConfig(dataframe_ops=True))  # pandas plan
            with force_polars(True):
                result = run_plan(ops)
        self.assertIsInstance(result, pd.DataFrame)

    def test_default_read_csv_plan_ignores_backend_flag(self):
        with csv_file(self.df) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv)
            with force_polars(True):
                ops, *_ = optimize(data, OptConfig(dataframe_ops=True))
            with force_polars(False):
                result = run_plan(ops)
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
