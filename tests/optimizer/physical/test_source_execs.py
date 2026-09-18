"""Physical source operators: lowering, implementation selection, and the
central guarantee that execution carries no operator-selection control flow.

The backend is not ambient state to flip: it belongs to the
:class:`FlagBasedSelector` a plan was built with, and is baked into the concrete
op class from there. So the branch-free tests build two plans from the *same*
logical DAG with two different selectors and execute both -- each keeps its own
backend, and re-executing either is stable, which is only true if no ``process``
consults anything outside the op.
"""
import unittest

import numpy as np
import pandas as pd
import polars as pl

import stratum as st
from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.physical._impl_selection import FlagBasedSelector
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.physical._registry import get_default_physical_registry
from stratum.optimizer.logical._source_ops import DataSourceOp
from stratum.optimizer.physical._source_execs import (
    InMemoryFrame, NumpyLoad, PandasInMemoryFrame, PandasReadCSV,
    PandasReadParquet, PolarsInMemoryFrame, PolarsReadCSV, PolarsReadParquet,
    ReadCSV, ReadParquet, lower_data_source)
from stratum.runtime._buffer_pool import BufferPool
from tests._helpers import csv_file, npy_file, parquet_file


def polars_conf(**kwargs):
    """An ``OptConfig`` that pins the plan to the polars impls."""
    return OptConfig(selector=FlagBasedSelector(backend="polars"), **kwargs)


def run_plan(ops, mode="fit_transform"):
    """Execute a linearized plan through a fresh BufferPool; return the sink value."""
    pool = BufferPool()
    for op in ops:
        inputs = [pool.pin(key) for key in op.inputs]
        pool.put(op, op.process(mode, inputs))
    return pool.pin(ops[-1])


class TestSourceImplSelection(unittest.TestCase):
    """Lowering + selection pick the backend-specific concrete source class."""

    def setUp(self):
        self.df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    def _read_csv_ops(self):
        return csv_file(self.df)

    def test_default_selector_prefers_pandas_in_memory_frame(self):
        ops, *_ = optimize(st.as_data_op(self.df))
        self.assertIsInstance(ops[0], PandasInMemoryFrame)

    def test_polars_pinned_selector_binds_polars_in_memory_frame(self):
        ops, *_ = optimize(st.as_data_op(self.df), polars_conf())
        self.assertIsInstance(ops[0], PolarsInMemoryFrame)

    def test_default_selector_prefers_pandas_read_csv(self):
        with self._read_csv_ops() as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv)
            ops, *_ = optimize(data, OptConfig(dataframe_ops=True))
            self.assertIsInstance(ops[-1], PandasReadCSV)

    def test_polars_pinned_selector_binds_polars_read_csv(self):
        with self._read_csv_ops() as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv)
            ops, *_ = optimize(data, polars_conf(dataframe_ops=True))
            self.assertIsInstance(ops[-1], PolarsReadCSV)

    def test_default_selector_prefers_pandas_read_parquet(self):
        with parquet_file(self.df) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_parquet)
            ops, *_ = optimize(data, OptConfig(dataframe_ops=True))
            self.assertIsInstance(ops[-1], PandasReadParquet)

    def test_polars_pinned_selector_binds_polars_read_parquet(self):
        with parquet_file(self.df) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_parquet)
            ops, *_ = optimize(data, polars_conf(dataframe_ops=True))
            self.assertIsInstance(ops[-1], PolarsReadParquet)

    def test_npy_is_single_impl(self):
        # np.load yields an ndarray; a single concrete impl serves both backends.
        with npy_file(np.array([1, 2, 3])) as path:
            data = st.as_data_op(path).skb.apply_func(np.load)
            ops, *_ = optimize(data, OptConfig(dataframe_ops=True))
            self.assertIsInstance(ops[-1], NumpyLoad)
            ops, *_ = optimize(data, polars_conf(dataframe_ops=True))
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
        op = ReadCSV(file_path="x.csv")
        ctx = PlanContext.from_flags()
        for backend, expected in (("pandas", PandasReadCSV),
                                  ("polars", PolarsReadCSV)):
            chosen = FlagBasedSelector(backend=backend).choose(op, candidates, ctx)
            self.assertIs(expected, chosen.impl_class)

    def test_flag_selector_falls_back_to_backend_agnostic(self):
        # NumpyLoad's only candidate is backend "numpy"; it is chosen under
        # either frame backend.
        candidates = list(self.registry.candidates_for(NumpyLoad))
        op = NumpyLoad(file_path="x.npy")
        ctx = PlanContext.from_flags()
        for backend in ("pandas", "polars"):
            chosen = FlagBasedSelector(backend=backend).choose(op, candidates, ctx)
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
    """The backend is fixed at plan time, by the selector the plan was built with.

    There is no run-time flag left to flip -- which is the point. What is still
    worth pinning is that the *same* logical DAG yields two independent plans
    under two selectors, each keeping its own backend through execution and
    across repeated runs.
    """

    def setUp(self):
        self.df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    def _both_plans(self, dag, conf_kwargs=None):
        conf_kwargs = conf_kwargs or {}
        pandas_ops, *_ = optimize(dag, OptConfig(**conf_kwargs))
        polars_ops, *_ = optimize(dag, polars_conf(**conf_kwargs))
        return pandas_ops, polars_ops

    def test_in_memory_plans_keep_their_own_backend(self):
        pandas_ops, polars_ops = self._both_plans(st.as_data_op(self.df))
        self.assertIsInstance(run_plan(pandas_ops), pd.DataFrame)
        self.assertIsInstance(run_plan(polars_ops), pl.DataFrame)
        # Re-running either plan is stable: nothing outside the op is consulted.
        self.assertIsInstance(run_plan(pandas_ops), pd.DataFrame)
        self.assertIsInstance(run_plan(polars_ops), pl.DataFrame)

    def test_read_csv_plans_keep_their_own_backend(self):
        with csv_file(self.df) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv)
            pandas_ops, polars_ops = self._both_plans(
                data, {"dataframe_ops": True})
            self.assertIsInstance(run_plan(pandas_ops), pd.DataFrame)
            self.assertIsInstance(run_plan(polars_ops), pl.DataFrame)
            self.assertIsInstance(run_plan(pandas_ops), pd.DataFrame)


class TestPolarsReadOptionTranslation(unittest.TestCase):
    """pandas-spelled read options reach the polars readers and must be rewritten.

    A source op can only ever have been built from ``pd.read_csv`` /
    ``pd.read_parquet`` (the only readers ``_READ_FORMATS`` recognises), so its
    options are always in pandas spelling -- polars renames most of them.
    """

    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6],
                                "unused": ["a", "b", "c"]})

    def test_usecols_becomes_columns(self):
        with csv_file(self.df) as path:
            op = PolarsReadCSV(file_path=path, read_kwargs={"usecols": ["x", "y"]})
            self.assertEqual(["x", "y"], op.process("fit_transform", []).columns)

    def test_sep_becomes_separator(self):
        with csv_file(self.df, sep=";") as path:
            op = PolarsReadCSV(file_path=path, read_kwargs={"sep": ";"})
            self.assertEqual(["x", "y", "unused"],
                             op.process("fit_transform", []).columns)

    def test_nrows_becomes_n_rows(self):
        with csv_file(self.df) as path:
            op = PolarsReadCSV(file_path=path, read_kwargs={"nrows": 2})
            self.assertEqual(2, len(op.process("fit_transform", [])))

    def test_header_none_becomes_has_header_false(self):
        with csv_file(self.df, header=False) as path:
            op = PolarsReadCSV(file_path=path, read_kwargs={"header": None})
            self.assertEqual(3, len(op.process("fit_transform", [])))

    def test_parquet_columns_pass_through(self):
        with parquet_file(self.df) as path:
            op = PolarsReadParquet(file_path=path, read_kwargs={"columns": ["x"]})
            self.assertEqual(["x"], op.process("fit_transform", []).columns)

    def test_untranslatable_option_fails_at_plan_time(self):
        # index_col has no polars counterpart. Selection binds the op, and
        # on_impl_selected rejects it there -- before any data is read.
        with csv_file(self.df) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv, index_col=0)
            with self.assertRaises(ValueError) as caught:
                optimize(data, polars_conf(dataframe_ops=True))
        self.assertIn("index_col", str(caught.exception))

    def test_untranslatable_option_is_fine_on_pandas(self):
        with csv_file(self.df) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv, index_col=0)
            ops, *_ = optimize(data, OptConfig(dataframe_ops=True))
        self.assertIsInstance(ops[-1], PandasReadCSV)

    def test_untranslatable_value_is_rejected(self):
        # header=2 has no has_header spelling; only the name survives the
        # plan-time check, so this one is caught when the value is resolved.
        with csv_file(self.df) as path:
            op = PolarsReadCSV(file_path=path, read_kwargs={"header": 2})
            with self.assertRaises(ValueError):
                op.process("fit_transform", [])

    def test_positional_read_options_are_rejected(self):
        # polars takes only the path positionally, so a pandas positional
        # option (pd.read_parquet's `engine`) would bind to something else.
        op = PolarsReadParquet(file_path="x.parquet", read_args=("pyarrow",))
        with self.assertRaises(ValueError):
            op.process("fit_transform", [])


if __name__ == "__main__":
    unittest.main()
