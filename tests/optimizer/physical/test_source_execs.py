"""Physical source operators: implementation selection, and the central
guarantee that execution carries no operator-selection control flow.

The backend is not ambient state to flip: it belongs to the
:class:`FlagBasedSelector` a plan was built with, and is baked into the concrete
op class from there. So the branch-free tests build two plans from the *same*
logical DAG with two different selectors and execute both -- each keeps its own
backend, and re-executing either is stable, which is only true if no ``process``
consults anything outside the op.
"""
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import polars as pl

import stratum as st
from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.logical._base import OperandRef
from stratum.optimizer.physical._impl_selection import (
    FlagBasedSelector, get_implementation_selector)
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.physical._registry import get_default_physical_registry
from stratum.optimizer.logical._source_ops import (
    InMemoryFrameOp, LoadNumpyOp, ReadCSVOp, ReadParquetOp)
from stratum.optimizer.physical._source_execs import (
    NumpyLoad, PandasInMemoryFrame, PandasReadCSV,
    PandasReadParquet, PolarsInMemoryFrame, PolarsReadCSV, PolarsReadParquet,
    _POLARS_CSV_TRANSLATORS, _POLARS_PARQUET_TRANSLATORS, _polars_cloud_path,
    _translate_storage_options)
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
    """Selection picks the backend-specific concrete source class."""

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

    def test_impls_subclass_the_logical_ops(self):
        for logical, impls in ((InMemoryFrameOp, (PandasInMemoryFrame, PolarsInMemoryFrame)),
                               (ReadCSVOp, (PandasReadCSV, PolarsReadCSV)),
                               (ReadParquetOp, (PandasReadParquet, PolarsReadParquet)),
                               (LoadNumpyOp, (NumpyLoad,))):
            for impl in impls:
                self.assertTrue(issubclass(impl, logical), impl.__name__)


class TestSourcesInRegistry(unittest.TestCase):
    """The source impls are registered in the default PhysicalRegistry and
    selection resolves through it (no per-class backend maps)."""

    def setUp(self):
        self.registry = get_default_physical_registry()

    def test_source_candidates_are_registered(self):
        for abstract, impls in ((ReadCSVOp, {PandasReadCSV, PolarsReadCSV}),
                                (ReadParquetOp, {PandasReadParquet, PolarsReadParquet}),
                                (InMemoryFrameOp, {PandasInMemoryFrame, PolarsInMemoryFrame})):
            candidates = self.registry.candidates_for(abstract)
            self.assertEqual(impls, {c.impl_class for c in candidates})
            self.assertEqual({"pandas", "polars"},
                             {c.backend_name for c in candidates})

    def test_numpy_load_is_registered(self):
        candidates = self.registry.candidates_for(LoadNumpyOp)
        self.assertEqual(("numpy",), tuple(c.backend_name for c in candidates))

    def test_flag_selector_picks_backend_match(self):
        candidates = list(self.registry.candidates_for(ReadCSVOp))
        op = ReadCSVOp(file_path="x.csv")
        ctx = PlanContext.from_flags()
        for backend, expected in (("pandas", PandasReadCSV),
                                  ("polars", PolarsReadCSV)):
            chosen = FlagBasedSelector(backend=backend).choose(op, candidates, ctx)
            self.assertIs(expected, chosen.impl_class)

    def test_flag_selector_falls_back_to_backend_agnostic(self):
        # NumpyLoad's only candidate is backend "numpy"; it is chosen under
        # either frame backend.
        candidates = list(self.registry.candidates_for(LoadNumpyOp))
        op = LoadNumpyOp(file_path="x.npy")
        ctx = PlanContext.from_flags()
        for backend in ("pandas", "polars"):
            chosen = FlagBasedSelector(backend=backend).choose(op, candidates, ctx)
            self.assertIs(NumpyLoad, chosen.impl_class)


class TestConcreteSourceProcess(unittest.TestCase):
    """Direct process() coverage per concrete source impl."""

    def test_numpy_read(self):
        with npy_file(np.array([1, 2, 3])) as path:
            op = NumpyLoad(file_path=path)
            result = op.process("fit_transform", [])
            np.testing.assert_array_equal(result, [1, 2, 3])

    def test_pandas_from_dataframe(self):
        df = pd.DataFrame({"a": [1, 2]})
        op = PandasInMemoryFrame(data=df)
        self.assertIs(df, op.process("fit_transform", []))

    def test_pandas_from_a_polars_dataframe(self):
        op = PandasInMemoryFrame(data=pl.DataFrame({"a": [1, 2]}))
        result = op.process("fit_transform", [])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual([1, 2], result["a"].tolist())

    def test_polars_from_dataframe(self):
        op = PolarsInMemoryFrame(data=pd.DataFrame({"a": [1, 2]}))
        self.assertIsInstance(op.process("fit_transform", []), pl.DataFrame)

    def test_polars_read_csv(self):
        with csv_file(pd.DataFrame({"a": [1, 2]})) as path:
            op = PolarsReadCSV(file_path=path)
            self.assertIsInstance(op.process("fit_transform", []), pl.DataFrame)

    def test_pandas_read_parquet(self):
        with parquet_file(pd.DataFrame({"a": [1, 2], "b": [3, 4]})) as path:
            op = PandasReadParquet(file_path=path)
            result = op.process("fit_transform", [])
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual([1, 2], result["a"].tolist())

    def test_polars_read_parquet(self):
        with parquet_file(pd.DataFrame({"a": [1, 2]})) as path:
            op = PolarsReadParquet(file_path=path)
            self.assertIsInstance(op.process("fit_transform", []), pl.DataFrame)

    def test_logical_sources_have_no_process(self):
        # The logical ops are plan-time data only; executing one is a bug.
        for op in (InMemoryFrameOp(data=pd.DataFrame({"a": [1]})),
                   ReadCSVOp(file_path="x.csv")):
            with self.subTest(type(op).__name__), self.assertRaises(NotImplementedError):
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

    A read op can only ever have been built from ``pd.read_csv`` /
    ``pd.read_parquet``, so its fields hold pandas' values -- polars renames most
    of the options and reads some values differently.
    """

    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6],
                                "unused": ["a", "b", "c"]})

    def test_usecols_becomes_columns(self):
        with csv_file(self.df) as path:
            op = PolarsReadCSV(file_path=path, usecols=["x", "y"])
            self.assertEqual(["x", "y"], op.process("fit_transform", []).columns)

    def test_sep_becomes_separator(self):
        with csv_file(self.df, sep=";") as path:
            op = PolarsReadCSV(file_path=path, sep=";")
            self.assertEqual(["x", "y", "unused"],
                             op.process("fit_transform", []).columns)

    def test_nrows_becomes_n_rows(self):
        with csv_file(self.df) as path:
            op = PolarsReadCSV(file_path=path, nrows=2)
            self.assertEqual(2, len(op.process("fit_transform", [])))

    def test_header_none_becomes_has_header_false(self):
        with csv_file(self.df, header=False) as path:
            op = PolarsReadCSV(file_path=path, header=None)
            self.assertEqual(3, len(op.process("fit_transform", [])))

    def test_names_without_header_keeps_first_row(self):
        with csv_file(self.df) as path:
            options = {"names": ["a", "b", "c"]}
            result = PolarsReadCSV(file_path=path, **options).process("fit_transform", [])
            expected = pd.read_csv(path, **options)
            self.assertEqual(expected.columns.tolist(), result.columns)
            self.assertEqual(expected.astype(str).values.tolist(),
                             [list(row) for row in result.cast(pl.String).rows()])

    def test_names_with_explicit_header_consumes_first_row(self):
        with csv_file(self.df) as path:
            options = {"names": ["a", "b", "c"], "header": 0}
            result = PolarsReadCSV(file_path=path, **options).process("fit_transform", [])
            expected = pd.read_csv(path, **options)
            self.assertEqual(expected.columns.tolist(), result.columns)
            self.assertEqual(expected.values.tolist(),
                             [list(row) for row in result.rows()])

    def test_parquet_columns_pass_through(self):
        with parquet_file(self.df) as path:
            op = PolarsReadParquet(file_path=path, columns=["x"])
            self.assertEqual(["x"], op.process("fit_transform", []).columns)

    def test_untranslatable_option_is_refused_at_plan_time(self):
        # index_col has no polars counterpart, so polars refuses the read in
        # supports() and a plan pinned to polars has no impl for it.
        with csv_file(self.df) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv, index_col=0)
            with self.assertRaises(NotImplementedError):
                optimize(data, polars_conf(dataframe_ops=True))
        op = ReadCSVOp(file_path="x.csv", pandas_options={"index_col": 0})
        self.assertFalse(PolarsReadCSV.supports(op, PlanContext.from_flags()))
        self.assertIn("index_col", op._unsupported_reason)

    def test_greedy_reads_an_untranslatable_source_on_pandas(self):
        with csv_file(self.df) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv, index_col=0)[["y"]]
            ops, *_ = optimize(data, OptConfig(selector=get_implementation_selector("greedy")))
            self.assertIsInstance(ops[0], PandasReadCSV)
            # The read's frame is converted for the polars projection after it.
            result = run_plan(ops)
            self.assertIsInstance(result, pl.DataFrame)
            self.assertEqual(pd.read_csv(path, index_col=0)["y"].tolist(),
                             result["y"].to_list())

    def test_an_untranslatable_literal_value_is_refused_at_plan_time(self):
        for op, reason in ((ReadCSVOp(file_path="x.csv", header=2), "header=2"),
                           (ReadCSVOp(file_path="x.csv", encoding="latin-1"), "encoding"),
                           (ReadParquetOp(file_path="x.parquet", filters=[("x", "like", "a")]),
                            "filter")):
            impl = PolarsReadParquet if isinstance(op, ReadParquetOp) else PolarsReadCSV
            with self.subTest(reason):
                self.assertFalse(impl.supports(op, PlanContext.from_flags()))
                self.assertIn(reason, op._unsupported_reason)
        # A graph-fed value is only known when the read runs.
        self.assertTrue(PolarsReadCSV.supports(
            ReadCSVOp(file_path="x.csv", header=OperandRef(0)), PlanContext.from_flags()))

    def test_untranslatable_option_is_fine_on_pandas(self):
        with csv_file(self.df) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv, index_col=0)
            ops, *_ = optimize(data, OptConfig(dataframe_ops=True))
        self.assertIsInstance(ops[-1], PandasReadCSV)

    def test_untranslatable_value_is_rejected(self):
        # header=2 has no has_header spelling; only the name survives the
        # plan-time check, so this one is caught when the value is resolved.
        with csv_file(self.df) as path:
            op = PolarsReadCSV(file_path=path, header=2)
            with self.assertRaises(ValueError):
                op.process("fit_transform", [])

    def test_parquet_filters_select_the_plan_backend(self):
        # pyarrow `filters` used to fail the whole polars plan at selection time.
        with parquet_file(self.df) as path:
            data = st.as_data_op(path).skb.apply_func(
                pd.read_parquet, filters=[("x", ">", 1)])
            ops, *_ = optimize(data, polars_conf(dataframe_ops=True))
        self.assertIsInstance(ops[-1], PolarsReadParquet)

    def test_untranslatable_parquet_filter_is_rejected(self):
        op = PolarsReadParquet(file_path="x.parquet", filters=[("x", "like", "a%")])
        with self.assertRaises(ValueError):
            op.process("fit_transform", [])

    def test_fsspec_storage_options_become_object_store_keys(self):
        # Passed verbatim, polars ignored gcsfs' `token` and read the bucket
        # unauthenticated (401).
        cases = [
            ({"token": "/keys/sa.json"}, {"service_account": "/keys/sa.json"}),
            ({"token": {"type": "service_account"}},
             {"service_account_key": '{"type": "service_account"}'}),
            ({"token": "anon"}, {"skip_signature": "true"}),
            ({"token": "google_default"}, {}),
            ({"key": "k", "secret": "s", "anon": False},
             {"aws_access_key_id": "k", "aws_secret_access_key": "s"}),
            ({"aws_region": "eu-west-1"}, {"aws_region": "eu-west-1"}),
        ]
        for pandas_options, polars_options in cases:
            with self.subTest(pandas_options):
                self.assertEqual(("storage_options", polars_options),
                                 _translate_storage_options(pandas_options))

    def test_cloud_dataset_directory_gets_trailing_slash(self):
        for path, expected in [
                ("gs://bucket/lake/table", "gs://bucket/lake/table/"),
                ("s3://bucket/table/", "s3://bucket/table/"),
                ("gs://bucket/table/part-0.parquet", "gs://bucket/table/part-0.parquet"),
                ("gs://bucket/table/**/*.parquet", "gs://bucket/table/**/*.parquet"),
                ("/local/table", "/local/table")]:
            with self.subTest(path):
                self.assertEqual(expected, _polars_cloud_path(path))

    def test_positional_read_options_bind_to_their_names(self):
        # pd.read_parquet's second positional is `engine`, which polars has no
        # counterpart for.
        with parquet_file(self.df) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_parquet, "pyarrow")
            ops, *_ = optimize(data, OptConfig(dataframe_ops=True))
        self.assertEqual({"engine": "pyarrow"}, ops[-1].pandas_options)
        self.assertFalse(PolarsReadParquet.supports(ops[-1], PlanContext.from_flags()))
        self.assertIn("engine", ops[-1]._unsupported_reason)


class TestPolarsReadOptionValues(unittest.TestCase):
    """pandas takes read option values that polars cannot, or reads differently.
    Polars refuses those at plan time, so pandas reads the file; every other value
    reads on polars exactly as on pandas."""

    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"], "z": [1.0, 2.5, 3.0]})

    def assert_matches_pandas(self, path, **options):
        op = PolarsReadCSV(file_path=path, **options)
        self.assertTrue(PolarsReadCSV.supports(op, PlanContext.from_flags()),
                        getattr(op, "_unsupported_reason", None))
        pd.testing.assert_frame_equal(pd.read_csv(path, **options),
                                      op.process("fit_transform", []).to_pandas(),
                                      check_dtype=False)

    def assert_refused(self, op, reason):
        impl = PolarsReadParquet if isinstance(op, ReadParquetOp) else PolarsReadCSV
        self.assertFalse(impl.supports(op, PlanContext.from_flags()))
        self.assertIn(reason, op._unsupported_reason)

    def test_every_option_has_a_translator(self):
        self.assertEqual(set(ReadCSVOp.OPTIONS), set(_POLARS_CSV_TRANSLATORS))
        self.assertEqual(set(ReadParquetOp.OPTIONS), set(_POLARS_PARQUET_TRANSLATORS))

    def test_usecols_keep_the_file_order(self):
        # polars returns a projection in the order given, pandas in the file's.
        with csv_file(self.df) as path:
            for usecols in (["z", "x"], [2, 0], {"z", "x"}, ["z", "x", "z"]):
                with self.subTest(usecols=usecols):
                    self.assert_matches_pandas(path, usecols=usecols)

    def test_translatable_values_match_pandas(self):
        with csv_file(self.df) as path:
            for options in ({"skiprows": 1}, {"nrows": 2}, {"names": ["p", "q", "r"]},
                            {"na_values": "b"}, {"na_values": ["b", "c"]},
                            {"na_values": {"y": "c"}}, {"na_values": {"y": ["c"]}},
                            {"quotechar": "'"}):
                with self.subTest(options):
                    self.assert_matches_pandas(path, **options)
        with csv_file(self.df, sep=";") as path:
            self.assert_matches_pandas(path, sep=";")

    def test_values_polars_cannot_read_are_refused_at_plan_time(self):
        for options, reason in (({"skiprows": [1]}, "skiprows"),
                                ({"skiprows": lambda i: i == 1}, "skiprows"),
                                ({"usecols": lambda name: name != "y"}, "usecols"),
                                ({"usecols": ["x", 0]}, "usecols"),
                                ({"na_values": {"y": ["b", "c"]}}, "na_values"),
                                ({"na_values": "1"}, "as a number"),
                                ({"na_values": 3}, "na_values"),
                                ({"sep": r"\s+"}, "sep"),
                                ({"quotechar": "''"}, "quotechar"),
                                ({"comment": "#"}, "comment"),
                                ({"names": ["p", 1, "r"]}, "names"),
                                ({"usecols": ["p"], "names": ["p", "q", "r"]}, "usecols with names")):
            with self.subTest(options):
                self.assert_refused(ReadCSVOp(file_path="x.csv", **options), reason)
        self.assert_refused(ReadParquetOp(file_path="x.parquet", columns=[0]), "columns")

    def test_names_must_fit_the_file_width(self):
        # pandas makes an index of the leading columns for fewer names, and pads
        # for more; polars reads only an exact fit.
        with csv_file(self.df) as path:
            for names in (["p", "q"], ["p", "q", "r", "s"]):
                with self.subTest(names=names):
                    self.assert_refused(ReadCSVOp(file_path=path, names=names), "columns")
        # A graph-fed path is only known when the read runs.
        self.assertTrue(PolarsReadCSV.supports(
            ReadCSVOp(file_path=OperandRef(0), names=["p", "q"]), PlanContext.from_flags()))

    def test_greedy_reads_a_refused_value_on_pandas(self):
        with csv_file(self.df) as path:
            data = st.as_data_op(path).skb.apply_func(pd.read_csv, skiprows=[1])[["x"]]
            ops, *_ = optimize(data, OptConfig(selector=get_implementation_selector("greedy")))
            self.assertIsInstance(ops[0], PandasReadCSV)
            self.assertEqual(pd.read_csv(path, skiprows=[1])["x"].tolist(),
                             run_plan(ops)["x"].to_list())


class TestPolarsParquetFilters(unittest.TestCase):
    """``PolarsReadParquet`` honours pyarrow ``filters`` exactly like
    ``pd.read_parquet``, on a hive-partitioned dataset."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        pd.DataFrame({"year": [2019, 2020, 2021, 2022] * 5, "x": range(20),
                      "kind": list("abcde") * 4}).to_parquet(
            self.root, partition_cols=["year"])

    def assert_matches_pandas(self, **options):
        expected = pd.read_parquet(self.root, **options)
        result = PolarsReadParquet(file_path=self.root, **options).process(
            "fit_transform", [])
        columns = options.get("columns") or ["x", "kind"]
        self.assertEqual(columns, [c for c in result.columns if c in columns])
        key = ["x"] if "x" in columns else columns
        self.assertEqual(
            expected[columns].sort_values(key).values.tolist(),
            [list(row) for row in result.select(columns).sort(key).rows()])

    def test_conjunction(self):
        self.assert_matches_pandas(filters=[("year", ">=", 2020), ("x", "<", 15)])

    def test_disjunction_of_conjunctions(self):
        self.assert_matches_pandas(filters=[[("year", "==", 2019)],
                                            [("year", "=", 2022), ("kind", "!=", "c")]])

    def test_membership(self):
        self.assert_matches_pandas(filters=[("kind", "in", ["a", "b"]),
                                            ("year", "not in", {2021})])

    def test_filter_on_column_left_out_of_projection(self):
        self.assert_matches_pandas(columns=["kind", "x"],
                                   filters=[("year", "<=", 2020)])

    def test_none_filters_read_everything(self):
        self.assert_matches_pandas(filters=None)


if __name__ == "__main__":
    unittest.main()
