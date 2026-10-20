import unittest

import pandas as pd
import polars as pl
import stratum as st
from stratum.optimizer._optimize import OptConfig
from skrub import selectors
from stratum.optimizer.logical._projection_ops import (
    ColumnProjectionOp, ColumnSelectorOp, DropOp, ProjectionOp, drop_targets_columns)
from stratum.optimizer.logical._ops import (
    GetItemOp, MethodCallOp, OperandRef, OutputType, TransformerOp)
from .test_dataframe_ops import (
    PolarsTestCase, _inp, _inputs_for, force_polars, optimize, run_op)


class TestProjectionRewrites(unittest.TestCase):
    """`optimize` rewrites column/frame projections produced by skrub DAGs."""

    def setUp(self):
        self.df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "datetime": ["2025-11-01 10:00:00",
                         "2025-11-02 15:30:00",
                         "2025-11-03 09:45:00"],
        })

    def test_projection_drop(self):
        ops = optimize(st.as_data_op(self.df).drop("y", axis=1))
        self.assertEqual(2, len(ops))
        self.assertIsInstance(ops[1], ProjectionOp)


class TestColumnProjectionOp(unittest.TestCase):
    """A column-selecting ``df[key]`` on a frame parses to a ColumnProjectionOp;
    the general GetItemOp stays the fallback for everything else."""

    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    def _ops(self, data_op, **config_kwargs):
        config_kwargs.setdefault("literal_inlining", False)
        return optimize(data_op, OptConfig(dataframe_ops=True, **config_kwargs))

    def test_single_column_parses_to_series_projection(self):
        ops = self._ops(st.as_data_op(self.df)["x"])
        proj = [o for o in ops if isinstance(o, ColumnProjectionOp)]
        self.assertEqual(1, len(proj))
        self.assertEqual("x", proj[0].key)
        self.assertIs(OutputType.SERIES, proj[0].output_type)
        self.assertFalse(any(isinstance(o, GetItemOp) for o in ops))

    def test_column_list_parses_to_frame_projection(self):
        ops = self._ops(st.as_data_op(self.df)[["x", "y"]])
        proj = [o for o in ops if isinstance(o, ColumnProjectionOp)]
        self.assertEqual(1, len(proj))
        self.assertEqual(["x", "y"], proj[0].key)
        self.assertIs(OutputType.FRAME, proj[0].output_type)

    def test_graph_fed_key_stays_getitem(self):
        # A key fed from another data-op is an OperandRef, not a literal column
        # name, so it is not a column projection -> stays a GetItemOp fallback.
        ops = self._ops(st.as_data_op(self.df)[st.as_data_op("x")])
        self.assertFalse(any(isinstance(o, ColumnProjectionOp) for o in ops))
        self.assertTrue(any(isinstance(o, GetItemOp) for o in ops))

    def test_toggle_off_keeps_getitem(self):
        with st.config(make_column_projection=False):
            ops = self._ops(st.as_data_op(self.df)["x"])
        self.assertFalse(any(isinstance(o, ColumnProjectionOp) for o in ops))
        self.assertTrue(any(isinstance(o, GetItemOp) for o in ops))

    def test_clone_derives_output_type_from_key(self):
        self.assertIs(OutputType.SERIES, ColumnProjectionOp(key="x").clone().output_type)
        self.assertIs(OutputType.FRAME, ColumnProjectionOp(key=["x", "y"]).clone().output_type)

    def test_execute_single_column_pandas(self):
        result = run_op(ColumnProjectionOp(key="x"), self.df)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual([1, 2, 3], result.tolist())

    def test_execute_column_list_pandas(self):
        result = run_op(ColumnProjectionOp(key=["x", "y"]), self.df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(["x", "y"], list(result.columns))

    def test_execute_single_column_polars(self):
        with force_polars():
            result = run_op(ColumnProjectionOp(key="x"), pl.from_pandas(self.df))
        self.assertIsInstance(result, pl.Series)
        self.assertEqual([1, 2, 3], result.to_list())

    def test_execute_column_list_polars(self):
        with force_polars():
            result = run_op(ColumnProjectionOp(key=["x", "y"]), pl.from_pandas(self.df))
        self.assertIsInstance(result, pl.DataFrame)
        self.assertEqual(["x", "y"], result.columns)


class TestDropOpPolars(PolarsTestCase):
    def test_drop_with_columns_kwarg(self):
        op = DropOp(args=(), kwargs={"columns": ["b"]})
        result = run_op(op, pl.DataFrame({"a": [1], "b": [2], "c": [3]}))
        self.assertNotIn("b", result.columns)

    def test_errors_ignore_skips_missing_columns(self):
        op = DropOp(args=(), kwargs={"columns": ["a", "missing"], "errors": "ignore"})
        result = run_op(op, pl.DataFrame({"a": [1], "b": [2]}))
        self.assertEqual(["b"], result.columns)

    def test_a_missing_column_raises_by_default(self):
        op = DropOp(args=(), kwargs={"columns": ["missing"]})
        with self.assertRaises(Exception):
            run_op(op, pl.DataFrame({"a": [1], "b": [2]}))


class TestColumnSelectorExtraction(unittest.TestCase):
    """skb.select(selector) -- an Apply of SelectCols -- becomes a ColumnSelectorOp."""

    def setUp(self):
        self.df = pd.DataFrame({"a": [1.0, 2.0], "s": ["x", "y"]})

    def _one(self, ops, cls):
        found = [o for o in ops if isinstance(o, cls)]
        self.assertEqual(1, len(found), f"expected exactly one {cls.__name__}")
        return found[0]

    def test_select_converts_to_column_selector(self):
        data = st.as_data_op(self.df)
        sel = selectors.numeric()
        ops = optimize(data.skb.select(sel), OptConfig(dataframe_ops=True))
        col_sel = self._one(ops, ColumnSelectorOp)
        self.assertIs(sel, col_sel.selector)
        self.assertIs(OutputType.FRAME, col_sel.output_type)
        # the SelectCols TransformerOp is fully replaced
        self.assertEqual([], [o for o in ops if isinstance(o, TransformerOp)])

    def test_select_by_names_converts(self):
        data = st.as_data_op(self.df)
        ops = optimize(data.skb.select(["a"]), OptConfig(dataframe_ops=True))
        self.assertEqual(["a"], self._one(ops, ColumnSelectorOp).selector)

    def test_select_cols_on_restricted_columns_is_not_a_projection(self):
        """Applied to part of the frame, skrub's SelectCols passes the rest through."""
        from skrub import SelectCols
        from tests.optimizer.physical.test_source_execs import run_plan
        data = st.as_data_op(self.df)
        for kwargs in ({"cols": ["a"]}, {"exclude_cols": ["a"]}):
            with self.subTest(**kwargs):
                dag = data.skb.apply(SelectCols([]), **kwargs)
                ops = optimize(dag, OptConfig(dataframe_ops=True))
                self.assertEqual([], [o for o in ops if isinstance(o, ColumnSelectorOp)])
                self.assertEqual(sorted(dag.skb.eval().columns),
                                 sorted(run_plan(ops).columns))


class TestColumnSelectorProcess(unittest.TestCase):
    """ColumnSelectorOp resolves the selector at fit and reuses the list at predict."""

    def test_fit_resolves_and_selects_pandas(self):
        op = ColumnSelectorOp(selector=selectors.numeric())
        result = run_op(op, pd.DataFrame({"a": [1.0], "s": ["x"]}))
        self.assertEqual(["a"], list(result.columns))
        self.assertEqual(["a"], op.selected_columns)

    def test_predict_reuses_stored_columns(self):
        # A new numeric column appearing at predict time is NOT picked up: the
        # fit-time resolution is what transforms (SelectCols semantics).
        op = ColumnSelectorOp(selector=selectors.numeric())
        run_op(op, pd.DataFrame({"a": [1.0], "s": ["x"]}))
        result = run_op(op, pd.DataFrame({"a": [2.0], "b": [3.0], "s": ["y"]}),
                        mode="predict")
        self.assertEqual(["a"], list(result.columns))

    def test_predict_before_fit_raises(self):
        op = ColumnSelectorOp(selector=selectors.numeric())
        with self.assertRaises(RuntimeError):
            run_op(op, pd.DataFrame({"a": [1.0]}), mode="predict")

    def test_polars(self):
        op = ColumnSelectorOp(selector=selectors.numeric())
        with force_polars():
            result = run_op(op, pl.DataFrame({"a": [1.0], "s": ["x"]}))
        self.assertEqual(["a"], result.columns)

    def test_clone_resets_resolution(self):
        # Cloning happens in the logical phase (choice unrolling), before impl
        # selection; a cloned ColumnSelectorOp is a fresh logical op that carries
        # no resolved column list.
        op = ColumnSelectorOp(selector=selectors.numeric())
        op.selected_columns = ["a"]  # even if the original were somehow resolved
        self.assertIsNone(op.clone().selected_columns)


class TestDropTargetsColumns(unittest.TestCase):
    """Which axis a ``drop`` call works on decides its logical family."""

    def test_axis_classification_matches_pandas(self):
        cases = [
            ({}, False),
            ({"axis": 0}, False),
            ({"axis": "index"}, False),
            ({"index": [0]}, False),
            ({"axis": 1}, True),
            ({"axis": "columns"}, True),
            ({"columns": ["a"]}, True),
            ({"columns": ["a"], "axis": 1}, None),     # pandas rejects axis with columns=
            ({"index": [0], "columns": ["a"]}, None),  # both axes
            ({"axis": OperandRef(1)}, None),           # graph-fed axis
        ]
        for kwargs, expected in cases:
            with self.subTest(kwargs=repr(kwargs)):
                self.assertIs(expected, drop_targets_columns(kwargs))
