import unittest

import numpy as np
import pandas as pd
import polars as pl
import stratum as st
from stratum.optimizer._optimize import OptConfig
from skrub import selectors
from stratum.optimizer.ir._projection_ops import (
    ApplyUDFOp, AssignOp, ColumnSelectorOp, DatetimeConversionOp, DropOp,
    GetAttrProjectionOp, MetadataOp, ProjectionOp, StringMethodOp,
    make_datetime_conversion_op, make_frame_get_attr, make_string_method_op,
    polars_datetime_kwargs)
from stratum.optimizer.ir._map_ops import AssignMapOp
from stratum.optimizer.ir._column_expr import Col, DtExpr
from stratum.optimizer.ir._ops import (
    CallOp, GetAttrOp, GetItemOp, MethodCallOp, Op, OperandRef, OutputType,
    TransformerOp)
from stratum.tests.logical_optimizer.test_dataframe_ops import (
    PolarsTestCase, _inp, _inputs_for, force_polars, make_map_op, optimize, run_op)


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
        # The column getitem and both fused .dt accessors fold into the assign map.
        self.assertEqual(5, len(ops))
        map_op = next(o for o in ops if isinstance(o, AssignMapOp))
        self.assertEqual({"year": DtExpr(Col("datetime"), "year"),
                          "month": DtExpr(Col("datetime"), "month")},
                         map_op.entries)
        self.assertIsInstance(ops[-1], MethodCallOp)  # the trailing .copy()


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

    def test_kwargs_can_be_omitted(self):
        op = ProjectionOp(method="copy", args=(), kwargs=None)
        self.assertIsNone(op.kwargs)

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

    def test_pandas_path_preserves_kwargs(self):
        op = DatetimeConversionOp(args=(), kwargs={"dayfirst": True})
        result = run_op(op, pd.Series(["01/02/2020"]))
        self.assertEqual(pd.Timestamp("2020-02-01"), result.iloc[0])

    def test_polars_unsupported_kwargs_use_pandas_semantics(self):
        with force_polars():
            op = DatetimeConversionOp(args=(), kwargs={"dayfirst": True})
            result = run_op(op, pl.Series("dt", ["01/02/2020"]))
        self.assertEqual([pd.Timestamp("2020-02-01")], result.to_list())

    def test_polars_positional_options_use_pandas_semantics(self):
        with force_polars():
            op = DatetimeConversionOp(args=("coerce",), kwargs={})
            result = run_op(op, pl.Series("dt", ["2025-01-01", "bad"]))
        self.assertEqual([pd.Timestamp("2025-01-01"), None], result.to_list())

    def test_polars_fallback_converts_datetime_index(self):
        with force_polars():
            op = DatetimeConversionOp(args=(), kwargs={"dayfirst": True})
            result = run_op(op, ["01/02/2020", "03/04/2021"])
        self.assertEqual(
            [pd.Timestamp("2020-02-01"), pd.Timestamp("2021-04-03")],
            result.to_list())

    def test_polars_fallback_keeps_scalar_result(self):
        with force_polars():
            op = DatetimeConversionOp(args=(), kwargs={"dayfirst": True})
            result = run_op(op, "01/02/2020")
        self.assertEqual(pd.Timestamp("2020-02-01"), result)

    def test_polars_datetime_kwargs_rejects_unsupported_errors(self):
        self.assertIsNone(
            polars_datetime_kwargs((), {"errors": "ignore"}))


class TestGetAttrProjectionOp(unittest.TestCase):
    def test_init_with_none(self):
        self.assertEqual([], GetAttrProjectionOp(attr_name=None).attr_name)

    def test_init_with_str(self):
        self.assertEqual(["dt"], GetAttrProjectionOp(attr_name="dt").attr_name)

    def _run_polars(self, dt_values, attr_name):
        with force_polars():
            s = pl.Series("dt", pd.to_datetime(dt_values))
            op = GetAttrProjectionOp(attr_name=attr_name, inputs=[_inp(s)], outputs=[])
            return op.process("fit_transform", _inputs_for(op))

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

    def test_polars_single_accessor_returns_namespace(self):
        result = self._run_polars(["2025-01-15"], ["dt"])
        self.assertEqual([2025], result.year().to_list())

    def test_shared_accessor_keeps_original_edge_when_fused(self):
        source = Op()
        source.output_type = OutputType.SERIES
        accessor = GetAttrProjectionOp(
            attr_name=["dt"], inputs=[source], outputs=[])
        current = GetAttrOp(attr_name="year")
        current.inputs = [accessor]
        other_consumer = Op(inputs=[accessor])
        accessor.outputs = [current, other_consumer]
        source.outputs = [accessor]

        new_op = make_frame_get_attr(None, current)

        self.assertEqual([other_consumer], accessor.outputs)
        self.assertIn(accessor, source.outputs)
        self.assertIn(new_op, source.outputs)


class TestStringMethodOp(unittest.TestCase):
    """`col.str.<method>(...)` fuses the .str accessor and the call into one op."""

    def setUp(self):
        self.df = pd.DataFrame({"s": ["a1", "bb", "c1"], "x": [1, 2, 3]})

    def _one(self, ops, cls):
        found = [o for o in ops if isinstance(o, cls)]
        self.assertEqual(1, len(found), f"expected exactly one {cls.__name__}")
        return found[0]

    def test_str_method_fuses_accessor_away(self):
        # A str call used as a column projection (here assigned) becomes a single
        # StringMethodOp; the GetAttrProjectionOp(["str"]) accessor drops out.
        # Map folding is disabled so the intermediate op stays observable (with it
        # on, the assign absorbs the call into a StrExpr entry -- see test_map_ops).
        data = st.as_data_op(self.df)
        with make_map_op(False):
            ops = optimize(data.assign(c=data["s"].str.upper()),
                           OptConfig(dataframe_ops=True))
        sm = self._one(ops, StringMethodOp)
        self.assertEqual("upper", sm.method)
        self.assertIs(OutputType.SERIES, sm.output_type)
        self.assertIsInstance(sm.inputs[0], GetItemOp)  # the column, not the accessor
        self.assertEqual([], [o for o in ops if isinstance(o, GetAttrProjectionOp)])

    def test_shared_accessor_detached_after_last_call(self):
        # When the same `.str` accessor feeds two calls, each folds to its own
        # StringMethodOp and the accessor is removed once its last consumer is fused.
        data = st.as_data_op(self.df)
        acc = data["s"].str
        with make_map_op(False):
            ops = optimize(data.assign(a=acc.count("1"), b=acc.upper()),
                           OptConfig(dataframe_ops=True))
        self.assertEqual(2, len([o for o in ops if isinstance(o, StringMethodOp)]))
        self.assertEqual([], [o for o in ops if isinstance(o, GetAttrProjectionOp)])

    def test_process_pandas(self):
        op = StringMethodOp(method="count", args=("1",))
        result = run_op(op, pd.Series(["a1", "bb", "c1"]))
        self.assertEqual([1, 0, 1], result.tolist())

    def test_process_polars_renames_method(self):
        # polars renames .str.count -> .str.count_matches (STR_POLARS_METHODS).
        with force_polars():
            op = StringMethodOp(method="count", args=("1",))
            result = run_op(op, pl.Series(["a1", "bb", "c1"]))
        self.assertEqual([1, 0, 1], result.to_list())

    def test_graph_argument_is_rewired_to_fused_method(self):
        column = Op()
        column.output_type = OutputType.SERIES
        accessor = GetAttrProjectionOp(
            attr_name=["str"], inputs=[column], outputs=[])
        argument = Op()
        call = MethodCallOp(
            method_name="contains", args=(OperandRef(1),), kwargs={})
        call.inputs = [accessor, argument]
        accessor.outputs = [call]
        argument.outputs = [call]
        column.outputs = [accessor]

        new_op = make_string_method_op(call)

        self.assertIn(new_op, argument.outputs)
        self.assertNotIn(call, argument.outputs)


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
        op = ColumnSelectorOp(selector=selectors.numeric())
        run_op(op, pd.DataFrame({"a": [1.0]}))
        self.assertIsNone(op.clone().selected_columns)


class TestMakeDatetimeConversionOp(unittest.TestCase):
    def test_extra_positional_args(self):
        op = CallOp(func=pd.to_datetime,
                    args=(OperandRef(0), "ISO8601"), kwargs={})
        new_op = make_datetime_conversion_op(op)
        self.assertEqual(("ISO8601",), tuple(new_op.args))


if __name__ == "__main__":
    unittest.main()
