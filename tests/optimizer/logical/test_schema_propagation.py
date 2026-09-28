import operator
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd
import polars as pl

import stratum as st
from stratum.optimizer._optimize import OptConfig, logical_optimize
from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer.logical._dataframe_ops import (
    AggregateOp, AssignMapOp, AssignOp, ColumnProjectionOp, ConcatOp, DataSourceOp,
    DatetimeConversionOp, DropOp, GetAttrProjectionOp, JoinOp, MetadataOp,
    SelectionKind, SelectionOp, SplitOp, SplitOutput)
from stratum.optimizer.logical._ops import (
    BinOp, ChoiceOp, GetItemOp, Op, OperandRef, OutputType, UnaryOp)
from stratum.optimizer.logical._projection_ops import StringMethodOp
from stratum.optimizer._optimize import optimize as optimize_full
from stratum.optimizer.logical._candidate_ops import ScoreCandidatesOp
from stratum.optimizer.logical._column_methods import ColumnMethodOp
from stratum.optimizer.logical._column_expr import (
    AggExpr, AllCols, BinOpExpr, Col, DtExpr)
from stratum.optimizer.logical._scoring import Metric
from tests._helpers import csv_file


def _logical_ops(dag):
    """Logical ops of `dag` after frame extraction + schema propagation.

    Schema propagation is a logical pass, so the assertions look at the logical
    DAG rather than at the linearized plan the full `optimize()` returns. The
    schema does survive lowering -- `install_lowered` carries it over -- but by
    then an op is addressed by its physical impl class, which makes the
    per-rule assertions here much harder to read.
    """
    root = logical_optimize(dag, OptConfig(dataframe_ops=True, propagate_schema=True))
    return list(topological_iterator(root))


def _schema_op(dag, op_type):
    """Propagate schemas over `dag` and return its single op of `op_type`."""
    found = [o for o in _logical_ops(dag) if isinstance(o, op_type)]
    assert len(found) == 1, f"expected exactly one {op_type.__name__}, got {len(found)}"
    return found[0]


def _stub(schema, output_type=OutputType.UNKNOWN):
    """A bare op standing in for an input that already carries `schema`."""
    op = Op()
    op.output_schema = schema
    op.output_type = output_type
    return op


class TestSchemaPropagation(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})

    # --- source ----------------------------------------------------------
    def test_source_schema_from_frame(self):
        src = _schema_op(st.as_data_op(self.df).drop(columns=["z"]), DataSourceOp)
        self.assertEqual(["x", "y", "z"], list(src.output_schema.keys()))

    def test_source_keeps_exact_dtype_for_typed_columns(self):
        op = DataSourceOp(data=pd.DataFrame({"i": [1, 2], "f": [1.5, 2.5]}))
        op.propagate_output_schema()
        self.assertEqual(pl.Int64, op.output_schema["i"])
        self.assertEqual(pl.Float64, op.output_schema["f"])

    def test_source_object_column_keeps_name_but_unknown_dtype(self):
        # an object column is element-typed only by scanning; a sample can be
        # confidently wrong, so the name is kept but the dtype is left Unknown.
        df = pd.DataFrame({"i": [1, 2, 3], "obj": pd.Series([1, 2, "x"], dtype=object)})
        op = DataSourceOp(data=df)
        op.propagate_output_schema()
        self.assertEqual(["i", "obj"], list(op.output_schema.keys()))
        self.assertEqual(pl.Int64, op.output_schema["i"])
        self.assertEqual(pl.Unknown, op.output_schema["obj"])

    def test_source_polars_frame_keeps_exact_schema(self):
        op = DataSourceOp(data=pl.DataFrame({"a": [1], "b": ["x"]}))
        op.propagate_output_schema()
        self.assertEqual(pl.Int64, op.output_schema["a"])
        self.assertEqual(pl.String, op.output_schema["b"])

    def test_csv_source_names_exact_dtypes_unknown(self):
        # Write via the path (not an open handle) so pandas controls newline
        # handling; a text-mode handle on Windows would translate \n -> \r\n and
        # leave a stray \r on the last column name.
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            self.df.to_csv(path, index=False)
            op = DataSourceOp(file_path=path, _format="csv")
            op.propagate_output_schema()
            self.assertEqual(["x", "y", "z"], list(op.output_schema.keys()))
            # dtypes need a full-file scan to be safe -> left Unknown.
            self.assertTrue(all(dt == pl.Unknown for dt in op.output_schema.values()))
        finally:
            os.unlink(path)

    def test_csv_source_honours_reader_options_that_change_the_columns(self):
        # read_kwargs are pandas reader options, and several of them change the
        # resulting column set. Each case asserts against what pandas actually
        # produces, so the rule can't drift from the reader it models.
        cases = [
            ({"sep": ";"}, {"sep": ";"}),
            ({"header": None}, {"header": False}),
            ({"sep": ";", "usecols": ["y"]}, {"sep": ";"}),
            ({"sep": ";", "index_col": 0}, {"sep": ";"}),
        ]
        for read_kwargs, write_kwargs in cases:
            with self.subTest(read_kwargs=read_kwargs), csv_file(self.df, **write_kwargs) as path:
                expected = list(pd.read_csv(path, **read_kwargs).columns)
                op = DataSourceOp(file_path=path, _format="csv", read_kwargs=read_kwargs)
                op.propagate_output_schema()
                self.assertEqual(expected, list(op.output_schema.keys()))

    def test_csv_source_header_none_keeps_pandas_integer_labels(self):
        # header=None makes pandas name the columns 0..n-1; the schema keeps those
        # labels verbatim rather than stringifying them.
        with csv_file(self.df, header=False) as path:
            op = DataSourceOp(file_path=path, _format="csv", read_kwargs={"header": None})
            op.propagate_output_schema()
            self.assertEqual([0, 1, 2], list(op.output_schema.keys()))

    def test_csv_source_graph_fed_reader_option_is_unknown(self):
        # a graph-fed option is still an OperandRef at plan time, so the column
        # set it implies isn't knowable.
        with csv_file(self.df) as path:
            op = DataSourceOp(file_path=path, _format="csv",
                              read_kwargs={"sep": OperandRef(0)})
            op.propagate_output_schema()
            self.assertIsNone(op.output_schema)

    def test_non_csv_source_format_is_unknown(self):
        op = DataSourceOp(file_path="/some/file.parquet", _format="parquet")
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_csv_source_unreadable_path_falls_back_to_unknown(self):
        op = DataSourceOp(file_path="/no/such/file.csv", _format="csv")
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_graph_fed_path_is_unknown(self):
        op = DataSourceOp(file_path=OperandRef(0), _format="csv")
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_in_memory_conversion_failure_falls_back_to_unknown(self):
        # an unconvertible pandas frame (e.g. an exotic/extension dtype) must not
        # crash optimize(); the pandas->polars conversion is caught and the schema
        # falls back to unknown.
        op = DataSourceOp(data=self.df)
        with mock.patch("polars.from_pandas", side_effect=Exception("boom")):
            op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    # --- column-changing ops --------------------------------------------
    def test_drop_removes_columns(self):
        op = _schema_op(st.as_data_op(self.df).drop(columns=["z"]), DropOp)
        self.assertEqual(["x", "y"], list(op.output_schema.keys()))

    def test_drop_positional_axis1_removes_columns(self):
        # positional form `df.drop(labels, axis=1)`: labels are columns.
        op = DropOp(args=[["z"]], kwargs={"axis": 1})
        op.inputs = [_stub(pl.Schema({"x": pl.Int64, "y": pl.Int64, "z": pl.Int64}))]
        op.propagate_output_schema()
        self.assertEqual(["x", "y"], list(op.output_schema.keys()))

    def test_row_drop_keeps_every_column(self):
        # `drop` defaults to axis=0, i.e. dropping *rows* by index label, which
        # leaves the column set untouched. Ground truth from pandas per spelling.
        frame = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        schema = pl.Schema({"x": pl.Int64, "y": pl.Int64})
        cases = [
            (dict(args=[[0]], kwargs={}), dict(labels=[0])),
            (dict(args=[[0]], kwargs={"axis": 0}), dict(labels=[0], axis=0)),
            (dict(args=[[0]], kwargs={"axis": "index"}), dict(labels=[0], axis="index")),
            (dict(args=[], kwargs={"index": [0]}), dict(index=[0])),
        ]
        for op_kwargs, drop_kwargs in cases:
            with self.subTest(drop_kwargs=drop_kwargs):
                op = DropOp(**op_kwargs)
                op.inputs = [_stub(schema)]
                op.propagate_output_schema()
                self.assertEqual(list(frame.drop(**drop_kwargs).columns),
                                 list(op.output_schema.keys()))

    def test_drop_with_a_string_column_axis_removes_columns(self):
        op = DropOp(args=[["y"]], kwargs={"axis": "columns"})
        op.inputs = [_stub(pl.Schema({"x": pl.Int64, "y": pl.Int64}))]
        op.propagate_output_schema()
        self.assertEqual(["x"], list(op.output_schema.keys()))

    def test_drop_with_a_graph_fed_axis_is_unknown(self):
        # the axis decides whether this touches the columns at all.
        op = DropOp(args=[["y"]], kwargs={"axis": OperandRef(1)})
        op.inputs = [_stub(pl.Schema({"x": pl.Int64, "y": pl.Int64}))]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_assign_adds_unknown_typed_column(self):
        op = _schema_op(st.as_data_op(self.df).assign(w=1), AssignMapOp)
        self.assertEqual(["x", "y", "z", "w"], list(op.output_schema.keys()))
        self.assertEqual(pl.Unknown, op.output_schema["w"])

    def test_rename_remaps_columns(self):
        op = _schema_op(st.as_data_op(self.df).rename(columns={"x": "a"}), MetadataOp)
        self.assertEqual(["a", "y", "z"], list(op.output_schema.keys()))

    def test_rename_via_axis1_mapper_remaps_columns(self):
        # rename(mapper, axis=1) is the positional equivalent of columns=mapper.
        op = MetadataOp(func="rename", kwargs={"mapper": {"x": "a"}, "axis": 1})
        op.inputs = [_stub(pl.Schema({"x": pl.Int64, "y": pl.Int64, "z": pl.Int64}))]
        op.propagate_output_schema()
        self.assertEqual(["a", "y", "z"], list(op.output_schema.keys()))

    def test_non_rename_metadata_is_unknown(self):
        # only `rename` is modelled; any other metadata op falls back to unknown.
        op = MetadataOp(func="reset_index")
        op.inputs = [_stub(pl.Schema({"x": pl.Int64, "y": pl.Int64}))]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_single_column_selection_is_one_column_schema(self):
        op = _schema_op(st.as_data_op(self.df)["x"], ColumnProjectionOp)
        self.assertEqual(["x"], list(op.output_schema.keys()))

    def test_multi_column_projection_selects_subschema(self):
        op = _schema_op(st.as_data_op(self.df)[["x", "y"]], ColumnProjectionOp)
        self.assertEqual(["x", "y"], list(op.output_schema.keys()))

    def test_row_slice_preserves_schema(self):
        # a slice key (`df[1:3]`) selects rows, so all columns are kept.
        op = GetItemOp(key=slice(1, 3))
        op.inputs = [_stub(pl.Schema({"x": pl.Int64, "y": pl.Int64}), OutputType.FRAME)]
        op.propagate_output_schema()
        self.assertEqual(["x", "y"], list(op.output_schema.keys()))

    def test_non_string_label_key_is_unknown(self):
        # a key that isn't string column labels (e.g. positional ints) can't be
        # resolved to named columns -> unknown.
        op = GetItemOp(key=[0, 1])
        op.inputs = [_stub(pl.Schema({"x": pl.Int64, "y": pl.Int64}), OutputType.FRAME)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_row_mask_preserves_schema(self):
        data = st.as_data_op(self.df)
        op = _schema_op(data[data["x"] > 1], SelectionOp)
        self.assertEqual(["x", "y", "z"], list(op.output_schema.keys()))

    def test_row_wise_dropna_preserves_schema(self):
        frame = pd.DataFrame({"a": [1.0, 2.0], "b": [None, 3.0]})
        for kwargs in ({}, {"axis": 0}, {"axis": "index"}):
            with self.subTest(kwargs=kwargs):
                op = _schema_op(st.as_data_op(frame).dropna(**kwargs), SelectionOp)
                self.assertEqual(["a", "b"], list(op.output_schema.keys()))

    def test_column_wise_dropna_is_unknown(self):
        # `dropna(axis=1)` drops the columns that contain nulls, so it is not a
        # relational selection at all, and which columns go is value-dependent.
        frame = pd.DataFrame({"a": [1.0, 2.0], "b": [None, 3.0]})
        self.assertEqual(["a"], list(frame.dropna(axis=1).columns))
        for kwargs in ({"axis": 1}, {"axis": "columns"}):
            with self.subTest(kwargs=kwargs):
                op = _schema_op(st.as_data_op(frame).dropna(**kwargs), SelectionOp)
                self.assertIsNone(op.output_schema)

    def test_selection_without_a_source_edge_is_unknown(self):
        # a detached selection has no input to pass a schema through.
        op = SelectionOp(kind=SelectionKind.HEAD)
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_duplicate_projection_labels_are_unknown(self):
        # `df[["a", "a"]]` really yields two `a` columns; a Schema cannot express
        # that, and narrowing it to one would understate the output.
        self.assertEqual(["a", "a"],
                         list(pd.DataFrame({"a": [1], "b": [2]})[["a", "a"]].columns))
        op = GetItemOp(key=["a", "a"])
        op.inputs = [_stub(pl.Schema({"a": pl.Int64, "b": pl.Int64}), OutputType.FRAME)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_rename_collision_is_unknown(self):
        # two columns renamed onto one label: pandas keeps both.
        self.assertEqual(["c", "c"], list(pd.DataFrame({"a": [1], "b": [2]})
                                          .rename(columns={"a": "c", "b": "c"}).columns))
        op = MetadataOp(func="rename", kwargs={"columns": {"a": "c", "b": "c"}})
        op.inputs = [_stub(pl.Schema({"a": pl.Int64, "b": pl.Float64}))]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_index_only_rename_keeps_the_columns(self):
        # a rename aimed at the index leaves the column names untouched.
        frame = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        schema = pl.Schema({"x": pl.Int64, "y": pl.Int64})
        for op_kwargs, rename_kwargs in [
            (dict(kwargs={"index": {0: 9}}), dict(index={0: 9})),
            (dict(args=[{0: 9}], kwargs={}), dict(mapper={0: 9})),
        ]:
            with self.subTest(rename_kwargs=rename_kwargs):
                op = MetadataOp(func="rename", **op_kwargs)
                op.inputs = [_stub(schema)]
                op.propagate_output_schema()
                self.assertEqual(list(frame.rename(**rename_kwargs).columns),
                                 list(op.output_schema.keys()))

    def test_rename_with_a_graph_fed_axis_is_unknown(self):
        op = MetadataOp(func="rename", args=[{"x": "a"}], kwargs={"axis": OperandRef(1)})
        op.inputs = [_stub(pl.Schema({"x": pl.Int64}))]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_score_candidates_schema_matches_the_table_it_builds(self):
        # `process` builds pl.DataFrame({"id": names, "scores": metric values}),
        # plus "vals" when emit_predictions.
        metric = Metric(name="acc", fn=lambda y_true, values: 0.5)
        real = pl.DataFrame({"id": ["c0", "c1"], "scores": [0.5, 0.7]}).schema

        op = ScoreCandidatesOp(candidate_names=["c0", "c1"], metric=metric)
        op.propagate_output_schema()
        self.assertEqual(real, op.output_schema)

        op = ScoreCandidatesOp(candidate_names=["c0", "c1"], metric=metric,
                               emit_predictions=True)
        op.propagate_output_schema()
        self.assertEqual(["id", "scores", "vals"], list(op.output_schema.keys()))
        self.assertEqual(pl.Unknown, op.output_schema["vals"])

    def test_projection_of_absent_column_is_unknown(self):
        # selecting a column the input schema doesn't carry can't be resolved
        # statically -> unknown (rather than an empty/partial schema).
        op = GetItemOp(key=["x", "missing"])
        op.inputs = [_stub(pl.Schema({"x": pl.Int64, "y": pl.Int64}), OutputType.FRAME)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_graph_fed_non_series_key_is_unknown(self):
        # a graph-fed key that isn't a series-shaped mask (e.g. a computed column
        # selector) can't be resolved statically -> unknown, not all columns.
        op = GetItemOp(key=OperandRef(1))
        op.inputs = [_stub(pl.Schema({"x": pl.Int64, "y": pl.Int64}), OutputType.FRAME),
                     _stub(None, OutputType.UNKNOWN)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_graph_fed_series_mask_preserves_schema(self):
        # a graph-fed SERIES key is a boolean row mask -> keeps all columns.
        op = GetItemOp(key=OperandRef(1))
        op.inputs = [_stub(pl.Schema({"x": pl.Int64, "y": pl.Int64}), OutputType.FRAME),
                     _stub(pl.Schema({"x": pl.Boolean}), OutputType.SERIES)]
        op.propagate_output_schema()
        self.assertEqual(["x", "y"], list(op.output_schema.keys()))

    # --- multi-input ops -------------------------------------------------
    def test_merge_collapses_shared_key(self):
        left = pd.DataFrame({"k": [1, 2], "a": [1, 2]})
        right = pd.DataFrame({"k": [1, 2], "b": [3, 4]})
        op = _schema_op(st.as_data_op(left).merge(st.as_data_op(right), on="k"), JoinOp)
        self.assertEqual(["k", "a", "b"], list(op.output_schema.keys()))

    def test_join_overlap_gets_suffixes(self):
        op = JoinOp(how="inner", left_on="k", right_on="k", suffixes=("_x", "_y"))
        op.inputs = [_stub(pl.Schema({"k": pl.Int64, "v": pl.Int64})),
                     _stub(pl.Schema({"k": pl.Int64, "v": pl.Int64}))]
        op.propagate_output_schema()
        # shared key `k` collapses; the overlapping non-key `v` is suffixed on both sides.
        self.assertEqual(["k", "v_x", "v_y"], list(op.output_schema.keys()))

    def test_merge_different_key_names_keeps_both(self):
        # left_on/right_on with different names: pandas keeps both key columns.
        left = pd.DataFrame({"a": [1, 2], "v": [1, 2]})
        right = pd.DataFrame({"b": [1, 2], "w": [3, 4]})
        op = _schema_op(
            st.as_data_op(left).merge(st.as_data_op(right), left_on="a", right_on="b"),
            JoinOp)
        self.assertEqual(["a", "v", "b", "w"], list(op.output_schema.keys()))

    def test_index_join_suffixes_overlapping_columns(self):
        # an index-based join has no left_on/right_on keys: every overlapping
        # column is suffixed on both sides, the rest are kept as-is.
        op = JoinOp(how="left", left_index=True, right_index=True, suffixes=("_l", "_r"))
        op.inputs = [_stub(pl.Schema({"a": pl.Int64, "v": pl.Int64})),
                     _stub(pl.Schema({"b": pl.Int64, "v": pl.Int64}))]
        op.propagate_output_schema()
        self.assertEqual(["a", "v_l", "b", "v_r"], list(op.output_schema.keys()))

    def test_concat_unions_columns(self):
        op = ConcatOp(first=OperandRef(0), others=[OperandRef(1)], axis=1)
        op.inputs = [_stub(pl.Schema({"a": pl.Int64})), _stub(pl.Schema({"b": pl.String}))]
        op.propagate_output_schema()
        self.assertEqual(["a", "b"], list(op.output_schema.keys()))

    def test_concat_includes_inline_constant_operand(self):
        # an operand that was a literal frame is stored in `others`, not as an
        # input edge, so reading `inputs` would silently drop its columns.
        op = ConcatOp(first=OperandRef(0), others=[pd.DataFrame({"c": [3]})], axis=1)
        op.inputs = [_stub(pl.Schema({"a": pl.Int64, "b": pl.Int64}))]
        op.propagate_output_schema()
        self.assertEqual(["a", "b", "c"], list(op.output_schema.keys()))

    def test_concat_of_only_constant_operands_is_not_an_empty_schema(self):
        # with no graph-fed operand there are no inputs at all; the result must be
        # the constants' columns, never a *known* zero-column schema.
        op = ConcatOp(first=pd.DataFrame({"c": [1]}),
                      others=[pd.DataFrame({"d": [2]})], axis=1)
        op.inputs = []
        op.propagate_output_schema()
        self.assertEqual(["c", "d"], list(op.output_schema.keys()))

    def test_concat_graph_fed_axis_is_unknown(self):
        # the axis selects the dtype rule, so an unresolvable axis is unknown.
        op = ConcatOp(first=OperandRef(0), others=[OperandRef(1)], axis=OperandRef(2))
        op.inputs = [_stub(pl.Schema({"a": pl.Int64})),
                     _stub(pl.Schema({"a": pl.Int64})), _stub(None)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_row_concat_drops_dtype_of_column_absent_from_an_operand(self):
        # ground truth: pandas null-fills the missing column, widening it to float.
        frames = [pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2], "b": [3]})]
        actual = pd.concat(frames).dtypes
        self.assertEqual("int64", str(actual["a"]))
        self.assertEqual("float64", str(actual["b"]))

        op = ConcatOp(first=OperandRef(0), others=[OperandRef(1)], axis=0)
        op.inputs = [_stub(pl.Schema({"a": pl.Int64})),
                     _stub(pl.Schema({"a": pl.Int64, "b": pl.Int64}))]
        op.propagate_output_schema()
        self.assertEqual(pl.Int64, op.output_schema["a"])     # in every operand
        self.assertEqual(pl.Unknown, op.output_schema["b"])   # widened by pandas

    def test_row_concat_drops_dtype_when_operands_disagree(self):
        op = ConcatOp(first=OperandRef(0), others=[OperandRef(1)], axis=0)
        op.inputs = [_stub(pl.Schema({"a": pl.Int64})), _stub(pl.Schema({"a": pl.Float64}))]
        op.propagate_output_schema()
        self.assertEqual(pl.Unknown, op.output_schema["a"])

    def test_column_concat_drops_every_dtype(self):
        # ground truth: a misaligned index null-fills *both* columns, and index
        # alignment can't be known statically -- so no dtype survives.
        actual = pd.concat([pd.DataFrame({"a": [1]}),
                            pd.DataFrame({"b": [2]}, index=[5])], axis=1).dtypes
        self.assertEqual(["float64", "float64"], [str(dt) for dt in actual])

        op = ConcatOp(first=OperandRef(0), others=[OperandRef(1)], axis=1)
        op.inputs = [_stub(pl.Schema({"a": pl.Int64})), _stub(pl.Schema({"b": pl.Int64}))]
        op.propagate_output_schema()
        self.assertEqual(["a", "b"], list(op.output_schema.keys()))
        self.assertTrue(all(dt == pl.Unknown for dt in op.output_schema.values()))

    def test_column_concat_with_a_shared_name_is_unknown(self):
        # pandas keeps both columns; a Schema can't express a duplicate name.
        self.assertEqual(["x", "x"], list(pd.concat(
            [pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]})], axis=1).columns))

        op = ConcatOp(first=OperandRef(0), others=[OperandRef(1)], axis=1)
        op.inputs = [_stub(pl.Schema({"x": pl.Int64})), _stub(pl.Schema({"x": pl.Int64}))]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    # --- split (X, y fan-out) -------------------------------------------
    def test_split_outputs_carry_their_input_schema(self):
        x = _stub(pl.Schema({"a": pl.Int64, "b": pl.Int64}), OutputType.FRAME)
        y = _stub(pl.Schema({"target": pl.Int64}), OutputType.FRAME)
        split = SplitOp(inputs=[x, y])
        out_x = SplitOutput(inputs=[split], is_x=True)
        out_y = SplitOutput(inputs=[split], is_x=False)

        # the split itself is a structural fan-out, not a single frame.
        split.propagate_output_schema()
        self.assertIsNone(split.output_schema)

        # each output keeps the columns of its matching split input.
        out_x.propagate_output_schema()
        out_y.propagate_output_schema()
        self.assertEqual(["a", "b"], list(out_x.output_schema.keys()))
        self.assertEqual(["target"], list(out_y.output_schema.keys()))

    def test_split_output_unknown_input_propagates(self):
        x = _stub(None, OutputType.FRAME)
        y = _stub(pl.Schema({"target": pl.Int64}), OutputType.FRAME)
        split = SplitOp(inputs=[x, y])
        out_x = SplitOutput(inputs=[split], is_x=True)
        out_x.propagate_output_schema()
        self.assertIsNone(out_x.output_schema)

    # --- unknown propagation --------------------------------------------
    def test_unknown_input_propagates(self):
        op = ConcatOp(first=OperandRef(0), others=[OperandRef(1)], axis=0)
        op.inputs = [_stub(pl.Schema({"a": pl.Int64})), _stub(None)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_projection_unknown_input_propagates(self):
        # a column projection over an unknown input schema stays unknown.
        op = GetItemOp(key=["x"])
        op.inputs = [_stub(None, OutputType.FRAME)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_assign_unknown_input_propagates(self):
        op = AssignOp(kwargs={"w": 1})
        op.inputs = [_stub(None)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_rename_unknown_input_propagates(self):
        op = MetadataOp(func="rename", kwargs={"columns": {"x": "a"}})
        op.inputs = [_stub(None)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_join_how_widens_the_nullable_sides_dtypes(self):
        # An unmatched row is null-filled, which widens int64 -> float64. Ground
        # truth comes from pandas for every `how`; the join key never gains nulls.
        left = pd.DataFrame({"id": [1, 2], "lv": [1, 2]})
        right = pd.DataFrame({"id": [1, 3], "rv": [9, 8]})
        left_schema = pl.Schema({"id": pl.Int64, "lv": pl.Int64})
        right_schema = pl.Schema({"id": pl.Int64, "rv": pl.Int64})

        for how in ("inner", "left", "right", "outer"):
            with self.subTest(how=how):
                actual = left.merge(right, on="id", how=how).dtypes
                op = JoinOp(how=how, left_on="id", right_on="id")
                op.inputs = [_stub(left_schema), _stub(right_schema)]
                op.propagate_output_schema()
                self.assertEqual(list(actual.index), list(op.output_schema.keys()))
                for name, real_dtype in actual.items():
                    # pandas kept int64 <=> the schema is allowed to keep Int64
                    if str(real_dtype) == "int64":
                        self.assertEqual(pl.Int64, op.output_schema[name], name)
                    else:
                        self.assertEqual(pl.Unknown, op.output_schema[name], name)

    def test_join_key_collapse_matches_pandas_for_every_key_spelling(self):
        # pandas pairs left_on/right_on *positionally* and collapses a pair only
        # when both names are equal; a bare merge() infers the keys from the common
        # columns. A set intersection models neither, so every spelling is checked
        # against the columns pandas actually produces.
        kv = pd.DataFrame({"k": [1, 2], "v": [1, 2]})
        kw = pd.DataFrame({"k": [1, 3], "w": [9, 8]})
        kv_s = pl.Schema({"k": pl.Int64, "v": pl.Int64})
        kw_s = pl.Schema({"k": pl.Int64, "w": pl.Int64})
        ab = pd.DataFrame({"a": [1], "b": [2]})
        ab_s = pl.Schema({"a": pl.Int64, "b": pl.Int64})

        cases = [
            # (JoinOp kwargs, merge kwargs, left frame/schema, right frame/schema)
            (dict(), dict(), kv, kw, kv_s, kw_s),
            (dict(left_on="k", right_on="k"), dict(on="k"), kv, kw, kv_s, kw_s),
            (dict(how="cross"), dict(how="cross"), kv, kw, kv_s, kw_s),
            (dict(left_index=True, right_index=True),
             dict(left_index=True, right_index=True), kv, kw, kv_s, kw_s),
            # shares both names, collapses neither: pandas emits a_x,b_x,a_y,b_y
            (dict(left_on=["a", "b"], right_on=["b", "a"]),
             dict(left_on=["a", "b"], right_on=["b", "a"]), ab, ab, ab_s, ab_s),
            (dict(left_on="a", right_on="b"),
             dict(left_on="a", right_on="b"), ab, ab, ab_s, ab_s),
        ]
        for join_kwargs, merge_kwargs, left, right, left_s, right_s in cases:
            with self.subTest(join_kwargs=join_kwargs):
                op = JoinOp(**join_kwargs)
                op.inputs = [_stub(left_s), _stub(right_s)]
                op.propagate_output_schema()
                self.assertEqual(list(left.merge(right, **merge_kwargs).columns),
                                 list(op.output_schema.keys()))

    def test_join_mixing_a_key_column_with_an_index_is_unknown(self):
        # pandas emits an extra key column for `left_on=... , right_index=True`
        # (k, k_x, k_y from two frames holding k), which isn't modelled.
        op = JoinOp(left_on="k", right_index=True)
        op.inputs = [_stub(pl.Schema({"k": pl.Int64, "v": pl.Int64})),
                     _stub(pl.Schema({"k": pl.Int64, "w": pl.Int64}))]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_join_graph_fed_keys_are_unknown(self):
        op = JoinOp(left_on=OperandRef(2), right_on="k")
        op.inputs = [_stub(pl.Schema({"k": pl.Int64})), _stub(pl.Schema({"k": pl.Int64}))]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_collapsed_key_keeps_its_dtype_only_when_both_sides_agree(self):
        # The collapsed key is the two key columns merged into one, and which side
        # wins depends on `how`: pandas keeps the left's int32 for inner/left but
        # takes the right's float64 for right/outer. No single dtype is correct for
        # a disagreeing pair, so it has to be Unknown.
        left = pd.DataFrame({"k": pd.Series([1, 2], dtype="int32"), "v": [1, 2]})
        right = pd.DataFrame({"k": pd.Series([1.0, 3.0], dtype="float64"), "w": [9, 8]})
        self.assertEqual(
            {"inner": "int32", "left": "int32", "right": "float64", "outer": "float64"},
            {how: str(left.merge(right, on="k", how=how).dtypes["k"])
             for how in ("inner", "left", "right", "outer")})

        for how in ("inner", "left", "right", "outer"):
            with self.subTest(how=how):
                op = JoinOp(how=how, left_on="k", right_on="k")
                op.inputs = [_stub(pl.Schema({"k": pl.Int32, "v": pl.Int64})),
                             _stub(pl.Schema({"k": pl.Float64, "w": pl.Int64}))]
                op.propagate_output_schema()
                self.assertEqual(pl.Unknown, op.output_schema["k"])

        # agreeing key dtypes still survive, even through an outer join
        op = JoinOp(how="outer", left_on="k", right_on="k")
        op.inputs = [_stub(pl.Schema({"k": pl.Int64, "v": pl.Int64})),
                     _stub(pl.Schema({"k": pl.Int64, "w": pl.Int64}))]
        op.propagate_output_schema()
        self.assertEqual(pl.Int64, op.output_schema["k"])

    def test_join_suffix_collision_is_unknown(self):
        # `.join()` defaults to empty suffixes; two overlapping columns would land
        # on the same label, which pandas rejects and a Schema cannot express.
        op = JoinOp(left_index=True, right_index=True, suffixes=("", ""))
        op.inputs = [_stub(pl.Schema({"a": pl.Int64})), _stub(pl.Schema({"a": pl.Int64}))]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_join_unknown_input_propagates(self):
        op = JoinOp(how="inner", left_on="k", right_on="k", suffixes=("_x", "_y"))
        op.inputs = [_stub(pl.Schema({"k": pl.Int64})), _stub(None)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_propagate_schema_disabled_leaves_schema_unset(self):
        # the pass is gated behind OptConfig.propagate_schema; with it off, no op
        # gets a schema (they keep the constructor default of None).
        dag = st.as_data_op(self.df).drop(columns=["z"])
        root = logical_optimize(dag, OptConfig(dataframe_ops=True, propagate_schema=False))
        ops = list(topological_iterator(root))
        self.assertTrue(all(o.output_schema is None for o in ops))

    def test_udf_falls_back_to_unknown(self):
        # apply() is a UDF: its output schema cannot be propagated -> None.
        from stratum.optimizer.logical._dataframe_ops import ApplyUDFOp
        op = _schema_op(st.as_data_op(self.df).apply(lambda c: c + 1), ApplyUDFOp)
        self.assertIsNone(op.output_schema)

    def test_unknown_poisons_downstream_known_op(self):
        # a UDF yields an unknown schema; a downstream drop (normally a known
        # column-changing op) can't recover columns from it -> stays unknown.
        dag = st.as_data_op(self.df).apply(lambda c: c + 1).drop(columns=["z"])
        op = _schema_op(dag, DropOp)
        self.assertIsNone(op.output_schema)

    # --- accessor projection (.dt.year, ...) ----------------------------
    def test_getattr_projection_keeps_columns_retypes(self):
        op = GetAttrProjectionOp(attr_name=["dt", "year"])
        op.inputs = [_stub(pl.Schema({"d": pl.Datetime("us")}))]
        op.propagate_output_schema()
        # column name is preserved; the dtype is no longer tracked.
        self.assertEqual(["d"], list(op.output_schema.keys()))
        self.assertEqual(pl.Unknown, op.output_schema["d"])

    def test_non_accessor_attribute_is_unknown(self):
        # make_frame_get_attr wraps *any* attribute of a frame-like input, but only
        # an accessor projection preserves the columns. `.T` transposes (its columns
        # become the old index labels), `.values` is an ndarray, `.shape` a tuple.
        self.assertEqual([0, 1], list(pd.DataFrame({"a": [1, 2], "b": [3, 4]}).T.columns))

        src = _stub(pl.Schema({"a": pl.Int64, "b": pl.Int64}))
        for attr_name in (["T"], ["values"], ["shape"], ["index"], ["columns"],
                          ["loc"], ["iloc"], ["dt"], ["dt", "year", "extra"]):
            with self.subTest(attr_name=attr_name):
                op = GetAttrProjectionOp(attr_name=attr_name, inputs=[src], outputs=[])
                op.propagate_output_schema()
                self.assertIsNone(op.output_schema)

    def test_getattr_projection_unknown_input_propagates(self):
        op = GetAttrProjectionOp(attr_name="dt")
        op.inputs = [_stub(None)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_dt_attributes_fanning_out_columns_are_unknown(self):
        # `.dt` is not uniformly column-preserving: these two are the whole-frame
        # members of the namespace, so the gate is per attribute, not per namespace.
        td = pd.Series(pd.to_timedelta(["1 days 2:03:04"]))
        self.assertEqual(7, len(td.dt.components.columns))
        self.assertEqual(3, len(pd.Series(pd.to_datetime(["2024-01-01"])).dt.isocalendar().columns))

        src = _stub(pl.Schema({"t": pl.Duration("us")}))
        for attr in ("components", "isocalendar"):
            with self.subTest(attr=attr):
                op = GetAttrProjectionOp(attr_name=["dt", attr], inputs=[src], outputs=[])
                op.propagate_output_schema()
                self.assertIsNone(op.output_schema)

    def test_dt_allow_list_holds_only_series_valued_properties(self):
        # Guards the allow-list against a member that isn't actually one-to-one.
        stamps = pd.Series(pd.to_datetime(["2024-03-05 01:02:03"]))
        spans = pd.Series(pd.to_timedelta(["1 days 2:03:04"]))
        for attr in GetAttrProjectionOp.ELEMENTWISE_ACCESSORS["dt"]:
            with self.subTest(attr=attr):
                values = [getattr(s.dt, attr, None) for s in (stamps, spans)]
                self.assertTrue(any(isinstance(v, pd.Series) for v in values))
                self.assertFalse(any(isinstance(v, pd.DataFrame) for v in values))

    # --- aggregation (groupby.agg) --------------------------------------
    @staticmethod
    def _aggregate(aggregations, grouping=(Col("g"),), **options):
        return AggregateOp(grouped=True, grouping=grouping,
                           aggregations=aggregations, options=options)

    def test_aggregate_named_entries_are_columns(self):
        # as_index defaults to True -> grouping key `g` goes to the index, not a column.
        op = self._aggregate(((None, AggExpr("sum", Col("v"))),))
        op.inputs = [_stub(pl.Schema({"g": pl.Int64, "v": pl.Int64}))]
        op.propagate_output_schema()
        self.assertEqual(["v"], list(op.output_schema.keys()))
        self.assertEqual(pl.Unknown, op.output_schema["v"])

    def test_aggregate_explicit_name_wins(self):
        op = self._aggregate((("total", AggExpr("sum", Col("v"))),))
        op.inputs = [_stub(pl.Schema({"g": pl.Int64, "v": pl.Int64}))]
        op.propagate_output_schema()
        self.assertEqual(["total"], list(op.output_schema.keys()))

    def test_aggregate_as_index_false_keeps_grouping_keys(self):
        op = self._aggregate(((None, AggExpr("sum", Col("v"))),), as_index=False)
        op.inputs = [_stub(pl.Schema({"g": pl.Int64, "v": pl.Int64}))]
        op.propagate_output_schema()
        self.assertEqual(["g", "v"], list(op.output_schema.keys()))

    def test_aggregate_as_index_false_skips_unknown_grouping_key(self):
        # as_index=False adds grouping keys as columns, but only those actually
        # present in the input schema; an unknown key is simply not emitted.
        op = self._aggregate(((None, AggExpr("sum", Col("v"))),), as_index=False)
        op.inputs = [_stub(pl.Schema({"v": pl.Int64}))]
        op.propagate_output_schema()
        self.assertEqual(["v"], list(op.output_schema.keys()))

    def test_aggregate_as_index_false_computed_key_is_unknown(self):
        # A computed key has no static column name to emit.
        op = self._aggregate(((None, AggExpr("sum", Col("v"))),),
                             grouping=(DtExpr(Col("d"), "year"),), as_index=False)
        op.inputs = [_stub(pl.Schema({"d": pl.Datetime("us"), "v": pl.Int64}))]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_aggregate_wildcard_is_unknown(self):
        # a bare function name aggregates every column -> not statically known.
        op = self._aggregate(((None, AggExpr("sum", AllCols())),))
        op.inputs = [_stub(pl.Schema({"g": pl.Int64, "v": pl.Int64}))]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_aggregate_unnamed_computed_child_has_a_logical_name(self):
        # Every backend uses the name fixed by the logical aggregation.
        child = BinOpExpr(operator.mul, Col("a"), Col("b"))
        op = self._aggregate(((None, AggExpr("sum", child)),))
        op.inputs = [_stub(pl.Schema({"g": pl.Int64, "a": pl.Int64, "b": pl.Int64}))]
        op.propagate_output_schema()
        self.assertEqual(pl.Schema({"_agg0": pl.Unknown}), op.output_schema)

    def test_aggregate_duplicate_output_name_is_unknown(self):
        op = self._aggregate((("v", AggExpr("sum", Col("v"))),
                              (None, AggExpr("mean", Col("v")))))
        op.inputs = [_stub(pl.Schema({"g": pl.Int64, "v": pl.Int64}))]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_aggregate_whole_object_reduction_is_unknown(self):
        op = AggregateOp(grouped=False, aggregations=((None, AggExpr("sum", AllCols())),))
        op.inputs = [_stub(pl.Schema({"v": pl.Int64}), OutputType.FRAME)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    # --- elementwise binary ops (df + 1, df["x"] > 0) -------------------
    def test_binop_frame_operand_keeps_schema(self):
        # an elementwise op over a frame keeps its columns (the dtype may change).
        op = BinOp(op=operator.gt, left=OperandRef(0), right=1)
        op.output_type = OutputType.SERIES
        op.inputs = [_stub(pl.Schema({"x": pl.Int64, "y": pl.Int64}), OutputType.FRAME)]
        op.propagate_output_schema()
        self.assertEqual(["x", "y"], list(op.output_schema.keys()))

    def test_binop_frame_frame_same_columns_keeps_schema(self):
        # df1 + df2 with identical columns keeps that shape.
        op = BinOp(op=operator.add, left=OperandRef(0), right=OperandRef(1))
        op.output_type = OutputType.FRAME
        op.inputs = [_stub(pl.Schema({"x": pl.Int64, "y": pl.Int64}), OutputType.FRAME),
                     _stub(pl.Schema({"x": pl.Int64, "y": pl.Int64}), OutputType.FRAME)]
        op.propagate_output_schema()
        self.assertEqual(["x", "y"], list(op.output_schema.keys()))

    def test_binop_frame_frame_differing_columns_is_unknown(self):
        # df1 + df2 over differing columns unions/aligns in pandas; guessing one
        # side's shape would be wrong, so fall back to unknown.
        op = BinOp(op=operator.add, left=OperandRef(0), right=OperandRef(1))
        op.output_type = OutputType.FRAME
        op.inputs = [_stub(pl.Schema({"x": pl.Int64, "y": pl.Int64}), OutputType.FRAME),
                     _stub(pl.Schema({"x": pl.Int64, "z": pl.Int64}), OutputType.FRAME)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_binop_dtype_matches_what_pandas_actually_returns(self):
        # The columns were always right; the *dtypes* used to be copied from the
        # operand, which lies for every operator that changes them. Each case
        # checks the schema's boolean-ness against pandas' real output dtype.
        frame = pd.DataFrame({"a": [1, 2]})
        schema = pl.Schema({"a": pl.Int64})
        for name in ("gt", "lt", "ge", "le", "eq", "ne", "add", "sub", "mul",
                     "truediv", "floordiv", "mod", "and_", "or_", "xor"):
            func = getattr(operator, name)
            with self.subTest(op=name):
                actual_is_bool = str(func(frame, 1).dtypes["a"]) == "bool"
                op = BinOp(op=func, left=OperandRef(0), right=1)
                op.output_type = OutputType.FRAME
                op.inputs = [_stub(schema, OutputType.FRAME)]
                op.propagate_output_schema()
                dtype = op.output_schema["a"]
                self.assertEqual(["a"], list(op.output_schema.keys()))
                if actual_is_bool:
                    self.assertEqual(pl.Boolean, dtype)
                else:
                    # not statically certain -> must not claim the operand's dtype
                    self.assertEqual(pl.Unknown, dtype)
                    self.assertNotEqual(pl.Int64, dtype)

    def test_logical_op_is_boolean_only_when_operands_are_boolean(self):
        # `&` is *logical* on booleans but *bitwise* on integers, so it only yields
        # a boolean when both operands already are.
        int_frame, bool_frame = pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [True, False]})
        self.assertEqual("int64", str((int_frame & int_frame).dtypes["a"]))
        self.assertEqual("bool", str((bool_frame & bool_frame).dtypes["a"]))

        for dtype, expected in [(pl.Int64, pl.Unknown), (pl.Boolean, pl.Boolean)]:
            with self.subTest(dtype=dtype):
                op = BinOp(op=operator.and_, left=OperandRef(0), right=OperandRef(1))
                op.output_type = OutputType.FRAME
                op.inputs = [_stub(pl.Schema({"a": dtype}), OutputType.FRAME),
                             _stub(pl.Schema({"a": dtype}), OutputType.FRAME)]
                op.propagate_output_schema()
                self.assertEqual(expected, op.output_schema["a"])

    def test_logical_op_with_a_constant_operand_is_unknown(self):
        # the constant's dtype isn't known, so boolean-ness can't be established.
        op = BinOp(op=operator.and_, left=OperandRef(0), right=1)
        op.output_type = OutputType.FRAME
        op.inputs = [_stub(pl.Schema({"a": pl.Boolean}), OutputType.FRAME)]
        op.propagate_output_schema()
        self.assertEqual(pl.Unknown, op.output_schema["a"])

    def test_unary_op_keeps_columns_and_derives_dtype(self):
        # UnaryOp had no rule at all, so `~mask` / `-df` lost the schema entirely.
        self.assertEqual("bool", str((~pd.DataFrame({"m": [True]})).dtypes["m"]))
        self.assertEqual("int64", str((-pd.DataFrame({"a": [1]})).dtypes["a"]))

        op = UnaryOp(op=operator.invert, operand=OperandRef(0))
        op.output_type = OutputType.SERIES
        op.inputs = [_stub(pl.Schema({"m": pl.Boolean}), OutputType.SERIES)]
        op.propagate_output_schema()
        self.assertEqual(pl.Boolean, op.output_schema["m"])

        op = UnaryOp(op=operator.neg, operand=OperandRef(0))
        op.output_type = OutputType.FRAME
        op.inputs = [_stub(pl.Schema({"a": pl.Int64}), OutputType.FRAME)]
        op.propagate_output_schema()
        self.assertEqual(["a"], list(op.output_schema.keys()))
        self.assertEqual(pl.Unknown, op.output_schema["a"])

    def test_unary_op_non_frame_output_is_unknown(self):
        op = UnaryOp(op=operator.neg, operand=OperandRef(0))
        op.output_type = OutputType.SCALAR
        op.inputs = [_stub(pl.Schema({"a": pl.Int64}), OutputType.FRAME)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_binop_non_frame_output_is_unknown(self):
        # a scalar-producing op (e.g. reduction) carries no column schema.
        op = BinOp(op=operator.add, left=OperandRef(0), right=OperandRef(1))
        op.output_type = OutputType.SCALAR
        op.inputs = [_stub(pl.Schema({"x": pl.Int64}), OutputType.FRAME), _stub(None)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_binop_unknown_frame_input_propagates(self):
        # frame output but the frame operand's schema is unknown -> unknown.
        op = BinOp(op=operator.add, left=OperandRef(0), right=1)
        op.output_type = OutputType.FRAME
        op.inputs = [_stub(None, OutputType.FRAME)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    # --- string accessor / choice ---------------------------------------
    def test_string_method_keeps_columns_retypes(self):
        # `.str.<method>` is elementwise: names survive, dtype varies by method
        # (contains->bool, len->int, lower->str) and the backends disagree, so it
        # is left Unknown.
        op = StringMethodOp(method="lower")
        op.inputs = [_stub(pl.Schema({"s": pl.String}))]
        op.propagate_output_schema()
        self.assertEqual(["s"], list(op.output_schema.keys()))
        self.assertEqual(pl.Unknown, op.output_schema["s"])

    def test_column_method_keeps_columns_and_derives_dtype(self):
        # a column method is shape-preserving, so only the dtype can change:
        # isin/notna are always boolean, the rest depend on the input.
        column = pd.Series([1, 2, 3])
        self.assertEqual("bool", str(column.isin([1]).dtype))
        self.assertEqual("bool", str(column.notna().dtype))
        self.assertEqual("float64", str(column.where(column > 1).dtype))

        for method, expected in [("isin", pl.Boolean), ("notna", pl.Boolean),
                                 ("fillna", pl.Unknown), ("clip", pl.Unknown),
                                 ("where", pl.Unknown), ("astype", pl.Unknown)]:
            with self.subTest(method=method):
                op = ColumnMethodOp(method=method)
                op.inputs = [_stub(pl.Schema({"x": pl.Int64}), OutputType.SERIES)]
                op.propagate_output_schema()
                self.assertEqual(["x"], list(op.output_schema.keys()))
                self.assertEqual(expected, op.output_schema["x"])

    def test_column_method_unknown_input_propagates(self):
        op = ColumnMethodOp(method="fillna")
        op.inputs = [_stub(None)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_string_method_unknown_input_propagates(self):
        op = StringMethodOp(method="lower")
        op.inputs = [_stub(None)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_string_method_fanning_out_columns_is_unknown(self):
        # These return a frame of new columns rather than a same-named column, so
        # reporting the operand's name would be a wrong name, not a coarse one.
        # `split`/`rsplit` do it under `expand=True`; `cat()` collapses to a str.
        for method in ("extract", "extractall", "get_dummies", "partition",
                       "rpartition", "split", "rsplit", "cat"):
            with self.subTest(method=method):
                op = StringMethodOp(method=method)
                op.inputs = [_stub(pl.Schema({"s": pl.String}))]
                op.propagate_output_schema()
                self.assertIsNone(op.output_schema)

    def test_string_method_one_to_one_methods_are_all_scalar(self):
        # Guards the allow-list against a member that isn't actually one-to-one.
        col = pd.Series(["a-b", "c-d"])
        for method in StringMethodOp.ONE_TO_ONE_METHODS:
            with self.subTest(method=method):
                func = getattr(col.str, method)
                result = None
                for args in ((), ("a",), (1,), ("a", "b")):
                    try:
                        result = func(*args)
                        break
                    except Exception:
                        continue
                self.assertIsInstance(result, pd.Series)

    def test_choice_is_unknown_by_design(self):
        # A choice picks one of several pipelines and sits at the end of the DAG,
        # so no operator downstream needs its schema. Unifying the outcomes'
        # schemas would be work with no consumer: it stays unknown even when every
        # outcome agrees.
        agreed = pl.Schema({"a": pl.Int64})
        op = ChoiceOp(outcome_names=[[("c", f"Opt{i}")] for i in range(2)])
        op.inputs = [_stub(agreed), _stub(agreed)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    # --- pass wiring ----------------------------------------------------
    def test_schema_survives_lowering_into_the_physical_plan(self):
        # Most families keep their schema for free, because implementation selection
        # rebinds op.__class__ in place. A lowering rule builds a *fresh* node, so
        # the source op would lose its schema without install_lowered copying it --
        # and only the full optimize() pipeline can show that.
        ops, *_ = optimize_full(st.as_data_op(self.df).drop(columns=["z"]), OptConfig())
        by_family = {type(o).__name__: o.output_schema for o in ops}
        self.assertTrue(any("InMemoryFrame" in n for n in by_family), by_family)
        source = next(s for n, s in by_family.items() if "InMemoryFrame" in n)
        drop = next(s for n, s in by_family.items() if "DropOp" in n)
        self.assertEqual(["x", "y", "z"], list(source.keys()))
        self.assertEqual(["x", "y"], list(drop.keys()))

    def test_every_plan_node_answers_to_the_pass(self):
        # the default rule lives on IRNode, not Op, so physical-only nodes (which
        # are PhysicalOps, not Ops) respond too.
        ops, *_ = optimize_full(st.as_data_op(self.df).drop(columns=["z"]), OptConfig())
        for op in ops:
            with self.subTest(op=type(op).__name__):
                self.assertTrue(hasattr(op, "propagate_output_schema"))

    # --- ops without a propagation rule fall back to unknown ------------
    def test_generic_op_falls_back_to_unknown(self):
        # the base Op (any non-frame / no-rule op) yields the unknown schema.
        op = Op()
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_datetime_conversion_is_datetime_typed(self):
        dt_df = pd.DataFrame({"d": ["2025-11-01", "2025-11-02"]})
        date = st.as_data_op(dt_df)["d"].skb.apply_func(pd.to_datetime)
        op = _schema_op(date, DatetimeConversionOp)
        self.assertEqual(["d"], list(op.output_schema.keys()))
        # The column *name* survives, but the Datetime unit is backend-dependent
        # (pandas -> ns, polars -> us) and pl.Schema has no unit-less Datetime, so
        # the dtype cannot be committed to at the logical layer.
        self.assertEqual(pl.Unknown, op.output_schema["d"])

    def test_datetime_conversion_over_a_frame_is_unknown(self):
        # `pd.to_datetime(frame)` is the assembly form: it reads year/month/day and
        # fans them in to one unnamed Series, so the input's names are not the
        # output's. Any other frame raises, so there is no frame-in/frame-out form.
        assembled = pd.to_datetime(pd.DataFrame({"year": [2024], "month": [1], "day": [2]}))
        self.assertIsInstance(assembled, pd.Series)
        with self.assertRaises(ValueError):
            pd.to_datetime(pd.DataFrame({"d1": ["2024-01-01"], "d2": ["2025-02-02"]}))

        for output_type in (OutputType.FRAME, OutputType.UNKNOWN):
            with self.subTest(output_type=output_type):
                op = DatetimeConversionOp()
                op.inputs = [_stub(pl.Schema({"year": pl.Int64, "month": pl.Int64,
                                              "day": pl.Int64}), output_type)]
                op.propagate_output_schema()
                self.assertIsNone(op.output_schema)

    def test_datetime_unit_is_not_expressible_in_a_schema(self):
        # Pins the reason the rule can't name a Datetime unit: a unit-less
        # pl.Datetime is rejected outright, and the two backends disagree on the
        # unit, so there is no correct single answer at the logical layer.
        with self.assertRaises(TypeError):
            pl.Schema({"d": pl.Datetime})
        self.assertNotEqual(pl.Datetime("ns"), pl.Datetime("us"))


if __name__ == "__main__":
    unittest.main()
