import operator
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd
import polars as pl

import stratum as st
from stratum.optimizer._optimize import OptConfig
from stratum.optimizer.ir._dataframe_ops import (
    AggregateOp, AssignOp, AssignMapOp, ConcatOp, DataSourceOp, DatetimeConversionOp, DropOp,
    GetAttrProjectionOp, JoinOp, MetadataOp, SplitOp, SplitOutput, ColumnProjectionOp,
    SelectionOp)
from stratum.optimizer.ir._ops import BinOp, GetItemOp, Op, OperandRef, OutputType
from stratum.optimizer.physical._source_execs import InMemoryFrame
from stratum.tests.logical_optimizer.test_dataframe_ops import optimize


def _schema_op(dag, op_type):
    """Optimize `dag` (with frame ops + schema propagation) and return the single op of `op_type`."""
    ops = optimize(dag, OptConfig(dataframe_ops=True, propagate_schema=True))
    found = [o for o in ops if isinstance(o, op_type)]
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
        # Lowering replaces DataSourceOp with an InMemoryFrame physical op; the
        # schema is copied across so the source still carries it.
        src = _schema_op(st.as_data_op(self.df).drop(columns=["z"]), InMemoryFrame)
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

    def test_drop_row_axis_is_unknown(self):
        # a positional row drop (`df.drop(labels)`, axis defaults to 0) names no
        # columns statically, so the schema can't be resolved -> unknown.
        op = DropOp(args=[[0, 1]], kwargs={"axis": 0})
        op.inputs = [_stub(pl.Schema({"x": pl.Int64, "y": pl.Int64}))]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_assign_adds_unknown_typed_column(self):
        # Constant/foldable assign becomes AssignMapOp under the map extractor.
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
        ops = optimize(data[data["x"] > 1], OptConfig(dataframe_ops=True))
        row_sel = [o for o in ops if isinstance(o, SelectionOp)]
        self.assertEqual(1, len(row_sel))
        self.assertEqual(["x", "y", "z"], list(row_sel[0].output_schema.keys()))

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

    def test_join_unknown_input_propagates(self):
        op = JoinOp(how="inner", left_on="k", right_on="k", suffixes=("_x", "_y"))
        op.inputs = [_stub(pl.Schema({"k": pl.Int64})), _stub(None)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_propagate_schema_disabled_leaves_schema_unset(self):
        # the pass is gated behind OptConfig.propagate_schema; with it off, no op
        # gets a schema (they keep the constructor default of None).
        dag = st.as_data_op(self.df).drop(columns=["z"])
        ops = optimize(dag, OptConfig(dataframe_ops=True, propagate_schema=False))
        self.assertTrue(all(o.output_schema is None for o in ops))

    def test_udf_falls_back_to_unknown(self):
        # apply() is a UDF: its output schema cannot be propagated -> None.
        from stratum.optimizer.ir._dataframe_ops import ApplyUDFOp
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

    def test_getattr_projection_unknown_input_propagates(self):
        op = GetAttrProjectionOp(attr_name="dt")
        op.inputs = [_stub(None)]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    # --- aggregation (groupby.agg) --------------------------------------
    def test_aggregate_dict_spec_keys_are_columns(self):
        # as_index defaults to True -> grouping key `g` goes to the index, not a column.
        op = AggregateOp(grouping_attributes="g", aggregations={"v": "sum"})
        op.inputs = [_stub(pl.Schema({"g": pl.Int64, "v": pl.Int64}))]
        op.propagate_output_schema()
        self.assertEqual(["v"], list(op.output_schema.keys()))
        self.assertEqual(pl.Unknown, op.output_schema["v"])

    def test_aggregate_as_index_false_keeps_grouping_keys(self):
        op = AggregateOp(grouping_attributes="g", aggregations={"v": "sum"},
                         groupby_kwargs={"as_index": False})
        op.inputs = [_stub(pl.Schema({"g": pl.Int64, "v": pl.Int64}))]
        op.propagate_output_schema()
        self.assertEqual(["g", "v"], list(op.output_schema.keys()))

    def test_aggregate_as_index_false_skips_unknown_grouping_key(self):
        # as_index=False adds grouping keys as columns, but only those actually
        # present in the input schema; an unknown key is simply not emitted.
        op = AggregateOp(grouping_attributes="g", aggregations={"v": "sum"},
                         groupby_kwargs={"as_index": False})
        op.inputs = [_stub(pl.Schema({"v": pl.Int64}))]
        op.propagate_output_schema()
        self.assertEqual(["v"], list(op.output_schema.keys()))

    def test_aggregate_string_spec_is_unknown(self):
        # a bare function name aggregates every column -> not statically known.
        op = AggregateOp(grouping_attributes="g", aggregations="sum")
        op.inputs = [_stub(pl.Schema({"g": pl.Int64, "v": pl.Int64}))]
        op.propagate_output_schema()
        self.assertIsNone(op.output_schema)

    def test_aggregate_list_value_spec_is_unknown(self):
        # a list value produces MultiIndex columns -> not representable -> unknown.
        op = AggregateOp(grouping_attributes="g", aggregations={"v": ["sum", "mean"]})
        op.inputs = [_stub(pl.Schema({"g": pl.Int64, "v": pl.Int64}))]
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
        self.assertIsInstance(op.output_schema["d"], pl.Datetime)

    def test_datetime_conversion_multi_column_retypes_all(self):
        # to_datetime over a frame keeps every column name and retypes to Datetime.
        op = DatetimeConversionOp()
        op.inputs = [_stub(pl.Schema({"d1": pl.String, "d2": pl.String}))]
        op.propagate_output_schema()
        self.assertEqual(["d1", "d2"], list(op.output_schema.keys()))
        self.assertTrue(all(isinstance(dt, pl.Datetime) for dt in op.output_schema.values()))


if __name__ == "__main__":
    unittest.main()
