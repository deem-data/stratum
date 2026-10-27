import unittest

import numpy as np
import pandas as pd

import stratum as st
from stratum.optimizer._optimize import OptConfig
from stratum.optimizer.logical._dataframe_ops import (
    ApplyUDFOp, AttributeReadOp, ColumnProjectionOp, DatetimeConversionOp, DatetimeFieldOp)
from stratum.optimizer.logical._ops import GetItemOp, OutputType, BinOp
from stratum.optimizer.logical._source_ops import LoadNumpyOp
from tests._helpers import npy_file
from .test_dataframe_ops import optimize


class TestOutputTypeInference(unittest.TestCase):
    """`extract_dataframe_op` infers FRAME vs SERIES (and MATRIX) per op."""

    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        self.dt_df = pd.DataFrame({
            "datetime": ["2025-11-01 10:00:00",
                         "2025-11-02 15:30:00",
                         "2025-11-03 09:45:00"],
        })

    def _find(self, ops, op_type):
        return [o for o in ops if isinstance(o, op_type)]

    def _one(self, ops, op_type):
        found = self._find(ops, op_type)
        self.assertEqual(1, len(found), f"expected exactly one {op_type.__name__}")
        return found[0]

    def test_column_selection_is_series(self):
        # df["x"] on a frame is a single-column projection -> a SERIES-typed
        # ColumnProjectionOp (the GetItem is rewritten away).
        ops = optimize(st.as_data_op(self.df)["x"], OptConfig(dataframe_ops=True))
        projections = self._find(ops, ColumnProjectionOp)
        self.assertEqual(1, len(projections))
        self.assertIs(OutputType.SERIES, projections[0].output_type)

    def test_multi_column_projection_is_frame(self):
        # df[["x", "y"]] selects a sub-frame -> a FRAME-typed ColumnProjectionOp.
        ops = optimize(st.as_data_op(self.df)[["x", "y"]], OptConfig(dataframe_ops=True))
        projections = self._find(ops, ColumnProjectionOp)
        self.assertEqual(1, len(projections))
        self.assertIs(OutputType.FRAME, projections[0].output_type)

    def test_comparison_on_column_is_series(self):
        # df["x"] > 1 : the column is a SERIES, so the comparison is a SERIES too.
        ops = optimize(st.as_data_op(self.df)["x"] > 1, OptConfig(dataframe_ops=True))
        binops = self._find(ops, BinOp)
        self.assertEqual(1, len(binops))
        self.assertIs(OutputType.SERIES, binops[0].output_type)

    def test_npy_source_is_matrix(self):
        # An npy read is a LoadNumpyOp, which is a MATRIX.
        with npy_file(np.array([1, 2, 3])) as path:
            data = st.as_data_op(path).skb.apply_func(np.load)
            ops = optimize(data, OptConfig(dataframe_ops=True))
        sources = [o for o in ops if isinstance(o, LoadNumpyOp)]
        self.assertEqual(1, len(sources))
        self.assertIs(OutputType.MATRIX, sources[0].output_type)

    def test_frame_comparison_is_frame(self):
        # df > 1 : the operand is a whole frame -> the comparison is a FRAME.
        # (Arithmetic BinOps are consumed by the numeric path; comparisons stay.)
        ops = optimize(st.as_data_op(self.df) > 1, OptConfig(dataframe_ops=True))
        self.assertIs(OutputType.FRAME, self._one(ops, BinOp).output_type)

    def test_datetime_conversion_on_column_is_series(self):
        # X["datetime"] is a SERIES; pd.to_datetime over it produces a
        # DatetimeConversionOp that must stay a SERIES (not reset to FRAME).
        date = st.as_data_op(self.dt_df)["datetime"].skb.apply_func(
            pd.to_datetime, format="%Y-%m-%d %H:%M:%S")
        ops = optimize(date, OptConfig(dataframe_ops=True))
        self.assertIs(OutputType.SERIES, self._one(ops, DatetimeConversionOp).output_type)

    def test_getattr_after_datetime_conversion_is_series(self):
        # date.dt.year : a `.dt` accessor only exists on a SERIES, and `.year`
        # keeps it a SERIES -> the fused DatetimeFieldOp is a SERIES.
        date = st.as_data_op(self.dt_df)["datetime"].skb.apply_func(
            pd.to_datetime, format="%Y-%m-%d %H:%M:%S")
        ops = optimize(date.dt.year, OptConfig(dataframe_ops=True))
        self.assertIs(OutputType.SERIES, self._one(ops, DatetimeFieldOp).output_type)

    def test_getattr_on_frame_stays_frame(self):
        # `.T` is a frame-level attribute (unlike `.str`/`.dt`, which are
        # series-only), so a GetAttr on a frame stays a FRAME.
        ops = optimize(st.as_data_op(self.df).T, OptConfig(dataframe_ops=True))
        self.assertIs(OutputType.FRAME, self._one(ops, AttributeReadOp).output_type)

    def test_apply_on_column_is_series(self):
        # df["x"].apply(f) operates on a column -> SERIES.
        ops = optimize(st.as_data_op(self.df)["x"].apply(lambda v: v + 1),
                       OptConfig(dataframe_ops=True))
        self.assertIs(OutputType.SERIES, self._one(ops, ApplyUDFOp).output_type)

    def test_apply_on_frame_is_frame(self):
        # df.apply(f) operates on the whole frame -> FRAME.
        ops = optimize(st.as_data_op(self.df).apply(lambda col: col + 1),
                       OptConfig(dataframe_ops=True))
        self.assertIs(OutputType.FRAME, self._one(ops, ApplyUDFOp).output_type)

    def test_column_arithmetic_is_series_before_numeric_folding(self):
        # Numeric-op parsing runs *after* frame parsing, so during frame parsing
        # `df["x"] + df["y"]` is still a BinOp over columns (pandas/polars world,
        # not a matrix) -> SERIES. Disable numeric_ops to observe it pre-folding.
        data = st.as_data_op(self.df)
        ops = optimize(data["x"] + data["y"],
                       OptConfig(dataframe_ops=True, numeric_ops=False))
        self.assertIs(OutputType.SERIES, self._one(ops, BinOp).output_type)

    def test_two_axis_loc_with_a_column_list_is_frame(self):
        # `df.loc[mask, ["x"]]` restricts rows *and* columns; a column list keeps a
        # frame, exactly as the one-axis `df[["x"]]` does.
        data = st.as_data_op(self.df)
        ops = optimize(data.loc[data["x"] > 1, ["x"]], OptConfig(dataframe_ops=True))
        self.assertIs(OutputType.FRAME, self._one(ops, GetItemOp).output_type)

    def test_two_axis_loc_with_a_single_column_is_series(self):
        # `df.loc[mask, "y"]` extracts one column, so the column indexer -- not the
        # tuple-ness of the key -- decides: SERIES, not the FRAME above.
        data = st.as_data_op(self.df)
        ops = optimize(data.loc[data["x"] > 1, "y"], OptConfig(dataframe_ops=True))
        self.assertIs(OutputType.SERIES, self._one(ops, GetItemOp).output_type)

    def test_two_axis_iloc_with_a_single_position_is_series(self):
        # `.iloc` indexes the same two axes by position: one position is one column.
        data = st.as_data_op(self.df)
        ops = optimize(data.iloc[0:2, 0], OptConfig(dataframe_ops=True))
        self.assertIs(OutputType.SERIES, self._one(ops, GetItemOp).output_type)

    def test_two_axis_iloc_with_a_position_list_is_frame(self):
        data = st.as_data_op(self.df)
        ops = optimize(data.iloc[0:2, [0]], OptConfig(dataframe_ops=True))
        self.assertIs(OutputType.FRAME, self._one(ops, GetItemOp).output_type)

    def test_the_two_axis_rule_does_not_reach_a_plain_getitem(self):
        # A tuple key means "one indexer per axis" only under `.loc`/`.iloc`. On a
        # plain `df[...]` it is a single MultiIndex label, so the second element is
        # not a column indexer and must not be read as one. MultiIndex labels are
        # otherwise unhandled: this pins the untouched default, not a correct kind
        # (pandas returns a SERIES here).
        df = pd.DataFrame({("a", "x"): [1, 2], ("a", "y"): [3, 4]})
        ops = optimize(st.as_data_op(df)[("a", "x")], OptConfig(dataframe_ops=True))
        self.assertIs(OutputType.FRAME, self._one(ops, GetItemOp).output_type)
