import unittest

import numpy as np
import pandas as pd

import stratum as st
from stratum.optimizer._optimize import OptConfig
from stratum.optimizer.ir._dataframe_ops import (
    ApplyUDFOp, DataSourceOp, DatetimeConversionOp, GetAttrProjectionOp)
from stratum.optimizer.ir._ops import GetItemOp, OutputType, BinOp
from stratum.tests.logical_optimizer.test_dataframe_ops import optimize, npy_file


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
        ops = optimize(st.as_data_op(self.df)["x"], OptConfig(dataframe_ops=True))
        getitems = self._find(ops, GetItemOp)
        self.assertEqual(1, len(getitems))
        self.assertIs(OutputType.SERIES, getitems[0].output_type)

    def test_multi_column_projection_is_frame(self):
        ops = optimize(st.as_data_op(self.df)[["x", "y"]], OptConfig(dataframe_ops=True))
        getitems = self._find(ops, GetItemOp)
        self.assertEqual(1, len(getitems))
        self.assertIs(OutputType.FRAME, getitems[0].output_type)

    def test_comparison_on_column_is_series(self):
        # df["x"] > 1 : the column is a SERIES, so the comparison is a SERIES too.
        ops = optimize(st.as_data_op(self.df)["x"] > 1, OptConfig(dataframe_ops=True))
        binops = self._find(ops, BinOp)
        self.assertEqual(1, len(binops))
        self.assertIs(OutputType.SERIES, binops[0].output_type)

    def test_npy_source_is_matrix(self):
        with npy_file(np.array([1, 2, 3])) as path:
            data = st.as_data_op(path).skb.apply_func(np.load)
            ops = optimize(data, OptConfig(dataframe_ops=True))
        sources = [o for o in ops if isinstance(o, DataSourceOp)]
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
        # keeps it a SERIES -> the fused GetAttrProjectionOp is a SERIES.
        date = st.as_data_op(self.dt_df)["datetime"].skb.apply_func(
            pd.to_datetime, format="%Y-%m-%d %H:%M:%S")
        ops = optimize(date.dt.year, OptConfig(dataframe_ops=True))
        self.assertIs(OutputType.SERIES, self._one(ops, GetAttrProjectionOp).output_type)

    def test_getattr_on_frame_stays_frame(self):
        # `.T` is a frame-level attribute (unlike `.str`/`.dt`, which are
        # series-only), so a GetAttr on a frame stays a FRAME.
        ops = optimize(st.as_data_op(self.df).T, OptConfig(dataframe_ops=True))
        self.assertIs(OutputType.FRAME, self._one(ops, GetAttrProjectionOp).output_type)

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
