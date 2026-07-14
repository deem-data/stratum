"""Shared test helpers for the dataframe-IR test suite, plus tests for the ops
that live in ``_dataframe_ops`` itself (the re-export hub).

The op-specific tests live alongside their module: ``test_source_ops``,
``test_projection_ops``, ``test_join_ops``, ``test_aggregation_ops``,
``test_split_ops`` and ``test_selection_ops``. They (and ``test_type_inference``)
import the helpers below from here, mirroring how ``_dataframe_ops`` re-exports the
per-category ops.
"""
import os
import tempfile
import unittest
from contextlib import contextmanager

import numpy as np
import polars as pl
from stratum._config import FLAGS
from stratum.optimizer._optimize import OptConfig, optimize as optimize_
from stratum.optimizer.ir._dataframe_ops import ConcatOp
from stratum.optimizer.ir._ops import OperandRef, OutputType, Op


def optimize(dag, conf=None, env=None):
    linearized_dag, *_ = optimize_(dag, conf, env)
    return linearized_dag


def _inp(val):
    op = Op()
    op.intermediate = val
    op.output_type = OutputType.FRAME
    return op


def _inputs_for(op):
    return [in_op.intermediate for in_op in op.inputs]


def run_op(op, *values, mode="fit_transform"):
    """Wire `values` as op.inputs (wrapped via `_inp`) and run `op.process`."""
    op.inputs = [_inp(v) for v in values]
    return op.process(mode, _inputs_for(op))


@contextmanager
def make_map_op(enabled=True):
    """Temporarily set `FLAGS.make_map_op`."""
    orig = FLAGS.make_map_op
    FLAGS.make_map_op = enabled
    try:
        yield
    finally:
        FLAGS.make_map_op = orig


@contextmanager
def force_polars(enabled=True):
    """Temporarily set `FLAGS.force_polars`."""
    orig = FLAGS.force_polars
    FLAGS.force_polars = enabled
    try:
        yield
    finally:
        FLAGS.force_polars = orig


@contextmanager
def csv_file(df, **to_csv_kwargs):
    """Write `df` to a temp .csv file and yield its path; cleaned up on exit."""
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp, index=False, **to_csv_kwargs)
    tmp.close()
    try:
        yield tmp.name
    finally:
        os.remove(tmp.name)


@contextmanager
def npy_file(arr):
    """Write `arr` to a temp .npy file and yield its path; cleaned up on exit."""
    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False, mode="wb")
    np.save(tmp, arr)
    tmp.close()
    try:
        yield tmp.name
    finally:
        os.remove(tmp.name)


@contextmanager
def parquet_file(df):
    """Write `df` to a temp .parquet file and yield its path; cleaned up on exit."""
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False, mode="wb")
    df.to_parquet(tmp.name)
    tmp.close()
    try:
        yield tmp.name
    finally:
        os.remove(tmp.name)


class PolarsTestCase(unittest.TestCase):
    """Base class that pins `FLAGS.force_polars=True` for every test."""

    def setUp(self):
        super().setUp()
        self._orig_force_polars = FLAGS.force_polars
        FLAGS.force_polars = True

    def tearDown(self):
        FLAGS.force_polars = self._orig_force_polars
        super().tearDown()


class TestConcatOpPolars(PolarsTestCase):
    def test_polars_concat(self):
        op = ConcatOp(first=OperandRef(0), others=[OperandRef(1)], axis=0)
        result = run_op(op, pl.DataFrame({"a": [1, 2]}), pl.DataFrame({"a": [3, 4]}))
        self.assertEqual(4, len(result))


if __name__ == "__main__":
    unittest.main()
