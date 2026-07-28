"""Shared test helpers for the dataframe-IR test suite, plus tests for the ops
that live in ``_dataframe_ops`` itself (the re-export hub).

The op-specific tests live alongside their module: ``test_source_ops``,
``test_projection_ops``, ``test_join_ops``, ``test_aggregation_ops``,
``test_split_ops`` and ``test_selection_ops``. They (and ``test_type_inference``)
import the helpers below from here, mirroring how ``_dataframe_ops`` re-exports the
per-category ops.
"""
import unittest
from contextlib import contextmanager

import polars as pl
from stratum._config import FLAGS
from stratum.optimizer._optimize import OptConfig, optimize as optimize_
from stratum.optimizer.ir._dataframe_ops import ConcatOp
from stratum.optimizer.ir._ops import OperandRef, OutputType, Op
from stratum.optimizer.physical import FlagBasedSelector
from stratum.optimizer.physical._impl_selection import bind_op
from stratum.optimizer.physical._plan_context import PlanContext


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
    """Wire `values` as op.inputs, bind the op's physical impl per the current
    flags, and run its ``process``.

    Binding mirrors what the optimizer's selection pass does: a migrated op
    (e.g. ``ConcatOp``) is swapped to its backend-specific physical impl chosen
    from ``FLAGS``; an un-migrated op keeps its own ``process``. This lets the
    same ``run_op(SomeOp(...), df)`` tests exercise the physical impls without
    each test having to construct the concrete class itself.
    """
    op.inputs = [_inp(v) for v in values]
    bind_op(op, PlanContext.from_flags(), selector=FlagBasedSelector())
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
