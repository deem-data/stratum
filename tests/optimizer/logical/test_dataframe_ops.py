"""Shared test helpers for the dataframe-IR test suite, plus tests for the ops
that live in ``_dataframe_ops`` itself (the re-export hub).

The op-specific tests live alongside their module: ``test_source_ops``,
``test_projection_ops``, ``test_join_ops``, ``test_aggregation_ops``,
``test_split_ops`` and ``test_selection_ops``. They (and ``test_type_inference``)
import the helpers below from here, mirroring how ``_dataframe_ops`` re-exports the
per-category ops.
"""
import copy
import unittest
from contextlib import contextmanager

import polars as pl
from stratum._config import FLAGS
from stratum.optimizer._optimize import OptConfig, optimize as optimize_
from stratum.optimizer.logical._dataframe_ops import ConcatOp
from stratum.optimizer.logical._ops import OperandRef, OutputType, Op
from stratum.optimizer.physical import FlagBasedSelector
from stratum.optimizer.physical._impl_selection import bind_op
from stratum.optimizer.physical._plan_context import PlanContext


#: Backend the helpers below plan against. Test-suite state, deliberately not a
#: `FLAGS` entry: which backend a plan runs on is decided by the implementation
#: selector, and pinning one is a testing tool rather than a user-facing knob
#: (users reach polars via `implementation_selector="greedy"`). `force_polars`
#: sets this; `optimize` and `run_op` turn it into the selector they plan with.
_BACKEND = "pandas"

#: Whether the query fast path may bid for a MASK selection. Same shape as
#: `_BACKEND`: a plan-level `OptConfig` setting, moved by `pandas_query` below.
_PANDAS_QUERY = False


def selector():
    """The selector the helpers plan with: the current `_BACKEND`, pinned."""
    return FlagBasedSelector(backend=_BACKEND)


def opt_config(conf=None):
    """`conf` with the helpers' plan settings filled in where it left them.

    Both settings have to be threaded in explicitly -- `OptConfig` is the only
    channel into implementation selection that no `set_config` parameter
    reaches, which is the point of them living there.
    """
    conf = copy.copy(conf) if conf is not None else OptConfig()
    if conf.selector is None:
        conf.selector = selector()
    if not conf.pandas_query:
        conf.pandas_query = _PANDAS_QUERY
    return conf


def plan_context(conf=None):
    """The plan context the helpers bind single ops against."""
    return PlanContext.from_flags(opt_config(conf))


def optimize(dag, conf=None, env=None):
    """Optimize `dag` under the helpers' current plan settings."""
    linearized_dag, *_ = optimize_(dag, opt_config(conf), env)
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
    bind_op(op, plan_context(), selector=selector())
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
    """Plan against the polars impls inside this block.

    Kept as a flag-shaped helper because that is how the ~60 call sites read,
    but it no longer sets anything global: it moves `_BACKEND`, which `optimize`
    and `run_op` turn into a `FlagBasedSelector`.
    """
    global _BACKEND
    orig = _BACKEND
    _BACKEND = "polars" if enabled else "pandas"
    try:
        yield
    finally:
        _BACKEND = orig


@contextmanager
def pandas_query(enabled=True):
    """Let the ``DataFrame.query()`` selection impl bid inside this block.

    Flag-shaped like `force_polars`, and for the same reason: it moves
    test-local state that `optimize` and `run_op` fold into their `OptConfig`,
    not anything global.
    """
    global _PANDAS_QUERY
    orig = _PANDAS_QUERY
    _PANDAS_QUERY = enabled
    try:
        yield
    finally:
        _PANDAS_QUERY = orig


class PolarsTestCase(unittest.TestCase):
    """Base class that plans every test against the polars impls."""

    def setUp(self):
        super().setUp()
        self._polars = force_polars(True)
        self._polars.__enter__()

    def tearDown(self):
        self._polars.__exit__(None, None, None)
        super().tearDown()


class TestConcatOpPolars(PolarsTestCase):
    def test_polars_concat(self):
        op = ConcatOp(first=OperandRef(0), others=[OperandRef(1)], axis=0)
        result = run_op(op, pl.DataFrame({"a": [1, 2]}), pl.DataFrame({"a": [3, 4]}))
        self.assertEqual(4, len(result))


if __name__ == "__main__":
    unittest.main()
