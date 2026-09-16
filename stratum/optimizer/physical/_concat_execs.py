"""Physical implementations of the ``ConcatOp`` frame concatenation.

``ConcatOp`` is a same-shape backend-variant family: the logical op is a pure
backend refinement of itself, so the concrete impls subclass it (``PandasConcatOp
(ConcatOp, PhysicalOp)``) rather than being a separate hierarchy. Implementation
selection swaps a logical ``ConcatOp`` to one of these in place; ``isinstance(op,
ConcatOp)`` therefore still identifies a concat anywhere in the plan.
"""
from __future__ import annotations

import pandas as pd
import polars as pl

from stratum.optimizer.logical._base import OperandRef
from stratum.optimizer.logical._dataframe_ops import ConcatOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl


class ConcatExec(ConcatOp, PhysicalOp):
    """Physical base: resolves the concat operands copied from the logical op."""

    def _resolve(self, inputs: list):
        first = inputs[self.first.k] if isinstance(self.first, OperandRef) else self.first
        others = [inputs[o.k] if isinstance(o, OperandRef) else o for o in self.others]
        axis = inputs[self.axis.k] if isinstance(self.axis, OperandRef) else self.axis
        return first, others, axis


@physical_impl(of=ConcatOp, backend="pandas")
class PandasConcatOp(ConcatExec):
    def process(self, mode: str, inputs: list):
        first, others, axis = self._resolve(inputs)
        return pd.concat([first, *others], axis=axis)


@physical_impl(of=ConcatOp, backend="polars")
class PolarsConcatOp(ConcatExec):
    # pandas concat axis (0=rows, 1=cols) -> polars `how`.
    axis_map = {0: "diagonal_relaxed", 1: "horizontal"}

    def process(self, mode: str, inputs: list):
        first, others, axis = self._resolve(inputs)
        return pl.concat([first, *others], how=self.axis_map[axis])
