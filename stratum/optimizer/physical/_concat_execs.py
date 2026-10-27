"""Physical implementations of the concat family.

``ConcatOp`` is a same-shape backend-variant family: the logical op is a pure
backend refinement of itself, so the concrete impls subclass it (``PandasConcatOp
(ConcatOp, PhysicalOp)``) rather than being a separate hierarchy. Implementation
selection swaps a logical ``ConcatOp`` to one of these in place; ``isinstance(op,
ConcatOp)`` therefore still identifies a concat anywhere in the plan.

``ArrayConcatOp`` has a single, backend-agnostic numpy impl.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from stratum.optimizer.logical._base import OperandRef, _resolve_operand
from stratum.optimizer.logical._dataframe_ops import ArrayConcatOp, ConcatOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl


def _operands(op: ConcatOp, inputs: list) -> tuple[list, object]:
    """The values to concatenate, in order, and the resolved axis."""
    return [inputs[ref.k] for ref in op.operands], _resolve_operand(op.axis, inputs)


class ConcatExec(ConcatOp, PhysicalOp):
    """Physical base of the frame concats."""

    def process(self, mode: str, inputs: list):
        operands, axis = _operands(self, inputs)
        if isinstance(operands[0], np.ndarray):
            # skrub concatenates arrays too (since 0.10), whatever the frame
            # backend; extraction could not tell this operand was one.
            return np.concatenate(operands, axis=axis)
        return self._concat_frames(operands, axis)

    def _concat_frames(self, operands: list, axis):
        raise NotImplementedError


@physical_impl(of=ConcatOp, backend="pandas")
class PandasConcatOp(ConcatExec):
    def _concat_frames(self, operands: list, axis):
        return pd.concat(operands, axis=axis)


@physical_impl(of=ConcatOp, backend="polars")
class PolarsConcatOp(ConcatExec):
    # pandas concat axis (0=rows, 1=cols) -> polars `how`.
    axis_map = {0: "diagonal_relaxed", 1: "horizontal"}

    @classmethod
    def supports(cls, op: ConcatOp, ctx) -> bool:
        if not isinstance(op.axis, OperandRef) and op.axis not in cls.axis_map:
            op._unsupported_reason = f"axis={op.axis!r}"
            return False
        return True

    def _concat_frames(self, operands: list, axis):
        return pl.concat(operands, how=self.axis_map[axis])


@physical_impl(of=ArrayConcatOp, backend="numpy")
class NumpyConcatOp(ArrayConcatOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        operands, axis = _operands(self, inputs)
        return np.concatenate(operands, axis=axis)
