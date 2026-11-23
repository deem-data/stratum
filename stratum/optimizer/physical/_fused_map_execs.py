"""Polars expression kernels selected by physical map fusion."""
from __future__ import annotations

import polars as pl
import polars.selectors as cs

from stratum.optimizer.logical._base import IndexEffect
from stratum.optimizer.logical._column_expr import ColumnExpr
from stratum.optimizer.physical._combine_execs import polars_assignment_value
from stratum.optimizer.physical._expressions import EvalContext, to_polars
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._representation import FrameLib


class PolarsFusedMapOp(PhysicalOp):
    """Evaluate a Series expression over a frame and materialized operands."""
    fields = ["expr", "result_name"]
    frame_lib = FrameLib.POLARS
    cacheable = True

    def process(self, inputs: list):
        expr = to_polars(self.expr, EvalContext.of(inputs, self.mode))
        if not isinstance(expr, pl.Expr):
            expr = pl.lit(expr)
        if self.result_name is not None:
            expr = expr.alias(self.result_name)
        return inputs[0].select(expr).to_series()


class PolarsFusedAssignOp(PhysicalOp):
    """Compute assigned columns in one with_columns or projected select call.

    ``columns`` is the original base projection, or None for the whole frame.
    Entry expressions read the input frame, including columns being overwritten.
    """
    fields = ["entries", "columns"]
    frame_lib = FrameLib.POLARS
    index_effect = IndexEffect.KEEP
    cacheable = True

    def process(self, inputs: list):
        frame = inputs[0]
        ctx = EvalContext.of(inputs, self.mode)
        values = {
            name: polars_assignment_value(to_polars(value, ctx)
                                           if isinstance(value, ColumnExpr) else value)
            for name, value in self.entries.items()
        }
        if self.columns is None:
            return frame.with_columns(**values)
        # An overwritten column must still exist in the original projection.
        for name in self.columns:
            if name not in frame.columns:
                raise pl.exceptions.ColumnNotFoundError(name)
        names = [*self.columns, *(name for name in values if name not in self.columns)]
        return frame.select([
            (pl.lit(values[name]) if not isinstance(values[name], pl.Expr) else values[name]).alias(name)
            if name in values else cs.by_name(name).as_expr()
            for name in names
        ])
