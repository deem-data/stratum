"""Physical implementations of folded column-map operators.

The concrete implementations subclass their logical map operator together with
:class:`PhysicalOp` and carry only the backend-specific kernel.
"""
from __future__ import annotations

import logging

import pandas as pd
import polars as pl

from stratum.optimizer.ir._map_ops import AssignMapOp, MissingMaskOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl

logger = logging.getLogger(__name__)


@physical_impl(of=AssignMapOp, backend="pandas")
class PandasAssignMapOp(AssignMapOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        ctx = self.make_context(mode, inputs)
        values = {name: expr.to_pandas(ctx) for name, expr in self.entries.items()}
        return ctx.frame.assign(**values)


@physical_impl(of=AssignMapOp, backend="polars")
class PolarsAssignMapOp(AssignMapOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        ctx = self.make_context(mode, inputs)
        columns = {}
        for name, expr in self.entries.items():
            result = expr.to_polars(ctx)
            if isinstance(result, (pd.Series, pd.DataFrame)):
                # An OperandLeaf can feed pandas data into a polars plan.
                logger.warning(f"Converting pandas object to polars object for column {name}")
                result = pl.from_pandas(result)
            elif isinstance(result, list):
                # Polars treats a list passed through the keyword API as one
                # list-valued scalar; assign semantics require a column.
                result = pl.Series(result)
            columns[name] = result
        # The keyword API accepts expressions, series, arrays and scalars,
        # broadcasting the latter just like pandas.DataFrame.assign.
        return ctx.frame.with_columns(**columns)


# Polars treats native NaN values separately from null values.
@physical_impl(of=MissingMaskOp, backend="pandas")
class PandasMissingMaskOp(MissingMaskOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        obj = inputs[0]
        return pd.isnull(obj) if self.positive else pd.notnull(obj)


@physical_impl(of=MissingMaskOp, backend="polars")
class PolarsMissingMaskOp(MissingMaskOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        obj = inputs[0]
        if isinstance(obj, pl.DataFrame):
            predicate = (pl.all().is_null() if self.positive
                         else pl.all().is_not_null())
            return obj.select(predicate)
        return obj.is_null() if self.positive else obj.is_not_null()
