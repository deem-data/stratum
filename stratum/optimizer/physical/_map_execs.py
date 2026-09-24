"""Physical implementations of ``AssignMapOp`` (folded column-map).

Same-shape backend-variant family: the concrete impls subclass ``AssignMapOp``
(+ :class:`PhysicalOp`) and carry the backend-specific kernel.

Entries are source-relative ``ColumnExpr`` DAGs; after assign-map fusion,
shared sub-expressions are common. At plan time each impl compiles its entries
into a :class:`~stratum.optimizer.physical._map_program.MapProgram`, so every
shared node is evaluated exactly once per call:

* **pandas** -- evaluate the steps in order into a list of values, then one
  ``assign`` with the outputs.
* **polars** -- maps with at most one ``MapProgram`` step run as one eager
  ``with_columns`` of the source-relative entries (lazy leveling costs more
  than it saves when there is little sharing). Deeper programs use one lazy
  plan: a ``with_columns`` per step level (steps of a level run in parallel),
  a final ``with_columns`` for the outputs no step is stored under, a
  ``drop`` of the private step columns, and a single ``collect``. A step that
  is exactly an output is staged under the output's name when the program
  allows it (see ``MapStep.output``), so a chain whose outputs are all shared
  prefixes needs no private columns and no copy. A ``select`` restores
  assignment column order when steps stored early would change it.

Every output is assigned simultaneously against the input frame, matching the
source-relative meaning of the entries.
"""
from __future__ import annotations

import logging

import pandas as pd
import polars as pl

from stratum.optimizer.logical._map_ops import AssignMapOp
from stratum.optimizer.physical._map_program import plan_map_program
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl

logger = logging.getLogger(__name__)


class _ProgramAssignMapOp(AssignMapOp, PhysicalOp):
    """Shared plan-time binding: compile ``entries`` into ``self.program``."""

    def on_impl_selected(self, ctx) -> None:
        self.program = plan_map_program(self.entries)


@physical_impl(of=AssignMapOp, backend="pandas")
class PandasAssignMapOp(_ProgramAssignMapOp):
    def process(self, mode: str, inputs: list):
        program = self.program
        ctx = self.make_context(mode, inputs)
        temps = ctx.temps = [None] * len(program.steps)
        for step in program.steps:
            temps[step.index] = step.expr.to_pandas(ctx)
        values = {name: expr.to_pandas(ctx)
                  for name, expr in program.outputs.items()}
        return ctx.frame.assign(**values)


def _as_polars_column(name: str, result):
    """Normalize a lowered map value to something ``with_columns`` accepts."""
    if isinstance(result, (pd.Series, pd.DataFrame)):
        # An OperandLeaf can feed pandas data into a polars plan.
        logger.warning(
            f"Converting pandas object to polars object for column {name}")
        return pl.from_pandas(result)
    if isinstance(result, list):
        # Polars treats a list passed through the keyword API as one
        # list-valued scalar; assign semantics require a column.
        return pl.Series(result)
    return result


def _polars_columns(outputs: dict, ctx) -> dict:
    # The keyword API accepts expressions, series, arrays and scalars,
    # broadcasting the latter just like pandas.DataFrame.assign.
    return {name: _as_polars_column(name, expr.to_polars(ctx))
            for name, expr in outputs.items()}


def _reordered_columns(columns: list[str], write_order: list[str],
                       assign_order: list[str]) -> list[str] | None:
    """Assignment column order, or ``None`` if the staged plan already has it.

    ``with_columns`` keeps existing columns in place and appends new ones in
    write order.
    """
    existing = set(columns)
    written = [n for n in write_order if n not in existing]
    expected = [n for n in assign_order if n not in existing]
    return None if written == expected else columns + expected


@physical_impl(of=AssignMapOp, backend="polars")
class PolarsAssignMapOp(_ProgramAssignMapOp):
    def on_impl_selected(self, ctx) -> None:
        super().on_impl_selected(ctx)
        program = self.program
        clash = set(program.temp_names).intersection(program.outputs)
        if clash:
            raise ValueError(
                f"Map outputs use reserved map column names {sorted(clash)}")
        self._levels = program.levels
        self._unstored = program.unstored_outputs
        # Steps stored under output names are written level by level, before
        # the remaining outputs, which can differ from assignment order.
        self._write_order = [step.output for level in self._levels
                             for step in level if step.output is not None]
        self._write_order += list(self._unstored)

    def process(self, mode: str, inputs: list):
        program = self.program
        ctx = self.make_context(mode, inputs)
        frame = ctx.frame
        # Zero or one materialised step: one eager with_columns of the entries.
        # The lazy leveled path pays for plan build / collect; that only pays
        # off once several shared nodes need staging.
        if len(program.steps) <= 1:
            return frame.with_columns(**_polars_columns(self.entries, ctx))

        temp_names = program.temp_names
        clash = set(temp_names).intersection(frame.columns)
        if clash:
            raise ValueError(
                f"Input frame contains reserved map column names {sorted(clash)}")
        ctx.temps = [pl.col(step.column) for step in program.steps]
        lf = frame.lazy()
        for level in self._levels:
            lf = lf.with_columns(**{
                step.column: _as_polars_column(step.column, step.expr.to_polars(ctx))
                for step in level})
        if self._unstored:
            lf = lf.with_columns(**_polars_columns(self._unstored, ctx))
        if temp_names:
            lf = lf.drop(temp_names)
        order = _reordered_columns(frame.columns, self._write_order,
                                   list(program.outputs))
        if order is not None:
            lf = lf.select(order)
        return lf.collect()
