"""Physical implementations of ``AssignMapOp`` (folded column-map).

Same-shape backend-variant family: the concrete impls subclass ``AssignMapOp``
(+ :class:`PhysicalOp`) and carry the backend-specific kernel.

Fusion has already inlined sequential assigns into source-relative
``ColumnExpr`` DAGs (shared by identity). Evaluation:

* **pandas** -- ``assign`` with structural ``EvalContext.memo`` so shared
  prefixes from fusion evaluate once in Python.
* **polars** -- rewrite subtrees that match earlier entries to
  ``pl.col(prior)``, then either a lazy chain of shallow ``with_columns``
  (when priors are used) or a single eager ``with_columns`` (independent
  columns). A single ``with_columns`` of deep inlined nests does not CSE
  across outputs cheaply; the prior-col chain stays linear and materializes
  once via ``collect``.
"""
from __future__ import annotations

import logging

import pandas as pd
import polars as pl

from stratum.optimizer.logical._column_expr import replace_prior_cols
from stratum.optimizer.logical._map_ops import AssignMapOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl

logger = logging.getLogger(__name__)


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


@physical_impl(of=AssignMapOp, backend="pandas")
class PandasAssignMapOp(AssignMapOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        ctx = self.make_context(mode, inputs)
        ctx.memo = {}
        values = {name: expr.to_pandas(ctx) for name, expr in self.entries.items()}
        return ctx.frame.assign(**values)


@physical_impl(of=AssignMapOp, backend="polars")
class PolarsAssignMapOp(AssignMapOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        ctx = self.make_context(mode, inputs)
        ctx.memo = {}
        priors: dict = {}
        lowered: dict[str, object] = {}
        needs_prior_cols = False
        for name, expr in self.entries.items():
            rewritten, used_prior = replace_prior_cols(expr, priors)
            needs_prior_cols = needs_prior_cols or used_prior
            lowered[name] = rewritten
            priors[expr] = name

        if needs_prior_cols:
            # Sibling columns are not visible inside one with_columns; chain
            # shallow pl.col(prior) steps in a lazy plan and collect once.
            lf = ctx.frame.lazy()
            for name, expr in lowered.items():
                result = _as_polars_column(name, expr.to_polars(ctx))
                lf = lf.with_columns(result.alias(name))
            return lf.collect()

        columns = {
            name: _as_polars_column(name, expr.to_polars(ctx))
            for name, expr in lowered.items()
        }
        return ctx.frame.with_columns(**columns)
