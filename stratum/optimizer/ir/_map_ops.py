"""Folded column-map operators (MapOp).

A MapOp computes new columns of one source frame from backend-agnostic
:class:`~stratum.optimizer.ir._column_expr.ColumnExpr` trees. The grammar is
restricted to natively-lazy computations (arithmetic, boolean logic,
``.str``/``.dt`` accessors, datetime parsing); on polars all entries compile
into one ``with_columns`` kernel. Anything outside the grammar stays in the
graph and feeds the map through an ``OperandLeaf`` input.

:class:`AssignMapOp` -- from ``df.assign(...)`` -- is the only map kind so far:
named, series-valued entries, with input columns passing through.
"""
from __future__ import annotations

import logging

import pandas as pd
import polars as pl

from stratum._config import FLAGS
from stratum.optimizer.ir._column_expr import (ColumnExpr, Const, EvalContext,
                                               _Folder)
from stratum.optimizer.ir._ops import (MethodCallOp, Op, OperandRef, OutputType)

logger = logging.getLogger(__name__)


class MapOp(Op):
    """Base for folded column-map operators."""

    def __init__(self, name: str, inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        self.output_type = OutputType.FRAME

    def make_context(self, mode: str, inputs: list) -> EvalContext:
        return EvalContext(frame=inputs[0], inputs=inputs, mode=mode)


class AssignMapOp(MapOp):
    """``df.assign(...)`` with each assigned column folded to a ``ColumnExpr``.

    ``entries`` maps a new column name to its series-valued expression; input
    columns pass through unchanged.
    """
    fields = ["entries"]

    def __init__(self, entries: dict[str, ColumnExpr],
                 inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=f"MAP(assign: {', '.join(entries)})",
                         inputs=inputs, outputs=outputs)
        self.entries = entries

    def process(self, mode: str, inputs: list):
        ctx = self.make_context(mode, inputs)
        if FLAGS.force_polars:
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
        values = {name: expr.to_pandas(ctx) for name, expr in self.entries.items()}
        return ctx.frame.assign(**values)


# --- Folding: assign subgraphs -> MapOp ---------------------------------------

def _detach_absorbed_and_rewire(op: Op, new_op: MapOp, folder: _Folder) -> None:
    """Detach absorbed ops and rewire kept producers to ``new_op``.

    Absorbed nodes are unlinked from their inputs and cleared; the source and
    each kept leaf op feed ``new_op`` in place of the folded consumer.
    Downstream consumers of ``op`` are rewired by the caller.
    """
    for node in folder.absorbed:
        for inp in node.inputs:
            inp.outputs = [o for o in inp.outputs if o is not node]
        node.inputs = []
        node.outputs = []
    for producer in new_op.inputs:
        producer.outputs = [o for o in producer.outputs if o is not op]
        producer.add_output(new_op)
    op.inputs = []


def _is_scalar_constant(value) -> bool:
    """Whether an assign kwarg constant can fold to a ``Const`` entry.

    Scalars broadcast identically on every backend and fold; sequence-like
    values (lists, arrays, series) keep the ``AssignOp`` fallback.
    """
    if isinstance(value, str):
        return True
    return not hasattr(value, "__len__") and not isinstance(
        value, (pd.Series, pd.DataFrame, pl.Series, pl.DataFrame))


def make_assign_map_op(op: MethodCallOp) -> AssignMapOp | None:
    """Fold ``df.assign(**kwargs)`` into an :class:`AssignMapOp`.

    All graph-fed kwargs share one folder, so a producer feeding several columns
    folds once. Returns ``None`` for non-foldable calls (positional args or a
    sequence-valued constant kwarg), leaving the plain ``AssignOp`` in place.
    """
    if op.args:
        return None
    kwargs = op.kwargs or {}
    if not kwargs:
        return None
    src = op.inputs[0]
    ref_names, roots, const_entries = [], [], {}
    for name, value in kwargs.items():
        if isinstance(value, OperandRef):
            ref_names.append(name)
            roots.append(op.inputs[value.k])
        elif _is_scalar_constant(value):
            const_entries[name] = Const(value)
        else:
            return None

    folder = _Folder(src)
    exprs = folder.fold_many(roots, root_consumer=op)
    # Preserve the kwargs' assignment order (later columns may overwrite earlier).
    entries = {name: (const_entries[name] if name in const_entries
                      else exprs[ref_names.index(name)])
               for name in kwargs}

    new_op = AssignMapOp(entries=entries,
                         inputs=[src, *folder.leaf_ops], outputs=list(op.outputs))
    _detach_absorbed_and_rewire(op, new_op, folder)
    return new_op
