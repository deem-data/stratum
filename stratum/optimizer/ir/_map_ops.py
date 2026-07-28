"""Folded column-map operators (MapOp).

A MapOp computes new columns of one source frame from backend-agnostic
:class:`~stratum.optimizer.ir._column_expr.ColumnExpr` trees. The grammar is
restricted to natively-lazy computations (arithmetic, boolean logic,
``.str``/``.dt`` accessors, datetime parsing); on polars all entries compile
into one ``with_columns`` kernel. Anything outside the grammar stays in the
graph and feeds the map through an ``OperandLeaf`` input.

The concrete map kinds currently are:

* :class:`AssignMapOp` for named, series-valued ``df.assign(...)`` entries.
* :class:`MissingMaskOp` for pandas missing-value predicates.
"""
from __future__ import annotations

import pandas as pd
import polars as pl

from stratum.optimizer.ir._column_expr import (ColumnExpr, Const, EvalContext,
                                               _Folder)
from stratum.optimizer.ir._ops import (CallOp, MethodCallOp, Op, OperandRef,
                                       OutputType)


MISSING_METHODS = ("isna", "isnull", "notna", "notnull")
POSITIVE_MISSING_METHODS = ("isna", "isnull")
MISSING_FUNCTIONS = (pd.isna, pd.isnull, pd.notna, pd.notnull)
POSITIVE_MISSING_FUNCTIONS = (pd.isna, pd.isnull)


class MapOp(Op):
    """Base for folded column-map operators."""
    logical_family = "Map"

    def __init__(self, name: str, inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        self.output_type = OutputType.FRAME

    def make_context(self, mode: str, inputs: list) -> EvalContext:
        return EvalContext(frame=inputs[0], inputs=inputs, mode=mode)


class MissingMaskOp(MapOp):
    """A missing-value predicate such as .isnull()/isna() on a frame or series."""

    fields = ["positive"]

    def __init__(self, positive: bool,
                 inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name="isnull" if positive else "notnull",
                         inputs=inputs, outputs=outputs)
        self.positive = positive
        if inputs:
            self.output_type = inputs[0].output_type


class AssignMapOp(MapOp):
    """``df.assign(...)`` with each assigned column folded to a ``ColumnExpr``.

    ``entries`` maps a new column name to its series-valued expression; input
    columns pass through unchanged.
    """
    fields = ["entries"]

    def __init__(self, entries: dict[str, ColumnExpr],
                 inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=f"assign: {', '.join(entries)}",
                         inputs=inputs, outputs=outputs)
        self.entries = entries


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


def _make_method_missing_mask_op(op: MethodCallOp, positive: bool) -> MissingMaskOp | None:
    """Build a mask from a bound call such as ``series.isna()``.

    The object before the method is stored as the call's only input.
    """
    if op.args or op.kwargs or len(op.inputs) != 1:
        return None

    new_op = MissingMaskOp(positive=positive, inputs=[op.inputs[0]], outputs=list(op.outputs))
    op.replace_output_of_inputs(new_op)
    return new_op


def _make_call_missing_mask_op(op: CallOp, positive: bool) -> MissingMaskOp | None:
    """Build a missing mask from ``pd.isna(obj)`` or ``pd.isna(obj=obj)``."""
    args = tuple(op.args or ())
    kwargs = dict(op.kwargs or {})

    if len(args) == 1 and not kwargs:
        operand_ref = args[0]
    elif not args and set(kwargs) == {"obj"}:
        operand_ref = kwargs["obj"]
    else:
        return None

    if not isinstance(operand_ref, OperandRef):
        return None

    new_op = MissingMaskOp(positive=positive, inputs=[op.inputs[operand_ref.k]], outputs=list(op.outputs))
    op.replace_output_of_inputs(new_op)
    return new_op


def make_missing_mask_op(op: MethodCallOp | CallOp) -> MissingMaskOp | None:
    """Extract bound-method and standalone pandas missing-value calls."""
    if isinstance(op, MethodCallOp):
        method_name = op.method_name
        if method_name not in MISSING_METHODS:
            return None
        positive = method_name in POSITIVE_MISSING_METHODS
        return _make_method_missing_mask_op(op, positive)

    if isinstance(op, CallOp):
        if op.func not in MISSING_FUNCTIONS:
            return None
        positive = op.func in POSITIVE_MISSING_FUNCTIONS
        return _make_call_missing_mask_op(op, positive)

    return None
