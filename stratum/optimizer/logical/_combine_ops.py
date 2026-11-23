"""Operators that combine several inputs into a frame or array.

Each operator defines its own column, row, and alignment semantics. In
particular, assignment does not evaluate the expressions producing its inputs.
"""
from __future__ import annotations

import numpy as np

from stratum.optimizer.logical._base import IndexEffect, config_key
from stratum.optimizer.logical._column_expr import Const
from stratum.optimizer.logical._ops import Op, OperandRef, OutputType, ValueOp, substitute_op
from stratum.optimizer.logical import _schema


class CombineOp(Op):
    """Base for assignment and concatenation."""
    logical_family = "Combine"


class AssignOp(CombineOp):
    """Assign named values to a frame, overwriting or appending columns.

    Input 0 is the base frame. Ordered ``kwargs`` bind column names to literals
    or input references. Series align by label; arrays assign by position and
    scalars broadcast. Callable values retain pandas' sequential evaluation.
    """
    fields = ["args", "kwargs"]
    index_effect = IndexEffect.KEEP
    cacheable = True

    def __init__(self, args=(), kwargs=None, inputs=None, outputs=None):
        super().__init__(name="assign", inputs=inputs, outputs=outputs)
        self.args = args
        self.kwargs = dict(kwargs or {})
        self.output_type = OutputType.FRAME

    def structure_key(self):
        # Column order and literal types affect the result (e.g. 1 versus 1.0).
        entries = tuple((name, value if isinstance(value, OperandRef) else Const(value))
                        for name, value in self.kwargs.items())
        return type(self), tuple(map(id, self.inputs)), config_key(self.args), entries

    def propagate_output_schema(self):
        self.output_schema = _schema.add_columns(self.inputs[0].output_schema,
                                                 list(self.kwargs))


class ConcatOp(CombineOp):
    """Concatenates frames: ``axis=0`` stacks their rows, ``axis=1`` places their
    columns side by side.

    ``operands`` are the ``OperandRef``s of the frames, in order; one input edge
    may feed several of them. ``axis`` is ``0`` or ``1`` (extraction normalizes
    pandas' names for them), or an ``OperandRef`` when graph-fed.

    Like skrub's ``.skb.concat`` it also stacks numpy arrays. Extraction turns a
    concat whose first operand is known to be an array into an
    :class:`ArrayConcatOp`. When the first operand's kind is unknown at plan time
    (an opaque call can return either), the frame impls stack arrays at run time.
    """
    fields = ["operands", "axis"]
    cacheable = True

    def __init__(self, operands: list = None, axis=0, inputs: list[Op] = None,
                 outputs: list[Op] = None):
        super().__init__(name="concat", inputs=inputs, outputs=outputs)
        self.operands = list(operands or [])
        self.axis = axis
        self.output_type = OutputType.FRAME

    @property
    def index_effect(self):
        # Column concatenation preserves a shared index. Row stacking can repeat
        # labels; a graph-fed axis cannot establish a shared index at plan time.
        return IndexEffect.COMMON if self.axis == 1 else IndexEffect.NEW

    @property
    def aligns_by_label(self):
        # Stacking rows pairs nothing; placing columns side by side aligns rows.
        return False if self.axis == 0 else None

    def propagate_output_schema(self):
        """See :func:`_schema.concat_schemas` for the per-axis rules."""
        if isinstance(self.axis, OperandRef):
            # The axis picks the dtype rule, so a graph-fed axis is unknown.
            self.output_schema = None
            return
        self.output_schema = _schema.concat_schemas(
            [self.inputs[ref.k].output_schema for ref in self.operands], self.axis)


class ArrayConcatOp(ConcatOp):
    """Joins numpy arrays along ``axis`` (``np.concatenate``): a concat whose first
    operand is known at plan time to be an array."""

    def __init__(self, operands: list = None, axis=0, inputs: list[Op] = None,
                 outputs: list[Op] = None):
        super().__init__(operands=operands, axis=axis, inputs=inputs, outputs=outputs)
        self.output_type = OutputType.MATRIX

    def propagate_output_schema(self):
        """An array has no named column schema."""
        self.output_schema = None


#: pandas' names for the two concat axes.
_CONCAT_AXES = {0: 0, "index": 0, "rows": 0, 1: 1, "columns": 1}


def _is_array(op: Op) -> bool:
    """Whether ``op`` is known at plan time to produce a numpy array."""
    return (op.output_type is OutputType.MATRIX
            or isinstance(op, ValueOp) and isinstance(op.value, np.ndarray))


def make_concat_op(op: ConcatOp) -> ConcatOp | None:
    """Normalize the axis of a recorded concat and pick its variant.

    The first operand decides, as in skrub: a known array makes an
    :class:`ArrayConcatOp`. Returns ``None`` when there is nothing to change.
    """
    axis = op.axis
    if not isinstance(axis, OperandRef):
        axis = _CONCAT_AXES.get(axis, axis)
    is_array = bool(op.operands) and _is_array(op.inputs[op.operands[0].k])
    normalized = axis == op.axis and type(axis) is type(op.axis)
    if type(op) is not ConcatOp or (normalized and not is_array):
        return None
    cls = ArrayConcatOp if is_array else ConcatOp
    return substitute_op(op, cls(operands=op.operands, axis=axis))
