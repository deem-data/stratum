"""Exact algebraic simplification of column expressions.

The expression-level counterparts of the identity rewrites in ``_numeric``: an
operation with its identity element is dropped (``x * 1``, ``x - 0``) and
``abs(abs(x))`` is one ``abs``. Only where the result is the same value, bit for
bit, of the same dtype, which the source frame's schema has to show: ``bool + 0``
is an int, ``int / 1`` a float, and ``-0.0 + 0`` is ``+0.0``.

The inverse and fused-function rewrites (``log(exp(x))``, ``exp(x) - 1``,
``sqrt(x * x)``) change float results, so they are not applied to expressions.
"""
from __future__ import annotations

import operator

from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer.logical import _schema
from stratum.optimizer.logical._column_expr import (
    BinOpExpr, Col, ColumnExpr, Const, UfuncExpr, map_exprs)
from stratum.optimizer.logical._map_ops import AssignMapOp, SeriesMapOp
from stratum.optimizer.logical._ops import Op
from stratum.optimizer.logical._selection_ops import MaskSelectionOp

#: Operator -> (its identity element, whether it may stand on the left too,
#: the AlgebraicRewritesConfig flag, whether an integer operand stays an integer).
_IDENTITIES = {
    operator.add: (0, True, "add_zero", True),
    operator.sub: (0, False, "identity_subtract", True),
    operator.mul: (1, True, "identity_op", True),
    operator.truediv: (1, False, "div_by_one", False),
    operator.pow: (1, False, "pow_by_one", True),
}


def simplify_expressions(root: Op, config) -> Op:
    """Simplify the expressions of every assign, series map and mask selection."""
    for op in topological_iterator(root):
        if not isinstance(op, (AssignMapOp, SeriesMapOp, MaskSelectionOp)):
            continue
        schema = op.inputs[0].output_schema
        simplify = _Simplifier(schema if _schema.is_known(schema) else {}, config)
        for name in op.fields:
            setattr(op, name, map_exprs(getattr(op, name), simplify))
    return root


class _Simplifier:
    """Simplifies an op's expressions bottom-up against its source's column dtypes.

    A node nothing simplifies is kept as it is, and each node is simplified once,
    so a subexpression the folder shared between entries stays one object.
    """

    def __init__(self, schema, config):
        self.schema = schema
        self.config = config
        self._done: dict[int, ColumnExpr] = {}

    def __call__(self, expr: ColumnExpr) -> ColumnExpr:
        if id(expr) not in self._done:
            self._done[id(expr)] = self._simplify(expr)
        return self._done[id(expr)]

    def _simplify(self, expr: ColumnExpr) -> ColumnExpr:
        rebuilt = expr.map_children(self)
        if all(new is old for new, old in zip(rebuilt.children(), expr.children())):
            rebuilt = expr
        while True:
            simpler = self._rewrite(rebuilt)
            if simpler is None:
                return rebuilt
            rebuilt = simpler

    def _rewrite(self, expr: ColumnExpr) -> ColumnExpr | None:
        if (self.config.abs_abs and isinstance(expr, UfuncExpr) and expr.func == "abs"
                and isinstance(expr.operand, UfuncExpr) and expr.operand.func == "abs"):
            return expr.operand
        if isinstance(expr, BinOpExpr) and expr.op in _IDENTITIES:
            identity, commutes, flag, keeps_int = _IDENTITIES[expr.op]
            if not getattr(self.config, flag):
                return None
            sides = [(expr.left, expr.right)] + ([(expr.right, expr.left)] if commutes else [])
            for operand, constant in sides:
                if self._is_identity(operand, constant, identity, keeps_int, expr.op):
                    return operand
        return None

    def _is_identity(self, operand, constant, identity, keeps_int, op) -> bool:
        """Whether ``operand <op> constant`` is ``operand``, value and dtype."""
        if not isinstance(constant, Const) or not isinstance(operand, Col):
            return False
        dtype = self.schema.get(operand.name)
        if dtype is None or dtype == _schema.UNKNOWN_DTYPE:
            return False
        # A plain Python number, so it takes the column's dtype (a numpy scalar can
        # widen it, and True is an int equal to 1).
        value = constant.value
        if type(value) is int:
            fits = dtype.is_float() or (dtype.is_integer() and keeps_int)
        elif type(value) is float:
            fits = dtype.is_float()
        else:
            return False
        # -0.0 + 0 is +0.0: adding zero to a float is not the float.
        if op is operator.add and dtype.is_float():
            return False
        return fits and value == identity
