"""Relational rewrites on the logical IR.

The algebraic rewrites simplify *expressions*; these change the plan's operator
*shape*, replacing one relational operator with another that means the same thing
but that the cost model and join reordering can reason about.
"""
import operator

from stratum.optimizer._op_utils import rewrite_pass
from stratum.optimizer.logical._column_expr import (
    Col, ColumnMethodExpr, OperandLeaf, UnaryOpExpr)
from stratum.optimizer.logical._join_ops import JoinOp
from stratum.optimizer.logical._ops import GetItemOp, Op, OutputType
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp
from stratum.optimizer.logical._selection_ops import SelectionKind, SelectionOp


def _is_isin(expr) -> bool:
    """``col.isin(values)`` with the single positional the column method admits."""
    return (isinstance(expr, ColumnMethodExpr) and expr.method == "isin"
            and len(expr.args) == 1 and not expr.kwargs)


def _membership_predicate(predicate):
    """``("semi" | "anti", (child, values))`` for a membership test, else ``None``."""
    how = "semi"
    if isinstance(predicate, UnaryOpExpr) and predicate.op is operator.invert:
        how, predicate = "anti", predicate.operand
    if not _is_isin(predicate):
        return None
    return how, (predicate.operand, predicate.args[0])


def _left_key(op: SelectionOp, child) -> str | None:
    """The left relation's key column for the membership test's subject.

    Usually the column folded into a ``Col``. It stays an operand instead when
    something else reads it too, which is exactly what the driving pipeline does:
    the column being filtered is also the one that was counted. That is still a
    column of the left relation, so the shape is promotable either way.
    """
    if isinstance(child, Col):
        return child.name
    if not isinstance(child, OperandLeaf):
        return None
    producer = op.inputs[child.ref.k]
    if (isinstance(producer, (ColumnProjectionOp, GetItemOp))
            and isinstance(producer.key, str)
            and producer.inputs and producer.inputs[0] is op.inputs[0]):
        return producer.key
    return None


def _match_isin_against_a_relation(op: Op):
    """Match ``frame[col.isin(<relation>)]``, the shape a filtering join replaces.

    Deliberately unmatched: ``isin`` against a literal collection, which is a
    predicate and nothing more, and a filtered *series*, which has no left
    relation to join. Telling the first case apart is the whole reason this is a
    rewrite rather than part of extraction: what separates a subquery from a
    constant is a property of the operand's *producer*, and extraction is local
    to one op by design.
    """
    if not (isinstance(op, SelectionOp) and op.kind is SelectionKind.MASK):
        return None
    if op.output_type is not OutputType.FRAME:
        return None
    matched = _membership_predicate(op.predicate)
    if matched is None:
        return None
    how, (child, values) = matched
    if not isinstance(values, OperandLeaf):
        return None
    key = _left_key(op, child)
    if key is None:
        return None
    # The predicate must read nothing beyond the key column and the build side;
    # anything else is a condition a filtering join has no place for.
    allowed = {values.ref.k}
    if isinstance(child, OperandLeaf):
        allowed.add(child.ref.k)
    if {ref.k for ref in op.predicate.iter_operand_refs()} != allowed:
        return None
    return (op, how, key, op.inputs[values.ref.k])


def _promote_to_filtering_join(op: SelectionOp, how: str, key: str, build: Op,
                               root: Op) -> Op:
    """Replace the selection with the equivalent semi/anti join, in place."""
    left = op.inputs[0]
    join = JoinOp(how=how, left_on=key,
                  # The build side is the sequence of key values itself, which is
                  # what an `isin` against a column produces; it has no key
                  # column of its own to name.
                  right_on=None,
                  inputs=[left, build], outputs=list(op.outputs))
    # Every producer loses the selection, including the column operand, which the
    # join reads by name instead and which stays in the graph for its other
    # consumers.
    seen: set[int] = set()
    for producer in op.inputs:
        if id(producer) in seen:
            continue
        seen.add(id(producer))
        producer.outputs = [out for out in producer.outputs if out is not op]
    for producer in (left, build):
        producer.add_output(join)
    op.replace_input_of_outputs(join)
    op.inputs = []
    op.outputs = []
    return join if root is op else root


promote_isin_to_filtering_join = rewrite_pass(_match_isin_against_a_relation,
                                              _promote_to_filtering_join)


def relational_rewrites(root: Op, semi_join: bool = True) -> Op:
    """Run the enabled relational rewrites, one pass each."""
    if semi_join:
        root = promote_isin_to_filtering_join(root)
    return root
