"""Relational rewrites on the logical IR.

The algebraic rewrites simplify *expressions*; these change the plan's operator
*shape*, replacing one relational operator with another that means the same thing
but that the cost model and join reordering can reason about.
"""
import operator

from stratum.optimizer._op_utils import rewrite_pass
from stratum.optimizer.logical._aggregation_ops import AggregateOp
from stratum.optimizer.logical._column_expr import (
    AggExpr, BinOpExpr, Col, ColumnMethodExpr, Const, OperandLeaf, UnaryOpExpr)
from stratum.optimizer.logical._index_ops import GroupKeysOp, IndexAccessOp
from stratum.optimizer.logical._join_ops import JoinOp
from stratum.optimizer.logical._ops import GetItemOp, Op, OperandRef, OutputType
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp
from stratum.optimizer.logical._selection_ops import SelectionKind, SelectionOp
from stratum.optimizer.logical._sort_ops import SortOp


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


def _match_counted_group_keys(op: Op):
    """Recognize proven value-count keys used only for membership.

    An arbitrary index is not a grouping key. The full producer shape and its
    sole consumer establish when the labels really are the counted values.
    """
    if not isinstance(op, IndexAccessOp) or len(op.inputs) != 1:
        return None
    if (len(op.outputs) != 1 or not isinstance(op.outputs[0], JoinOp)
            or op.outputs[0].how not in ("semi", "anti")
            or op.outputs[0].inputs[1] is not op):
        return None
    selection = op.inputs[0]
    if (not isinstance(selection, SelectionOp)
            or selection.kind is not SelectionKind.MASK
            or selection.output_type is not OutputType.SERIES
            or selection.outputs != [op] or len(selection.inputs) != 1):
        return None
    predicate = selection.predicate
    if (not isinstance(predicate, BinOpExpr) or predicate.op is not operator.ge
            or not isinstance(predicate.left, OperandLeaf)
            or predicate.left.ref.k != 0
            or not isinstance(predicate.right, Const)
            or type(predicate.right.value) not in (int, float)
            or predicate.right.value <= 0):
        return None
    upstream = selection.inputs[0]
    sort = upstream if isinstance(upstream, SortOp) else None
    if sort is not None:
        if (sort.outputs != [selection] or sort.by
                or len(sort.inputs) != 1):
            return None
        upstream = sort.inputs[0]
    agg = upstream
    if (not isinstance(agg, AggregateOp) or not agg.grouped
            or agg.output_type is not OutputType.SERIES
            or agg.outputs != [sort or selection]
            or len(agg.inputs) != 1
            or agg.options != {"sort": False, "dropna": True,
                               "observed": False, "sort_categories": True}):
        return None
    values = OperandLeaf(OperandRef(0))
    if (agg.grouping != (values,)
            or agg.aggregations != (("count", AggExpr("size", values)),)):
        return None
    projection = agg.inputs[0]
    if (not isinstance(projection, ColumnProjectionOp)
            or not isinstance(projection.key, str)
            or projection.outputs != [agg]
            or len(projection.inputs) != 1
            or projection.inputs[0].output_type is not OutputType.FRAME):
        return None
    return op, selection, sort, agg, projection


def _replace_counted_group_keys(op: IndexAccessOp, selection: SelectionOp,
                                sort: SortOp | None, agg: AggregateOp,
                                projection: ColumnProjectionOp, root: Op) -> Op:
    source = projection.inputs[0]
    key = projection.key
    count_name = "count" if key != "count" else "__count"
    count = AggregateOp(
        grouped=True, grouping=(Col(key),),
        aggregations=((count_name, AggExpr("size", Col(key))),),
        options={"sort": False, "dropna": agg.options["dropna"]},
        output_type=OutputType.FRAME, inputs=[source])
    filtered = SelectionOp(
        kind=SelectionKind.MASK,
        predicate=BinOpExpr(operator.ge, Col(count_name),
                            Const(selection.predicate.right.value)),
        inputs=[count])
    keys = GroupKeysOp(key=key, inputs=[filtered], outputs=list(op.outputs))
    source.outputs = [out for out in source.outputs if out is not projection]
    source.add_output(count)
    count.outputs = [filtered]
    filtered.outputs = [keys]
    op.replace_input_of_outputs(keys)
    # Detach the replaced cone so later graph passes see only live consumers.
    for old in (op, selection, sort, agg, projection):
        if old is not None:
            old.inputs = []
            old.outputs = []
    return keys if root is op else root


promote_counted_group_keys = rewrite_pass(_match_counted_group_keys,
                                          _replace_counted_group_keys)


def relational_rewrites(root: Op, semi_join: bool = True) -> Op:
    """Run the enabled relational rewrites, one pass each."""
    if semi_join:
        root = promote_isin_to_filtering_join(root)
        root = promote_counted_group_keys(root)
    return root
