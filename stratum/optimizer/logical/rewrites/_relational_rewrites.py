"""Relational rewrites on the logical IR.

Replace relational operator patterns with equivalent operations, including
semi/anti joins.

They run after schema propagation, and each one keeps the schemas current: the
subgraph a rewrite installs produces what the replaced one did. Each rewrite
updates schemas for operators it creates or changes, inputs first. A later
rewrite in this stage can then read any op's ``output_schema``.
"""
import operator

from stratum.optimizer._op_utils import rewrite_pass
from stratum.optimizer.logical._aggregation_ops import AggregateOp
from stratum.optimizer.logical._column_expr import (
    AggExpr, BinOpExpr, Col, Const, IsinExpr, OperandLeaf, UnaryOpExpr)
from stratum.optimizer.logical._index_ops import GroupKeysOp, IndexAccessOp
from stratum.optimizer.logical._join_ops import FilteringJoinOp
from stratum.optimizer.logical._ops import GetItemOp, Op, OperandRef, OutputType
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp
from stratum.optimizer.logical import _schema
from stratum.optimizer.logical._selection_ops import MaskSelectionOp
from stratum.optimizer.logical._sort_ops import SortOp


def _membership_predicate(predicate):
    """``(anti, (child, values))`` for a membership test, else ``None``."""
    anti = False
    if isinstance(predicate, UnaryOpExpr) and predicate.op is operator.invert:
        anti, predicate = True, predicate.operand
    if not isinstance(predicate, IsinExpr):
        return None
    return anti, (predicate.operand, predicate.values)


def _left_key(op: MaskSelectionOp, child) -> str | None:
    """The left relation's key column for the membership test's subject.

    Usually the column is folded into a ``Col``. A shared column projection may
    remain an operand, for example when the same column feeds an aggregation.
    Both representations qualify if they read a column of the left relation.
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

    Keep literal collections as predicates and leave filtered Series unchanged.
    This rewrite inspects operand producers to recognize a relation; extraction
    classifies only the individual operator.
    """
    if not isinstance(op, MaskSelectionOp):
        return None
    if op.output_type is not OutputType.FRAME:
        return None
    matched = _membership_predicate(op.predicate)
    if matched is None:
        return None
    anti, (child, values) = matched
    if not isinstance(values, OperandLeaf):
        return None
    key = _left_key(op, child)
    if key is None:
        return None
    # This filtering-join representation supports only the key column and build
    # side; additional predicate operands cannot be represented here.
    allowed = {values.ref.k}
    if isinstance(child, OperandLeaf):
        allowed.add(child.ref.k)
    if {ref.k for ref in op.predicate.iter_operand_refs()} != allowed:
        return None
    return (op, anti, key, op.inputs[values.ref.k])


def _promote_to_filtering_join(op: MaskSelectionOp, anti: bool, key: str, build: Op,
                               root: Op) -> Op:
    """Replace the selection with the equivalent semi/anti join, in place."""
    left = op.inputs[0]
    join = FilteringJoinOp(anti=anti, left_on=key,
                           # The build side is the sequence of key values itself,
                           # which is what an `isin` against a column produces; it
                           # has no key column of its own to name.
                           right_on=None,
                           inputs=[left, build], outputs=list(op.outputs))
    # Every producer loses the selection, including the column operand, which the
    # join reads by name instead and which stays in the graph for its other
    # consumers.
    for producer in (left, build):
        producer.add_output(join)
    op.replace_input_of_outputs(join)
    op.detach()
    join.propagate_output_schema()
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
    if (len(op.outputs) != 1 or not isinstance(op.outputs[0], FilteringJoinOp)
            or op.outputs[0].inputs[1] is not op):
        return None
    selection = op.inputs[0]
    if (not isinstance(selection, MaskSelectionOp)
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
            or (agg.sort_keys, agg.drop_null_keys, agg.observed, agg.keys_as_columns,
                agg.level, agg.sort_categories) != (False, True, False, False, None, True)):
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


def _replace_counted_group_keys(op: IndexAccessOp, selection: MaskSelectionOp,
                                sort: SortOp | None, agg: AggregateOp,
                                projection: ColumnProjectionOp, root: Op) -> Op:
    source = projection.inputs[0]
    key = projection.key
    count_name = "count" if key != "count" else "__count"
    count = AggregateOp(
        grouped=True, grouping=(Col(key),),
        aggregations=((count_name, AggExpr("size", Col(key))),),
        sort_keys=False, drop_null_keys=agg.drop_null_keys,
        output_type=OutputType.FRAME, inputs=[source])
    filtered = MaskSelectionOp(
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
            old.detach()
    for new in (count, filtered, keys):
        new.propagate_output_schema()
    return keys if root is op else root


promote_counted_group_keys = rewrite_pass(_match_counted_group_keys,
                                          _replace_counted_group_keys)


def relational_rewrites(root: Op, semi_join: bool = True) -> Op:
    """Run the enabled relational rewrites, one pass each."""
    if semi_join:
        root = promote_isin_to_filtering_join(root)
        root = promote_counted_group_keys(root)
    return root
