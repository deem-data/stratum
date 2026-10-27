"""Relational rewrites on the logical IR.

The algebraic rewrites simplify *expressions*; these change the plan's operator
*shape*, replacing one relational operator with another that means the same thing
but that the cost model and join reordering can reason about.

They run after schema propagation, and each one keeps the schemas current: the
subgraph a rewrite installs produces what the replaced one did, so only the ops it
creates need a schema, which the rewrite propagates itself (inputs first). A later
rewrite in this stage can then read any op's ``output_schema``.
"""
import operator

from stratum.optimizer._op_utils import rewrite_pass
from stratum.optimizer.logical._aggregation_ops import AggregateOp
from stratum.optimizer.logical._column_expr import (
    AggExpr, BinOpExpr, Col, Const, IsinExpr, OperandLeaf, UnaryOpExpr)
from stratum.optimizer.logical._index_ops import GroupKeysOp, IndexAccessOp
from stratum.optimizer.logical._join_ops import FilteringJoinOp
from stratum.optimizer.logical._map_ops import AssignMapOp
from stratum.optimizer.logical._ops import GetItemOp, Op, OperandRef, OutputType
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp, DropOp
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
    # The predicate must read nothing beyond the key column and the build side;
    # anything else is a condition a filtering join has no place for.
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


def _kept_columns(projection: Op) -> list[str] | None:
    """The columns a pure column projection keeps, in order, or ``None``.

    Only projections that keep every row and pick columns by name qualify. A literal
    ``df[[...]]`` names its columns itself. A ``drop`` needs the schema: the kept
    columns are whatever its input has besides the dropped ones, and every dropped
    one must exist there, since dropping a missing column raises.
    """
    if len(projection.inputs) != 1:
        return None
    if isinstance(projection, ColumnProjectionOp):
        names = _schema.as_column_list(projection.key)
        if isinstance(projection.key, str) or names is None or len(set(names)) != len(names):
            return None
        return names
    if isinstance(projection, DropOp):
        source = projection.inputs[0].output_schema
        dropped = _schema.as_column_list(projection.columns)
        if (not _schema.is_known(source) or dropped is None
                or not set(dropped) <= set(source)
                or projection.level is not None or projection.errors != "raise"):
            return None
        return [name for name in source if name not in dropped]
    return None


def _column_of(leaf: Op, sources: dict[int, set[str] | None]) -> str | None:
    """The column name ``leaf`` reads, if it is one column of a frame in ``sources``.

    ``sources`` maps a frame's id to the columns it can be read for, ``None`` for all.
    """
    if not (isinstance(leaf, (ColumnProjectionOp, GetItemOp))
            and isinstance(leaf.key, str) and len(leaf.inputs) == 1
            and id(leaf.inputs[0]) in sources):
        return None
    allowed = sources[id(leaf.inputs[0])]
    return leaf.key if allowed is None or leaf.key in allowed else None


def _match_projection_under_map(op: Op):
    """Match a map over a column projection whose leaves read the unprojected frame.

    ``df[["a", "b"]].assign(c=df["x"] * 2)`` folds against the projection, so
    ``df["x"]`` stays an input of the map and is materialised on its own. The map
    can run on ``df`` itself, where ``x`` is a column, with the projection after it.
    """
    if not isinstance(op, AssignMapOp):
        return None
    projection = op.inputs[0]
    kept = _kept_columns(projection)
    if kept is None:
        return None
    source = projection.inputs[0]
    if source.output_type is not OutputType.FRAME:
        return None
    if any(ref.k == 0 for expr in op.entries.values() for ref in expr.iter_operand_refs()):
        return None  # the projected frame itself is an operand, not just a column source
    # A column of the projection is the same column of its source.
    sources = {id(source): None, id(projection): set(kept)}
    columns = {k: name for k, leaf in enumerate(op.inputs[1:], start=1)
               if (name := _column_of(leaf, sources)) is not None}
    if not columns:
        return None
    return op, projection, kept, columns


def _push_projection_through_map(op: AssignMapOp, projection: Op, kept: list[str],
                                 columns: dict[int, str], root: Op) -> Op:
    """Run the map on the projection's source and project its result instead.

    The map evaluates all entries against its input frame before assigning, so a
    column read through a leaf is the same column read from the source. Projected
    columns keep their order and entries overwrite in place or append, as pandas
    ``assign`` does on the projected frame.
    """
    source = projection.inputs[0]
    old_inputs = list(op.inputs)
    inputs = [source]
    mapping = {0: 0}
    for k, leaf in enumerate(old_inputs[1:], start=1):
        if k in columns:
            mapping[k] = Col(columns[k])
        elif any(in_ is leaf for in_ in inputs):
            mapping[k] = next(i for i, in_ in enumerate(inputs) if in_ is leaf)
        else:
            mapping[k] = len(inputs)
            inputs.append(leaf)
    op.entries = {name: expr.remap_operand_refs(mapping)
                  for name, expr in op.entries.items()}
    for producer in dict.fromkeys(old_inputs):
        producer.outputs = [out for out in producer.outputs if out is not op]
    op.inputs = inputs
    for producer in dict.fromkeys(inputs):
        producer.add_output(op)

    keep = kept + [name for name in op.entries if name not in kept]
    pushed = ColumnProjectionOp(key=keep, inputs=[op], outputs=list(op.outputs))
    pushed.is_X, pushed.is_y = op.is_X, op.is_y
    op.is_X = op.is_y = False
    for consumer in dict.fromkeys(op.outputs):
        while any(in_ is op for in_ in consumer.inputs):
            consumer.replace_input(op, pushed)
    op.outputs = [pushed]

    # Every surviving producer has its consumer now, so the dead ones can go.
    for old in dict.fromkeys(old_inputs):
        if not old.outputs:
            old.detach_and_prune()
    op.propagate_output_schema()
    pushed.propagate_output_schema()
    return pushed if root is op else root


push_projection_through_map = rewrite_pass(_match_projection_under_map,
                                           _push_projection_through_map)


def relational_rewrites(root: Op, semi_join: bool = True,
                        projection_pushup: bool = True) -> Op:
    """Run the enabled relational rewrites, one pass each."""
    if semi_join:
        root = promote_isin_to_filtering_join(root)
        root = promote_counted_group_keys(root)
    if projection_pushup:
        root = push_projection_through_map(root)
    return root
