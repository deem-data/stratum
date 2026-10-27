"""Folding an op subgraph into a :class:`~stratum.optimizer.logical._column_expr.ColumnExpr`.

Mask selections, assign maps and aggregations absorb the row-local ops feeding
them into one expression tree, so the whole computation is one operator (and one
kernel on a lazy backend). This module is the shared folder; each family's
factory decides which roots to fold and installs the operator it builds.
"""
from __future__ import annotations
import datetime
import decimal

import numpy as np

from stratum.optimizer.logical._ops import (OperandRef, BinOp, UnaryOp, GetItemOp, Op,
                                            ValueOp)
from stratum.optimizer.logical._column_expr import (
    BINARY_SYMBOLS, UNARY_SYMBOLS, BinOpExpr, Col, ColumnExpr, Const,
    DatetimeExpr, DtExpr, OperandLeaf, StrExpr, UnaryOpExpr)
from stratum.optimizer.logical._column_methods import ColumnMethodOp
from stratum.optimizer.logical._map_ops import (
    DatetimeConversionOp, DatetimeFieldOp, StringMethodOp)
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp

# Values a ``ValueOp`` may carry into a ``Const``. A ``Const`` is a scalar
# broadcast to every row: a list would become one list-valued cell repeated per
# row and a pandas Series is not a scalar at all. Container values therefore
# stay ``OperandLeaf``s. pandas Timestamp/Timedelta subclass the datetime types.
CONST_SCALAR_TYPES = (
    bool, int, float, str,
    datetime.date, datetime.time, datetime.timedelta,
    decimal.Decimal, np.generic,
)


class _Folder:
    """Fold operator subgraphs into ``ColumnExpr`` trees.

    Three passes: :meth:`_discover` collects the foldable subgraph,
    :meth:`_absorbable` keeps nodes without external consumers, :meth:`_build`
    materialises the tree (an :class:`OperandLeaf` for the rest). The new
    operator's inputs are ``[src, *leaf_ops]``; :meth:`install` wires it in.

    ``fold_many`` folds several roots against one shared cone and memo, so a
    producer feeding two roots is absorbed once and both trees share the
    sub-expression.
    """

    def __init__(self, src: Op):
        self.src = src
        self.absorbed: list[Op] = []
        self._absorbed_ids: set[int] = set()
        self.leaf_ops: list[Op] = []
        self._leaf_index: dict[int, int] = {}

    def fold(self, root: Op, root_consumer: Op) -> ColumnExpr:
        return self.fold_many([root], root_consumer)[0]

    def fold_many(self, roots: list[Op], root_consumer: Op) -> list[ColumnExpr]:
        for root in roots:
            assert any(o is root_consumer for o in root.outputs)
        subgraph, child_ops = self._discover(roots)
        absorbable = self._absorbable(roots, root_consumer, subgraph, child_ops)
        memo: dict[int, ColumnExpr] = {}
        return [self._build(root, absorbable, child_ops, memo) for root in roots]

    def install(self, new_op: Op, replaced: list) -> None:
        """Take the absorbed ops out of the graph and wire ``new_op`` in.

        ``new_op.inputs`` must be ``[src, *leaf_ops]``. Each of those producers
        feeds ``new_op`` in place of the ``replaced`` ops (``None`` entries are
        skipped), which lose their inputs. Downstream consumers of the replaced
        ops are rewired by the caller.
        """
        for node in self.absorbed:
            node.detach()
        replaced = [node for node in replaced if node is not None]
        replaced_ids = {id(node) for node in replaced}
        for producer in new_op.inputs:
            producer.outputs = [o for o in producer.outputs if id(o) not in replaced_ids]
            producer.add_output(new_op)
        for node in replaced:
            if node is not new_op:
                node.inputs = []

    # --- structural classification -------------------------------------------

    def _is_foldable(self, node: Op) -> bool:
        """Return whether ``node`` can be represented as a ``ColumnExpr``."""
        if isinstance(node, ValueOp):
            # A graph-fed constant (a Var resolved from env, or as_data_op(5)).
            return isinstance(node.value, CONST_SCALAR_TYPES)
        if isinstance(node, BinOp):
            return node.op in BINARY_SYMBOLS
        if isinstance(node, UnaryOp):
            return node.op in UNARY_SYMBOLS
        if isinstance(node, (GetItemOp, ColumnProjectionOp)):
            # A single column of the source frame: df["col"] (a bare GetItem, or
            # its rewritten ColumnProjectionOp form). Anything else (a chained
            # getitem, a list/sub-frame key, a non-string key) is not a Col leaf.
            return (isinstance(node.key, str) and bool(node.inputs)
                    and node.inputs[0] is self.src)
        if isinstance(node, StringMethodOp):
            # A graph-fed arg isn't representable in the expr; such a call stays a leaf.
            return not any(isinstance(v, OperandRef) for v in node.params.values())
        if isinstance(node, DatetimeConversionOp):
            # A graph-fed option isn't representable in the expr; the op stays a leaf.
            return not any(isinstance(getattr(node, f), OperandRef) for f in node.fields)
        if isinstance(node, ColumnMethodOp):
            return True
        if isinstance(node, DatetimeFieldOp):
            return True
        return False

    def _producer_ops(self, node: Op) -> list[Op]:
        """Return foldable operand producers for ``node``."""
        if isinstance(node, BinOp):
            return [node.inputs[r.k] for r in (node.left, node.right)
                    if isinstance(r, OperandRef)]
        if isinstance(node, UnaryOp):
            if isinstance(node.operand, OperandRef):
                return [node.inputs[node.operand.k]]
            return []
        if isinstance(node, (StringMethodOp, DatetimeConversionOp,
                             DatetimeFieldOp)):
            return [node.inputs[0]]
        if isinstance(node, ColumnMethodOp):
            # Every remaining input is referenced by a field. Returning all
            # inputs lets expression-valued method arguments join the fold cone;
            # non-foldable/external producers become OperandLeafs later.
            return list(node.inputs)
        return []

    # --- pass 1: discover the foldable subgraph -----------------------------------

    def _discover(self, roots: list[Op]) -> tuple[dict[int, Op], dict[int, list[Op]]]:
        """Collect the foldable subgraph rooted at ``roots``."""
        subgraph: dict[int, Op] = {}
        child_ops: dict[int, list[Op]] = {}
        stack = list(roots)
        while stack:
            node = stack.pop()
            if id(node) in subgraph or not self._is_foldable(node):
                continue
            subgraph[id(node)] = node
            children = [p for p in self._producer_ops(node)
                        if p is not self.src and self._is_foldable(p)]
            child_ops[id(node)] = children
            stack.extend(children)
        return subgraph, child_ops

    # --- pass 2: which subgraph nodes have no external consumers ---------------

    def _absorbable(self, roots: list[Op], root_consumer: Op,
                    subgraph: dict[int, Op], child_ops: dict[int, list[Op]]) -> set[int]:
        """Return foldable nodes with no external consumers."""
        root_ids = {id(r) for r in roots}
        dropped: set[int] = set()
        stack: list[Op] = []
        for nid, node in subgraph.items():
            for consumer in node.outputs:
                internal = (id(consumer) in subgraph
                            or (nid in root_ids and consumer is root_consumer))
                if not internal:
                    dropped.add(nid)
                    stack.append(node)
                    break
        while stack:
            node = stack.pop()
            for child in child_ops[id(node)]:
                if id(child) not in dropped:
                    dropped.add(id(child))
                    stack.append(child)
        return {nid for nid in subgraph if nid not in dropped}

    # --- pass 3: materialise the expression bottom-up -------------------------

    def _build(self, root: Op, absorbable: set[int], child_ops: dict[int, list[Op]],
               memo: dict[int, ColumnExpr]) -> ColumnExpr:
        """Build the expression tree from absorbable nodes."""
        if id(root) not in absorbable:
            return self._leaf(root)
        if id(root) in memo:
            return memo[id(root)]
        # Iterative post-order over the absorbed sub-DAG: an operand is built (and
        # memoised) before the node that consumes it, so shared nodes fold once --
        # also across roots, since the memo is shared by ``fold_many``.
        order: list[Op] = []
        visited: set[int] = set(memo)
        stack = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for child in child_ops[id(node)]:
                if id(child) in absorbable and id(child) not in visited:
                    stack.append((child, False))
        for node in order:
            memo[id(node)] = self._make_expr(node, absorbable, memo)
            self._absorb(node)
        return memo[id(root)]

    def _make_expr(self, node: Op, absorbable: set[int],
                   memo: dict[int, ColumnExpr]) -> ColumnExpr:
        if isinstance(node, ValueOp):
            return Const(node.value)
        if isinstance(node, BinOp):
            return BinOpExpr(node.op,
                             self._operand(node.left, node, absorbable, memo),
                             self._operand(node.right, node, absorbable, memo))
        if isinstance(node, UnaryOp):
            return UnaryOpExpr(node.op,
                               self._operand(node.operand, node, absorbable, memo))
        if isinstance(node, (GetItemOp, ColumnProjectionOp)):
            return Col(node.key)
        if isinstance(node, StringMethodOp):
            # The .str accessor was fused away in frame extraction, so the column is
            # just inputs[0]; the params are literals (checked in _is_foldable).
            operand = self._resolve(node.inputs[0], absorbable, memo)
            return StrExpr(operand, node.method, node.params)
        if isinstance(node, DatetimeConversionOp):
            operand = self._resolve(node.inputs[0], absorbable, memo)
            return DatetimeExpr(operand, node.format, node.exact, node.cache,
                                node.errors)
        if isinstance(node, ColumnMethodOp):
            operand = self._resolve(node.inputs[0], absorbable, memo)
            return node.expr_class(operand, **{
                name: self._method_value(getattr(node, name), node, absorbable, memo)
                for name in node.fields})
        if isinstance(node, DatetimeFieldOp):
            operand = self._resolve(node.inputs[0], absorbable, memo)
            return DtExpr(operand, node.field)
        raise AssertionError(f"unfoldable node reached _make_expr: {node!r}")

    def _operand(self, operand, parent: Op, absorbable: set[int],
                 memo: dict[int, ColumnExpr]) -> ColumnExpr:
        if isinstance(operand, OperandRef):
            return self._resolve(parent.inputs[operand.k], absorbable, memo)
        return Const(operand)

    def _method_value(self, value, parent: Op, absorbable: set[int],
                      memo: dict[int, ColumnExpr]):
        if isinstance(value, OperandRef):
            return self._resolve(parent.inputs[value.k], absorbable, memo)
        if isinstance(value, tuple):
            return tuple(self._method_value(item, parent, absorbable, memo)
                         for item in value)
        if isinstance(value, list):
            return [self._method_value(item, parent, absorbable, memo)
                    for item in value]
        if isinstance(value, dict):
            return {key: self._method_value(item, parent, absorbable, memo)
                    for key, item in value.items()}
        return value

    def _resolve(self, node: Op, absorbable: set[int],
                 memo: dict[int, ColumnExpr]) -> ColumnExpr:
        """Return the folded expression or an ``OperandLeaf``."""
        if id(node) in absorbable:
            return memo[id(node)]
        return self._leaf(node)

    def _absorb(self, node: Op) -> None:
        if id(node) not in self._absorbed_ids:
            self._absorbed_ids.add(id(node))
            self.absorbed.append(node)

    def _leaf(self, node: Op) -> OperandLeaf:
        if node is self.src:
            return OperandLeaf(OperandRef(0))
        idx = self._leaf_index.get(id(node))
        if idx is None:
            idx = len(self.leaf_ops)
            self.leaf_ops.append(node)
            self._leaf_index[id(node)] = idx
        return OperandLeaf(OperandRef(1 + idx))
