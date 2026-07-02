"""Backend-agnostic column expression tree.

Used by selections (boolean predicates) and maps (computed columns).
Expressions are immutable value types and compare structurally.
"""
from __future__ import annotations
import operator

import polars as pl

from stratum.optimizer.ir._ops import OperandRef, BinOp, UnaryOp, GetItemOp, Op
from stratum.optimizer.ir._projection_ops import StringMethodOp, STR_POLARS_METHODS

# operator callable -> symbol. A binary/unary op whose callable is not in the
# corresponding map is not foldable into a column expression.
BINARY_SYMBOLS = {
    operator.gt: ">", operator.lt: "<", operator.ge: ">=", operator.le: "<=",
    operator.eq: "==", operator.ne: "!=",
    operator.and_: "&", operator.or_: "|", operator.xor: "^",
    operator.add: "+", operator.sub: "-", operator.mul: "*",
    operator.truediv: "/", operator.floordiv: "//", operator.mod: "%",
    operator.pow: "**",
}
UNARY_SYMBOLS = {operator.invert: "~", operator.neg: "-", operator.pos: "+"}


class ColumnExpr:
    """Base class for column-expression nodes."""
    __slots__ = ()

    def _key(self):
        raise NotImplementedError

    def __eq__(self, other):
        return type(self) is type(other) and self._key() == other._key()

    def __hash__(self):
        return hash((type(self).__name__, self._key()))

    def to_pandas(self, frame, inputs):
        """Evaluate the expression against a pandas frame."""
        raise NotImplementedError

    def to_polars(self, inputs):
        """Convert the expression to a Polars expression."""
        raise NotImplementedError

    def to_pandas_query(self, params: dict) -> str | None:
        """Return a pandas ``query()`` string, or ``None`` if unsupported.

        Literals are bound into ``params`` and referenced as ``@p<i>``. A ``None``
        anywhere falls back to the boolean-mask path (``to_pandas``).
        """
        return None

    def iter_operand_refs(self):
        """Yield all referenced ``OperandRef`` objects."""
        return iter(())

    def remap_operand_refs(self, mapping: dict) -> "ColumnExpr":
        """Return a copy with operand references remapped."""
        return self


class Col(ColumnExpr):
    """Reference to a source-frame column."""
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def _key(self):
        return self.name

    def __repr__(self):
        return f"Col({self.name!r})"

    def to_pandas(self, frame, inputs):
        return frame[self.name]

    def to_polars(self, inputs):
        return pl.col(self.name)

    def to_pandas_query(self, params):
        # Backtick the name so spaces / keywords / dots stay valid inside the query.
        return f"`{self.name}`"


class Const(ColumnExpr):
    """Literal scalar value."""
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def _key(self):
        try:
            hash(self.value)
        except TypeError:
            return ("__id__", id(self.value))
        return self.value

    def __repr__(self):
        return f"Const({self.value!r})"

    def to_pandas(self, frame, inputs):
        return self.value

    def to_polars(self, inputs):
        return pl.lit(self.value)

    def to_pandas_query(self, params):
        # Bind the literal as a real object (referenced as @p<i>) rather than
        # stringifying it -- keeps timestamps/strings/NaN intact in the query.
        name = f"p{len(params)}"
        params[name] = self.value
        return f"@{name}"


class OperandLeaf(ColumnExpr):
    """Reference to an operator input that was not folded."""
    __slots__ = ("ref",)

    def __init__(self, ref):
        self.ref = ref

    def _key(self):
        return self.ref

    def __repr__(self):
        return f"OperandLeaf({self.ref})"

    def to_pandas(self, frame, inputs):
        return inputs[self.ref.k]

    def to_polars(self, inputs):
        return inputs[self.ref.k]

    def iter_operand_refs(self):
        yield self.ref

    def remap_operand_refs(self, mapping):
        return OperandLeaf(OperandRef(mapping[self.ref.k]))


class BinOpExpr(ColumnExpr):
    """Binary operation on two expressions."""
    __slots__ = ("op", "left", "right")

    def __init__(self, op, left: ColumnExpr, right: ColumnExpr):
        self.op = op
        self.left = left
        self.right = right

    def _key(self):
        return (self.op, self.left, self.right)

    def __repr__(self):
        return f"({self.left!r} {BINARY_SYMBOLS.get(self.op, self.op)} {self.right!r})"

    def to_pandas(self, frame, inputs):
        return self.op(self.left.to_pandas(frame, inputs),
                       self.right.to_pandas(frame, inputs))

    def to_polars(self, inputs):
        return self.op(self.left.to_polars(inputs), self.right.to_polars(inputs))

    def to_pandas_query(self, params):
        sym = BINARY_SYMBOLS.get(self.op)
        if sym is None:
            return None
        left = self.left.to_pandas_query(params)
        right = self.right.to_pandas_query(params)
        if left is None or right is None:
            return None
        return f"({left} {sym} {right})"

    def iter_operand_refs(self):
        yield from self.left.iter_operand_refs()
        yield from self.right.iter_operand_refs()

    def remap_operand_refs(self, mapping):
        return BinOpExpr(self.op, self.left.remap_operand_refs(mapping),
                         self.right.remap_operand_refs(mapping))


class UnaryOpExpr(ColumnExpr):
    """Unary operation on an expression."""
    __slots__ = ("op", "operand")

    def __init__(self, op, operand: ColumnExpr):
        self.op = op
        self.operand = operand

    def _key(self):
        return (self.op, self.operand)

    def __repr__(self):
        return f"{UNARY_SYMBOLS.get(self.op, self.op)}({self.operand!r})"

    def to_pandas(self, frame, inputs):
        return self.op(self.operand.to_pandas(frame, inputs))

    def to_polars(self, inputs):
        return self.op(self.operand.to_polars(inputs))

    def to_pandas_query(self, params):
        sym = UNARY_SYMBOLS.get(self.op)
        if sym is None:
            return None
        operand = self.operand.to_pandas_query(params)
        if operand is None:
            return None
        return f"({sym}{operand})"

    def iter_operand_refs(self):
        yield from self.operand.iter_operand_refs()

    def remap_operand_refs(self, mapping):
        return UnaryOpExpr(self.op, self.operand.remap_operand_refs(mapping))


class StrExpr(ColumnExpr):
    """String accessor call (``.str.<method>()``)."""
    __slots__ = ("operand", "method", "args", "kwargs")

    def __init__(self, operand: ColumnExpr, method: str, args=(), kwargs=None):
        self.operand = operand
        self.method = method
        self.args = tuple(args)
        self.kwargs = kwargs or {}

    def _key(self):
        return (self.operand, self.method, self.args, frozenset(self.kwargs.items()))

    def __repr__(self):
        inner = ", ".join([repr(self.operand)]
                          + [repr(a) for a in self.args]
                          + [f"{k}={v!r}" for k, v in self.kwargs.items()])
        return f"str.{self.method}({inner})"

    def to_pandas(self, frame, inputs):
        obj = self.operand.to_pandas(frame, inputs)
        return getattr(obj.str, self.method)(*self.args, **self.kwargs)

    def to_polars(self, inputs):
        obj = self.operand.to_polars(inputs)
        name = STR_POLARS_METHODS.get(self.method, self.method)
        return getattr(obj.str, name)(*self.args, **self.kwargs)

    def iter_operand_refs(self):
        yield from self.operand.iter_operand_refs()

    def remap_operand_refs(self, mapping):
        return StrExpr(self.operand.remap_operand_refs(mapping),
                       self.method, self.args, self.kwargs)


# --- Conversion: op subgraph -> ColumnExpr -----------------------------------

class _Folder:
    """Fold an operator subgraph into a ``ColumnExpr``.

    Runs three passes: :meth:`_discover` collects the foldable cone,
    :meth:`_absorbable` keeps only nodes private to it, and :meth:`_build`
    materialises the tree, emitting an :class:`OperandLeaf` for the rest. The
    new operator's inputs are ``[src, *leaf_ops]``.
    """

    def __init__(self, src: Op):
        self.src = src
        self.absorbed: list[Op] = []
        self._absorbed_ids: set[int] = set()
        self.leaf_ops: list[Op] = []
        self._leaf_index: dict[int, int] = {}

    def fold(self, root: Op, root_consumer: Op) -> ColumnExpr:
        assert any(o is root_consumer for o in root.outputs)
        cone, child_ops = self._discover(root)
        absorbable = self._absorbable(root, root_consumer, cone, child_ops)
        return self._build(root, absorbable, child_ops)

    # --- structural classification -------------------------------------------

    def _is_foldable(self, node: Op) -> bool:
        """Return whether ``node`` can be represented as a ``ColumnExpr``."""
        if isinstance(node, BinOp):
            return node.op in BINARY_SYMBOLS
        if isinstance(node, UnaryOp):
            return node.op in UNARY_SYMBOLS
        if isinstance(node, GetItemOp):
            # A column of the source frame: df["col"]. Anything else (a chained
            # getitem, a non-string key) is not a Col leaf.
            return (isinstance(node.key, str) and bool(node.inputs)
                    and node.inputs[0] is self.src)
        if isinstance(node, StringMethodOp):
            # A graph-fed arg isn't representable in StrExpr; such a call stays a leaf.
            return (not any(isinstance(a, OperandRef) for a in (node.args or ()))
                    and not any(isinstance(v, OperandRef)
                                for v in (node.kwargs or {}).values()))
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
        if isinstance(node, StringMethodOp):
            return [node.inputs[0]]
        return []

    # --- pass 1: discover the foldable cone -----------------------------------

    def _discover(self, root: Op) -> tuple[dict[int, Op], dict[int, list[Op]]]:
        """Collect the foldable subgraph rooted at ``root``."""
        cone: dict[int, Op] = {}
        child_ops: dict[int, list[Op]] = {}
        stack = [root]
        while stack:
            node = stack.pop()
            if id(node) in cone or not self._is_foldable(node):
                continue
            cone[id(node)] = node
            children = [p for p in self._producer_ops(node)
                        if p is not self.src and self._is_foldable(p)]
            child_ops[id(node)] = children
            stack.extend(children)
        return cone, child_ops

    # --- pass 2: which cone nodes are fully private ---------------------------

    def _absorbable(self, root: Op, root_consumer: Op,
                    cone: dict[int, Op], child_ops: dict[int, list[Op]]) -> set[int]:
        """Return foldable nodes with no external consumers."""
        dropped: set[int] = set()
        stack: list[Op] = []
        for nid, node in cone.items():
            for consumer in node.outputs:
                internal = (id(consumer) in cone
                            or (node is root and consumer is root_consumer))
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
        return {nid for nid in cone if nid not in dropped}

    # --- pass 3: materialise the expression bottom-up -------------------------

    def _build(self, root: Op, absorbable: set[int],
               child_ops: dict[int, list[Op]]) -> ColumnExpr:
        """Build the expression tree from absorbable nodes."""
        if id(root) not in absorbable:
            return self._leaf(root)
        # Iterative post-order over the absorbed sub-DAG: an operand is built (and
        # memoised) before the node that consumes it, so shared nodes fold once.
        order: list[Op] = []
        visited: set[int] = set()
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
        memo: dict[int, ColumnExpr] = {}
        for node in order:
            memo[id(node)] = self._make_expr(node, absorbable, memo)
            self._absorb(node)
        return memo[id(root)]

    def _make_expr(self, node: Op, absorbable: set[int],
                   memo: dict[int, ColumnExpr]) -> ColumnExpr:
        if isinstance(node, BinOp):
            return BinOpExpr(node.op,
                             self._operand(node.left, node, absorbable, memo),
                             self._operand(node.right, node, absorbable, memo))
        if isinstance(node, UnaryOp):
            return UnaryOpExpr(node.op,
                               self._operand(node.operand, node, absorbable, memo))
        if isinstance(node, GetItemOp):
            return Col(node.key)
        # StringMethodOp: the .str accessor was fused away in frame extraction, so the
        # column is just inputs[0]; args/kwargs are literals (checked in _is_foldable).
        operand = self._resolve(node.inputs[0], absorbable, memo)
        return StrExpr(operand, node.method,
                       tuple(node.args or ()), dict(node.kwargs or {}))

    def _operand(self, operand, parent: Op, absorbable: set[int],
                 memo: dict[int, ColumnExpr]) -> ColumnExpr:
        if isinstance(operand, OperandRef):
            return self._resolve(parent.inputs[operand.k], absorbable, memo)
        return Const(operand)

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


def fold_column_expr(root_node: Op, src: Op, root_consumer: Op):
    """Fold an operator subgraph into a column expression.

    Returns ``(expr, absorbed_ops, leaf_ops)``.
    """
    folder = _Folder(src)
    expr = folder.fold(root_node, root_consumer)
    return expr, folder.absorbed, folder.leaf_ops
