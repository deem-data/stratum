"""Backend-agnostic column expression tree.

Used by selections (boolean predicates) and maps (computed columns).
Expressions are immutable value types and compare structurally.

Every node maps 1:1 onto a polars expression (``pl.col``/``pl.lit``,
arithmetic/boolean operators, the ``.str``/``.dt`` namespaces, datetime
parsing), so a whole tree compiles into a single backend kernel. Anything
outside the grammar (fitted transformers, UDFs, data from other frames) is
referenced through an :class:`OperandLeaf`.

Evaluation goes through an :class:`EvalContext` carrying the source frame, the
op's resolved inputs and the execution mode.

Trees may share sub-expressions by identity (a DAG). Structural walks
(:func:`iter_postorder`, :func:`rewrite_leaves`, operand-ref iteration and
remapping) visit every shared node once, so their cost is linear in the number
of distinct nodes rather than in the size of the unfolded tree.
"""
from __future__ import annotations
import math
import operator
from typing import Callable, Iterable, Mapping

import polars as pl
import pandas as pd

from stratum.optimizer.logical._ops import OperandRef, BinOp, UnaryOp, GetItemOp, Op
from stratum.optimizer.logical._column_methods import (
    ColumnMethodOp, get_column_method_spec, polars_dtype)
from stratum.optimizer.logical._projection_ops import (
    ColumnProjectionOp, DatetimeConversionOp, GetAttrProjectionOp, StringMethodOp,
    STR_POLARS_METHODS, polars_datetime_kwargs)

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


class EvalContext:
    """Everything a column expression needs at evaluation time.

    ``frame`` is evaluated against (the op's primary operand); ``inputs`` are the
    op's resolved input values (read by :class:`OperandLeaf`); ``mode`` is
    ``fit_transform`` or ``predict`` (unused by the current stateless grammar).
    ``temps`` holds, per step of a planned map program, what a kernel exposes
    to ``TempRef``: the evaluated value on pandas, a column reference into the
    lazy plan on polars. It is ``None`` outside such kernels.
    """
    __slots__ = ("frame", "inputs", "mode", "temps")

    def __init__(self, frame, inputs, mode: str = "fit_transform", temps=None):
        self.frame = frame
        self.inputs = inputs
        self.mode = mode
        self.temps = temps


class ColumnExpr:
    """Base class for column-expression nodes."""
    __slots__ = ("_hash_cache",)

    def _key(self):
        raise NotImplementedError

    def __eq__(self, other):
        if self is other:
            return True
        return type(self) is type(other) and self._key() == other._key()

    def __hash__(self):
        # Cached: the key embeds the children, so an uncached hash re-walks the
        # whole sub-DAG on every lookup.
        try:
            return self._hash_cache
        except AttributeError:
            h = hash((type(self).__name__, self._key()))
            self._hash_cache = h
            return h

    def children(self) -> tuple["ColumnExpr", ...]:
        """Direct sub-expressions, in a fixed order matching :meth:`with_children`."""
        raise TypeError(
            f"ColumnExpr node {type(self).__name__} does not declare its children")

    def with_children(self, children: tuple["ColumnExpr", ...]) -> "ColumnExpr":
        """Return a copy of this node with ``children`` replacing :meth:`children`."""
        raise TypeError(
            f"ColumnExpr node {type(self).__name__} does not declare its children")

    # TODO we should move this to the physical operator selection later
    def to_pandas(self, ctx: EvalContext):
        """Evaluate the expression against ``ctx.frame`` (a pandas frame)."""
        raise NotImplementedError

    def to_polars(self, ctx: EvalContext):
        """Evaluate on the polars backend, returning a lazy ``pl.Expr``."""
        raise NotImplementedError

    def to_pandas_query(self, params: dict) -> str | None:
        """Return a pandas ``query()`` string, or ``None`` if unsupported.

        Literals are bound into ``params`` and referenced as ``@p<i>``. A ``None``
        anywhere falls back to the boolean-mask path (``to_pandas``).
        """
        return None

    def iter_operand_refs(self):
        """Yield the ``OperandRef`` of every distinct ``OperandLeaf`` node."""
        for node in iter_postorder([self]):
            if isinstance(node, OperandLeaf):
                yield node.ref

    def remap_operand_refs(self, mapping: dict) -> "ColumnExpr":
        """Return a copy with operand references remapped (``self`` if unchanged)."""
        def remap(leaf):
            if isinstance(leaf, OperandLeaf):
                return OperandLeaf(OperandRef(mapping[leaf.ref.k]))
            return None
        return rewrite_leaves([self], remap)[0]


class _Leaf(ColumnExpr):
    """A node without sub-expressions."""
    __slots__ = ()

    def children(self):
        return ()

    def with_children(self, children):
        if children:
            raise TypeError(f"{type(self).__name__} has no children")
        return self


class Col(_Leaf):
    """Reference to a source-frame column."""
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def _key(self):
        return self.name

    def __repr__(self):
        return f"Col({self.name!r})"

    def to_pandas(self, ctx):
        return ctx.frame[self.name]

    def to_polars(self, ctx):
        return pl.col(self.name)

    def to_pandas_query(self, params):
        # Backtick the name so spaces / keywords / dots stay valid inside the query.
        return f"`{self.name}`"


def _scalar_key(value):
    """Key of a hashable literal that tells apart values Python considers equal.

    1 == 1.0 == True and 0.0 == -0.0 compare equal in Python but give different
    result dtypes / signs, so the type and the float sign are part of the key.
    """
    if isinstance(value, float):
        return (type(value), value, math.copysign(1.0, value))
    return (type(value), value)


def _literal_key(value):
    """Hashable key of a node's literal arguments, typed like :func:`_scalar_key`.

    Nested expressions are keyed by themselves, containers are recursed into and
    unhashable leaves fall back to identity.
    """
    if isinstance(value, ColumnExpr):
        return value
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_literal_key(v) for v in value))
    if isinstance(value, dict):
        return ("__dict__", frozenset((k, _literal_key(v)) for k, v in value.items()))
    if isinstance(value, (set, frozenset)):
        return ("__set__", frozenset(_literal_key(v) for v in value))
    try:
        hash(value)
    except TypeError:
        return ("__id__", id(value))
    return _scalar_key(value)


class Const(_Leaf):
    """Literal scalar value."""
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def _key(self):
        value = self.value
        try:
            hash(value)
        except TypeError:
            return ("__id__", id(value))
        return _scalar_key(value)

    def __repr__(self):
        return f"Const({self.value!r})"

    def to_pandas(self, ctx):
        return self.value

    def to_polars(self, ctx):
        return pl.lit(self.value)

    def to_pandas_query(self, params):
        # Bind the literal as a real object (referenced as @p<i>) rather than
        # stringifying it -- keeps timestamps/strings/NaN intact in the query.
        name = f"p{len(params)}"
        params[name] = self.value
        return f"@{name}"


class OperandLeaf(_Leaf):
    """Reference to an operator input that was not folded."""
    __slots__ = ("ref",)

    def __init__(self, ref):
        self.ref = ref

    def _key(self):
        return self.ref

    def __repr__(self):
        return f"OperandLeaf({self.ref})"

    def to_pandas(self, ctx):
        return ctx.inputs[self.ref.k]

    def to_polars(self, ctx):
        return ctx.inputs[self.ref.k]


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

    def to_pandas(self, ctx):
        return self.op(self.left.to_pandas(ctx), self.right.to_pandas(ctx))

    def to_polars(self, ctx):
        return self.op(self.left.to_polars(ctx), self.right.to_polars(ctx))

    def to_pandas_query(self, params):
        sym = BINARY_SYMBOLS.get(self.op)
        if sym is None:
            return None
        left = self.left.to_pandas_query(params)
        right = self.right.to_pandas_query(params)
        if left is None or right is None:
            return None
        return f"({left} {sym} {right})"

    def children(self):
        return (self.left, self.right)

    def with_children(self, children):
        left, right = children
        return BinOpExpr(self.op, left, right)


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

    def to_pandas(self, ctx):
        return self.op(self.operand.to_pandas(ctx))

    def to_polars(self, ctx):
        return self.op(self.operand.to_polars(ctx))

    def to_pandas_query(self, params):
        sym = UNARY_SYMBOLS.get(self.op)
        if sym is None:
            return None
        operand = self.operand.to_pandas_query(params)
        if operand is None:
            return None
        return f"({sym}{operand})"

    def children(self):
        return (self.operand,)

    def with_children(self, children):
        (operand,) = children
        return UnaryOpExpr(self.op, operand)


class StrExpr(ColumnExpr):
    """String accessor call (``.str.<method>()``)."""
    __slots__ = ("operand", "method", "args", "kwargs")

    def __init__(self, operand: ColumnExpr, method: str, args=(), kwargs=None):
        self.operand = operand
        self.method = method
        self.args = tuple(args)
        self.kwargs = kwargs or {}

    def _key(self):
        return (self.operand, self.method, _literal_key(self.args),
                _literal_key(self.kwargs))

    def __repr__(self):
        inner = ", ".join([repr(self.operand)]
                          + [repr(a) for a in self.args]
                          + [f"{k}={v!r}" for k, v in self.kwargs.items()])
        return f"str.{self.method}({inner})"

    def to_pandas(self, ctx):
        obj = self.operand.to_pandas(ctx)
        return getattr(obj.str, self.method)(*self.args, **self.kwargs)

    def to_polars(self, ctx):
        obj = self.operand.to_polars(ctx)
        name = STR_POLARS_METHODS.get(self.method, self.method)
        return getattr(obj.str, name)(*self.args, **self.kwargs)

    def children(self):
        return (self.operand,)

    def with_children(self, children):
        (operand,) = children
        return StrExpr(operand, self.method, self.args, self.kwargs)


class DtExpr(ColumnExpr):
    """Datetime accessor attribute (``.dt.<attr>``).

    pandas reads the attribute off ``.dt``; polars calls a method, remapping a
    few names via ``GetAttrProjectionOp.POLARS_ATTR_NAME_MAP``.
    """
    __slots__ = ("operand", "attr")

    def __init__(self, operand: ColumnExpr, attr: str):
        self.operand = operand
        self.attr = attr

    def _key(self):
        return (self.operand, self.attr)

    def __repr__(self):
        return f"dt.{self.attr}({self.operand!r})"

    def to_pandas(self, ctx):
        obj = self.operand.to_pandas(ctx)
        return getattr(obj.dt, self.attr)

    def to_polars(self, ctx):
        obj = self.operand.to_polars(ctx)
        if self.attr == "is_month_end":
            return obj.dt.month_end() == obj
        name = GetAttrProjectionOp.POLARS_ATTR_NAME_MAP.get(self.attr, self.attr)
        return getattr(obj.dt, name)()

    def children(self):
        return (self.operand,)

    def with_children(self, children):
        (operand,) = children
        return DtExpr(operand, self.attr)


class DatetimeExpr(ColumnExpr):
    """Datetime conversion (``pd.to_datetime`` / ``.str.to_datetime``).

    ``args``/``kwargs`` are literals; a graph-fed argument keeps the conversion
    op as a leaf instead.
    """
    __slots__ = ("operand", "args", "kwargs")

    def __init__(self, operand: ColumnExpr, args=(), kwargs=None):
        self.operand = operand
        self.args = tuple(args)
        self.kwargs = kwargs or {}

    def _key(self):
        return (self.operand, _literal_key(self.args), _literal_key(self.kwargs))

    def __repr__(self):
        return f"to_datetime({self.operand!r})"

    def to_pandas(self, ctx):
        obj = self.operand.to_pandas(ctx)
        return pd.to_datetime(obj, *self.args, **self.kwargs)

    def to_polars(self, ctx):
        obj = self.operand.to_polars(ctx)
        translated = polars_datetime_kwargs(self.args, self.kwargs)
        if translated is None:
            raise NotImplementedError(
                "DatetimeExpr contains options unsupported by Polars")
        # TODO: Support already-datetime and numeric operands natively; the
        # Polars string namespace only accepts string input.
        return obj.str.to_datetime(**translated)

    def children(self):
        return (self.operand,)

    def with_children(self, children):
        (operand,) = children
        return DatetimeExpr(operand, self.args, self.kwargs)


# --- DAG walks -----------------------------------------------------------------

def iter_postorder(roots: Iterable[ColumnExpr]) -> list[ColumnExpr]:
    """Return every distinct node reachable from ``roots``, children first.

    Nodes are deduplicated by identity, so a sub-expression shared by several
    parents appears once. Iterative, so deep towers do not hit the recursion
    limit.
    """
    order: list[ColumnExpr] = []
    seen: set[int] = set()
    for root in roots:
        stack = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            stack.extend((child, False) for child in reversed(node.children()))
    return order


def rewrite_leaves(roots: list[ColumnExpr],
                   rewrite: Callable[[ColumnExpr], ColumnExpr | None],
                   memo: dict[int, ColumnExpr] | None = None) -> list[ColumnExpr]:
    """Rebuild ``roots`` bottom-up, replacing each leaf ``x`` by ``rewrite(x)``.

    ``rewrite`` returns the replacement, or ``None`` to keep the leaf.
    Replacements are inserted as-is and not walked. Unchanged sub-DAGs keep
    their identity and shared nodes are rebuilt once, so sharing in the input
    carries over to the output. Pass the same ``memo`` across calls to keep
    sharing between separately rewritten roots.
    """
    if memo is None:
        memo = {}
    for node in iter_postorder(roots):
        if id(node) in memo:
            continue
        kids = node.children()
        if not kids:
            replacement = rewrite(node)
            memo[id(node)] = node if replacement is None else replacement
            continue
        new_kids = tuple(memo[id(c)] for c in kids)
        unchanged = all(a is b for a, b in zip(kids, new_kids))
        memo[id(node)] = node if unchanged else node.with_children(new_kids)
    return [memo[id(root)] for root in roots]


def _collect_value_exprs(value, found: list) -> None:
    """Append every ``ColumnExpr`` nested in ``value`` to ``found``, in order."""
    if isinstance(value, ColumnExpr):
        found.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_value_exprs(item, found)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_value_exprs(item, found)


def _replace_value_exprs(value, replacements):
    """Rebuild ``value`` taking its nested expressions from ``replacements``,
    in :func:`_collect_value_exprs` order."""
    if isinstance(value, ColumnExpr):
        return next(replacements)
    if isinstance(value, tuple):
        return tuple(_replace_value_exprs(item, replacements) for item in value)
    if isinstance(value, list):
        return [_replace_value_exprs(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_value_exprs(item, replacements)
                for key, item in value.items()}
    return value


def _eval_expr_value(value, ctx, backend):
    if isinstance(value, ColumnExpr):
        return getattr(value, f"to_{backend}")(ctx)
    if isinstance(value, tuple):
        return tuple(_eval_expr_value(item, ctx, backend) for item in value)
    if isinstance(value, list):
        return [_eval_expr_value(item, ctx, backend) for item in value]
    if isinstance(value, dict):
        return {key: _eval_expr_value(item, ctx, backend)
                for key, item in value.items()}
    return value


def _polars_expr_dtype(expr, ctx):
    # FIXME(#216): this resolves only the two shapes it pattern-matches and returns
    # None for every other operand, which is what makes the NaN handling in
    # _polars_fillna / _polars_notna quietly backend-dependent. Asking polars
    # for the expression's own output dtype (a schema-only resolve, e.g.
    # ctx.frame.lazy().select(expr).collect_schema()) would answer for any expr.
    if isinstance(expr, Col) and isinstance(ctx.frame, pl.DataFrame):
        return ctx.frame.schema.get(expr.name)
    if isinstance(expr, ColumnMethodExpr) and expr.method == "astype":
        dtype = expr.args[0] if expr.args else expr.kwargs.get("dtype")
        if not isinstance(dtype, ColumnExpr):
            return polars_dtype(dtype)
    return None


def has_static_polars_dtype(expr: ColumnExpr) -> bool:
    """Whether :func:`_polars_expr_dtype` can resolve ``expr``'s dtype from its
    shape alone: a bare ``Col`` or an ``astype`` with a literal dtype."""
    if isinstance(expr, Col):
        return True
    if isinstance(expr, ColumnMethodExpr) and expr.method == "astype":
        dtype = expr.args[0] if expr.args else expr.kwargs.get("dtype")
        return not isinstance(dtype, ColumnExpr)
    return False


class ColumnMethodExpr(ColumnExpr):
    """A registry-backed, row-local Series method call."""

    __slots__ = ("operand", "method", "args", "kwargs")

    def __init__(self, operand: ColumnExpr, method: str, args=(), kwargs=None):
        self.operand = operand
        self.method = method
        self.args = tuple(args or ())
        self.kwargs = dict(kwargs or {})

    def _key(self):
        return (
            self.operand,
            self.method,
            _literal_key(self.args),
            _literal_key(self.kwargs),
        )

    def __repr__(self):
        return f"{self.method}({self.operand!r})"

    def to_pandas(self, ctx):
        spec = get_column_method_spec(self.method)
        assert spec is not None
        obj = self.operand.to_pandas(ctx)
        args = _eval_expr_value(self.args, ctx, "pandas")
        kwargs = _eval_expr_value(self.kwargs, ctx, "pandas")
        return spec.pandas_eval(obj, list(args), kwargs, None)

    def to_polars(self, ctx):
        spec = get_column_method_spec(self.method)
        assert spec is not None
        obj = self.operand.to_polars(ctx)
        args = _eval_expr_value(self.args, ctx, "polars")
        kwargs = _eval_expr_value(self.kwargs, ctx, "polars")
        dtype = _polars_expr_dtype(self.operand, ctx)
        return spec.polars_eval(obj, list(args), kwargs, dtype)

    @property
    def reads_operand_dtype(self) -> bool:
        """Whether the polars result depends on the operand's dtype.

        ``_polars_expr_dtype`` resolves it only for a bare ``Col`` or a literal
        ``astype`` operand, so rewrites must not replace such an operand with
        anything else.
        """
        return get_column_method_spec(self.method).reads_operand_dtype

    def children(self):
        found = [self.operand]
        _collect_value_exprs((self.args, self.kwargs), found)
        return tuple(found)

    def with_children(self, children):
        replacements = iter(children)
        operand = next(replacements)
        args, kwargs = _replace_value_exprs((self.args, self.kwargs), replacements)
        return ColumnMethodExpr(operand, self.method, args, kwargs)


# --- Conversion: op subgraph -> ColumnExpr -----------------------------------

class _Folder:
    """Fold operator subgraphs into ``ColumnExpr`` trees.

    Three passes: :meth:`_discover` collects the foldable subgraph,
    :meth:`_absorbable` keeps nodes without external consumers, :meth:`_build`
    materialises the tree (an :class:`OperandLeaf` for the rest). The new
    operator's inputs are ``[src, *leaf_ops]``.

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

    # --- structural classification -------------------------------------------

    def _is_foldable(self, node: Op) -> bool:
        """Return whether ``node`` can be represented as a ``ColumnExpr``."""
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
            return self._has_literal_call_args(node)
        if isinstance(node, DatetimeConversionOp):
            # Only absorb calls whose pandas options have an equivalent Polars
            # spelling. The unfused op handles the rest through pandas.
            return (self._has_literal_call_args(node)
                    and polars_datetime_kwargs(node.args, node.kwargs) is not None)
        if isinstance(node, ColumnMethodOp):
            return True
        if isinstance(node, GetAttrProjectionOp):
            # Only the fused datetime accessor (.dt.<attr>); .str is already fused
            # into StringMethodOp during frame extraction.
            return len(node.attr_name) == 2 and node.attr_name[0] == "dt"
        return False

    @staticmethod
    def _has_literal_call_args(node: Op) -> bool:
        return (not any(isinstance(a, OperandRef) for a in (node.args or ()))
                and not any(isinstance(v, OperandRef)
                            for v in (node.kwargs or {}).values()))

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
                             GetAttrProjectionOp)):
            return [node.inputs[0]]
        if isinstance(node, ColumnMethodOp):
            # Every remaining input is referenced by args/kwargs. Returning all
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
            # just inputs[0]; args/kwargs are literals (checked in _is_foldable).
            operand = self._resolve(node.inputs[0], absorbable, memo)
            return StrExpr(operand, node.method,
                           tuple(node.args or ()), dict(node.kwargs or {}))
        if isinstance(node, DatetimeConversionOp):
            operand = self._resolve(node.inputs[0], absorbable, memo)
            return DatetimeExpr(operand, tuple(node.args or ()),
                                dict(node.kwargs or {}))
        if isinstance(node, ColumnMethodOp):
            operand = self._resolve(node.inputs[0], absorbable, memo)
            args = self._method_value(node.args or (), node, absorbable, memo)
            kwargs = self._method_value(node.kwargs or {}, node, absorbable, memo)
            return ColumnMethodExpr(operand, node.method, args, kwargs)
        if isinstance(node, GetAttrProjectionOp):
            operand = self._resolve(node.inputs[0], absorbable, memo)
            return DtExpr(operand, node.attr_name[1])
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


def fold_column_expr(root_node: Op, src: Op, root_consumer: Op):
    """Fold an operator subgraph into a column expression.

    Returns ``(expr, absorbed_ops, leaf_ops)``.
    """
    folder = _Folder(src)
    expr = folder.fold(root_node, root_consumer)
    return expr, folder.absorbed, folder.leaf_ops


def substitute_cols(expr: ColumnExpr, bindings: Mapping[str, ColumnExpr],
                    memo: dict[int, ColumnExpr] | None = None) -> ColumnExpr:
    """Replace ``Col(name)`` with ``bindings[name]`` when present.

    Binding values are already source-relative and are not re-walked, which
    preserves simultaneous-assign semantics inside one original map. Bindings
    are reused by identity, so fused towers share prefixes instead of
    deep-cloning. Rewriting several expressions against the same bindings with
    one shared ``memo`` keeps the sub-expressions they share shared.
    """
    def bind(leaf):
        return bindings.get(leaf.name) if isinstance(leaf, Col) else None
    return rewrite_leaves([expr], bind, memo)[0]
