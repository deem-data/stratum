"""Backend-agnostic column expression tree.

Used by selections (boolean predicates), maps (computed columns) and
aggregations (reductions). Expressions are immutable value types and compare
structurally.

The grammar is restricted to natively lazy, row-local computations (column
references, literals, arithmetic and boolean operators, the ``.str``/``.dt``
namespaces, datetime parsing, registered column methods, and one reduction at
the root of an aggregation entry), so a whole tree compiles into a single
backend kernel. Anything outside the grammar (fitted transformers, UDFs, data
from other frames) is referenced through an :class:`OperandLeaf`.

The nodes are pure data. Running them is the physical layer's job
(``physical/_expressions.py`` compiles a tree for pandas and polars), and
folding an op subgraph into a tree is ``_expr_folding.py``'s.
"""
from __future__ import annotations
import operator

from stratum.optimizer.logical._base import OperandRef, config_key

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
    """Base class for column-expression nodes.

    Every node's ``__init__`` takes its fields in ``__slots__`` order. The
    structural walks below rely on that, so a node only defines its fields,
    its key and its repr.
    """
    __slots__ = ()

    def _key(self):
        raise NotImplementedError

    def __eq__(self, other):
        return type(self) is type(other) and self._key() == other._key()

    def __hash__(self):
        return hash((type(self).__name__, self._key()))

    def children(self):
        """The expressions nested in this node's fields, in field order."""
        for name in type(self).__slots__:
            yield from _nested_exprs(getattr(self, name))

    def map_children(self, fn) -> "ColumnExpr":
        """This node with ``fn`` applied to each nested child expression.

        A node without children (a leaf) is returned as is.
        """
        if next(self.children(), None) is None:
            return self
        return type(self)(*(map_exprs(getattr(self, name), fn)
                            for name in type(self).__slots__))

    def iter_operand_refs(self):
        """Yield all referenced ``OperandRef`` objects."""
        for child in self.children():
            yield from child.iter_operand_refs()

    def remap_operand_refs(self, mapping: dict) -> "ColumnExpr":
        """Return a copy with operand references remapped."""
        return self.map_children(lambda child: child.remap_operand_refs(mapping))

    def has_aggregate(self) -> bool:
        """Whether this subtree contains a reduction (an :class:`AggExpr`).

        The grammar is row-wise everywhere else, so an aggregate is only legal at
        the root of an aggregation entry.
        """
        return any(child.has_aggregate() for child in self.children())


def _nested_exprs(value):
    if isinstance(value, ColumnExpr):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _nested_exprs(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _nested_exprs(item)


def map_exprs(value, fn):
    """``value`` with ``fn`` applied to every expression nested in its
    tuples/lists/dicts; anything else is kept."""
    if isinstance(value, ColumnExpr):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(map_exprs(item, fn) for item in value)
    if isinstance(value, list):
        return [map_exprs(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: map_exprs(item, fn) for key, item in value.items()}
    return value


class Col(ColumnExpr):
    """Reference to a source-frame column."""
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def _key(self):
        return self.name

    def __repr__(self):
        return f"Col({self.name!r})"


class AllCols(ColumnExpr):
    """Every column of the source frame.

    Lets ``df.groupby(k).sum()`` stay one aggregation entry instead of needing one
    per column, which the logical layer cannot enumerate without schema
    propagation. Inside a grouped aggregation both backends exclude the grouping
    keys from the wildcard, so the two agree.
    """
    __slots__ = ()

    def _key(self):
        return ()

    def __repr__(self):
        return "AllCols()"


class Const(ColumnExpr):
    """Literal scalar value."""
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def _key(self):
        try:
            hash(self.value)
        except TypeError:
            return (type(self.value), "__id__", id(self.value))
        return (type(self.value), self.value)

    def __repr__(self):
        return f"Const({self.value!r})"


class OperandLeaf(ColumnExpr):
    """Reference to an operator input that was not folded."""
    __slots__ = ("ref",)

    def __init__(self, ref):
        self.ref = ref

    def _key(self):
        return self.ref

    def __repr__(self):
        return f"OperandLeaf({self.ref})"

    def iter_operand_refs(self):
        yield self.ref

    def remap_operand_refs(self, mapping):
        # A ColumnExpr target substitutes the leaf, e.g. a projected input that a
        # rewrite turns into a column of the frame (see push_projection_through_map).
        target = mapping[self.ref.k]
        if isinstance(target, ColumnExpr):
            return target
        return OperandLeaf(OperandRef(target))


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


class StrExpr(ColumnExpr):
    """String accessor call (``.str.<method>()``), in pandas' spelling: the
    ``method`` and its ``params`` by pandas' parameter names."""
    __slots__ = ("operand", "method", "params")

    def __init__(self, operand: ColumnExpr, method: str, params=None):
        self.operand = operand
        self.method = method
        self.params = dict(params or {})

    def _key(self):
        return (self.operand, self.method, config_key(self.params))

    def __repr__(self):
        inner = ", ".join([repr(self.operand)]
                          + [f"{k}={v!r}" for k, v in self.params.items()])
        return f"str.{self.method}({inner})"


class DtExpr(ColumnExpr):
    """Datetime accessor attribute (``.dt.<attr>``), with pandas' meaning."""
    __slots__ = ("operand", "attr")

    def __init__(self, operand: ColumnExpr, attr: str):
        self.operand = operand
        self.attr = attr

    def _key(self):
        return (self.operand, self.attr)

    def __repr__(self):
        return f"dt.{self.attr}({self.operand!r})"


class DatetimeExpr(ColumnExpr):
    """Datetime parsing (``pd.to_datetime``), with the fields of
    :class:`~stratum.optimizer.logical._map_ops.DatetimeConversionOp`."""
    __slots__ = ("operand", "format", "exact", "cache", "errors")

    def __init__(self, operand: ColumnExpr, format=None, exact=None, cache=True,
                 errors="raise"):
        self.operand = operand
        self.format = format
        self.exact = exact
        self.cache = cache
        self.errors = errors

    def _key(self):
        return (self.operand, self.format, self.exact, self.cache, self.errors)

    def __repr__(self):
        return f"to_datetime({self.operand!r})"


# --- Aggregation --------------------------------------------------------------

def _expand_agg_params(spec: dict) -> dict:
    """Flatten ``{(func, ...): params}`` into ``{func: frozenset(params)}``."""
    out: dict[str, frozenset] = {}
    for funcs, params in spec.items():
        for func in funcs:
            if func in out:
                raise ValueError(f"duplicate parameter entry for {func!r}")
            out[func] = frozenset(params)
    return out


# Parameters each reduction accepts. Ported from the signature survey in #192 and
# re-verified against pandas 3.0.2. Only parameters that change the *result* live
# here; *how* a reduction runs is the physical layer's concern, so execution hints
# have no logical spelling at all.
AGG_PARAMS = _expand_agg_params({
    ("sum", "prod", "min", "max", "first", "last"):
                                          ("numeric_only", "min_count", "skipna"),
    ("mean", "median"):                   ("numeric_only", "skipna"),
    ("std", "var", "sem"):                ("ddof", "numeric_only", "skipna"),
    ("skew", "kurt", "idxmin", "idxmax"): ("skipna", "numeric_only"),
    ("all", "any"):                       ("skipna",),
    ("quantile",):                        ("q", "interpolation", "numeric_only"),
    ("nunique",):                         ("dropna",),
    ("count", "size"):                    (),
})


class AggExpr(ColumnExpr):
    """Reduction of a row-wise expression, e.g. ``SUM(a * b)``.

    ``func`` is the canonical reduction name, so ``.sum()`` and ``.agg("sum")``
    normalize to the same node and compare equal -- which is what lets CSE merge
    the two spellings. ``child`` is the row-wise expression being reduced;
    ``params`` holds only result-affecting options, validated against
    :data:`AGG_PARAMS`.

    Evaluating a *grouped* aggregation is the operator's job, not the
    expression's: the grouped pandas path reads ``func``/``params`` off the node
    instead of compiling it.
    """
    __slots__ = ("func", "child", "params")

    def __init__(self, func: str, child: ColumnExpr, params: dict | None = None):
        allowed = AGG_PARAMS.get(func)
        if allowed is None:
            raise NotImplementedError(f"unsupported aggregation {func!r}")
        params = dict(params or {})
        unsupported = sorted(set(params) - allowed)
        if unsupported:
            raise NotImplementedError(
                f"unsupported parameters for {func!r}: {', '.join(unsupported)}")
        if child.has_aggregate():
            # The grammar below an aggregate is row-wise; a nested reduction has
            # no meaning and would break the one-kernel property.
            raise ValueError(f"nested aggregation in {func!r}")
        self.func = func
        self.child = child
        self.params = params

    def _key(self):
        return (self.func, self.child, frozenset(self.params.items()))

    def __repr__(self):
        inner = ", ".join([repr(self.child)]
                          + [f"{k}={v!r}" for k, v in self.params.items()])
        return f"{self.func}({inner})"

    def has_aggregate(self):
        return True


class ColumnMethodExpr(ColumnExpr):
    """Base of the row-local column methods, in pandas' spelling.

    Each subclass is one pandas method, named by ``method``. Its fields are the
    ``operand`` followed by the fields of its
    :class:`~stratum.optimizer.logical._column_methods.ColumnMethodOp`, under the
    same names. Expression-valued fields are row-wise operands too, e.g. the other
    side of ``a.where(cond, b)``.
    """
    __slots__ = ()
    method: str = None

    def _key(self):
        return tuple(config_key(getattr(self, name)) for name in type(self).__slots__)

    def __repr__(self):
        return f"{self.method}({', '.join(repr(getattr(self, name)) for name in type(self).__slots__)})"


class AstypeExpr(ColumnMethodExpr):
    """``astype(dtype)``; ``dtype`` is the pandas spelling."""
    __slots__ = ("operand", "dtype")
    method = "astype"

    def __init__(self, operand: ColumnExpr, dtype):
        self.operand = operand
        self.dtype = dtype


class FillnaExpr(ColumnMethodExpr):
    """``fillna(value)``: a missing value (null, or NaN in a float column) becomes ``value``."""
    __slots__ = ("operand", "value")
    method = "fillna"

    def __init__(self, operand: ColumnExpr, value):
        self.operand = operand
        self.value = value


class ClipExpr(ColumnMethodExpr):
    """``clip(lower, upper)``; a ``None`` bound is open."""
    __slots__ = ("operand", "lower", "upper")
    method = "clip"

    def __init__(self, operand: ColumnExpr, lower=None, upper=None):
        self.operand = operand
        self.lower = lower
        self.upper = upper


class WhereExpr(ColumnMethodExpr):
    """``where(cond, other)``: the operand where ``cond`` holds, ``other`` elsewhere."""
    __slots__ = ("operand", "cond", "other")
    method = "where"

    def __init__(self, operand: ColumnExpr, cond, other):
        self.operand = operand
        self.cond = cond
        self.other = other


class IsinExpr(ColumnMethodExpr):
    """``isin(values)``: whether each value is one of ``values``."""
    __slots__ = ("operand", "values")
    method = "isin"

    def __init__(self, operand: ColumnExpr, values):
        self.operand = operand
        self.values = values


class IsnaExpr(ColumnMethodExpr):
    """``isna()``: whether each value is missing (null, or NaN in a float column)."""
    __slots__ = ("operand",)
    method = "isna"

    def __init__(self, operand: ColumnExpr):
        self.operand = operand


class NotnaExpr(ColumnMethodExpr):
    """``notna()``: the negation of :class:`IsnaExpr`."""
    __slots__ = ("operand",)
    method = "notna"

    def __init__(self, operand: ColumnExpr):
        self.operand = operand


class MapValuesExpr(ColumnMethodExpr):
    """``map(mapping, na_action)``: a dict lookup; a value it lacks becomes missing."""
    __slots__ = ("operand", "mapping", "na_action")
    method = "map"

    def __init__(self, operand: ColumnExpr, mapping, na_action=None):
        self.operand = operand
        self.mapping = mapping
        self.na_action = na_action
