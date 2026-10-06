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
"""
from __future__ import annotations
import datetime
import decimal
import operator

import numpy as np
import polars as pl
import pandas as pd

from stratum.optimizer.logical._ops import OperandRef, BinOp, UnaryOp, GetItemOp, Op, ValueOp
from stratum.optimizer.logical._base import config_key
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

# Values a ``ValueOp`` may carry into a ``Const``. Const compiles to ``pl.lit()``,
# which is only correct for a scalar: a list becomes one list-valued cell repeated
# per row and a pandas Series is rejected outright. Container values therefore stay
# ``OperandLeaf``s, where Polars column inputs are converted before expression
# evaluation. pandas Timestamp/Timedelta subclass the datetime types.
CONST_SCALAR_TYPES = (
    bool, int, float, str,
    datetime.date, datetime.time, datetime.timedelta,
    decimal.Decimal, np.generic,
)


class EvalContext:
    """Everything a column expression needs at evaluation time.

    ``frame`` is evaluated against (the op's primary operand); ``inputs`` are the
    op's resolved input values (read by :class:`OperandLeaf`); ``mode`` is
    ``fit_transform`` or ``predict`` (unused by the current stateless grammar).
    """
    __slots__ = ("frame", "inputs", "mode")

    def __init__(self, frame, inputs, mode: str = "fit_transform"):
        self.frame = frame
        self.inputs = inputs
        self.mode = mode


class ColumnExpr:
    """Base class for column-expression nodes."""
    __slots__ = ()

    def _key(self):
        raise NotImplementedError

    def __eq__(self, other):
        return type(self) is type(other) and self._key() == other._key()

    def __hash__(self):
        return hash((type(self).__name__, self._key()))

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
        """Yield all referenced ``OperandRef`` objects."""
        return iter(())

    def remap_operand_refs(self, mapping: dict) -> "ColumnExpr":
        """Return a copy with operand references remapped."""
        return self

    def has_aggregate(self) -> bool:
        """Whether this subtree contains a reduction (an :class:`AggExpr`).

        The grammar is row-wise everywhere else, so an aggregate is only legal at
        the root of an aggregation entry. Composite nodes recurse; the leaves keep
        this default.
        """
        return False


class Col(ColumnExpr):
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


class AllCols(ColumnExpr):
    """Every column of the source frame (polars ``pl.all()``).

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

    def to_pandas(self, ctx):
        return ctx.frame

    def to_polars(self, ctx):
        return pl.all()


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


class OperandLeaf(ColumnExpr):
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
        value = ctx.inputs[self.ref.k]
        if isinstance(value, pd.Series):
            return pl.from_pandas(value)
        if isinstance(value, list):
            return pl.Series(value)
        return value

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

    def iter_operand_refs(self):
        yield from self.left.iter_operand_refs()
        yield from self.right.iter_operand_refs()

    def remap_operand_refs(self, mapping):
        return BinOpExpr(self.op, self.left.remap_operand_refs(mapping),
                         self.right.remap_operand_refs(mapping))

    def has_aggregate(self):
        return self.left.has_aggregate() or self.right.has_aggregate()


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

    def iter_operand_refs(self):
        yield from self.operand.iter_operand_refs()

    def remap_operand_refs(self, mapping):
        return UnaryOpExpr(self.op, self.operand.remap_operand_refs(mapping))

    def has_aggregate(self):
        return self.operand.has_aggregate()



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

    def to_pandas(self, ctx):
        obj = self.operand.to_pandas(ctx)
        return getattr(obj.str, self.method)(*self.args, **self.kwargs)

    def to_polars(self, ctx):
        obj = self.operand.to_polars(ctx)
        name = STR_POLARS_METHODS.get(self.method, self.method)
        return getattr(obj.str, name)(*self.args, **self.kwargs)

    def iter_operand_refs(self):
        yield from self.operand.iter_operand_refs()

    def remap_operand_refs(self, mapping):
        return StrExpr(self.operand.remap_operand_refs(mapping),
                       self.method, self.args, self.kwargs)

    def has_aggregate(self):
        return self.operand.has_aggregate()


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

    def iter_operand_refs(self):
        yield from self.operand.iter_operand_refs()

    def remap_operand_refs(self, mapping):
        return DtExpr(self.operand.remap_operand_refs(mapping), self.attr)

    def has_aggregate(self):
        return self.operand.has_aggregate()


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
        return (self.operand, self.args, frozenset(self.kwargs.items()))

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

    def iter_operand_refs(self):
        yield from self.operand.iter_operand_refs()

    def remap_operand_refs(self, mapping):
        return DatetimeExpr(self.operand.remap_operand_refs(mapping),
                            self.args, self.kwargs)

    def has_aggregate(self):
        return self.operand.has_aggregate()


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

# Reduction -> polars ``Expr`` method. `sem` is deliberately absent (polars has no
# equivalent) and `nunique` is handled separately, because polars' ``n_unique``
# counts null as a distinct value while pandas' default drops it.
#
# idxmin/idxmax require original index labels, including inside each group.
# They are refused until an implementation can preserve that information.
_AGG_POLARS_METHODS = {
    "sum": "sum", "prod": "product", "min": "min", "max": "max",
    "first": "first", "last": "last", "mean": "mean", "median": "median",
    "std": "std", "var": "var", "skew": "skew", "kurt": "kurtosis",
    "idxmin": "arg_min", "idxmax": "arg_max", "all": "all", "any": "any",
    "quantile": "quantile", "count": "count", "size": "len",
}
# Our parameter name -> the polars keyword. Anything outside this map has no
# polars spelling on the reduction call.
_AGG_POLARS_PARAMS = {"ddof": "ddof", "q": "quantile",
                      "interpolation": "interpolation"}
# Parameters that are a no-op on the polars side when left at their pandas
# default, so they can be dropped instead of refused.
_AGG_POLARS_NOOP_DEFAULTS = {"skipna": True, "numeric_only": False,
                             "min_count": 0}
# Fixed polars keywords needed to match pandas. pandas' skew/kurt are the
# bias-corrected (sample) moments, while polars defaults to the biased
# population form; `fisher=True` (excess kurtosis) is already polars' default and
# matches pandas.
_AGG_POLARS_FIXED_KWARGS = {"skew": {"bias": False}, "kurt": {"bias": False}}


class AggExpr(ColumnExpr):
    """Reduction of a row-wise expression, e.g. ``SUM(a * b)``.

    ``func`` is the canonical reduction name, so ``.sum()`` and ``.agg("sum")``
    normalize to the same node and compare equal -- which is what lets CSE merge
    the two spellings. ``child`` is the row-wise expression being reduced;
    ``params`` holds only result-affecting options, validated against
    :data:`AGG_PARAMS`.

    Evaluating a *grouped* aggregation is the operator's job, not the
    expression's: ``to_polars`` returns the same ``pl.Expr`` in either case (it
    goes inside ``group_by(...).agg(...)`` or a bare ``select``), while
    ``to_pandas`` performs the whole-object reduction, and the grouped pandas
    path reads ``func``/``params`` off the node instead.
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

    def to_pandas(self, ctx):
        obj = self.child.to_pandas(ctx)
        # `numeric_only` picks columns out of a frame; the child is a single
        # column by construction, so it never applies here.
        params = {k: v for k, v in self.params.items() if k != "numeric_only"}
        if self.func == "size":
            return obj.size
        if self.func in ("first", "last"):
            # pandas 3.0 has no Series.first/last; they only exist on a groupby.
            return obj.iloc[0 if self.func == "first" else -1]
        return getattr(obj, self.func)(**params)

    def supports_polars(self):
        # arg_min/arg_max return positions within a group, not index labels.
        if self.func in {"idxmin", "idxmax", "sem"}:
            return False
        params = {k: v for k, v in self.params.items()
                  if _AGG_POLARS_NOOP_DEFAULTS.get(k, object()) != v}
        allowed = {"dropna"} if self.func == "nunique" else set(_AGG_POLARS_PARAMS)
        return not (set(params) - allowed)

    def to_polars(self, ctx):
        if self.func in {"idxmin", "idxmax", "sem"}:
            raise NotImplementedError(f"AggExpr({self.func!r}) has no equivalent Polars lowering")
        obj = self.child.to_polars(ctx)
        if self.func in {"first", "last"} and self.params.get("skipna", True):
            obj = obj.drop_nulls()
        params = {k: v for k, v in self.params.items()
                  if _AGG_POLARS_NOOP_DEFAULTS.get(k, object()) != v}
        if self.func == "nunique":
            # polars counts null as a distinct value; pandas drops it by default.
            if params.pop("dropna", True):
                obj = obj.drop_nulls()
            return obj.n_unique()
        method = _AGG_POLARS_METHODS.get(self.func)
        if method is None:
            raise NotImplementedError(
                f"AggExpr({self.func!r}) has no Polars equivalent")
        unsupported = sorted(set(params) - set(_AGG_POLARS_PARAMS))
        if unsupported:
            raise NotImplementedError(
                f"AggExpr({self.func!r}) parameters unsupported by Polars: "
                f"{', '.join(unsupported)}")
        if self.func == "quantile":
            # pandas defaults q=0.5 / interpolation="linear"; polars defaults
            # interpolation="nearest", so both are passed explicitly.
            params.setdefault("q", 0.5)
            params.setdefault("interpolation", "linear")
        kwargs = {_AGG_POLARS_PARAMS[k]: v for k, v in params.items()}
        kwargs.update(_AGG_POLARS_FIXED_KWARGS.get(self.func, {}))
        return getattr(obj, method)(**kwargs)

    def iter_operand_refs(self):
        yield from self.child.iter_operand_refs()

    def remap_operand_refs(self, mapping):
        return AggExpr(self.func, self.child.remap_operand_refs(mapping),
                       self.params)

    def has_aggregate(self):
        return True


def _iter_value_refs(value):
    if isinstance(value, ColumnExpr):
        yield from value.iter_operand_refs()
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_value_refs(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_value_refs(item)


def _value_has_aggregate(value) -> bool:
    if isinstance(value, ColumnExpr):
        return value.has_aggregate()
    if isinstance(value, (list, tuple)):
        return any(_value_has_aggregate(item) for item in value)
    if isinstance(value, dict):
        return any(_value_has_aggregate(item) for item in value.values())
    return False


def _remap_expr_value(value, mapping):
    if isinstance(value, ColumnExpr):
        return value.remap_operand_refs(mapping)
    if isinstance(value, tuple):
        return tuple(_remap_expr_value(item, mapping) for item in value)
    if isinstance(value, list):
        return [_remap_expr_value(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _remap_expr_value(item, mapping)
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
    # Fast paths for the two operand shapes whose dtype is known without
    # asking polars; everything else falls through to the schema-only probe
    # in ``to_polars``.
    if isinstance(expr, Col) and isinstance(ctx.frame, pl.DataFrame):
        return ctx.frame.schema.get(expr.name)
    if isinstance(expr, ColumnMethodExpr) and expr.method == "astype":
        dtype = expr.args[0] if expr.args else expr.kwargs.get("dtype")
        if not isinstance(dtype, ColumnExpr):
            return polars_dtype(dtype)
    return None


def _probe_expr_dtype(expr, ctx):
    """Schema-only output dtype of a derived operand, or None (#216).

    A ``pl.Expr`` carries no dtype of its own, so the NaN handling in
    ``_polars_fillna`` / ``_polars_notna`` needs polars to say what the
    expression produces. Selecting the operand against an empty projection of
    the frame is a schema resolve: no data is read. Returns None when the
    operand does not resolve against the frame, which leaves the evaluators
    with the null-only (NaN-blind) behavior they had before #216.
    """
    if not isinstance(expr, pl.Expr):
        return None
    frame = ctx.frame
    if isinstance(frame, pl.DataFrame):
        lazy = frame.lazy()
    elif isinstance(frame, pl.LazyFrame):
        lazy = frame
    else:
        return None
    probe = "_stratum_expr_dtype_probe"
    try:
        return lazy.select(expr.alias(probe)).collect_schema().get(probe)
    except Exception:
        return None


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
            config_key(self.args),
            config_key(self.kwargs),
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
        if dtype is None:
            dtype = _probe_expr_dtype(obj, ctx)
        return spec.polars_eval(obj, list(args), kwargs, dtype)

    def iter_operand_refs(self):
        yield from self.operand.iter_operand_refs()
        yield from _iter_value_refs(self.args)
        yield from _iter_value_refs(self.kwargs)

    def remap_operand_refs(self, mapping):
        return ColumnMethodExpr(
            self.operand.remap_operand_refs(mapping),
            self.method,
            _remap_expr_value(self.args, mapping),
            _remap_expr_value(self.kwargs, mapping),
        )

    def has_aggregate(self):
        # Expression-valued arguments are row-wise operands too, e.g. the other
        # side of `a.where(cond, b)`.
        return (self.operand.has_aggregate() or _value_has_aggregate(self.args)
                or _value_has_aggregate(self.kwargs))


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
