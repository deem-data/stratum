"""Running a :class:`~stratum.optimizer.logical._column_expr.ColumnExpr` on a backend.

The logical expression nodes are pure data; this module compiles them:

* :func:`to_pandas` evaluates a tree eagerly against a pandas frame;
* :func:`to_pandas_query` compiles a predicate to a ``DataFrame.query()`` string,
  or returns ``None`` when it cannot;
* :func:`to_polars` compiles a tree into one lazy ``pl.Expr``;
* :func:`polars_unsupported` says why ``to_polars`` cannot express a tree
  exactly, so a polars impl can refuse it at plan time (#226).

Each is a :func:`functools.singledispatch` function over the node types.
Evaluation reads an :class:`EvalContext`.
"""
from __future__ import annotations

from functools import singledispatch

import pandas as pd
import polars as pl

from stratum.optimizer.logical._base import IRNode
from stratum.optimizer.logical._column_expr import (
    BINARY_SYMBOLS, UNARY_SYMBOLS, AggExpr, AllCols, AstypeExpr, BinOpExpr, Col,
    ColumnExpr, ColumnMethodExpr, Const, DatetimeExpr, DtExpr, OperandLeaf, StrExpr, UnaryOpExpr,
    map_exprs)
from stratum.optimizer.logical._column_methods import normalize_dtype
from stratum.optimizer.logical._ops import apply_binary
from stratum.optimizer.physical._polars_translate import (
    polars_column_method, polars_dt_attribute,
    polars_dt_attribute_unsupported, polars_str_method_unsupported,
    polars_string_method, polars_to_datetime)


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

    @classmethod
    def of(cls, inputs: list, mode: str) -> "EvalContext":
        """The context of an op whose source frame is its first input."""
        return cls(frame=inputs[0], inputs=inputs, mode=mode)


# --- pandas -----------------------------------------------------------------------

@singledispatch
def to_pandas(expr: ColumnExpr, ctx: EvalContext):
    """Evaluate ``expr`` against ``ctx.frame`` (a pandas frame)."""
    raise NotImplementedError(f"no pandas evaluation for {type(expr).__name__}")


@to_pandas.register
def _(expr: Col, ctx):
    return ctx.frame[expr.name]


@to_pandas.register
def _(expr: AllCols, ctx):
    return ctx.frame


@to_pandas.register
def _(expr: Const, ctx):
    return expr.value


@to_pandas.register
def _(expr: OperandLeaf, ctx):
    return ctx.inputs[expr.ref.k]


@to_pandas.register
def _(expr: BinOpExpr, ctx):
    return expr.op(to_pandas(expr.left, ctx), to_pandas(expr.right, ctx))


@to_pandas.register
def _(expr: UnaryOpExpr, ctx):
    return expr.op(to_pandas(expr.operand, ctx))


@to_pandas.register
def _(expr: StrExpr, ctx):
    obj = to_pandas(expr.operand, ctx)
    return getattr(obj.str, expr.method)(**expr.params)


@to_pandas.register
def _(expr: DtExpr, ctx):
    return getattr(to_pandas(expr.operand, ctx).dt, expr.attr)


def pandas_to_datetime(obj, format=None, exact=None, cache=True, errors="raise"):
    """``pd.to_datetime`` with the fields of ``DatetimeConversionOp``;
    ``exact=None`` keeps pandas' default."""
    kwargs = {"format": format, "cache": cache, "errors": errors}
    if exact is not None:
        kwargs["exact"] = exact
    return pd.to_datetime(obj, **kwargs)


@to_pandas.register
def _(expr: DatetimeExpr, ctx):
    return pandas_to_datetime(to_pandas(expr.operand, ctx), expr.format, expr.exact,
                              expr.cache, expr.errors)


@to_pandas.register
def _(expr: AggExpr, ctx):
    # The whole-object reduction; a grouped pandas aggregation reads func/params
    # off the node instead (see PandasAggregateOp).
    obj = to_pandas(expr.child, ctx)
    # `numeric_only` picks columns out of a frame; the child is a single
    # column by construction, so it never applies here.
    params = {k: v for k, v in expr.params.items() if k != "numeric_only"}
    if expr.func == "size":
        return obj.size
    if expr.func in ("first", "last"):
        # pandas 3.0 has no Series.first/last; they only exist on a groupby.
        return obj.iloc[0 if expr.func == "first" else -1]
    return getattr(obj, expr.func)(**params)


def method_fields(expr: ColumnMethodExpr) -> dict:
    """The fields of a column method expression after its operand, by name."""
    return {name: getattr(expr, name) for name in type(expr).__slots__[1:]}


#: Column method fields named apart from their pandas parameter.
_PANDAS_PARAMETERS = {"mapping": "func"}


def pandas_column_method(obj, method: str, fields: dict):
    """Call the pandas column method ``method`` on ``obj``, passing ``fields`` by name."""
    return getattr(obj, method)(**{_PANDAS_PARAMETERS.get(name, name): value
                                   for name, value in fields.items()})


@to_pandas.register
def _(expr: ColumnMethodExpr, ctx):
    obj = to_pandas(expr.operand, ctx)
    fields = map_exprs(method_fields(expr), lambda e: to_pandas(e, ctx))
    return pandas_column_method(obj, expr.method, fields)


@singledispatch
def to_pandas_query(expr: ColumnExpr, params: dict) -> str | None:
    """Return a pandas ``query()`` string for ``expr``, or ``None`` if unsupported.

    Literals are bound into ``params`` and referenced as ``@p<i>``. A ``None``
    anywhere falls back to the boolean-mask path (``to_pandas``).
    """
    return None


@to_pandas_query.register
def _(expr: Col, params):
    # Backtick the name so spaces / keywords / dots stay valid inside the query.
    return f"`{expr.name}`"


@to_pandas_query.register
def _(expr: Const, params):
    # Bind the literal as a real object (referenced as @p<i>) rather than
    # stringifying it -- keeps timestamps/strings/NaN intact in the query.
    name = f"p{len(params)}"
    params[name] = expr.value
    return f"@{name}"


@to_pandas_query.register
def _(expr: BinOpExpr, params):
    sym = BINARY_SYMBOLS.get(expr.op)
    if sym is None:
        return None
    left = to_pandas_query(expr.left, params)
    right = to_pandas_query(expr.right, params)
    if left is None or right is None:
        return None
    return f"({left} {sym} {right})"


@to_pandas_query.register
def _(expr: UnaryOpExpr, params):
    sym = UNARY_SYMBOLS.get(expr.op)
    if sym is None:
        return None
    operand = to_pandas_query(expr.operand, params)
    if operand is None:
        return None
    return f"({sym}{operand})"


# --- polars -----------------------------------------------------------------------

@singledispatch
def to_polars(expr: ColumnExpr, ctx: EvalContext):
    """Compile ``expr`` into a lazy ``pl.Expr`` over ``ctx.frame``."""
    raise NotImplementedError(f"no polars compilation for {type(expr).__name__}")


@to_polars.register
def _(expr: Col, ctx):
    return pl.col(expr.name)


@to_polars.register
def _(expr: AllCols, ctx):
    return pl.all()


@to_polars.register
def _(expr: Const, ctx):
    return pl.lit(expr.value)


@to_polars.register
def _(expr: OperandLeaf, ctx):
    value = ctx.inputs[expr.ref.k]
    if isinstance(value, pd.Series):
        return pl.from_pandas(value)
    if isinstance(value, list):
        return pl.Series(value)
    return value


@to_polars.register
def _(expr: BinOpExpr, ctx):
    return apply_binary(expr.op, to_polars(expr.left, ctx), to_polars(expr.right, ctx))


@to_polars.register
def _(expr: UnaryOpExpr, ctx):
    return expr.op(to_polars(expr.operand, ctx))


@to_polars.register
def _(expr: StrExpr, ctx):
    return polars_string_method(to_polars(expr.operand, ctx), expr.method, expr.params)


@to_polars.register
def _(expr: DtExpr, ctx):
    return polars_dt_attribute(to_polars(expr.operand, ctx), expr.attr)


@to_polars.register
def _(expr: DatetimeExpr, ctx):
    return polars_to_datetime(to_polars(expr.operand, ctx), expr.format, expr.exact,
                              expr.cache, expr.errors)


# Reduction -> polars ``Expr`` method. `sem` is deliberately absent (polars has no
# equivalent) and `nunique` is handled separately, because polars' ``n_unique``
# counts null as a distinct value while pandas' default drops it.
#
# idxmin/idxmax require original index labels, including inside each group, so
# they have no entry: polars' arg_min/arg_max return positions.
_AGG_POLARS_METHODS = {
    "sum": "sum", "prod": "product", "min": "min", "max": "max",
    "first": "first", "last": "last", "mean": "mean", "median": "median",
    "std": "std", "var": "var", "skew": "skew", "kurt": "kurtosis",
    "all": "all", "any": "any",
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
_AGG_POLARS_REFUSED = frozenset({"idxmin", "idxmax", "sem"})


def _agg_polars_params(expr: AggExpr) -> dict:
    """``expr``'s parameters, minus the ones that are no-ops on polars."""
    return {k: v for k, v in expr.params.items()
            if _AGG_POLARS_NOOP_DEFAULTS.get(k, object()) != v}


def _agg_supports_polars(expr: AggExpr) -> bool:
    if expr.func in _AGG_POLARS_REFUSED:
        return False
    allowed = {"dropna"} if expr.func == "nunique" else set(_AGG_POLARS_PARAMS)
    return not (set(_agg_polars_params(expr)) - allowed)


@to_polars.register
def _(expr: AggExpr, ctx):
    # The same pl.Expr serves a grouped aggregation (inside group_by().agg())
    # and a whole-frame one (inside select()). Reductions and parameters polars
    # cannot express were refused at plan time (see _agg_supports_polars).
    obj = to_polars(expr.child, ctx)
    if expr.func in {"first", "last"} and expr.params.get("skipna", True):
        obj = obj.drop_nulls()
    params = _agg_polars_params(expr)
    if expr.func == "nunique":
        # polars counts null as a distinct value; pandas drops it by default.
        if params.pop("dropna", True):
            obj = obj.drop_nulls()
        return obj.n_unique()
    if expr.func == "quantile":
        # pandas defaults q=0.5 / interpolation="linear"; polars defaults
        # interpolation="nearest", so both are passed explicitly.
        params.setdefault("q", 0.5)
        params.setdefault("interpolation", "linear")
    kwargs = {_AGG_POLARS_PARAMS[k]: v for k, v in params.items()}
    kwargs.update(_AGG_POLARS_FIXED_KWARGS.get(expr.func, {}))
    return getattr(obj, _AGG_POLARS_METHODS[expr.func])(**kwargs)


def _polars_expr_dtype(expr, ctx):
    # FIXME(#216): this resolves only the two shapes it pattern-matches and returns
    # None for every other operand, which is what makes the NaN handling in
    # _polars_fillna / _polars_notna quietly backend-dependent. Asking polars
    # for the expression's own output dtype (a schema-only resolve, e.g.
    # ctx.frame.lazy().select(expr).collect_schema()) would answer for any expr.
    if isinstance(expr, Col) and isinstance(ctx.frame, pl.DataFrame):
        return ctx.frame.schema.get(expr.name)
    if isinstance(expr, AstypeExpr) and not isinstance(expr.dtype, ColumnExpr):
        return normalize_dtype(expr.dtype)
    return None


@to_polars.register
def _(expr: ColumnMethodExpr, ctx):
    obj = to_polars(expr.operand, ctx)
    fields = map_exprs(method_fields(expr), lambda e: to_polars(e, ctx))
    return polars_column_method(expr.method, obj, fields,
                                _polars_expr_dtype(expr.operand, ctx))


def polars_unsupported(expr: ColumnExpr) -> str | None:
    """Why :func:`to_polars` cannot evaluate ``expr`` exactly, or ``None``.

    Checked at plan time by the polars impls' ``supports`` (#226), so an
    expression the polars compiler cannot express binds the pandas impl
    instead of failing -- or silently differing -- during execution. The
    first offending node is named.
    """
    reason = _polars_unsupported_here(expr)
    if reason is not None:
        return f"{expr!r}: {reason}"
    for child in expr.children():
        reason = polars_unsupported(child)
        if reason is not None:
            return reason
    return None


@singledispatch
def _polars_unsupported_here(expr: ColumnExpr) -> str | None:
    """This node's own polars restriction; children are checked separately."""
    return None


@_polars_unsupported_here.register
def _(expr: Const):
    try:
        pl.lit(expr.value)
    except Exception as exc:  # noqa: BLE001 - any refusal means no literal
        first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        return f"no polars literal for {type(expr.value).__name__} ({first_line})"
    return None


@_polars_unsupported_here.register
def _(expr: StrExpr):
    return polars_str_method_unsupported(expr.method, expr.params)


@_polars_unsupported_here.register
def _(expr: DtExpr):
    return polars_dt_attribute_unsupported(expr.attr)


@_polars_unsupported_here.register
def _(expr: AggExpr):
    return None if _agg_supports_polars(expr) else f"no polars lowering for {expr.func!r}"


def refuse_unsupported_exprs(op: IRNode, exprs) -> bool:
    """``supports`` helper for polars impls; records the refusal reason on ``op``."""
    for expr in exprs:
        reason = polars_unsupported(expr)
        if reason is not None:
            op._unsupported_reason = reason
            return False
    return True
