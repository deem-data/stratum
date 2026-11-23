"""Value transformations over one or more explicit inputs.

Maps return their computed value. Assignment belongs to the Combine family;
projections and producers of map operands remain separate logical operators.
Expression maps use OperandLeaf references without a distinguished source frame.
"""
from __future__ import annotations

import pandas as pd
import polars as pl

from stratum.optimizer.logical._base import IndexEffect
from stratum.optimizer.logical._ops import (BinOp, UnaryOp, CallOp, GetAttrOp, MethodCallOp, Op,
                                            OperandRef, OutputType, substitute_op)
from stratum.optimizer.logical._column_expr import (
    BINARY_SYMBOLS, UNARY_SYMBOLS, UFUNCS, BinOpExpr, ColumnExpr, Const,
    OperandLeaf, StrExpr, UnaryOpExpr, UfuncExpr)
from stratum.optimizer.logical._index_ops import IndexAccessOp
from stratum.optimizer.logical._metadata_ops import make_structure_read_op
from stratum.frontend._pandas_calls import bind_pandas_function
from stratum.optimizer.logical import _schema


class MapOp(Op):
    """Base for value transformations; each variant defines its alignment rules."""
    logical_family = "Map"
    index_effect = IndexEffect.KEEP

    def __init__(self, name: str = "", inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        self.output_type = OutputType.FRAME


# Unknown at compile time, distinct from an unnamed pandas Series (None).
UNKNOWN_NAME = object()
_NO_NAME = object()


def series_name(op: Op):
    """The known Series name, without reading intermediate values."""
    from stratum.optimizer.logical._projection_ops import ColumnProjectionOp
    from stratum.optimizer.logical._ops import GetItemOp, ValueOp

    if isinstance(op, ExpressionMapOp):
        return op.result_name()
    if isinstance(op, (ColumnProjectionOp, GetItemOp)) and isinstance(op.key, str):
        return op.key
    if isinstance(op, ValueOp):
        return op.value.name if isinstance(op.value, (pd.Series, pl.Series)) else _NO_NAME
    if isinstance(op, (DatetimeConversionOp, DatetimeFieldOp)):
        return series_name(op.inputs[0])
    schema = op.output_schema
    return next(iter(schema)) if schema is not None and len(schema) == 1 else UNKNOWN_NAME


class ExpressionMapOp(MapOp):
    """Compute an expression over explicit inputs, without retaining a base frame.

    ``OperandLeaf(OperandRef(k))`` reads input k. Projection is a separate
    operator; expressions have no implicit source columns. Binary operations
    retain their operands' label-alignment semantics.
    """
    fields = ["expr"]
    cacheable = True

    def __init__(self, expr: ColumnExpr, inputs=None, outputs=None,
                 output_type: OutputType = OutputType.SERIES):
        super().__init__(name=repr(expr), inputs=inputs, outputs=outputs)
        self.expr = expr
        self.output_type = output_type

    @property
    def index_effect(self):
        return self.expr.index_effect

    @property
    def aligns_by_label(self):
        return self.expr.aligns_by_label

    def clone(self):
        cloned = super().clone()
        cloned.output_type = self.output_type
        return cloned

    def result_name(self):
        """Pandas' result name, or UNKNOWN_NAME when it cannot be inferred."""
        def name(expr):
            if isinstance(expr, Const):
                return _NO_NAME
            if isinstance(expr, OperandLeaf):
                source = self.inputs[expr.ref.k]
                return _NO_NAME if source.output_type is OutputType.SCALAR else series_name(source)
            if isinstance(expr, StrExpr) and not expr.preserves_columns:
                return UNKNOWN_NAME
            if isinstance(expr, BinOpExpr):
                left, right = name(expr.left), name(expr.right)
                if left is _NO_NAME:
                    return right
                if right is _NO_NAME:
                    return left
                if left is UNKNOWN_NAME or right is UNKNOWN_NAME:
                    return UNKNOWN_NAME
                return left if left == right else None
            return name(expr.operand)
        return name(self.expr)

    def propagate_output_schema(self):
        if self.expr.preserves_columns:
            dtype = pl.Boolean if self.expr.boolean_result else _schema.UNKNOWN_DTYPE
            self.output_schema = _schema.cast_columns(self.inputs[0].output_schema, dtype)
            return

        def dtype(column):
            if not isinstance(self.expr, (BinOpExpr, UnaryOpExpr)):
                return _schema.UNKNOWN_DTYPE
            operands = list(self.expr.children())
            schemas = [self.inputs[e.ref.k].output_schema for e in operands
                       if isinstance(e, OperandLeaf)]
            dtypes = ([schema[column] for schema in schemas]
                      if len(schemas) == len(operands)
                      and all(schema is not None and column in schema for schema in schemas)
                      else None)
            return _schema.elementwise_result_dtype(self.expr.op, dtypes)

        if self.output_type is OutputType.SERIES:
            name = self.result_name()
            self.output_schema = pl.Schema({name: dtype(name)}) if isinstance(name, str) else None
        else:
            schemas = [op.output_schema for op in self.inputs
                       if op.output_type is OutputType.FRAME]
            if (any(op.output_type in (OutputType.SERIES, OutputType.UNKNOWN) for op in self.inputs)
                    or not schemas or any(s is None for s in schemas)
                    or any(list(s) != list(schemas[0]) for s in schemas)):
                self.output_schema = None
            else:
                self.output_schema = pl.Schema({name: dtype(name) for name in schemas[0]})


def ufunc_name(op: Op) -> str | None:
    """Recognize a supported unary NumPy call without additional options."""
    if not isinstance(op, CallOp) or op.kwargs or list(op.args or ()) != [OperandRef(0)]:
        return None
    return next((name for name, func in UFUNCS.items() if op.func is func), None)


def make_expression_map_op(op: Op) -> ExpressionMapOp | None:
    """Extract one elementwise operation, leaving all producers in the DAG."""
    from stratum.optimizer.logical._ops import ValueOp
    from stratum.optimizer.logical._expr_folding import CONST_SCALAR_TYPES

    kinds = {i.output_type for i in op.inputs}
    if not kinds.intersection((OutputType.FRAME, OutputType.SERIES)):
        return None
    inputs = list(op.inputs)

    def operand(value):
        if isinstance(value, OperandRef):
            producer = inputs[value.k]
            if isinstance(producer, ValueOp) and isinstance(producer.value, CONST_SCALAR_TYPES):
                return Const(producer.value)
            return OperandLeaf(value)
        if pd.api.types.is_scalar(value):
            return Const(value)
        # Literal containers are data operands, not scalar expression constants.
        inputs.append(ValueOp(value))
        return OperandLeaf(OperandRef(len(inputs) - 1))

    if isinstance(op, BinOp) and op.op in BINARY_SYMBOLS:
        expr = BinOpExpr(op.op, operand(op.left), operand(op.right))
    elif isinstance(op, UnaryOp) and op.op in UNARY_SYMBOLS:
        expr = UnaryOpExpr(op.op, operand(op.operand))
    elif (func := ufunc_name(op)) is not None:
        expr = UfuncExpr(OperandLeaf(OperandRef(0)), func)
    else:
        return None
    frame_result = (OutputType.FRAME in kinds
                    or any(isinstance(i, ValueOp) and isinstance(i.value, (pd.DataFrame, pl.DataFrame))
                           for i in inputs))
    kind = OutputType.FRAME if frame_result else OutputType.SERIES
    used = {ref.k for ref in expr.iter_operand_refs()}
    kept = [k for k in range(len(inputs)) if k in used]
    expr = expr.remap_operand_refs({k: new for new, k in enumerate(kept)})
    new_op = substitute_op(op, ExpressionMapOp(expr, output_type=kind))
    new_op.inputs = [inputs[k] for k in kept]
    for producer in inputs[len(op.inputs):]:
        producer.add_output(new_op)
    for k, producer in enumerate(op.inputs):
        if k not in used:
            producer.outputs = [out for out in producer.outputs if out is not new_op]
    return new_op


class CallMapOp(MapOp):
    """Base for the maps that replay a recorded pandas call on their receiver.

    ``args``/``kwargs`` are the call's arguments after the receiver, which is
    operand 0. Only the subclasses are built; each says which call it replays.
    """
    fields = ["args", "kwargs"]

    def __init__(self, args: tuple | list = (), kwargs: dict | None = None,
                 inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(inputs=inputs, outputs=outputs)
        if kwargs is not None:
            self.check_kwargs(kwargs)
        self.args = args
        self.kwargs = kwargs


class ApplyUDFOp(CallMapOp):
    """``df.apply(func, ...)`` or ``series.apply(func, ...)``."""


class DatetimeConversionOp(MapOp):
    """Parses a column of strings into datetimes (``pd.to_datetime``).

    ``format``: the ``strftime`` pattern, ``None`` to infer it. ``exact``: whether
    the pattern must match the whole string (``None``: pandas' default).
    ``cache``: reuse the result for repeated strings. ``errors``: ``"raise"`` on
    an unparseable string, or ``"coerce"`` it to a missing value. Each may be an
    ``OperandRef`` when graph-fed.
    """
    fields = ["format", "exact", "cache", "errors"]
    cacheable = True
    # Additional inputs are parsing options, not row-aligned data operands.
    aligns_by_label = False

    ERRORS = ("raise", "coerce")

    def __init__(self, format=None, exact=None, cache=True, errors="raise",
                 inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name="TO_DATETIME", inputs=inputs, outputs=outputs)
        if not isinstance(errors, OperandRef) and errors not in self.ERRORS:
            raise ValueError(f"errors must be one of {self.ERRORS}, got {errors!r}")
        self.format = format
        self.exact = exact
        self.cache = cache
        self.errors = errors

    def propagate_output_schema(self):
        """Converting a column keeps its name and leaves the dtype Unknown.

        The result's datetime unit can depend on the input and backend, which is
        selected during physical planning. This schema does not represent a
        datetime dtype with an unknown unit.

        Only a column operand is column-preserving. Given a *frame*,
        ``pd.to_datetime`` is the assembly form: it reads year/month/day columns
        and fans them in to one unnamed Series (any other frame raises), so the
        input's names are not the output's. Extraction leaves that form opaque.
        """
        if self.inputs[0].output_type is not OutputType.SERIES:
            self.output_schema = None
            return
        self.output_schema = _schema.cast_columns(self.inputs[0].output_schema)


class DatetimeFieldOp(MapOp):
    """Reads one field of each datetime, ``col.dt.<field>`` (``year``, ``hour``, ...).

    ``field`` keeps pandas' name and meaning; the polars impl translates it (see
    ``physical/_polars_translate.py``).
    """
    fields = ["field"]
    cacheable = True

    # The `.dt` properties that yield one value per datetime, so the read keeps
    # the input's columns and only changes their dtype. Extraction sees *any*
    # attribute read, so this is an allow-list: `.dt.components` and
    # `.dt.isocalendar` fan one column out into several (7 and 3 respectively),
    # `.dt.tz` and `.dt.freq` are a single value, and a `.dt` method read without
    # its call is a bound method. Those stay an AttributeReadOp.
    FIELDS = frozenset({
        "date", "day", "day_of_week", "day_of_year", "dayofweek", "dayofyear",
        "days", "days_in_month", "daysinmonth", "hour", "is_leap_year",
        "is_month_end", "is_month_start", "is_quarter_end", "is_quarter_start",
        "is_year_end", "is_year_start", "microsecond", "microseconds",
        "minute", "month", "nanosecond", "nanoseconds", "quarter", "second",
        "seconds", "time", "timetz", "weekday", "year",
    })

    def __init__(self, field: str = None, inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=f"dt.{field}", inputs=inputs, outputs=outputs)
        self.field = field

    def propagate_output_schema(self):
        """A field read keeps the column names, dtype Unknown."""
        self.output_schema = _schema.cast_columns(self.inputs[0].output_schema)


class AttributeReadOp(Op):
    """An attribute read off a frame or column that the optimizer does not model:
    ``.loc``, ``.T``, ``.values``, the ``.str``/``.dt``/``.cat`` namespaces, a ``.dt``
    property that is not one value per datetime, ...

    ``attr_name`` is the chain of names read, e.g. ``["cat", "codes"]``. Opaque and
    pandas-only: it reads pandas' attributes as recorded. These can change shape
    (``.T``) or produce non-tabular values (``.values``).
    """
    logical_family = "Attribute"
    fields = ["attr_name"]
    # `.T` turns the columns into rows and `.values` drops the labels.
    index_effect = IndexEffect.NEW

    def __init__(self, attr_name: list[str] = None, inputs: list[Op] = None,
                 outputs: list[Op] = None):
        self.attr_name = list(attr_name or [])
        super().__init__(name=".".join(self.attr_name) or "?", inputs=inputs, outputs=outputs)


# --- Extraction: recorded calls -> call-based maps ----------------------------

#: The ``pd.to_datetime`` arguments a :class:`DatetimeConversionOp` holds.
_DATETIME_OPTIONS = frozenset({"format", "exact", "cache", "errors"})


def make_datetime_conversion_op(op: CallOp) -> DatetimeConversionOp | None:
    """Parse ``pd.to_datetime(column, ...)`` into a :class:`DatetimeConversionOp`.

    Returns ``None`` (the call stays opaque and runs on pandas as recorded) for
    anything else: options other than ``format``, ``exact``, ``cache`` and
    ``errors="raise"``/``"coerce"``, and a frame operand, which is the assembly
    form. An option's value may be graph-fed.
    """
    call = bind_pandas_function(pd.to_datetime, op.args, op.kwargs)
    if call is None:
        return None
    arg = call.pop("arg", None)
    if (not isinstance(arg, OperandRef) or arg.k != 0
            or op.inputs[0].output_type is not OutputType.SERIES
            or set(call) - _DATETIME_OPTIONS):
        return None
    errors = call.get("errors", "raise")
    if not isinstance(errors, OperandRef) and errors not in DatetimeConversionOp.ERRORS:
        return None
    return substitute_op(op, DatetimeConversionOp(
        format=call.get("format"), exact=call.get("exact"),
        cache=call.get("cache", True), errors=errors), keep_kind=True)


def make_frame_get_attr(op: GetAttrOp) -> Op:
    """Parse an attribute read from a dataframe or series.

    ``.index`` reads the row labels, ``.shape`` and ``.columns`` the structure, and
    ``.dt.<field>`` a datetime field. Any other read becomes an opaque
    :class:`AttributeReadOp`; a read off one extends its chain of names.
    """
    if op.attr_name == "index" or op.attr_name == ["index"]:
        return substitute_op(op, IndexAccessOp())
    structure_op = make_structure_read_op(op)
    if structure_op is not None:
        return structure_op
    input_ = op.inputs[0]
    attr_name = op.attr_name if isinstance(op.attr_name, list) else [op.attr_name]
    if not isinstance(input_, AttributeReadOp):
        new_op = substitute_op(op, AttributeReadOp(attr_name=attr_name))
        # Retain the source kind as an approximation for subsequent extraction.
        new_op.output_type = input_.output_type
        return new_op

    # Fuse the chained read with the one it reads off.
    chain = [*input_.attr_name, *attr_name]
    source = input_.inputs[0]
    if len(chain) == 2 and chain[0] == "dt" and chain[1] in DatetimeFieldOp.FIELDS:
        new_op = DatetimeFieldOp(field=chain[1], inputs=[source], outputs=op.outputs)
    else:
        new_op = AttributeReadOp(attr_name=chain, inputs=[source], outputs=op.outputs)
    new_op.output_type = input_.output_type
    if len(input_.outputs) > 1:
        input_.outputs.remove(op)
        source.add_output(new_op)
    else:
        source.replace_output(input_, new_op)
    return new_op
