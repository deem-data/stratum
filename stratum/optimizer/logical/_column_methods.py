"""Shape-preserving column methods (the ColumnMethodOp family).

Each pandas method extraction admits is its own subclass, whose fields are the
method's arguments, bound once from the recorded call and named as pandas names
them; ``method`` is the pandas method it stands for. A call outside the shapes
admitted here stays an opaque method call that runs on pandas as recorded.

An op with external consumers runs on its own; otherwise the folder absorbs it
into an AssignMap/Selection ``ColumnExpr`` (each op names its expression class).
The kernels that run a method are in ``physical/_polars_translate.py`` (polars)
and the pandas impls, which call the pandas method with the fields.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from stratum.optimizer.logical._base import OutputType
from stratum.optimizer.logical import _schema
from stratum.optimizer.logical._column_expr import (
    AstypeExpr, ClipExpr, FillnaExpr, IsinExpr, IsnaExpr, MapValuesExpr, NotnaExpr,
    WhereExpr)
from stratum.optimizer.logical._map_ops import MapOp
from stratum.optimizer.logical._ops import MethodCallOp, OperandRef, substitute_op


# The dtypes a folded ``astype`` admits, as the schema's dtypes (``_schema``
# describes columns with polars dtypes).
_DTYPES = {
    str: pl.String,
    bool: pl.Boolean,
    int: pl.Int64,
    float: pl.Float64,
    "str": pl.String,
    "string": pl.String,
    "bool": pl.Boolean,
    "boolean": pl.Boolean,
    "int8": pl.Int8,
    "int16": pl.Int16,
    "int32": pl.Int32,
    "int64": pl.Int64,
    "uint8": pl.UInt8,
    "uint16": pl.UInt16,
    "uint32": pl.UInt32,
    "uint64": pl.UInt64,
    "float32": pl.Float32,
    "float64": pl.Float64,
}


def normalize_dtype(dtype):
    """The schema dtype for a pandas ``astype`` target, or ``None`` if not admitted."""
    if isinstance(dtype, pl.DataType):
        return dtype
    try:
        translated = _DTYPES.get(dtype)
    except TypeError:
        translated = None
    if translated is not None:
        return translated
    try:
        numpy_dtype = np.dtype(dtype)
    except (TypeError, ValueError):
        return None
    return _DTYPES.get(numpy_dtype.name)


class ColumnMethodOp(MapOp):
    """Abstract base of the column methods: keeps the shape of a Series or
    DataFrame and computes each value from the one in its place.

    Never instantiated directly. Impls register on the concrete subclasses.
    """
    #: The pandas method this op stands for.
    method: str = None
    #: The :class:`~stratum.optimizer.logical._column_expr.ColumnMethodExpr` the
    #: folder turns this op into: the operand, then this op's fields by name.
    expr_class = None
    #: Whether the result is a boolean mask whatever the input dtype is. The
    #: others all depend on it: ``fillna`` upcasts on a float fill value, ``clip``
    #: on a float bound, ``where`` null-pads the rows it masks out, and ``astype``
    #: is handed its target, so none of them is certain here.
    boolean_result = False

    def __init__(self, inputs=None, outputs=None):
        super().__init__(name=self.method, inputs=inputs, outputs=outputs)

    def propagate_output_schema(self):
        """A column method is shape-preserving, so the column names carry over and
        only the dtype can change. See :attr:`boolean_result`."""
        dtype = pl.Boolean if self.boolean_result else _schema.UNKNOWN_DTYPE
        self.output_schema = _schema.cast_columns(self.inputs[0].output_schema, dtype)


class AstypeOp(ColumnMethodOp):
    """Casts every value to ``dtype``, a pandas spelling :func:`normalize_dtype` admits."""
    method, expr_class = "astype", AstypeExpr
    fields = ["dtype"]

    def __init__(self, dtype=None, inputs=None, outputs=None):
        super().__init__(inputs=inputs, outputs=outputs)
        self.dtype = dtype


class FillnaOp(ColumnMethodOp):
    """Replaces missing values (null, and NaN in a float column) with ``value``.

    On a Series ``value`` is a scalar; on a DataFrame it maps column names to
    the scalar for that column (absent columns are ignored).
    """
    method, expr_class = "fillna", FillnaExpr
    fields = ["value"]

    def __init__(self, value=None, inputs=None, outputs=None):
        super().__init__(inputs=inputs, outputs=outputs)
        self.value = value


class ClipOp(ColumnMethodOp):
    """Bounds every value to ``[lower, upper]``; a ``None`` bound is open."""
    method, expr_class = "clip", ClipExpr
    fields = ["lower", "upper"]

    def __init__(self, lower=None, upper=None, inputs=None, outputs=None):
        super().__init__(inputs=inputs, outputs=outputs)
        self.lower = lower
        self.upper = upper


class WhereOp(ColumnMethodOp):
    """Keeps each value where ``cond`` holds and puts ``other`` elsewhere."""
    method, expr_class = "where", WhereExpr
    fields = ["cond", "other"]

    def __init__(self, cond=None, other=None, inputs=None, outputs=None):
        super().__init__(inputs=inputs, outputs=outputs)
        self.cond = cond
        self.other = other


class IsinOp(ColumnMethodOp):
    """Whether each value is one of ``values``."""
    method, expr_class = "isin", IsinExpr
    fields = ["values"]
    boolean_result = True

    def __init__(self, values=None, inputs=None, outputs=None):
        super().__init__(inputs=inputs, outputs=outputs)
        self.values = values


class IsnaOp(ColumnMethodOp):
    """Whether each value is missing (null, or NaN in a float column)."""
    method, expr_class = "isna", IsnaExpr
    fields = []
    boolean_result = True


class NotnaOp(ColumnMethodOp):
    """Whether each value is present: the negation of :class:`IsnaOp`."""
    method, expr_class = "notna", NotnaExpr
    fields = []
    boolean_result = True


class MapValuesOp(ColumnMethodOp):
    """Looks each value up in ``mapping``, a plain dict; a value it lacks becomes
    missing. ``na_action="ignore"`` keeps missing values missing."""
    method, expr_class = "map", MapValuesExpr
    fields = ["mapping", "na_action"]

    def __init__(self, mapping=None, na_action=None, inputs=None, outputs=None):
        super().__init__(inputs=inputs, outputs=outputs)
        self.mapping = mapping
        self.na_action = na_action


# --- Extraction: recorded pandas call -> column method fields -------------------
#
# Each parser takes the call bound against pandas' signature and whether the
# receiver is a DataFrame (a method can mean something else there: ``fillna``
# with a dict fills per column on a frame, per index label on a Series). It
# returns the op's fields, or ``None`` to keep the call opaque.

def _is_ref_or_scalar(value) -> bool:
    if isinstance(value, OperandRef):
        return True
    if isinstance(value, (pd.Series, pd.DataFrame, pl.Series, pl.DataFrame)):
        return False
    return isinstance(value, str) or not hasattr(value, "__len__")


def _astype_fields(call: dict, frame: bool) -> dict | None:
    dtype = call.get("dtype")
    if (call.keys() != {"dtype"} or isinstance(dtype, OperandRef)
            or normalize_dtype(dtype) is None):
        return None
    return dict(dtype=dtype)


def _fillna_fields(call: dict, frame: bool) -> dict | None:
    value = call.get("value")
    if call.keys() != {"value"}:
        return None
    if frame:
        # A plain dict of column name -> scalar fill value.
        if type(value) is not dict or not all(
                isinstance(k, str) and v is not None and not pd.isna(v)
                and _is_ref_or_scalar(v)
                for k, v in value.items() if not isinstance(v, OperandRef)):
            return None
    elif not _is_ref_or_scalar(value):
        return None
    return dict(value=value)


def _clip_fields(call: dict, frame: bool) -> dict | None:
    if (frame or not call or call.keys() - {"lower", "upper"}
            or not all(_is_ref_or_scalar(v) for v in call.values())):
        return None
    return dict(lower=call.get("lower"), upper=call.get("upper"))


def _where_fields(call: dict, frame: bool) -> dict | None:
    # Without `other`, pandas fills NaN, which not every polars dtype can hold.
    if (frame or call.keys() != {"cond", "other"}
            or not all(_is_ref_or_scalar(v) for v in call.values())):
        return None
    return dict(cond=call["cond"], other=call["other"])


def _isin_fields(call: dict, frame: bool) -> dict | None:
    values = call.get("values")
    if frame or call.keys() != {"values"} or not (
            isinstance(values, OperandRef)
            or isinstance(values, (list, tuple, set, frozenset, pd.Index, np.ndarray))):
        return None
    return dict(values=values)


def _no_fields(call: dict, frame: bool) -> dict | None:
    return None if frame or call else {}


def _map_fields(call: dict, frame: bool) -> dict | None:
    """A plain dict of scalar keys and values: a lookup that misses to NaN.

    Refused: functions and Series (arbitrary code / index alignment), dict
    subclasses (``__missing__`` makes them a function), missing keys, and keys
    or values that mix types beyond int/float, whose pandas result dtype
    Polars would not reproduce.
    """
    if frame:
        return None
    # pandas 3 renamed `arg` to `func`; the old name still arrives as a keyword.
    given = {**call.pop("kwargs", {}), **call}
    names = given.keys() & {"func", "arg"}
    if len(names) != 1 or given.keys() - {"func", "arg", "na_action"}:
        return None
    mapping, na_action = given[names.pop()], given.get("na_action")
    if type(mapping) is not dict or na_action not in (None, "ignore"):
        return None

    def homogeneous(values):
        types = {type(v) for v in values if v is not None}
        return len(types) <= 1 or types <= {int, float}

    if not (all(type(k) in (str, int, float) and not pd.isna(k) for k in mapping)
            and all(v is None or type(v) in (str, int, float, bool)
                    for v in mapping.values())
            and homogeneous(mapping) and homogeneous(mapping.values())):
        return None
    return dict(mapping=mapping, na_action=na_action)


# pandas method -> (column method subclass, parser of the bound call)
COLUMN_METHODS = {
    "astype": (AstypeOp, _astype_fields),
    "fillna": (FillnaOp, _fillna_fields),
    "clip": (ClipOp, _clip_fields),
    "where": (WhereOp, _where_fields),
    "isin": (IsinOp, _isin_fields),
    "isna": (IsnaOp, _no_fields),
    "isnull": (IsnaOp, _no_fields),
    "notna": (NotnaOp, _no_fields),
    "notnull": (NotnaOp, _no_fields),
    "map": (MapValuesOp, _map_fields),
}


def make_column_method_op(op: MethodCallOp) -> ColumnMethodOp | None:
    """Parse a column method call into its :class:`ColumnMethodOp` subclass.

    Returns ``None`` (the call stays opaque) when the call is not one we model.
    """
    cls, parse = COLUMN_METHODS[op.method_name]
    call = op.bind_pandas()
    if call is None:
        return None
    fields = parse(call, op.inputs[0].output_type is OutputType.FRAME)
    if fields is None:
        return None
    return substitute_op(op, cls(**fields), keep_kind=True)
