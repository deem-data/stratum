"""Typed, backend-neutral column method operators.

The registry in this module is the single admission point for pandas methods
that are safe to preserve as row-local column expressions.  A registered call
can execute as its own :class:`ColumnMethodOp` when it has external consumers,
or be absorbed into an AssignMap/Selection ``ColumnExpr`` by the folder.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import polars as pl

from stratum.optimizer.logical._base import (
    OutputType, _resolve_args, _resolve_kwargs)
from stratum.optimizer.logical import _schema
from stratum.optimizer.logical._ops import MethodCallOp, Op, OperandRef


CallValidator = Callable[[tuple, dict], bool]
MethodEvaluator = Callable[[object, list, dict, object | None], object]


@dataclass(frozen=True, slots=True)
class ColumnMethodSpec:
    """Semantics shared by standalone and folded column-method execution."""

    name: str
    validate: CallValidator
    pandas_eval: MethodEvaluator
    polars_eval: MethodEvaluator
    supports_frame: bool = False


def _is_ref_or_scalar(value) -> bool:
    if isinstance(value, OperandRef):
        return True
    if isinstance(value, (pd.Series, pd.DataFrame, pl.Series, pl.DataFrame)):
        return False
    return isinstance(value, str) or not hasattr(value, "__len__")


_POLARS_DTYPES = {
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


def polars_dtype(dtype):
    """Translate the conservative dtype subset accepted by folded ``astype``."""
    if isinstance(dtype, pl.DataType):
        return dtype
    try:
        translated = _POLARS_DTYPES.get(dtype)
    except TypeError:
        translated = None
    if translated is not None:
        return translated
    try:
        numpy_dtype = np.dtype(dtype)
    except (TypeError, ValueError):
        return None
    return _POLARS_DTYPES.get(numpy_dtype.name)


def _validate_astype(args, kwargs):
    if len(args) == 1 and not kwargs:
        dtype = args[0]
    elif not args and set(kwargs) == {"dtype"}:
        dtype = kwargs["dtype"]
    else:
        return False
    return not isinstance(dtype, OperandRef) and polars_dtype(dtype) is not None


def _validate_fillna(args, kwargs):
    if len(args) == 1 and not kwargs:
        value = args[0]
    elif not args and set(kwargs) == {"value"}:
        value = kwargs["value"]
    else:
        return False
    return _is_ref_or_scalar(value)


def _validate_clip(args, kwargs):
    return (
        len(args) <= 2
        and set(kwargs) <= {"lower", "upper"}
        and len(args) + len(kwargs) > 0
        and all(_is_ref_or_scalar(v) for v in (*args, *kwargs.values()))
    )


def _validate_where(args, kwargs):
    if not (1 <= len(args) <= 2) or set(kwargs) - {"other"}:
        return False
    if len(args) == 2 and "other" in kwargs:
        return False
    if len(args) == 1 and "other" not in kwargs:
        # pandas' implicit NaN cannot be represented for every Polars dtype.
        return False
    return all(
        isinstance(v, OperandRef) or _is_ref_or_scalar(v)
        for v in (*args, *kwargs.values())
    )


def _validate_isin(args, kwargs):
    if len(args) != 1 or kwargs:
        return False
    values = args[0]
    return isinstance(values, OperandRef) or isinstance(
        values, (list, tuple, set, frozenset, pd.Index, np.ndarray))


def _validate_notna(args, kwargs):
    return not args and not kwargs


def _pandas_method(name):
    def evaluate(obj, args, kwargs, dtype=None):
        return getattr(obj, name)(*args, **kwargs)
    return evaluate


def _arg(args, kwargs, position, name, default=None):
    if len(args) > position:
        return args[position]
    return kwargs.get(name, default)


def _polars_astype(obj, args, kwargs, dtype=None):
    target = _arg(args, kwargs, 0, "dtype")
    return obj.cast(polars_dtype(target), strict=True)


def _polars_fillna(obj, args, kwargs, dtype=None):
    # pandas fills NaN as well as null, polars needs both calls. The fill_nan
    # half runs whenever the operand's dtype resolves as float -- including
    # derived operands, via the schema-only probe in _column_expr (#216).
    value = _arg(args, kwargs, 0, "value")
    result = obj.fill_null(value)
    actual_dtype = getattr(obj, "dtype", None) or dtype
    if actual_dtype is not None and actual_dtype.is_float():
        result = result.fill_nan(value)
    return result


def _polars_clip(obj, args, kwargs, dtype=None):
    lower = _arg(args, kwargs, 0, "lower")
    upper = _arg(args, kwargs, 1, "upper")
    return obj.clip(lower_bound=lower, upper_bound=upper)


def _polars_where(obj, args, kwargs, dtype=None):
    condition = args[0]
    other = _arg(args, kwargs, 1, "other", np.nan)
    if isinstance(obj, pl.Series):
        if not isinstance(other, pl.Series):
            # pandas promotes the result to the supertype of column and fill
            # value, so the fill value must not be pinned to the column dtype
            # (#216): an int column's .where(cond, 1.5) is float in pandas and
            # raises here otherwise. repeat() infers the scalar's own dtype
            # and zip_with supertypes the pair, matching pandas.
            other = pl.repeat(other, len(obj), eager=True)
        return obj.zip_with(condition, other)
    return pl.when(condition).then(obj).otherwise(other)


def _polars_isin(obj, args, kwargs, dtype=None):
    values = list(args[0]) if isinstance(args[0], (set, frozenset)) else args[0]
    return obj.is_in(values)


def _polars_notna(obj, args, kwargs, dtype=None):
    # Same float/null semantics as _polars_fillna: pandas reports NaN as
    # missing, polars needs is_not_nan on top of is_not_null whenever the
    # operand's dtype resolves as float (#216).
    result = obj.is_not_null()
    actual_dtype = getattr(obj, "dtype", None) or dtype
    if actual_dtype is not None and actual_dtype.is_float():
        result = result & obj.is_not_nan()
    return result


def _spec(name, validate, polars_eval, *, supports_frame=False):
    return ColumnMethodSpec(
        name=name,
        validate=validate,
        pandas_eval=_pandas_method(name),
        polars_eval=polars_eval,
        supports_frame=supports_frame,
    )


COLUMN_METHOD_SPECS = {
    spec.name: spec
    for spec in (
        _spec("astype", _validate_astype, _polars_astype, supports_frame=True),
        _spec("fillna", _validate_fillna, _polars_fillna),
        _spec("clip", _validate_clip, _polars_clip),
        _spec("where", _validate_where, _polars_where),
        _spec("isin", _validate_isin, _polars_isin),
        _spec("notna", _validate_notna, _polars_notna),
    )
}


def get_column_method_spec(name: str) -> ColumnMethodSpec | None:
    return COLUMN_METHOD_SPECS.get(name)


def is_supported_column_method(op: MethodCallOp) -> bool:
    spec = get_column_method_spec(op.method_name)
    if spec is None or not op.inputs:
        return False
    if (op.inputs[0].output_type is OutputType.FRAME
            and not spec.supports_frame):
        return False
    return spec.validate(tuple(op.args or ()), dict(op.kwargs or {}))


class ColumnMethodOp(Op):
    """A supported shape-preserving method on a Series or DataFrame."""

    logical_family = "ColumnMethod"
    fields = ["method", "args", "kwargs"]

    def __init__(self, method: str, args=(), kwargs=None,
                 inputs=None, outputs=None):
        super().__init__(name=method, inputs=inputs, outputs=outputs)
        self.method = method
        self.args = tuple(args or ())
        self.kwargs = dict(kwargs or {})

    #: Methods whose result is a boolean mask whatever the input dtype is. The
    #: others all depend on it: ``fillna`` upcasts on a float fill value, ``clip``
    #: on a float bound, ``where`` null-pads the rows it masks out, and ``astype``
    #: is handed its target (possibly per column), so none of them is certain here.
    BOOLEAN_METHODS = frozenset({"isin", "notna"})

    def propagate_output_schema(self):
        """A column method is shape-preserving, so the column names carry over and
        only the dtype can change. See :data:`BOOLEAN_METHODS`."""
        dtype = (pl.Boolean if self.method in self.BOOLEAN_METHODS
                 else _schema.UNKNOWN_DTYPE)
        self.output_schema = _schema.cast_columns(self.inputs[0].output_schema, dtype)

    def resolved_call(self, inputs):
        return (
            inputs[0],
            _resolve_args(self.args, inputs),
            _resolve_kwargs(self.kwargs, inputs),
        )


def make_column_method_op(op: MethodCallOp) -> ColumnMethodOp:
    new_op = ColumnMethodOp(
        method=op.method_name,
        args=op.args,
        kwargs=op.kwargs,
        inputs=op.inputs,
        outputs=op.outputs,
    )
    new_op.output_type = op.inputs[0].output_type
    op.replace_output_of_inputs(new_op)
    return new_op
