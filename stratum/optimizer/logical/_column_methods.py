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
from stratum.optimizer.logical._map_ops import MapOp
from stratum.optimizer.logical._ops import MethodCallOp, OperandRef


CallValidator = Callable[[tuple, dict], bool]
MethodEvaluator = Callable[[object, list, dict, object | None], object]


@dataclass(frozen=True, slots=True)
class ColumnMethodSpec:
    """Semantics shared by standalone and folded column-method execution."""

    name: str
    validate: CallValidator
    pandas_eval: MethodEvaluator
    polars_eval: MethodEvaluator
    #: Validates the call on a DataFrame, whose arguments can mean something else
    #: than on a Series (``fillna`` with a dict fills per column on a frame, per
    #: index label on a Series). ``None``: the method only folds on a Series.
    validate_frame: CallValidator | None = None


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


def _validate_frame_fillna(args, kwargs):
    """``DataFrame.fillna`` with a plain dict of column name -> scalar fill value."""
    if len(args) == 1 and not kwargs:
        value = args[0]
    elif not args and set(kwargs) == {"value"}:
        value = kwargs["value"]
    else:
        return False
    return (type(value) is dict and all(
        isinstance(k, str) and v is not None and not pd.isna(v) and _is_ref_or_scalar(v)
        for k, v in value.items() if not isinstance(v, OperandRef)))


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


def _map_mapping(args, kwargs):
    """The mapping of a ``Series.map`` call (pandas 3 renamed ``arg`` to ``func``)."""
    if len(args) == 1 and not set(kwargs) - {"na_action"}:
        return args[0]
    keys = set(kwargs) & {"arg", "func"}
    if not args and len(keys) == 1 and not set(kwargs) - {"arg", "func", "na_action"}:
        return kwargs[keys.pop()]
    return None


def _validate_map(args, kwargs):
    """A plain dict of scalar keys and values: a lookup that misses to NaN.

    Refused: functions and Series (arbitrary code / index alignment), dict
    subclasses (``__missing__`` makes them a function), missing keys, and keys
    or values that mix types beyond int/float, whose pandas result dtype
    Polars would not reproduce.
    """
    mapping = _map_mapping(args, kwargs)
    if type(mapping) is not dict or kwargs.get("na_action") not in (None, "ignore"):
        return False

    def homogeneous(values):
        types = {type(v) for v in values if v is not None}
        return len(types) <= 1 or types <= {int, float}

    return (all(type(k) in (str, int, float) and not pd.isna(k) for k in mapping)
            and all(v is None or type(v) in (str, int, float, bool)
                    for v in mapping.values())
            and homogeneous(mapping) and homogeneous(mapping.values()))


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
    # FIXME(#216): pandas fills NaN as well as null, polars needs both calls, and the
    # fill_nan half is skipped whenever the dtype is unknown. A pl.Expr carries
    # no dtype, and `_polars_expr_dtype` only resolves a bare Col or an astype,
    # so any derived operand ((a / b).fillna(0), a chained clip/where/fillna)
    # silently keeps its NaNs here while pandas replaces them.
    value = _arg(args, kwargs, 0, "value")
    if isinstance(obj, pl.DataFrame):
        return _polars_frame_fillna(obj, value)
    result = obj.fill_null(value)
    actual_dtype = getattr(obj, "dtype", None) or dtype
    if actual_dtype is not None and actual_dtype.is_float():
        result = result.fill_nan(value)
    return result


def _fill_value_fits(value, dtype) -> bool:
    """Whether filling ``value`` keeps the column's kind, as pandas does then."""
    if isinstance(value, bool):
        return dtype == pl.Boolean
    if isinstance(value, (int, float, np.number)):
        return dtype.is_numeric()
    if isinstance(value, str):
        return dtype in (pl.String, pl.Categorical)
    return False


def _polars_frame_fillna(frame, values: dict):
    columns = []
    for name, value in values.items():
        if name not in frame.columns:  # pandas ignores unknown columns
            continue
        dtype = frame.schema[name]
        if not _fill_value_fits(value, dtype):
            # FIXME(#227): fall back to pandas, which upcasts to object here.
            raise NotImplementedError(
                f"fillna({{{name!r}: {value!r}}}) on a {dtype} column has no exact "
                f"polars translation: pandas would mix types in an object column.")
        columns.append(_polars_fillna(pl.col(name), [value], {}, dtype).alias(name))
    return frame.with_columns(columns) if columns else frame


def _polars_clip(obj, args, kwargs, dtype=None):
    lower = _arg(args, kwargs, 0, "lower")
    upper = _arg(args, kwargs, 1, "upper")
    return obj.clip(lower_bound=lower, upper_bound=upper)


def _polars_where(obj, args, kwargs, dtype=None):
    condition = args[0]
    other = _arg(args, kwargs, 1, "other", np.nan)
    if isinstance(obj, pl.Series):
        if not isinstance(other, pl.Series):
            # FIXME(#216): two problems on this line. Pinning `other` to the column
            # dtype raises where pandas promotes to a supertype, so an int
            # column's .where(cond, 1.5) fails here and returns floats in
            # pandas; and the list comprehension materialises one Python object
            # per row, where pl.repeat(other, len(obj), eager=True) would not.
            # The pl.Expr branch below has neither problem.
            other = pl.Series([other] * len(obj), dtype=obj.dtype)
        return obj.zip_with(condition, other)
    return pl.when(condition).then(obj).otherwise(other)


def _polars_isin(obj, args, kwargs, dtype=None):
    values = list(args[0]) if isinstance(args[0], (set, frozenset)) else args[0]
    return obj.is_in(values)


def _polars_notna(obj, args, kwargs, dtype=None):
    # FIXME(#216): same unknown-dtype gap as _polars_fillna. Without a dtype the
    # is_not_nan half is skipped, so (a / b).notna() reports True for a NaN
    # that pandas reports as False. _polars_isna has the mirror-image gap.
    result = obj.is_not_null()
    actual_dtype = getattr(obj, "dtype", None) or dtype
    if actual_dtype is not None and actual_dtype.is_float():
        result = result & obj.is_not_nan()
    return result


def _polars_isna(obj, args, kwargs, dtype=None):
    # pandas counts NaN as missing, polars only null. See _polars_notna's FIXME(#216).
    result = obj.is_null()
    actual_dtype = getattr(obj, "dtype", None) or dtype
    if actual_dtype is not None and actual_dtype.is_float():
        result = result | obj.is_nan()
    return result


def _polars_map(obj, args, kwargs, dtype=None):
    # Misses become missing, like pandas (unlike replace(), which keeps them).
    # A missing input misses too, so na_action="ignore" needs nothing extra.
    mapping = _map_mapping(args, kwargs)
    if {type(v) for v in mapping.values() if v is not None} == {int, float}:
        # pandas gives float64; Polars types the lookup by its first value.
        mapping = {k: None if v is None else float(v) for k, v in mapping.items()}
    return obj.replace_strict(mapping, default=None)


def _spec(name, validate, polars_eval, *, validate_frame=None):
    return ColumnMethodSpec(
        name=name,
        validate=validate,
        pandas_eval=_pandas_method(name),
        polars_eval=polars_eval,
        validate_frame=validate_frame,
    )


COLUMN_METHOD_SPECS = {
    spec.name: spec
    for spec in (
        _spec("astype", _validate_astype, _polars_astype, validate_frame=_validate_astype),
        _spec("fillna", _validate_fillna, _polars_fillna,
              validate_frame=_validate_frame_fillna),
        _spec("clip", _validate_clip, _polars_clip),
        _spec("where", _validate_where, _polars_where),
        _spec("isin", _validate_isin, _polars_isin),
        _spec("isna", _validate_notna, _polars_isna),
        _spec("isnull", _validate_notna, _polars_isna),
        _spec("notna", _validate_notna, _polars_notna),
        _spec("notnull", _validate_notna, _polars_notna),
        _spec("map", _validate_map, _polars_map),
    )
}


def get_column_method_spec(name: str) -> ColumnMethodSpec | None:
    return COLUMN_METHOD_SPECS.get(name)


def is_supported_column_method(op: MethodCallOp) -> bool:
    spec = get_column_method_spec(op.method_name)
    if spec is None or not op.inputs:
        return False
    validate = (spec.validate_frame if op.inputs[0].output_type is OutputType.FRAME
                else spec.validate)
    return validate is not None and validate(tuple(op.args or ()), dict(op.kwargs or {}))


class ColumnMethodOp(MapOp):
    """A supported shape-preserving method on a Series or DataFrame."""

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
    BOOLEAN_METHODS = frozenset({"isin", "isna", "isnull", "notna", "notnull"})

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
