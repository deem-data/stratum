"""Physical implementations of the map operators.

Each logical map op is a same-shape backend-variant family: the concrete
``Pandas*`` / ``Polars*`` impls subclass the logical op (+ :class:`PhysicalOp`)
and carry only the backend-specific kernel. Selection swaps the logical op to
the matching impl in place, so ``isinstance(op, StringMethodOp)`` etc. still
holds across the plan. For ``AssignMapOp`` every folded entry runs in one
``with_columns`` call on polars and goes through ``assign`` on pandas (see
``_expressions``). The call-based maps replay their recorded pandas call on
pandas and translate it on polars (see ``_polars_translate``).
"""
from __future__ import annotations

import logging

import pandas as pd
import polars as pl
from numpy import sin, cos

from stratum.optimizer.logical._base import (
    OperandRef, _resolve_args, _resolve_kwargs, _resolve_operand)
from stratum.optimizer.logical._column_methods import (
    AstypeOp, ClipOp, ColumnMethodOp, FillnaOp, IsinOp, IsnaOp, MapValuesOp, NotnaOp,
    WhereOp)
from stratum.optimizer.logical._map_ops import (
    ApplyUDFOp, AssignMapOp, AssignOp, AttributeReadOp, DatetimeConversionOp,
    DatetimeFieldOp, StringMethodOp)
from stratum.optimizer.physical._expressions import (
    EvalContext, pandas_column_method, pandas_to_datetime, refuse_unsupported_exprs,
    to_pandas, to_polars)
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._polars_translate import (
    polars_column_method, polars_dt_attribute, polars_dt_attribute_unsupported,
    polars_frame_fillna_unsupported, polars_str_method_unsupported, polars_string_method,
    polars_to_datetime)
from stratum.optimizer.physical._registry import physical_impl

logger = logging.getLogger(__name__)


def _resolved_call(op, inputs: list):
    """The receiver (operand 0) and the recorded call's resolved arguments."""
    return inputs[0], _resolve_args(op.args, inputs), _resolve_kwargs(op.kwargs, inputs)


# --- AssignMapOp (folded df.assign) ------------------------------------------

@physical_impl(of=AssignMapOp, backend="pandas")
class PandasAssignMapOp(AssignMapOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        ctx = EvalContext.of(inputs, mode)
        values = {name: to_pandas(expr, ctx) for name, expr in self.entries.items()}
        return ctx.frame.assign(**values)


@physical_impl(of=AssignMapOp, backend="polars")
class PolarsAssignMapOp(AssignMapOp, PhysicalOp):
    @classmethod
    def supports(cls, op: AssignMapOp, ctx) -> bool:
        return refuse_unsupported_exprs(op, op.entries.values())

    def process(self, mode: str, inputs: list):
        ctx = EvalContext.of(inputs, mode)
        columns = {}
        for name, expr in self.entries.items():
            result = to_polars(expr, ctx)
            if isinstance(result, list):
                # Polars treats a list passed through the keyword API as one
                # list-valued scalar; assign semantics require a column.
                result = pl.Series(result)
            columns[name] = result
        # The keyword API accepts expressions, series, arrays and scalars,
        # broadcasting the latter just like pandas.DataFrame.assign.
        return ctx.frame.with_columns(**columns)


# --- ColumnMethodOp (astype/fillna/clip/where/isin/isna/notna/map) ---------------

def _method_fields(op: ColumnMethodOp, inputs: list) -> dict:
    """The op's fields by name, graph-fed ones resolved against ``inputs``."""
    return {name: _resolve_operand(getattr(op, name), inputs) for name in op.fields}


class _PandasColumnMethodExec(ColumnMethodOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        return pandas_column_method(inputs[0], self.method, _method_fields(self, inputs))


class _PolarsColumnMethodExec(ColumnMethodOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        obj = inputs[0]
        return polars_column_method(self.method, obj, _method_fields(self, inputs),
                                    getattr(obj, "dtype", None))


@physical_impl(of=AstypeOp, backend="pandas")
class PandasAstypeOp(AstypeOp, _PandasColumnMethodExec):
    pass


@physical_impl(of=AstypeOp, backend="polars")
class PolarsAstypeOp(AstypeOp, _PolarsColumnMethodExec):
    pass


@physical_impl(of=FillnaOp, backend="pandas")
class PandasFillnaOp(FillnaOp, _PandasColumnMethodExec):
    pass


@physical_impl(of=FillnaOp, backend="polars")
class PolarsFillnaOp(FillnaOp, _PolarsColumnMethodExec):
    @classmethod
    def supports(cls, op: FillnaOp, ctx) -> bool:
        # A frame's per-column fill must keep each column's kind.
        reason = polars_frame_fillna_unsupported(op.value, op.inputs[0].output_schema)
        if reason is not None:
            op._unsupported_reason = reason
            return False
        return True


@physical_impl(of=ClipOp, backend="pandas")
class PandasClipOp(ClipOp, _PandasColumnMethodExec):
    pass


@physical_impl(of=ClipOp, backend="polars")
class PolarsClipOp(ClipOp, _PolarsColumnMethodExec):
    pass


@physical_impl(of=WhereOp, backend="pandas")
class PandasWhereOp(WhereOp, _PandasColumnMethodExec):
    pass


@physical_impl(of=WhereOp, backend="polars")
class PolarsWhereOp(WhereOp, _PolarsColumnMethodExec):
    pass


@physical_impl(of=IsinOp, backend="pandas")
class PandasIsinOp(IsinOp, _PandasColumnMethodExec):
    pass


@physical_impl(of=IsinOp, backend="polars")
class PolarsIsinOp(IsinOp, _PolarsColumnMethodExec):
    pass


@physical_impl(of=IsnaOp, backend="pandas")
class PandasIsnaOp(IsnaOp, _PandasColumnMethodExec):
    pass


@physical_impl(of=IsnaOp, backend="polars")
class PolarsIsnaOp(IsnaOp, _PolarsColumnMethodExec):
    pass


@physical_impl(of=NotnaOp, backend="pandas")
class PandasNotnaOp(NotnaOp, _PandasColumnMethodExec):
    pass


@physical_impl(of=NotnaOp, backend="polars")
class PolarsNotnaOp(NotnaOp, _PolarsColumnMethodExec):
    pass


@physical_impl(of=MapValuesOp, backend="pandas")
class PandasMapValuesOp(MapValuesOp, _PandasColumnMethodExec):
    pass


@physical_impl(of=MapValuesOp, backend="polars")
class PolarsMapValuesOp(MapValuesOp, _PolarsColumnMethodExec):
    pass


# --- ApplyUDFOp (df.apply / column UDF) ---------------------------------------

@physical_impl(of=ApplyUDFOp, backend="pandas")
class PandasApplyUDFOp(ApplyUDFOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = _resolved_call(self, inputs)
        return _obj.apply(*_args, **_kwargs)


@physical_impl(of=ApplyUDFOp, backend="polars")
class PolarsApplyUDFOp(ApplyUDFOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = _resolved_call(self, inputs)
        if not isinstance(_obj, pl.Series):
            return _obj.map_rows(*_args, **_kwargs)
        if _args[0] == sin:
            logger.debug("Rewrite UDF sin to polars sin")
            return _obj.sin()
        if _args[0] == cos:
            logger.debug("Rewrite UDF cos to polars cos")
            return _obj.cos()
        return _obj.map_elements(*_args, **_kwargs)


# --- AssignOp (df.assign, non-foldable) ---------------------------------------

@physical_impl(of=AssignOp, backend="pandas")
class PandasAssignOp(AssignOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = _resolved_call(self, inputs)
        return _obj.assign(*_args, **_kwargs)


@physical_impl(of=AssignOp, backend="polars")
class PolarsAssignOp(AssignOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = _resolved_call(self, inputs)
        checked_kwargs = {}
        for k, v in _kwargs.items():
            if isinstance(v, pd.Series) or isinstance(v, pd.DataFrame):
                # A literal pandas value recorded in the call, not a plan edge.
                logger.warning(f"Converting pandas object to polars object for column {k}")
                checked_kwargs[k] = pl.from_pandas(v)
            elif isinstance(v, list):
                checked_kwargs[k] = pl.Series(v)
            else:
                checked_kwargs[k] = v
        return _obj.with_columns(*_args, **checked_kwargs)


# --- DatetimeConversionOp (pd.to_datetime) ------------------------------------

def _to_datetime_fields(op: DatetimeConversionOp, inputs: list) -> tuple:
    """The op's fields, graph-fed ones resolved against ``inputs``."""
    return tuple(_resolve_operand(getattr(op, f), inputs) for f in op.fields)


@physical_impl(of=DatetimeConversionOp, backend="pandas")
class PandasDatetimeConversionOp(DatetimeConversionOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        return pandas_to_datetime(inputs[0], *_to_datetime_fields(self, inputs))


@physical_impl(of=DatetimeConversionOp, backend="polars")
class PolarsDatetimeConversionOp(DatetimeConversionOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        return polars_to_datetime(inputs[0], *_to_datetime_fields(self, inputs))


# --- StringMethodOp (col.str.<method>) ----------------------------------------

@physical_impl(of=StringMethodOp, backend="pandas")
class PandasStringMethodOp(StringMethodOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        return getattr(inputs[0].str, self.method)(**_resolve_kwargs(self.params, inputs))


@physical_impl(of=StringMethodOp, backend="polars")
class PolarsStringMethodOp(StringMethodOp, PhysicalOp):
    @classmethod
    def supports(cls, op: StringMethodOp, ctx) -> bool:
        # An untranslatable call binds the pandas impl. With plan-valued
        # arguments only the method can be checked now; replace's arguments
        # are then checked when the call runs.
        if not any(isinstance(v, OperandRef) for v in op.params.values()):
            reason = polars_str_method_unsupported(op.method, op.params)
        elif op.method != "replace":
            reason = polars_str_method_unsupported(op.method)
        else:
            reason = None
        if reason is not None:
            op._unsupported_reason = f"str.{op.method}: {reason}"
            return False
        return True

    def process(self, mode: str, inputs: list):
        return polars_string_method(inputs[0], self.method,
                                    _resolve_kwargs(self.params, inputs))


# --- DatetimeFieldOp (col.dt.year, ...) --------------------------------------

@physical_impl(of=DatetimeFieldOp, backend="pandas")
class PandasDatetimeFieldOp(DatetimeFieldOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        return getattr(inputs[0].dt, self.field)


@physical_impl(of=DatetimeFieldOp, backend="polars")
class PolarsDatetimeFieldOp(DatetimeFieldOp, PhysicalOp):
    @classmethod
    def supports(cls, op: DatetimeFieldOp, ctx) -> bool:
        """Refuses a field polars has no method for (see
        :func:`polars_dt_attribute_unsupported`)."""
        reason = polars_dt_attribute_unsupported(op.field)
        if reason is not None:
            op._unsupported_reason = f"dt.{op.field}: {reason}"
            return False
        return True

    def process(self, mode: str, inputs: list):
        return polars_dt_attribute(inputs[0], self.field)


# --- AttributeReadOp (df.loc, df.T, col.cat.codes, ...) -----------------------
# pandas-only: polars has no index labels to `.loc`, no `.T` and no namespaces
# but its own, so a polars plan reads these on pandas.

@physical_impl(of=AttributeReadOp, backend="pandas")
class PandasAttributeReadOp(AttributeReadOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        result = inputs[0]
        for attr in self.attr_name:
            result = getattr(result, attr)
        return result
