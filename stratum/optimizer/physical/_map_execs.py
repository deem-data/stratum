"""Physical implementations of the map operators.

Each logical map op is a same-shape backend-variant family: the concrete
``Pandas*`` / ``Polars*`` impls subclass the logical op (+ :class:`PhysicalOp`)
and carry only the backend-specific kernel. Selection swaps the logical op to
the matching impl in place, so ``isinstance(op, StringMethodOp)`` etc. still
holds across the plan. For ``AssignMapOp`` every folded entry runs in one
``with_columns`` call on polars and goes through ``assign`` on pandas; the
backend-agnostic ``ColumnExpr`` folding and ``make_context`` stay on the
logical side. The call-based maps share ``_extract_args_and_kwargs`` from the
logical ``CallMapOp`` base (it is backend-agnostic operand plumbing).
"""
from __future__ import annotations

import logging

import pandas as pd
import polars as pl
from numpy import sin, cos

from stratum.optimizer.logical._base import OperandRef
from stratum.optimizer.logical._column_methods import (
    ColumnMethodOp, get_column_method_spec)
from stratum.optimizer.logical._map_ops import (
    AccessorMapOp, ApplyUDFOp, AssignMapOp, AssignOp, CallMapOp,
    DatetimeConversionOp, StringMethodOp, polars_datetime_kwargs,
    polars_dt_attribute, polars_dt_attribute_unsupported, polars_string_method,
    polars_str_method_unsupported)
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._representation import refuse_unsupported_exprs
from stratum.optimizer.physical._registry import physical_impl

logger = logging.getLogger(__name__)


# --- AssignMapOp (folded df.assign) ------------------------------------------

@physical_impl(of=AssignMapOp, backend="pandas")
class PandasAssignMapOp(AssignMapOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        ctx = self.make_context(mode, inputs)
        values = {name: expr.to_pandas(ctx) for name, expr in self.entries.items()}
        return ctx.frame.assign(**values)


@physical_impl(of=AssignMapOp, backend="polars")
class PolarsAssignMapOp(AssignMapOp, PhysicalOp):
    @classmethod
    def supports(cls, op: AssignMapOp, ctx) -> bool:
        return refuse_unsupported_exprs(op, op.entries.values())

    def process(self, mode: str, inputs: list):
        ctx = self.make_context(mode, inputs)
        columns = {}
        for name, expr in self.entries.items():
            result = expr.to_polars(ctx)
            if isinstance(result, list):
                # Polars treats a list passed through the keyword API as one
                # list-valued scalar; assign semantics require a column.
                result = pl.Series(result)
            columns[name] = result
        # The keyword API accepts expressions, series, arrays and scalars,
        # broadcasting the latter just like pandas.DataFrame.assign.
        return ctx.frame.with_columns(**columns)


# --- CallMapOp (generic func/method map) -------------------------------------

@physical_impl(of=CallMapOp, backend="pandas")
class PandasCallMapOp(CallMapOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        if self.method is not None:
            return getattr(_obj, self.method)(*_args, **_kwargs)
        if self.func is not None:
            return self.func(_obj, *_args, **_kwargs)
        raise TypeError("CallMapOp requires either `func` or `method` to be set.")


@physical_impl(of=CallMapOp, backend="polars")
class PolarsCallMapOp(CallMapOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        if self.method is not None:
            raise ValueError(f"Unsupported method: {self.method}")
        if self.func is not None:
            return self.func(_obj, *_args, **_kwargs)
        raise TypeError("CallMapOp requires either `func` or `method` to be set.")


# --- ColumnMethodOp (astype/fillna/clip/where/isin/notna) --------------------

@physical_impl(of=ColumnMethodOp, backend="pandas")
class PandasColumnMethodOp(ColumnMethodOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        obj, args, kwargs = self.resolved_call(inputs)
        spec = get_column_method_spec(self.method)
        return spec.pandas_eval(obj, args, kwargs, getattr(obj, "dtype", None))


@physical_impl(of=ColumnMethodOp, backend="polars")
class PolarsColumnMethodOp(ColumnMethodOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        obj, args, kwargs = self.resolved_call(inputs)
        spec = get_column_method_spec(self.method)
        return spec.polars_eval(obj, args, kwargs, getattr(obj, "dtype", None))


# --- ApplyUDFOp (df.apply / column UDF) ---------------------------------------

@physical_impl(of=ApplyUDFOp, backend="pandas")
class PandasApplyUDFOp(ApplyUDFOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        if self.columns:
            _obj = _obj[self.columns]
        return _obj.apply(*_args, **_kwargs)


@physical_impl(of=ApplyUDFOp, backend="polars")
class PolarsApplyUDFOp(ApplyUDFOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        n_cols = None
        if self.columns:
            _obj = _obj[self.columns]
            n_cols = 1 if type(self.columns) == str else len(self.columns)
        if isinstance(_obj, pl.Series):
            n_cols = 1
        if n_cols == 1:
            if _args[0] == sin:
                logger.debug("Rewrite UDF sin to polars sin")
                return _obj.sin()
            elif _args[0] == cos:
                logger.debug("Rewrite UDF cos to polars cos")
                return _obj.cos()
            else:
                return _obj.map_elements(*_args, **_kwargs)
        return _obj.map_rows(*_args, **_kwargs)


# --- AssignOp (df.assign, non-foldable) ---------------------------------------

@physical_impl(of=AssignOp, backend="pandas")
class PandasAssignOp(AssignOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        return _obj.assign(*_args, **_kwargs)


@physical_impl(of=AssignOp, backend="polars")
class PolarsAssignOp(AssignOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        checked_kwargs = {}
        for k, v in _kwargs.items():
            if isinstance(v, OperandRef):
                raise NotImplementedError("Is not yet suppoerted, please report this issue")
            elif isinstance(v, pd.Series) or isinstance(v, pd.DataFrame):
                # A literal pandas value recorded in the call, not a plan edge.
                logger.warning(f"Converting pandas object to polars object for column {k}")
                checked_kwargs[k] = pl.from_pandas(v)
            elif isinstance(v, list):
                checked_kwargs[k] = pl.Series(v)
            else:
                checked_kwargs[k] = v
        return _obj.with_columns(*_args, **checked_kwargs)


# --- DatetimeConversionOp (pd.to_datetime) ------------------------------------

def _to_datetime_via_pandas(obj, args, kwargs):
    """Preserve pandas datetime semantics for options Polars cannot express."""
    name = getattr(obj, "name", None)
    if isinstance(obj, (pl.Series, pl.DataFrame)):
        obj = obj.to_pandas()
    result = pd.to_datetime(obj, *args, **kwargs)
    if isinstance(result, (pd.Series, pd.DataFrame)):
        return pl.from_pandas(result)
    if isinstance(result, pd.Index):
        return pl.Series(name, result)
    return result


@physical_impl(of=DatetimeConversionOp, backend="pandas")
class PandasDatetimeConversionOp(DatetimeConversionOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        return pd.to_datetime(_obj, *_args, **_kwargs)


@physical_impl(of=DatetimeConversionOp, backend="polars")
class PolarsDatetimeConversionOp(DatetimeConversionOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        translated = polars_datetime_kwargs(_args, _kwargs)
        if translated is not None:
            # TODO: Support already-datetime and numeric operands natively;
            # the Polars string namespace only accepts string input.
            return _obj.str.to_datetime(**translated)
        return _to_datetime_via_pandas(_obj, _args, _kwargs)


# --- StringMethodOp (col.str.<method>) ----------------------------------------

@physical_impl(of=StringMethodOp, backend="pandas")
class PandasStringMethodOp(StringMethodOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        return getattr(_obj.str, self.method)(*_args, **_kwargs)


@physical_impl(of=StringMethodOp, backend="polars")
class PolarsStringMethodOp(StringMethodOp, PhysicalOp):
    @classmethod
    def supports(cls, op: StringMethodOp, ctx) -> bool:
        # An untranslatable call binds the pandas impl. With plan-valued
        # arguments only the method can be checked now; replace's arguments
        # are then checked when the call runs.
        if not _has_operand_refs(op.args, op.kwargs):
            reason = polars_str_method_unsupported(op.method, op.args, op.kwargs)
        elif op.method != "replace":
            reason = polars_str_method_unsupported(op.method)
        else:
            reason = None
        if reason is not None:
            op._unsupported_reason = f"str.{op.method}: {reason}"
            return False
        return True

    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        return polars_string_method(_obj, self.method, _args, _kwargs)


def _has_operand_refs(args, kwargs) -> bool:
    return any(isinstance(v, OperandRef) for v in (*args, *kwargs.values()))


# --- AccessorMapOp (col.dt.year, ...) ----------------------------------------

@physical_impl(of=AccessorMapOp, backend="pandas")
class PandasAccessorMapOp(AccessorMapOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        tmp = inputs[0]
        for attr in self.attr_name:
            tmp = getattr(tmp, attr)
        return tmp


@physical_impl(of=AccessorMapOp, backend="polars")
class PolarsAccessorMapOp(AccessorMapOp, PhysicalOp):

    @classmethod
    def supports(cls, op: AccessorMapOp, ctx) -> bool:
        """Allow only a ``.dt`` field polars can read, e.g. ``col.dt.year``.

        Extraction wraps *any* pandas attribute read on frame-like data (see
        :func:`make_frame_get_attr`), so this is an allow-list: ``.iloc``,
        ``.loc``, ``.T``, ``.values``, ``.cat`` and ``.str[...]`` have no
        polars translation, and polars frames have no ``.index``. A refusal
        binds the pandas impl.

        That includes the bare ``.dt`` namespace. Only a ``.dt`` method call
        reads it, and that call runs pandas' method and arguments verbatim,
        which a polars namespace mostly lacks (``floor``, ``normalize``) or
        reads differently (``round("h")`` wants ``"1h"``, ``strftime`` formats
        ``%f`` as nanoseconds).

        A listed field is still refused when polars has no method for it (see
        :func:`polars_dt_attribute_unsupported`).
        """
        attr_name = op.attr_name
        if (len(attr_name) != 2 or attr_name[0] != "dt"
                or attr_name[1] not in AccessorMapOp.ELEMENTWISE_ACCESSORS["dt"]):
            op._unsupported_reason = (f"pandas attribute `.{'.'.join(attr_name)}` "
                                      "has no polars translation")
            return False
        reason = polars_dt_attribute_unsupported(attr_name[1])
        if reason is not None:
            op._unsupported_reason = f"dt.{attr_name[1]}: {reason}"
            return False
        return True

    def process(self, mode: str, inputs: list):
        # `supports` admits only `["dt", <field>]`.
        return polars_dt_attribute(inputs[0], self.attr_name[1])
