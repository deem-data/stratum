"""Physical implementations of the projection operators.

Each logical projection op (``MetadataOp``, ``DropOp``, ``ColumnSelectorOp``,
``ApplyUDFOp``, ``AssignOp``, ``DatetimeConversionOp``, ``StringMethodOp``,
``GetAttrProjectionOp``) is a same-shape backend-variant family: the concrete
``Pandas*`` / ``Polars*`` impls subclass the logical op (+ :class:`PhysicalOp`)
and carry only the backend-specific ``process``. Selection swaps the logical op
to the matching impl in place, so ``isinstance(op, DropOp)`` etc. still holds
across the plan. The shared ``_extract_args_and_kwargs`` helper stays on the
logical ``ProjectionOp`` base (it is backend-agnostic operand plumbing).
"""
from __future__ import annotations

import logging

import pandas as pd
import polars as pl
from numpy import sin, cos

from stratum.optimizer.logical._base import OperandRef, _resolve_args, _resolve_kwargs
from stratum.optimizer.logical._projection_ops import (
    ApplyUDFOp, AssignOp, ColumnProjectionOp, ColumnSelectorOp,
    DatetimeConversionOp, DropOp, GetAttrProjectionOp, MetadataOp, ProjectionOp,
    StringMethodOp, STR_POLARS_METHODS, polars_datetime_kwargs,
    resolve_selector_columns)
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl

logger = logging.getLogger(__name__)


# --- ProjectionOp (generic func/method projection) ----------------------------

@physical_impl(of=ProjectionOp, backend="pandas")
class PandasProjectionOp(ProjectionOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        if self.method is not None:
            return getattr(_obj, self.method)(*_args, **_kwargs)
        if self.func is not None:
            return self.func(_obj, *_args, **_kwargs)
        raise TypeError("ProjectionOp requires either `func` or `method` to be set.")


@physical_impl(of=ProjectionOp, backend="polars")
class PolarsProjectionOp(ProjectionOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        if self.method is not None:
            raise ValueError(f"Unsupported method: {self.method}")
        if self.func is not None:
            return self.func(_obj, *_args, **_kwargs)
        raise TypeError("ProjectionOp requires either `func` or `method` to be set.")


# --- MetadataOp (e.g. rename) -------------------------------------------------

@physical_impl(of=MetadataOp, backend="pandas")
class PandasMetadataOp(MetadataOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj = inputs[0]
        _args = _resolve_args(self.args, inputs)
        _kwargs = _resolve_kwargs(self.kwargs, inputs)
        return getattr(_obj, self.func)(*_args, **_kwargs)


@physical_impl(of=MetadataOp, backend="polars")
class PolarsMetadataOp(MetadataOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj = inputs[0]
        _args = _resolve_args(self.args, inputs)
        _kwargs = _resolve_kwargs(self.kwargs, inputs)
        if "columns" in _kwargs:
            _args.append(_kwargs["columns"])
        return getattr(_obj, self.func)(*_args)


# --- DropOp -------------------------------------------------------------------

@physical_impl(of=DropOp, backend="pandas")
class PandasDropOp(DropOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        return _obj.drop(*_args, **_kwargs)


@physical_impl(of=DropOp, backend="polars")
class PolarsDropOp(DropOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        if "columns" in _kwargs:
            _args.append(_kwargs["columns"])
        if "ignore_errors" in _kwargs:
            _args.append(_kwargs["ignore_errors"] == "raise")
        return _obj.drop(*_args)


# --- ColumnSelectorOp (skb.select) --------------------------------------------

class ColumnSelectorExec(ColumnSelectorOp, PhysicalOp):
    """Physical base: resolve the (deferred) selector at fit time and reuse the
    stored column list at predict time -- backend-agnostic; only the final
    indexing differs per backend."""

    def _resolved_columns(self, frame, mode: str):
        if mode == "fit_transform":
            self.selected_columns = resolve_selector_columns(frame, self.selector)
        elif self.selected_columns is None:
            raise RuntimeError(
                f"{self} was asked to transform before the selector was resolved; "
                f"run fit_transform first.")
        return self.selected_columns


@physical_impl(of=ColumnSelectorOp, backend="pandas")
class PandasColumnSelectorOp(ColumnSelectorExec):
    def process(self, mode: str, inputs: list):
        frame = inputs[0]
        return frame[self._resolved_columns(frame, mode)]


@physical_impl(of=ColumnSelectorOp, backend="polars")
class PolarsColumnSelectorOp(ColumnSelectorExec):
    def process(self, mode: str, inputs: list):
        frame = inputs[0]
        return frame.select(self._resolved_columns(frame, mode))


# --- ColumnProjectionOp (df["a"] / df[["a", "b"]]) ----------------------------
# Both backends index by literal name(s) the same way: a str yields a column
# (Series), a list of str a sub-frame. No fit-time resolution (the columns are
# given verbatim), unlike the deferred-selector ColumnSelectorOp above.

@physical_impl(of=ColumnProjectionOp, backend="pandas")
class PandasColumnProjectionOp(ColumnProjectionOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        return inputs[0][self.key]


@physical_impl(of=ColumnProjectionOp, backend="polars")
class PolarsColumnProjectionOp(ColumnProjectionOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        return inputs[0][self.key]


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
        return pd.to_datetime(inputs[0], *self.args, **self.kwargs)


@physical_impl(of=DatetimeConversionOp, backend="polars")
class PolarsDatetimeConversionOp(DatetimeConversionOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        translated = polars_datetime_kwargs(self.args, self.kwargs)
        if translated is not None:
            # TODO: Support already-datetime and numeric operands natively;
            # the Polars string namespace only accepts string input.
            return inputs[0].str.to_datetime(**translated)
        return _to_datetime_via_pandas(inputs[0], self.args, self.kwargs)


# --- StringMethodOp (col.str.<method>) ----------------------------------------

@physical_impl(of=StringMethodOp, backend="pandas")
class PandasStringMethodOp(StringMethodOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        return getattr(_obj.str, self.method)(*_args, **_kwargs)


@physical_impl(of=StringMethodOp, backend="polars")
class PolarsStringMethodOp(StringMethodOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        name = STR_POLARS_METHODS.get(self.method, self.method)
        return getattr(_obj.str, name)(*_args, **_kwargs)


# --- GetAttrProjectionOp (col.dt.year, ...) -----------------------------------

@physical_impl(of=GetAttrProjectionOp, backend="pandas")
class PandasGetAttrProjectionOp(GetAttrProjectionOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        tmp = inputs[0]
        for attr in self.attr_name:
            tmp = getattr(tmp, attr)
        return tmp


@physical_impl(of=GetAttrProjectionOp, backend="polars")
class PolarsGetAttrProjectionOp(GetAttrProjectionOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        result = inputs[0]
        tmp = result
        for attr in self.attr_name:
            attr = self.POLARS_ATTR_NAME_MAP.get(attr, attr)
            # TODO find better way to handle this
            if attr == "is_month_end":
                return result.dt.month_end() == result
            # polars implements dt.day as a method, not an attribute; getattr
            # handles both attributes and methods.
            tmp = getattr(tmp, attr)
        if len(self.attr_name) == 2:
            return tmp()
        return tmp
