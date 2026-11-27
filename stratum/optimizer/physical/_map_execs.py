"""Physical implementations of map operators.

The pandas and Polars implementations subclass their logical operator and
``PhysicalOp``. Selection changes the node's class in place, preserving its
identity and ``isinstance`` checks against the logical class.

- ``ExpressionMapOp`` evaluates an expression over explicit input values.
- Call-based maps replay the recorded pandas call or translate it for Polars
  (see ``_polars_translate``).
"""
from __future__ import annotations

import logging
import operator

import pandas as pd
import polars as pl
from numpy import sin, cos

from stratum.optimizer.logical._base import (
    OutputType, OperandRef, _resolve_args, _resolve_kwargs, _resolve_operand)
from stratum.optimizer.logical._column_expr import BinOpExpr, ColumnMethodExpr, StrExpr, UfuncExpr
from stratum.optimizer.logical._ops import ValueOp
from stratum.optimizer.logical._map_ops import (
    ApplyUDFOp, ExpressionMapOp, AttributeReadOp, DatetimeConversionOp,
    DatetimeFieldOp)
from stratum.optimizer.physical._expressions import (
    EvalContext, pandas_to_datetime, refuse_unsupported_exprs,
    to_pandas, to_polars)
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._polars_translate import (
    polars_dt_attribute, polars_dt_attribute_unsupported,
    polars_to_datetime)
from stratum.optimizer.physical._registry import physical_impl

logger = logging.getLogger(__name__)


def _resolved_call(op, inputs: list):
    """The receiver (operand 0) and the recorded call's resolved arguments."""
    return inputs[0], _resolve_args(op.args, inputs), _resolve_kwargs(op.kwargs, inputs)


@physical_impl(of=ExpressionMapOp, backend="pandas")
class PandasExpressionMapOp(ExpressionMapOp, PhysicalOp):
    def process(self, inputs: list):
        return to_pandas(self.expr, EvalContext(None, inputs, self.mode))


def _map_name_is_read(op, memo=None):
    """Whether a consumer observes the name, possibly through arithmetic maps."""
    from stratum.optimizer.logical._aggregation_ops import AggregateOp
    from stratum.optimizer.logical._combine_ops import AssignOp
    from stratum.optimizer.logical._selection_ops import MaskSelectionOp
    from stratum.optimizer.physical._aggregation_execs import PolarsScalarAggregateOp

    memo = {} if memo is None else memo
    if op in memo:
        return memo[op]

    def reads_name(out):
        if isinstance(out, (ExpressionMapOp, DatetimeConversionOp, DatetimeFieldOp)):
            return _map_name_is_read(out, memo)
        if isinstance(out, AggregateOp) and PolarsScalarAggregateOp.supports(out):
            return False
        return not (isinstance(out, (AssignOp, MaskSelectionOp)) and out.inputs[0] is not op)

    result = not op.outputs or any(reads_name(out) for out in op.outputs)
    memo[op] = result
    return result


@physical_impl(of=ExpressionMapOp, backend="polars")
class PolarsExpressionMapOp(ExpressionMapOp, PhysicalOp):
    @classmethod
    def supports(cls, op, ctx):
        if isinstance(op.expr, (ColumnMethodExpr, StrExpr)):
            return refuse_unsupported_exprs(op, [op.expr])
        if any(i.output_type is OutputType.UNKNOWN
               and not (isinstance(i, ValueOp)
                        and not isinstance(i.value, (pd.Series, pd.DataFrame)))
               for i in op.inputs):
            op._unsupported_reason = "an operand's label-alignment semantics are unknown"
            return False
        if op.output_type is OutputType.SERIES:
            if _map_name_is_read(op) and not isinstance(op.result_name(), str):
                op._unsupported_reason = "the Series name cannot be represented in Polars"
                return False
        else:
            if isinstance(op.expr, UfuncExpr):
                op._unsupported_reason = "NumPy frame calls require pandas to preserve column labels"
                return False
            if isinstance(op.expr, BinOpExpr) and op.expr.op in (
                    operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge):
                op._unsupported_reason = "frame comparisons require pandas missing-value semantics"
                return False
            frames = [i for i in op.inputs if i.output_type is OutputType.FRAME]
            if any(i.output_type is OutputType.SERIES for i in op.inputs):
                op._unsupported_reason = "frame/Series arithmetic requires pandas column alignment"
                return False
            if len(frames) > 1 and (any(i.output_schema is None for i in frames)
                                   or any(list(i.output_schema) != list(frames[0].output_schema)
                                          for i in frames[1:])):
                op._unsupported_reason = "frame operands require matching columns in Polars"
                return False
        return refuse_unsupported_exprs(op, [op.expr])

    def process(self, inputs: list):
        result = to_polars(self.expr, EvalContext(None, inputs, self.mode))
        if isinstance(result, pl.Expr):
            result = pl.select(result).to_series()
        return result


# --- ApplyUDFOp (df.apply / column UDF) ---------------------------------------

@physical_impl(of=ApplyUDFOp, backend="pandas")
class PandasApplyUDFOp(ApplyUDFOp, PhysicalOp):
    def process(self, inputs: list):
        _obj, _args, _kwargs = _resolved_call(self, inputs)
        return _obj.apply(*_args, **_kwargs)


@physical_impl(of=ApplyUDFOp, backend="polars")
class PolarsApplyUDFOp(ApplyUDFOp, PhysicalOp):
    def process(self, inputs: list):
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


# --- DatetimeConversionOp (pd.to_datetime) ------------------------------------

def _to_datetime_fields(op: DatetimeConversionOp, inputs: list) -> tuple:
    """The op's fields, graph-fed ones resolved against ``inputs``."""
    return tuple(_resolve_operand(getattr(op, f), inputs) for f in op.fields)


@physical_impl(of=DatetimeConversionOp, backend="pandas")
class PandasDatetimeConversionOp(DatetimeConversionOp, PhysicalOp):
    def process(self, inputs: list):
        return pandas_to_datetime(inputs[0], *_to_datetime_fields(self, inputs))


@physical_impl(of=DatetimeConversionOp, backend="polars")
class PolarsDatetimeConversionOp(DatetimeConversionOp, PhysicalOp):
    def process(self, inputs: list):
        return polars_to_datetime(inputs[0], *_to_datetime_fields(self, inputs))


# --- DatetimeFieldOp (col.dt.year, ...) --------------------------------------

@physical_impl(of=DatetimeFieldOp, backend="pandas")
class PandasDatetimeFieldOp(DatetimeFieldOp, PhysicalOp):
    def process(self, inputs: list):
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

    def process(self, inputs: list):
        return polars_dt_attribute(inputs[0], self.field)


# --- AttributeReadOp (df.loc, df.T, col.cat.codes, ...) -----------------------
# These accesses use pandas APIs such as .loc, .T and .cat.codes. Convert
# inputs to pandas before resolving the recorded attribute path.

@physical_impl(of=AttributeReadOp, backend="pandas")
class PandasAttributeReadOp(AttributeReadOp, PhysicalOp):
    def process(self, inputs: list):
        result = inputs[0]
        for attr in self.attr_name:
            result = getattr(result, attr)
        return result
