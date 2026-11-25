"""Physical assignment of already-computed columns and literal values."""
import pandas as pd
import polars as pl

from stratum.optimizer.logical._base import OperandRef, OutputType, _resolve_args, _resolve_kwargs
from stratum.optimizer.logical._column_expr import Const
from stratum.optimizer.logical._ops import ValueOp
from stratum.optimizer.logical._combine_ops import AssignOp
from stratum.optimizer.physical._expressions import polars_unsupported
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl


# --- Assignment of computed values --------------------------------------------

def polars_assignment_value(value):
    """Keep sequences column-valued and scalars literal, including strings."""
    if isinstance(value, list):
        return pl.Series(value)
    return pl.lit(value) if pd.api.types.is_scalar(value) else value


@physical_impl(of=AssignOp, backend="pandas")
class PandasAssignOp(AssignOp, PhysicalOp):
    def process(self, inputs: list):
        _obj = inputs[0]
        _args = _resolve_args(self.args, inputs)
        _kwargs = _resolve_kwargs(self.kwargs, inputs)
        return _obj.assign(*_args, **_kwargs)


@physical_impl(of=AssignOp, backend="polars")
class PolarsAssignOp(AssignOp, PhysicalOp):
    @classmethod
    def supports(cls, op, ctx):
        for value in op.kwargs.values():
            if isinstance(value, OperandRef):
                source = op.inputs[value.k]
                if isinstance(source, ValueOp):
                    value = source.value
                elif source.output_type is OutputType.UNKNOWN:
                    op._unsupported_reason = "an assignment operand may require label alignment"
                    return False
            if callable(value):
                op._unsupported_reason = "callable assignments require pandas evaluation"
                return False
            if isinstance(value, (pd.Series, pd.DataFrame)):
                op._unsupported_reason = "literal pandas assignments require label alignment"
                return False
            if pd.api.types.is_scalar(value):
                reason = polars_unsupported(Const(value))
                if reason is not None:
                    op._unsupported_reason = reason
                    return False
        return True

    def process(self, inputs: list):
        _obj = inputs[0]
        _args = _resolve_args(self.args, inputs)
        _kwargs = _resolve_kwargs(self.kwargs, inputs)
        checked_kwargs = {name: polars_assignment_value(value) for name, value in _kwargs.items()}
        return _obj.with_columns(*_args, **checked_kwargs)
