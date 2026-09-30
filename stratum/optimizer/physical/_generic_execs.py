"""Physical execution for generic values and Python calls."""

from polars import Series as PlSeries

from stratum.optimizer.logical._base import _resolve_args, _resolve_kwargs
from stratum.optimizer.logical._ops import CallOp, MethodCallOp, ValueOp
from stratum.optimizer.physical._lowering import lowering_rule
from stratum.optimizer.physical._physical_ops import PhysicalOp


class ValueExec(ValueOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        out = self.value
        self.value = None
        return out


class MethodCallExec(MethodCallOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        obj = inputs[0]
        args = _resolve_args(self.args, inputs)
        kwargs = _resolve_kwargs(self.kwargs, inputs)
        if self.method_name == "apply" and isinstance(obj, PlSeries):
            return obj.map_elements(*args, **kwargs)
        return obj.__getattribute__(self.method_name)(*args, **kwargs)


class CallExec(CallOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        args = _resolve_args(self.args, inputs)
        kwargs = _resolve_kwargs(self.kwargs, inputs)
        return self.func(*args, **kwargs)


@lowering_rule(ValueOp, MethodCallOp, CallOp)
def lower_generic(op, ctx):
    # These are pure execution refinements: keep the node identity and all state.
    op.__class__ = {
        ValueOp: ValueExec,
        MethodCallOp: MethodCallExec,
        CallOp: CallExec,
    }[type(op)]
    return op
