"""Physical implementations of ``GetItemOp`` (``obj[key]`` indexing).

Mostly backend-agnostic; the only backend-specific case is a boolean-mask row
filter, which polars spells ``frame.filter(mask)`` rather than ``frame[mask]``.
``is_filter`` is intrinsic op config (set during extraction), so the polars
impl branches on it as data, not on a backend flag.
"""
from __future__ import annotations

from stratum.optimizer.logical._base import OperandRef
from stratum.optimizer.logical._ops import GetItemOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl


@physical_impl(of=GetItemOp, backend="pandas")
class PandasGetItemOp(GetItemOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        key = inputs[self.key.k] if isinstance(self.key, OperandRef) else self.key
        return inputs[0][key]


@physical_impl(of=GetItemOp, backend="polars")
class PolarsGetItemOp(GetItemOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        key = inputs[self.key.k] if isinstance(self.key, OperandRef) else self.key
        if self.is_filter:
            return inputs[0].filter(key)
        return inputs[0][key]
