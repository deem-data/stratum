"""Physical implementations of ``GetItemOp`` (``obj[key]`` indexing).

Mostly backend-agnostic; the only backend-specific case is a boolean-mask row
filter, which polars spells ``frame.filter(mask)`` rather than ``frame[mask]``.
``is_filter`` is intrinsic op config (set during extraction), so the polars
impl branches on it as data, not on a backend flag.

The key is resolved through ``_resolve_operand`` rather than a plain
``OperandRef`` check: a two-axis indexer (``frame.loc[mask, cols]``) carries a
*tuple* key whose axes may each be graph-fed, so the refs to substitute can sit
nested inside it.
"""
from __future__ import annotations

import pandas as pd

from stratum.optimizer.logical._base import _resolve_operand
from stratum.optimizer.logical._ops import GetItemOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl


@physical_impl(of=GetItemOp, backend="pandas")
class PandasGetItemOp(GetItemOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        key = _resolve_operand(self.key, inputs)
        return inputs[0][key]


@physical_impl(of=GetItemOp, backend="polars")
class PolarsGetItemOp(GetItemOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        key = _resolve_operand(self.key, inputs)
        if self.is_filter:
            return inputs[0].filter(key)
        if isinstance(key, pd.Index):
            # Column labels read off a frame (`df[df.columns[:2]]`): polars
            # indexes by a list of names, not by a pandas Index.
            key = key.tolist()
        return inputs[0][key]
