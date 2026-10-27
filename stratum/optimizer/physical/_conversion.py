"""Explicit frame-library conversions on plan edges (#227).

pandas -> polars is always admitted (polars has no index). polars -> pandas
yields a RangeIndex, so it is admitted only where the pandas plan also has a
default index (:func:`has_default_index`); otherwise the producer
runs on pandas. Strings convert to pandas' ``str`` dtype.
"""
from __future__ import annotations

import pandas as pd
import polars as pl

from stratum.optimizer.logical._base import FRAME_TYPES, IRNode, IndexEffect
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._representation import FrameLib


class ConversionOp(PhysicalOp):
    """Converts a frame or series between frame libraries."""

    logical_family = "Conversion"
    index_effect = IndexEffect.KEEP

    def __init__(self, source: FrameLib, target: FrameLib, inputs=None, outputs=None):
        super().__init__(name=f"{source}->{target}", inputs=inputs, outputs=outputs)
        self.source = source
        self.target = target
        self.frame_lib = target

    @classmethod
    def accepts_input_lib(cls, op, index: int, lib) -> bool:
        return lib == op.source

    def process(self, mode: str, inputs: list):
        value = inputs[0]
        if self.target is FrameLib.POLARS:
            return to_polars(value)
        return to_pandas(value)


def to_polars(value):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return pl.from_pandas(value)
    return value


def to_pandas(value):
    """polars -> pandas via Arrow defaults: dates as ``datetime.date``, strings as ``str``."""
    if isinstance(value, pl.Series):
        return to_pandas(value.to_frame()).iloc[:, 0]
    if isinstance(value, pl.DataFrame):
        frame = value.to_arrow().to_pandas()
        strings = [name for name, dtype in value.schema.items() if dtype == pl.String]
        return frame.astype({name: "str" for name in strings}) if strings else frame
    return value


# --- index lineage ------------------------------------------------------------
#
# Index origin: the op that last determined a value's row set and order, or
# DEFAULT for a fresh RangeIndex. Row-preserving ops inherit it. Equal origins
# imply positional pairing (polars) == label alignment (pandas).

DEFAULT = "default"


def has_default_index(op: IRNode, memo: dict | None = None) -> bool:
    """Whether ``op``'s pandas output has a default RangeIndex."""
    return index_origin(op, memo) == DEFAULT


def index_origin(op: IRNode, memo: dict | None = None):
    """``DEFAULT`` or the id of the origin op. Unknown ops start a new origin."""
    memo = {} if memo is None else memo
    key = id(op)
    if key not in memo:
        memo[key] = key  # cycle guard
        memo[key] = _index_origin(op, memo)
    return memo[key]


def _index_origin(op: IRNode, memo: dict):
    """Follow ``op``'s declared :class:`IndexEffect`."""
    effect = op.index_effect
    if effect is IndexEffect.DEFAULT:
        return DEFAULT
    if effect is IndexEffect.COMMON:
        return _common_origin(op, memo)
    first = index_origin(op.inputs[0], memo) if op.inputs else id(op)
    if effect is IndexEffect.KEEP:
        return first
    if effect is IndexEffect.PREFIX and first == DEFAULT:
        return DEFAULT
    return id(op)


def _common_origin(op: IRNode, memo: dict):
    """Elementwise op: differing origins align on a label union (new origin)."""
    origins = {index_origin(i, memo) for i in op.inputs
               if i.output_type in FRAME_TYPES or i.output_type.name == "UNKNOWN"}
    return origins.pop() if len(origins) == 1 else id(op)
