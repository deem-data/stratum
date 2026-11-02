"""Explicit frame-library conversions on plan edges (#227).

pandas -> polars is always admitted: where the index is tracked (see
``PhysicalOp.carries_index``) the polars value holds it beside it. polars ->
pandas gives that index back, or a RangeIndex where none is held, so it is
admitted where the pandas plan also has a default index
(:func:`has_default_index`), where the index is not used (see ``_index_use``),
or where the polars producer carries it; otherwise the producer runs on pandas.
Strings convert to pandas' ``str`` dtype.
"""
from __future__ import annotations

import pandas as pd

from stratum.optimizer.logical._base import FRAME_TYPES, IRNode, IndexEffect
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._representation import FrameLib
from stratum.runtime._frames import PolarsIndex, polars_index_of, to_pandas, to_polars, unwrap


class ConversionOp(PhysicalOp):
    """Converts a frame or series between frame libraries."""

    logical_family = "Conversion"
    index_effect = IndexEffect.KEEP
    reads_index = False
    cacheable = True

    def __init__(self, source: FrameLib, target: FrameLib, inputs=None, outputs=None):
        super().__init__(name=f"{source}->{target}", inputs=inputs, outputs=outputs)
        self.source = source
        self.target = target
        self.frame_lib = target

    def lineage_config(self):
        return str(self.source), str(self.target)

    @classmethod
    def accepts_input_lib(cls, op, index: int, lib) -> bool:
        return lib == op.source

    def process(self, mode: str, inputs: list):
        return to_polars(inputs[0]) if self.target is FrameLib.POLARS else to_pandas(inputs[0])

    def process_indexed(self, mode: str, inputs: list):
        value = unwrap(inputs[0])
        if self.target is FrameLib.POLARS:
            held = self._tracks_index and isinstance(value, (pd.DataFrame, pd.Series))
            return to_polars(value), PolarsIndex.of(value.index) if held else None
        result = to_pandas(value)
        index = polars_index_of(inputs[0])
        if index is not None:
            result.index = index.to_pandas()
        return result, None


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
    stands_for = getattr(op, "stands_for", None)
    if stands_for is not None:
        # A BoundaryOp: the index is that of the op before the split it reads.
        return index_origin(stands_for, memo)
    labels_source = getattr(op, "labels_source", None)
    if labels_source is not None:
        # It takes another op's labels, row by row (``set_axis(x.index)``).
        return index_origin(labels_source, memo)
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
    origins = {index_origin(i, memo) for k, i in enumerate(op.inputs)
               if (i.output_type in FRAME_TYPES or i.output_type.name == "UNKNOWN")
               and not op.reads_state(k)}
    return origins.pop() if len(origins) == 1 else id(op)
