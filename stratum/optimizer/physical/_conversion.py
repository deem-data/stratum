"""Explicit frame-library conversions on plan edges (#227).

Index checks allow pandas -> Polars conversions, which discard pandas labels.
Polars -> pandas yields a RangeIndex, so it is allowed only where the pandas
plan also has a default index (:func:`has_default_index`); otherwise the
producer must run on pandas. Strings convert to pandas' ``str`` dtype.
"""
from __future__ import annotations

from typing import NamedTuple

import pandas as pd
import polars as pl

from stratum.optimizer.logical._base import FRAME_TYPES, IRNode, IndexEffect
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._representation import FrameLib


class ConversionOp(PhysicalOp):
    """Converts a frame or series between frame libraries."""

    logical_family = "Conversion"
    index_effect = IndexEffect.KEEP
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

    def process(self, inputs: list):
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
# Index identity and default labels are separate properties: two RangeIndexes
# can have different lengths. Row-preserving operators inherit both properties.


class _IndexOrigin(NamedTuple):
    """Index identity, with a proof of default labels when available."""
    producer: int
    default: bool = False


def has_default_index(op: IRNode, memo: dict | None = None) -> bool:
    """Whether ``op``'s pandas output is known to have a default RangeIndex."""
    return index_origin(op, memo).default


def index_origin(op: IRNode, memo: dict | None = None) -> _IndexOrigin:
    """Index provenance; equality proves matching row labels and order."""
    memo = {} if memo is None else memo
    key = id(op)
    if key not in memo:
        memo[key] = _IndexOrigin(key)  # cycle guard
        memo[key] = _index_origin(op, memo)
    return memo[key]


def _index_origin(op: IRNode, memo: dict):
    """Follow ``op``'s declared :class:`IndexEffect`."""
    labels_source = getattr(op, "labels_source", None)
    if labels_source is not None:
        # set_axis(x.index) takes x's labels in the same order.
        return index_origin(labels_source, memo)
    effect = op.index_effect
    if effect is IndexEffect.DEFAULT:
        return _IndexOrigin(id(op), default=True)
    if effect is IndexEffect.COMMON:
        return _common_origin(op, memo)
    first = index_origin(op.inputs[0], memo) if op.inputs else _IndexOrigin(id(op))
    if effect is IndexEffect.KEEP:
        return first
    if effect is IndexEffect.PREFIX:
        return _IndexOrigin(id(op), default=first.default)
    return _IndexOrigin(id(op))


def _structure_value(op: IRNode) -> bool:
    """Shape or column metadata, optionally indexed or converted to a list."""
    from stratum.optimizer.logical._metadata_ops import ColumnNamesOp, ShapeOp
    from stratum.optimizer.logical._ops import GetItemOp, MethodCallOp

    while op.inputs and (isinstance(op, GetItemOp)
                         or isinstance(op, MethodCallOp) and op.method_name == "tolist"):
        op = op.inputs[0]
    return isinstance(op, (ColumnNamesOp, ShapeOp))


def index_inputs(op: IRNode):
    """Data operands that carry, or may carry, pandas row labels."""
    from stratum.optimizer.logical._ops import ValueOp

    for k, source in enumerate(op.inputs):
        if op.reads_state(k):
            continue
        if source.output_type not in FRAME_TYPES and source.output_type.name != "UNKNOWN":
            continue
        if _structure_value(source):
            continue
        # Scalars and positional array/list operands carry no row labels.
        if (isinstance(source, ValueOp)
                and not isinstance(source.value, (pd.Series, pd.DataFrame, pl.Series, pl.DataFrame))):
            continue
        yield source


def _common_origin(op: IRNode, memo: dict):
    """A common index, or the new index formed by label alignment."""
    origins = {index_origin(source, memo) for source in index_inputs(op)}
    if len(origins) == 1:
        return origins.pop()
    # A union of 0..n-1 indexes remains default, but its length may change.
    return _IndexOrigin(id(op), default=bool(origins) and all(o.default for o in origins))
