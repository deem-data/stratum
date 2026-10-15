"""Explicit frame-library conversions on plan edges (#227).

pandas -> polars is always admitted (polars has no index). polars -> pandas
yields a RangeIndex, so it is admitted only where the pandas plan also has a
default index (:func:`has_default_index`); otherwise the producer
runs on pandas. Strings convert to pandas' ``str`` dtype.
"""
from __future__ import annotations

import pandas as pd
import polars as pl

from stratum.optimizer.logical._base import FRAME_TYPES, IRNode
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._representation import FrameLib


class ConversionOp(PhysicalOp):
    """Converts a frame or series between frame libraries."""

    logical_family = "Conversion"

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


def _is_default_index(data) -> bool:
    if isinstance(data, (pl.DataFrame, pl.Series)):
        return True
    index = getattr(data, "index", None)
    return (isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1)


def _index_origin(op: IRNode, memo: dict):
    # Lazy imports: avoid import cycles.
    from stratum.optimizer.logical._aggregation_ops import AggregateOp
    from stratum.optimizer.logical._join_ops import FILTERING_JOINS, JoinOp
    from stratum.optimizer.logical._map_ops import MapOp
    from stratum.optimizer.logical._numeric_ops import NumericOp
    from stratum.optimizer.logical._ops import (BaseEstimatorOp, BinOp, ImplOp,
                                                MethodCallOp, SplitXOp, UnaryOp,
                                                ValueOp)
    from stratum.optimizer.logical._projection_ops import (
        ColumnProjectionOp, ColumnSelectorOp, DropOp, MetadataOp)
    from stratum.optimizer.logical._selection_ops import SelectionKind, SelectionOp
    from stratum.optimizer.logical._sort_ops import SortOp
    from stratum.optimizer.physical._source_execs import FileReadOp, InMemoryFrame

    new = id(op)

    def first_input():
        return index_origin(op.inputs[0], memo) if op.inputs else new

    if isinstance(op, ConversionOp):
        return first_input()
    if isinstance(op, FileReadOp):
        return DEFAULT
    if isinstance(op, InMemoryFrame):
        return DEFAULT if _is_default_index(op.data) else new
    if isinstance(op, ValueOp):
        return DEFAULT if _is_default_index(op.value) else new
    if isinstance(op, MetadataOp):
        if op.func == "reset_index":
            return DEFAULT if dict(op.kwargs or {}).get("drop") is True else new
        return first_input()  # rename
    if isinstance(op, SelectionOp):
        # Keeps row labels; only head() of a default index stays default.
        if op.kind is SelectionKind.HEAD and first_input() == DEFAULT:
            return DEFAULT
        if dict(op.kwargs or {}).get("ignore_index") is True:
            return DEFAULT
        return new
    if isinstance(op, SortOp):
        return DEFAULT if op.ignore_index else new
    if isinstance(op, JoinOp):
        # merge -> RangeIndex; index and semi/anti joins keep labels.
        if op.left_index or op.right_index or op.how in FILTERING_JOINS:
            return new
        return DEFAULT
    if isinstance(op, AggregateOp):
        return DEFAULT if op.grouped and op.options.get("as_index", True) is False else new
    if isinstance(op, (MapOp, ColumnProjectionOp, ColumnSelectorOp, DropOp,
                       BaseEstimatorOp, ImplOp, SplitXOp)):
        # Row-preserving. A DropOp only drops columns; a row drop is a
        # SelectionOp(DROP) and starts a new origin above.
        return first_input()
    if isinstance(op, MethodCallOp) and op.method_name in _ROW_PRESERVING_METHODS:
        return _common_origin(op, memo)
    if isinstance(op, (BinOp, UnaryOp, NumericOp)):
        return _common_origin(op, memo)
    return new


#: Row-preserving pandas methods.
_ROW_PRESERVING_METHODS = frozenset({
    "where", "mask", "fillna", "astype", "map", "clip", "abs", "round", "replace",
    "isna", "notna", "isnull", "notnull", "isin", "between", "rename", "copy",
})


def _common_origin(op: IRNode, memo: dict):
    """Elementwise op: differing origins align on a label union (new origin)."""
    origins = {index_origin(i, memo) for i in op.inputs
               if i.output_type in FRAME_TYPES or i.output_type.name == "UNKNOWN"}
    return origins.pop() if len(origins) == 1 else id(op)
