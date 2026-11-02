"""Physical implementations of the metadata operators.

Each impl translates the op's backend-neutral fields into its library's call.
polars rows carry no index labels, so the polars impls refuse an index rename
and new labels, and treat dropping the index as a no-op. The structure reads
(``ShapeOp``, ``ColumnNamesOp``) return the same value on both backends and carry
no frame library, so they put no library constraint on the frame they read.
"""
from __future__ import annotations

import pandas as pd

from stratum.optimizer.logical._base import _resolve_args, _resolve_kwargs
from stratum.optimizer.logical._metadata_ops import (
    ColumnNamesOp, RenameOp, RowLabelsOp, ShapeOp)
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl


# --- RenameOp -------------------------------------------------------------------

@physical_impl(of=RenameOp, backend="pandas")
class PandasRenameOp(RenameOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        if self.as_frame:
            # to_frame(None) labels the column None; to_frame() falls back to 0.
            return inputs[0].to_frame(*(self.label or ()))
        if self.label is not None:
            return inputs[0].rename(self.label[0])
        # Only the axes being renamed: a Series takes no ``columns``.
        axes = {k: v for k, v in (("columns", self.columns), ("index", self.index))
                if v is not None}
        return inputs[0].rename(**_resolve_kwargs({**axes, "errors": self.errors}, inputs))


@physical_impl(of=RenameOp, backend="polars")
class PolarsRenameOp(RenameOp, PhysicalOp):
    @classmethod
    def supports(cls, op: RenameOp, ctx) -> bool:
        if op.index is not None:
            op._unsupported_reason = "renames index labels"
            return False
        if op.label is not None and not isinstance(op.label[0], str):
            op._unsupported_reason = "polars names columns by strings only"
            return False
        if op.as_frame and op.label is None:
            # pandas labels the column by the series' name, or 0 without one. A
            # polars series always has a name, and a derived one's need not be
            # pandas' (`a + b` keeps a's name in polars, has none in pandas).
            op._unsupported_reason = "labels the column by the series' pandas name"
            return False
        return True

    def process(self, mode: str, inputs: list):
        if self.as_frame:
            return inputs[0].to_frame(self.label[0])
        if self.label is not None:
            return inputs[0].rename(self.label[0])
        kwargs = _resolve_kwargs({"columns": self.columns, "errors": self.errors}, inputs)
        return inputs[0].rename(kwargs["columns"], strict=kwargs["errors"] == "raise")


# --- RowLabelsOp ----------------------------------------------------------------

@physical_impl(of=RowLabelsOp, backend="pandas")
class PandasRowLabelsOp(RowLabelsOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        if self.keys is not None:
            return inputs[0].set_index(list(self.keys))
        if self.labels is not None:
            return inputs[0].set_axis(_resolve_args([self.labels], inputs)[0], axis=0)
        return inputs[0].reset_index(drop=True)


@physical_impl(of=RowLabelsOp, backend="polars")
class PolarsRowLabelsOp(RowLabelsOp, PhysicalOp):
    @classmethod
    def supports(cls, op: RowLabelsOp, ctx) -> bool:
        if op.keys is not None or op.labels is not None:
            op._unsupported_reason = "polars rows carry no labels to set"
            return False
        return True

    def process(self, mode: str, inputs: list):
        # polars rows carry no labels to drop, so there is nothing to do.
        return inputs[0]


# --- ShapeOp --------------------------------------------------------------------

@physical_impl(of=ShapeOp, backend="pandas")
class PandasShapeOp(ShapeOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        return inputs[0].shape


@physical_impl(of=ShapeOp, backend="polars")
class PolarsShapeOp(ShapeOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        # polars shapes match pandas': (n_rows, n_cols), or (n,) for a series.
        return inputs[0].shape


# --- ColumnNamesOp ----------------------------------------------------------------

class _ColumnNamesExec(ColumnNamesOp, PhysicalOp):
    @classmethod
    def output_frame_lib(cls, op, input_libs: list):
        # The names are a pandas Index on both backends, not a frame, so they
        # carry no frame library: a pandas consumer such as `.tolist()` must not
        # pin the producer to pandas, and a polars consumer must not get a
        # pandas -> polars conversion, which leaves an Index as it is.
        return None


@physical_impl(of=ColumnNamesOp, backend="pandas")
class PandasColumnNamesOp(_ColumnNamesExec):
    def process(self, mode: str, inputs: list):
        return inputs[0].columns


@physical_impl(of=ColumnNamesOp, backend="polars")
class PolarsColumnNamesOp(_ColumnNamesExec):
    def process(self, mode: str, inputs: list):
        # polars lists the names as plain strings; pandas callers expect an Index.
        return pd.Index(inputs[0].columns)
