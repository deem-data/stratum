"""Physical implementations of ``SortOp`` (row ordering).

Same-shape backend-variant family: the concrete impls subclass the logical
``SortOp``.

Both impls pin a *stable* sort. That is not a tuning choice: with an unstable
sort the two backends are free to order tied keys differently, so a plan would
stop being reproducible across backends. Stability is how the result is pinned,
which is why it lives here and not as a field on the logical op.
"""
from __future__ import annotations

import polars as pl

from stratum.optimizer.logical._base import OutputType
from stratum.optimizer.logical._sort_ops import SortOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl


@physical_impl(of=SortOp, backend="pandas")
class PandasSortOp(SortOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        obj = inputs[0]
        ascending = (self.ascending if isinstance(self.ascending, bool)
                     else list(self.ascending))
        kwargs = {"ascending": ascending, "na_position": self.na_position,
                  "kind": "stable", "ignore_index": self.ignore_index}
        if self.by:
            return obj.sort_values(by=list(self.by), **kwargs)
        # A series orders by its own values, which take no `by`.
        return obj.sort_values(**kwargs)


@physical_impl(of=SortOp, backend="polars")
class PolarsSortOp(SortOp, PhysicalOp):
    @classmethod
    def supports(cls, op: SortOp, ctx) -> bool:
        # A pandas Series fallback (notably value_counts) keeps pandas's index.
        return not (op.output_type is OutputType.SERIES and op.inputs
                    and getattr(op.inputs[0], "_selected_backend", None) == "pandas")

    def process(self, mode: str, inputs: list):
        obj = inputs[0]
        descending = (not self.ascending if isinstance(self.ascending, bool)
                      else [not flag for flag in self.ascending])
        # polars puts nulls first by default, pandas last, so this is always
        # passed rather than left to the backend.
        kwargs = {"descending": descending,
                  "nulls_last": self.na_position == "last", "maintain_order": True}
        # pandas places NaN with the missing values; polars sorts NaN as the
        # largest float, i.e. first in a descending sort. Sort floats by a key
        # that turns NaN into null, which leaves the values themselves alone.
        if isinstance(obj, pl.Series):
            if not obj.dtype.is_float():
                return obj.sort(descending=descending, nulls_last=kwargs["nulls_last"])
            frame = obj.to_frame()
            return frame.sort(_nan_last_key(frame, frame.columns[0]), **kwargs).to_series()
        return obj.sort([_nan_last_key(obj, key) for key in self.by], **kwargs)


def _nan_last_key(frame, name):
    column = pl.col(name)
    return column.fill_nan(None) if frame.schema[name].is_float() else column
