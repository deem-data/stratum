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

from stratum.optimizer.logical._sort_ops import SortOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl
from stratum.runtime._frames import (polars_index_of, take_index, unwrap,
                                     with_row_positions)


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
    # A pandas Series (notably value_counts' pandas fallback) keeps its index on
    # pandas: its frame library is pandas, which this impl does not accept (#226).

    @classmethod
    def carries_index(cls, op: SortOp) -> bool:
        return True

    def process(self, mode: str, inputs: list):
        obj = inputs[0]
        if isinstance(obj, pl.Series) and not obj.dtype.is_float():
            descending, nulls_last = self._order()
            return obj.sort(descending=descending, nulls_last=nulls_last)
        frame = obj.to_frame() if isinstance(obj, pl.Series) else obj
        result = self._sort(frame, self._keys(frame))
        return result.to_series() if isinstance(obj, pl.Series) else result

    def process_indexed(self, mode: str, inputs: list):
        obj = unwrap(inputs[0])
        if not self._tracks_index:
            return self.process(mode, [obj]), None
        frame = obj.to_frame() if isinstance(obj, pl.Series) else obj
        # Stable, so tied rows keep their index's order too.
        result, rows = with_row_positions(
            frame, lambda numbered, _: self._sort(numbered, self._keys(frame)))
        result = result.to_series() if isinstance(obj, pl.Series) else result
        return result, take_index(polars_index_of(inputs[0]), rows)

    def _order(self):
        descending = (not self.ascending if isinstance(self.ascending, bool)
                      else [not flag for flag in self.ascending])
        # polars puts nulls first by default, pandas last, so this is always
        # passed rather than left to the backend.
        return descending, self.na_position == "last"

    def _keys(self, frame) -> list[str]:
        # A series orders by its own values.
        return list(self.by) or [frame.columns[0]]

    def _sort(self, frame, keys):
        descending, nulls_last = self._order()
        # pandas places NaN with the missing values; polars sorts NaN as the
        # largest float, i.e. first in a descending sort. Sort floats by a key
        # that turns NaN into null, which leaves the values themselves alone.
        return frame.sort([_nan_last_key(frame, key) for key in keys], descending=descending,
                          nulls_last=nulls_last, maintain_order=True)


def _nan_last_key(frame, name):
    column = pl.col(name)
    return column.fill_nan(None) if frame.schema[name].is_float() else column
