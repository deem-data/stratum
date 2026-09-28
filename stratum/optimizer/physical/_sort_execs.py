"""Physical implementations of ``SortOp`` (row ordering).

Same-shape backend-variant family: the concrete impls subclass the logical
``SortOp``.

Both impls pin a *stable* sort. That is not a tuning choice: with an unstable
sort the two backends are free to order tied keys differently, so a plan would
stop being reproducible across backends. Stability is how the result is pinned,
which is why it lives here and not as a field on the logical op.
"""
from __future__ import annotations

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
                  "kind": "stable"}
        if self.by:
            return obj.sort_values(by=list(self.by), **kwargs)
        # A series orders by its own values, which take no `by`.
        return obj.sort_values(**kwargs)


@physical_impl(of=SortOp, backend="polars")
class PolarsSortOp(SortOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        obj = inputs[0]
        descending = (not self.ascending if isinstance(self.ascending, bool)
                      else [not flag for flag in self.ascending])
        # polars puts nulls first by default, pandas last, so this is always
        # passed rather than left to the backend.
        kwargs = {"descending": descending,
                  "nulls_last": self.na_position == "last"}
        if self.by:
            return obj.sort(list(self.by), maintain_order=True, **kwargs)
        return obj.sort(**kwargs)
