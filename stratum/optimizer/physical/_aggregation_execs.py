"""Physical implementations of ``AggregateOp`` (fused groupby-aggregate).

Same-shape backend-variant family: the concrete impls subclass the logical
``AggregateOp``. Only pandas is implemented; the polars impl raises (as the
logical op did under ``force_polars``) until a polars backend lands.
"""
from __future__ import annotations

from stratum.optimizer.logical._aggregation_ops import AggregateOp
from stratum.optimizer.logical._base import OperandRef
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl


class AggregateExec(AggregateOp, PhysicalOp):
    """Physical base: resolves grouping/aggregation specs copied from the logical op."""

    def _resolve(self, inputs: list):
        obj = inputs[0]
        grouping = (inputs[self.grouping_attributes.k]
                    if isinstance(self.grouping_attributes, OperandRef)
                    else self.grouping_attributes)
        aggregations = (inputs[self.aggregations.k]
                        if isinstance(self.aggregations, OperandRef)
                        else self.aggregations)
        return obj, grouping, aggregations


@physical_impl(of=AggregateOp, backend="pandas")
class PandasAggregateOp(AggregateExec):
    def process(self, mode: str, inputs: list):
        obj, grouping, aggregations = self._resolve(inputs)
        return obj.groupby(grouping, **self.groupby_kwargs).agg(aggregations)


@physical_impl(of=AggregateOp, backend="polars")
class PolarsAggregateOp(AggregateExec):
    def process(self, mode: str, inputs: list):
        raise NotImplementedError("AggregateOp Polars backend is not implemented yet.")
