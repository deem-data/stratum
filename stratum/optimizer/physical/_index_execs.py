"""Physical execution for index access and grouped-key extraction."""

from stratum.optimizer.logical._index_ops import GroupKeysOp, IndexAccessOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import pandas_impl, polars_impl


@pandas_impl(of=IndexAccessOp)
class PandasIndexAccessOp(IndexAccessOp, PhysicalOp):
    @classmethod
    def supports(cls, op: IndexAccessOp, ctx) -> bool:
        return (not op.inputs or
                getattr(op.inputs[0], "_selected_backend", None) != "polars")

    def process(self, mode: str, inputs: list):
        return inputs[0].index


@pandas_impl(of=GroupKeysOp)
class PandasGroupKeysOp(GroupKeysOp, PhysicalOp):
    @classmethod
    def supports(cls, op: GroupKeysOp, ctx) -> bool:
        return (not op.inputs or
                getattr(op.inputs[0], "_selected_backend", None) != "polars")

    def process(self, mode: str, inputs: list):
        return inputs[0].index


@polars_impl(of=GroupKeysOp)
class PolarsGroupKeysOp(GroupKeysOp, PhysicalOp):
    @classmethod
    def supports(cls, op: GroupKeysOp, ctx) -> bool:
        return (not op.inputs or
                getattr(op.inputs[0], "_selected_backend", None) == "polars")

    def process(self, mode: str, inputs: list):
        return inputs[0].get_column(self.key)
