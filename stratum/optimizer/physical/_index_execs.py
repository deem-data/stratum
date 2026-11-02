"""Physical execution for index access and grouped-key extraction."""

from stratum.optimizer.logical._index_ops import GroupKeysOp, IndexAccessOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import pandas_impl, polars_impl
from stratum.runtime._frames import polars_index_of


@pandas_impl(of=IndexAccessOp)
class PandasIndexAccessOp(IndexAccessOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        return inputs[0].index


@pandas_impl(of=GroupKeysOp)
class PandasGroupKeysOp(GroupKeysOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        return inputs[0].index


@polars_impl(of=GroupKeysOp)
class PolarsGroupKeysOp(GroupKeysOp, PhysicalOp):
    @classmethod
    def supports(cls, op: GroupKeysOp, ctx) -> bool:
        # The key is a level of the aggregate's index, held as polars columns.
        if op.key in (cls.input_index_names(op, 0) or ()):
            return True
        op._unsupported_reason = "the key is not a level of an index held in polars columns"
        return False

    @classmethod
    def carries_index(cls, op: GroupKeysOp) -> bool:
        # pandas gives an Index, which has no index of its own.
        return True

    @classmethod
    def reads_polars_index(cls, op: GroupKeysOp) -> bool:
        return True

    def process(self, mode: str, inputs: list):
        raise RuntimeError(f"{self} reads its input's index: it runs process_indexed")

    def process_indexed(self, mode: str, inputs: list):
        return polars_index_of(inputs[0]).columns.get_column(self.key), None
