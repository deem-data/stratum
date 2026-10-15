"""Physical implementations of the metadata operators (``RenameOp``, ``ResetIndexOp``).

Each impl translates the op's backend-neutral fields into its library's call.
polars rows carry no index labels, so the polars impls refuse an index rename
and treat dropping the index as a no-op.
"""
from __future__ import annotations

from stratum.optimizer.logical._base import _resolve_kwargs
from stratum.optimizer.logical._metadata_ops import RenameOp, ResetIndexOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl


# --- RenameOp -------------------------------------------------------------------

@physical_impl(of=RenameOp, backend="pandas")
class PandasRenameOp(RenameOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
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
        return True

    def process(self, mode: str, inputs: list):
        kwargs = _resolve_kwargs({"columns": self.columns, "errors": self.errors}, inputs)
        return inputs[0].rename(kwargs["columns"], strict=kwargs["errors"] == "raise")


# --- ResetIndexOp ---------------------------------------------------------------

@physical_impl(of=ResetIndexOp, backend="pandas")
class PandasResetIndexOp(ResetIndexOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        return inputs[0].reset_index(drop=True)


@physical_impl(of=ResetIndexOp, backend="polars")
class PolarsResetIndexOp(ResetIndexOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        # polars rows carry no labels to drop, so there is nothing to do.
        return inputs[0]
