"""Physical implementations of the projection operators.

Each logical projection op (``DropOp``, ``ColumnSelectorOp``,
``ColumnProjectionOp``) is a same-shape backend-variant family: the concrete
``Pandas*`` / ``Polars*`` impls subclass the logical op (+ :class:`PhysicalOp`)
and carry only the backend-specific ``process``. Selection swaps the logical op
to the matching impl in place, so ``isinstance(op, DropOp)`` etc. still holds
across the plan.
"""
from __future__ import annotations

from stratum.optimizer.logical._projection_ops import (
    ColumnProjectionOp, ColumnSelectorOp, DropOp, resolve_selector_columns)
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl


# --- DropOp -------------------------------------------------------------------

@physical_impl(of=DropOp, backend="pandas")
class PandasDropOp(DropOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        return _obj.drop(*_args, **_kwargs)


@physical_impl(of=DropOp, backend="polars")
class PolarsDropOp(DropOp, PhysicalOp):
    @classmethod
    def supports(cls, op: DropOp, ctx) -> bool:
        # Extraction only builds a DropOp for a column drop on a frame; row drops
        # are DropLabelsSelectionOp.
        unsupported = set(op.kwargs or {}) - {"columns", "labels", "axis", "errors"}
        if unsupported:
            op._unsupported_reason = f"drop options {sorted(unsupported)}"
            return False
        return True

    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        columns = _kwargs.get("columns", _kwargs.get("labels", _args[0] if _args else []))
        columns = [columns] if isinstance(columns, str) else list(columns)
        # pandas' errors="ignore" skips labels that are not there.
        return _obj.drop(columns, strict=_kwargs.get("errors", "raise") == "raise")


# --- ColumnSelectorOp (skb.select) --------------------------------------------

class ColumnSelectorExec(ColumnSelectorOp, PhysicalOp):
    """Physical base: resolve the (deferred) selector at fit time and reuse the
    stored column list at predict time -- backend-agnostic; only the final
    indexing differs per backend."""

    def _resolved_columns(self, frame, mode: str):
        if mode == "fit_transform":
            self.selected_columns = resolve_selector_columns(frame, self.selector)
        elif self.selected_columns is None:
            raise RuntimeError(
                f"{self} was asked to transform before the selector was resolved; "
                f"run fit_transform first.")
        return self.selected_columns


@physical_impl(of=ColumnSelectorOp, backend="pandas")
class PandasColumnSelectorOp(ColumnSelectorExec):
    def process(self, mode: str, inputs: list):
        frame = inputs[0]
        return frame[self._resolved_columns(frame, mode)]


@physical_impl(of=ColumnSelectorOp, backend="polars")
class PolarsColumnSelectorOp(ColumnSelectorExec):
    def process(self, mode: str, inputs: list):
        frame = inputs[0]
        return frame.select(self._resolved_columns(frame, mode))


# --- ColumnProjectionOp (df["a"] / df[["a", "b"]]) ----------------------------
# Both backends index by literal name(s) the same way: a str yields a column
# (Series), a list of str a sub-frame. No fit-time resolution (the columns are
# given verbatim), unlike the deferred-selector ColumnSelectorOp above.

@physical_impl(of=ColumnProjectionOp, backend="pandas")
class PandasColumnProjectionOp(ColumnProjectionOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        return inputs[0][self.key]


@physical_impl(of=ColumnProjectionOp, backend="polars")
class PolarsColumnProjectionOp(ColumnProjectionOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        return inputs[0][self.key]
