"""Physical implementations of ``SelectionOp`` (relational row selection).

Same-shape backend-variant family. The MASK kind evaluates a backend-agnostic
``ColumnExpr`` predicate; method-based kinds (dropna, head, ...) map to a
per-backend method name.

The pandas backend has two mutually exclusive impls: ``PandasQuerySelectionOp``
routes an expressible MASK predicate through ``DataFrame.query()``, and
``PandasIndexSelectionOp`` handles everything else (boolean masking + the
method-based kinds). Which one runs is decided entirely at plan time by
``supports(op, ctx)`` -- gated on ``ctx.pandas_query`` and whether the predicate
is query-expressible -- so neither ``process`` carries selection control flow.
"""
from __future__ import annotations

from stratum.optimizer.logical._base import _resolve_args, _resolve_kwargs
from stratum.optimizer.logical._column_expr import EvalContext
from stratum.optimizer.logical._selection_ops import (
    SelectionKind, SelectionOp, _SELECTION_PANDAS_METHOD, _SELECTION_POLARS_METHOD)
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl


def _query_selectable(op: SelectionOp) -> bool:
    """Whether ``op`` can run through ``DataFrame.query()``.

    Only a MASK predicate that compiles to a query string qualifies; an
    ``OperandLeaf`` or ``.str`` accessor yields ``None`` from ``to_pandas_query``
    and must go through boolean masking instead.
    """
    return (op.kind is SelectionKind.MASK
            and op.predicate is not None
            and op.predicate.to_pandas_query({}) is not None)


@physical_impl(of=SelectionOp, backend="pandas")
class PandasQuerySelectionOp(SelectionOp, PhysicalOp):
    """MASK selection via ``DataFrame.query()`` (pandas fast path).

    Chosen only when ``ctx.pandas_query`` is set and the predicate is
    query-expressible, so ``process`` never has to fall back.
    """

    @classmethod
    def supports(cls, op: SelectionOp, ctx) -> bool:
        return ctx.pandas_query and _query_selectable(op)

    def on_impl_selected(self, ctx) -> None:
        # Compile the query string and bind its literals at plan time; supports()
        # guarantees the predicate is expressible, so this never yields None.
        params: dict = {}
        self.query = self.predicate.to_pandas_query(params)
        self.query_params = params

    def process(self, mode: str, inputs: list):
        return inputs[0].query(self.query, local_dict=self.query_params)


@physical_impl(of=SelectionOp, backend="pandas")
class PandasIndexSelectionOp(SelectionOp, PhysicalOp):
    """Boolean-mask indexing and method-based selections on pandas.

    Handles every pandas selection the query fast path doesn't take.
    """

    @classmethod
    def supports(cls, op: SelectionOp, ctx) -> bool:
        return not (ctx.pandas_query and _query_selectable(op))

    def process(self, mode: str, inputs: list):
        _obj = inputs[0]
        if self.kind is SelectionKind.MASK:
            ctx = EvalContext(frame=_obj, inputs=inputs, mode=mode)
            return _obj[self.predicate.to_pandas(ctx)]
        _args = _resolve_args(self.args, inputs) if self.args else []
        _kwargs = _resolve_kwargs(self.kwargs, inputs) if self.kwargs else {}
        method = _SELECTION_PANDAS_METHOD.get(self.kind)
        if method is None:
            raise NotImplementedError(
                f"SelectionOp.process is not implemented for kind {self.kind.name}.")
        return getattr(_obj, method)(*_args, **_kwargs)


@physical_impl(of=SelectionOp, backend="polars")
class PolarsSelectionOp(SelectionOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj = inputs[0]
        if self.kind is SelectionKind.MASK:
            ctx = EvalContext(frame=_obj, inputs=inputs, mode=mode)
            return _obj.filter(self.predicate.to_polars(ctx))
        _args = _resolve_args(self.args, inputs) if self.args else []
        _kwargs = _resolve_kwargs(self.kwargs, inputs) if self.kwargs else {}
        method = _SELECTION_POLARS_METHOD.get(self.kind)
        if method is None:
            raise NotImplementedError(
                f"SelectionOp.process is not implemented for kind {self.kind.name} "
                f"on the Polars backend.")
        return getattr(_obj, method)(*_args, **_kwargs)
