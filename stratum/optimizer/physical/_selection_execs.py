"""Physical implementations of ``SelectionOp`` (relational row selection).

Same-shape backend-variant family. The MASK kind evaluates a backend-agnostic
``ColumnExpr`` predicate; method-based kinds (dropna, head, ...) map to a
per-backend method name.

The pandas MASK implementations are mutually exclusive:
``PandasQuerySelectionOp`` routes an expressible predicate through
``DataFrame.query()``, while ``PandasIndexSelectionOp`` performs boolean
indexing. ``PandasMethodBasedSelectionOp`` handles every pandas method-based kind,
while Polars uses dedicated implementations with narrower feasibility checks.
Every choice is made at plan time by ``supports(op, ctx)``.
"""

from __future__ import annotations

from numbers import Integral

from stratum.optimizer.ir._base import _resolve_args, _resolve_kwargs
from stratum.optimizer.ir._column_expr import EvalContext
from stratum.optimizer.ir._selection_ops import (
    _SELECTION_PANDAS_METHOD,
    _SELECTION_POLARS_METHOD,
    SelectionKind,
    SelectionOp,
)
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl


def _query_selectable(op: SelectionOp) -> bool:
    """Whether ``op`` can run through ``DataFrame.query()``.

    Only a MASK predicate that compiles to a query string qualifies; an
    ``OperandLeaf`` or ``.str`` accessor yields ``None`` from ``to_pandas_query``
    and must go through boolean masking instead.
    """
    return (
        op.kind is SelectionKind.MASK
        and op.predicate is not None
        and op.predicate.to_pandas_query({}) is not None
    )


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
    """Boolean-mask indexing when the query fast path does not apply."""

    @classmethod
    def supports(cls, op: SelectionOp, ctx) -> bool:
        return op.kind is SelectionKind.MASK and not (ctx.pandas_query and _query_selectable(op))

    def process(self, mode: str, inputs: list):
        _obj = inputs[0]
        ctx = EvalContext(frame=_obj, inputs=inputs, mode=mode)
        return _obj[self.predicate.to_pandas(ctx)]


@physical_impl(of=SelectionOp, backend="pandas")
class PandasMethodBasedSelectionOp(SelectionOp, PhysicalOp):
    @classmethod
    def supports(cls, op: SelectionOp, ctx) -> bool:
        return op.kind in _SELECTION_PANDAS_METHOD

    def process(self, mode: str, inputs: list):
        _obj = inputs[0]
        _args = _resolve_args(self.args, inputs) if self.args else []
        _kwargs = _resolve_kwargs(self.kwargs, inputs) if self.kwargs else {}
        method_name = _SELECTION_PANDAS_METHOD.get(self.kind)
        if method_name is None:
            raise NotImplementedError(f"SelectionOp.process is not implemented for kind {self.kind.name}.")
        return getattr(_obj, method_name)(*_args, **_kwargs)


@physical_impl(of=SelectionOp, backend="polars")
class PolarsMaskSelectionOp(SelectionOp, PhysicalOp):
    @classmethod
    def supports(cls, op: SelectionOp, ctx) -> bool:
        return op.kind is SelectionKind.MASK

    def process(self, mode: str, inputs: list):
        _obj = inputs[0]
        ctx = EvalContext(frame=_obj, inputs=inputs, mode=mode)
        return _obj.filter(self.predicate.to_polars(ctx))


def get_param(op: SelectionOp, pos: int, key: str):
    if op.args and len(op.args) > pos:
        return op.args[pos]
    if op.kwargs and key in op.kwargs:
        return op.kwargs[key]
    return None


def get_polars_sample_kwargs(op: SelectionOp, inputs: list):
    n = get_param(op, 0, "n")
    fraction = get_param(op, 1, "frac")
    with_replacement = get_param(op, 2, "replace")
    seed = get_param(op, 4, "random_state")

    n, fraction = _resolve_args((n, fraction), inputs)
    if None not in (n, fraction):
        raise ValueError("PolarsSampleSelectionOp: provide n or frac exclusively, not both")

    kwargs = dict()
    if n is not None:
        kwargs["n"] = n

    if fraction is not None:
        kwargs["n"] = round(fraction * len(inputs[0]))

    if with_replacement is not None:
        kwargs["with_replacement"] = _resolve_args((with_replacement,), inputs)[0]

    if seed is not None:
        kwargs["seed"] = _resolve_args((seed,), inputs)[0]

    return kwargs


@physical_impl(of=SelectionOp, backend="polars")
class PolarsSampleSelectionOp(SelectionOp, PhysicalOp):
    """Reject any configuration that cannot be mirrored with Polars."""

    @classmethod
    def supports(cls, op: SelectionOp, ctx) -> bool:
        random_state = get_param(op, 4, "random_state")
        weights = get_param(op, 3, "weights")
        axis = get_param(op, 5, "axis")
        return op.kind is SelectionKind.SAMPLE and weights is None and (isinstance(random_state, Integral) or random_state is None) and axis in (None, 0, "index", "rows")

    def process(self, mode: str, inputs: list):
        _obj = inputs[0]
        method_name = _SELECTION_POLARS_METHOD.get(self.kind)
        kwargs = get_polars_sample_kwargs(self, inputs)
        return getattr(_obj, method_name)(**kwargs)


@physical_impl(of=SelectionOp, backend="polars")
class PolarsSliceSelectionOp(SelectionOp, PhysicalOp):
    @classmethod
    def supports(cls, op: SelectionOp, ctx) -> bool:
        return op.kind in (SelectionKind.HEAD, SelectionKind.TAIL)

    def process(self, mode: str, inputs: list):
        n = get_param(self, 0, "n")
        _obj = inputs[0]

        method_name = _SELECTION_POLARS_METHOD[self.kind]
        if n is not None:
            n = _resolve_args((n,), inputs)[0]
            return getattr(_obj, method_name)(n)

        return getattr(_obj, method_name)()


def get_polars_unique_kwargs(op: SelectionOp, inputs: list):
    subset = get_param(op, 0, "subset")
    keep = get_param(op, 1, "keep")
    kwargs = dict()

    if subset is not None:
        subset = _resolve_args((subset,), inputs)[0]
        kwargs["subset"] = subset
    if keep is not None:
        keep = _resolve_args((keep,), inputs)[0]
    if keep is False:
        keep = "none"
    elif keep is None:
        keep = "first"

    kwargs["keep"] = keep
    kwargs["maintain_order"] = True

    return kwargs


@physical_impl(of=SelectionOp, backend="polars")
class PolarsDropDuplicatesSelectionOp(SelectionOp, PhysicalOp):
    @classmethod
    def supports(cls, op: SelectionOp, ctx) -> bool:
        inplace = get_param(op, 2, "inplace")
        return op.kind is SelectionKind.DROP_DUPLICATES and (inplace is None or inplace is False)

    def process(self, mode: str, inputs: list):
        _obj = inputs[0]
        method_name = _SELECTION_POLARS_METHOD[self.kind]
        kwargs = get_polars_unique_kwargs(self, inputs)

        return getattr(_obj, method_name)(**kwargs)


def _resolve_dropna_kwargs(op: SelectionOp, inputs: list):
    return _resolve_kwargs(op.kwargs, inputs) if op.kwargs else {}


def _dropna_predicate(kwargs: dict):
    import polars as pl
    from polars import selectors as cs

    if "how" in kwargs and "thresh" in kwargs:
        raise TypeError("You cannot set both the how and thresh arguments at the same time.")

    subset = kwargs.get("subset")
    selected = cs.all() if subset is None else cs.by_name(subset)
    present = ((selected & cs.float()).fill_nan(None).is_not_null(), (selected - cs.float()).is_not_null())

    if "thresh" in kwargs:
        return pl.sum_horizontal(*(expr.cast(pl.UInt32) for expr in present)) >= kwargs["thresh"]

    how = kwargs.get("how", "any")
    if how == "any":
        return pl.all_horizontal(*present)
    if how == "all":
        return pl.any_horizontal(*present)
    raise ValueError(f"invalid how option: {how}")


@physical_impl(of=SelectionOp, backend="polars")
class PolarsDropnaSelectionOp(SelectionOp, PhysicalOp):
    @classmethod
    def supports(cls, op: SelectionOp, ctx) -> bool:
        kwargs = op.kwargs or {}
        axis = kwargs.get("axis")
        inplace = kwargs.get("inplace")
        return op.kind is SelectionKind.DROPNA and not op.args and axis in (None, 0, "index", "rows") and inplace in (None, False)

    def process(self, mode: str, inputs: list):
        _obj = inputs[0]
        kwargs = _resolve_dropna_kwargs(self, inputs)
        return _obj.filter(_dropna_predicate(kwargs))
