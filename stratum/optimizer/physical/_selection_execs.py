"""Physical implementations of the ``SelectionOp`` family (relational row selection).

Impls register on the concrete selection subclasses and translate their
backend-neutral fields into the library's own call; ``supports(op, ctx)`` only
decides whether the backend can express those field values. Field values may be
``OperandRef``s (graph-fed), resolved against ``inputs`` at execution.

``MaskSelectionOp`` has two mutually exclusive pandas impls:
``PandasQuerySelectionOp`` routes an expressible predicate through
``DataFrame.query()``, while ``PandasIndexSelectionOp`` performs boolean indexing.
``DropLabelsSelectionOp`` is pandas-only: polars rows carry no index labels.
"""

from __future__ import annotations

from numbers import Integral

import polars as pl

from stratum.optimizer.logical._base import OutputType, _resolve_kwargs, _resolve_operand
from stratum.optimizer.logical._selection_ops import (
    DropDuplicatesSelectionOp,
    DropLabelsSelectionOp,
    DropMissingSelectionOp,
    MaskSelectionOp,
    SampleSelectionOp,
    SliceSelectionOp,
)
from stratum.optimizer.physical._expressions import (
    EvalContext, refuse_unsupported_exprs, to_pandas, to_pandas_query,
    to_polars)
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl
from stratum.runtime._frames import (polars_index_of, row_numbers, take_index, unused_name,
                                     unwrap, with_row_positions)


# --- MaskSelectionOp ------------------------------------------------------------

def _query_selectable(op: MaskSelectionOp) -> bool:
    """Whether ``op`` can run through ``DataFrame.query()``.

    Only a predicate that compiles to a query string qualifies; an
    ``OperandLeaf`` or ``.str`` accessor yields ``None`` from ``to_pandas_query``
    and must go through boolean masking instead. ``query`` is also frame-only, so
    a masked series never takes this path; today that is implied (a predicate over
    a series always references the series itself, which compiles to an
    ``OperandLeaf``) but the check does not rely on it.
    """
    return (op.output_type is OutputType.FRAME
            and op.predicate is not None
            and to_pandas_query(op.predicate, {}) is not None)


@physical_impl(of=MaskSelectionOp, backend="pandas")
class PandasQuerySelectionOp(MaskSelectionOp, PhysicalOp):
    """Mask selection via ``DataFrame.query()`` (pandas fast path).

    Chosen only when ``ctx.pandas_query`` is set and the predicate is
    query-expressible, so ``process`` never has to fall back.
    """

    @classmethod
    def supports(cls, op: MaskSelectionOp, ctx) -> bool:
        return ctx.pandas_query and _query_selectable(op)

    def on_impl_selected(self, ctx) -> None:
        # Compile the query string and bind its literals at plan time; supports()
        # guarantees the predicate is expressible, so this never yields None.
        params: dict = {}
        self.query = to_pandas_query(self.predicate, params)
        self.query_params = params

    def process(self, mode: str, inputs: list):
        return inputs[0].query(self.query, local_dict=self.query_params)


@physical_impl(of=MaskSelectionOp, backend="pandas")
class PandasIndexSelectionOp(MaskSelectionOp, PhysicalOp):
    """Boolean-mask indexing when the query fast path does not apply."""

    @classmethod
    def supports(cls, op: MaskSelectionOp, ctx) -> bool:
        return not (ctx.pandas_query and _query_selectable(op))

    def process(self, mode: str, inputs: list):
        return inputs[0][to_pandas(self.predicate, EvalContext.of(inputs, mode))]


@physical_impl(of=MaskSelectionOp, backend="polars")
class PolarsMaskSelectionOp(MaskSelectionOp, PhysicalOp):
    @classmethod
    def supports(cls, op: MaskSelectionOp, ctx) -> bool:
        # A pandas Series's labelled values stay on pandas: its frame library is
        # pandas, which this impl does not accept (#226).
        return op.predicate is None or refuse_unsupported_exprs(op, [op.predicate])

    @classmethod
    def carries_index(cls, op: MaskSelectionOp) -> bool:
        return True

    def process(self, mode: str, inputs: list):
        predicate = to_polars(self.predicate, EvalContext.of(inputs, mode))
        if isinstance(inputs[0], pl.DataFrame):
            return inputs[0].filter(predicate)
        return inputs[0].filter(_mask(inputs[0], predicate))

    def process_indexed(self, mode: str, inputs: list):
        plain = [unwrap(value) for value in inputs]
        if not self._tracks_index:
            return self.process(mode, plain), None
        mask = _mask(plain[0], to_polars(self.predicate, EvalContext.of(plain, mode)))
        return _filter_indexed(self, inputs, mask)


def _filter_indexed(op, inputs: list, mask: pl.Series):
    """``process_indexed`` of a selection that keeps the rows ``mask`` holds."""
    data = unwrap(inputs[0])
    if not op._tracks_index:
        return data.filter(mask), None
    rows = row_numbers(len(data)).filter(mask)
    return data.filter(mask), take_index(polars_index_of(inputs[0]), rows)


def _mask(frame, predicate) -> pl.Series:
    """``predicate`` over ``frame``'s rows, as ``filter`` reads it: a missing
    value keeps no row. A series' expression reads it as a one-column frame."""
    if not isinstance(predicate, pl.Series):
        table = frame.to_frame() if isinstance(frame, pl.Series) else frame
        name = unused_name(table.columns)
        predicate = table.with_columns(predicate.alias(name)).get_column(name)
    return predicate.fill_null(False)


# --- SliceSelectionOp -----------------------------------------------------------

def _slice(op: SliceSelectionOp, inputs: list):
    """``head``/``tail`` mean the same on pandas and polars, negative ``n`` included."""
    n = _resolve_operand(op.n, inputs)
    return inputs[0].tail(n) if op.from_end else inputs[0].head(n)


@physical_impl(of=SliceSelectionOp, backend="pandas")
class PandasSliceSelectionOp(SliceSelectionOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        return _slice(self, inputs)


@physical_impl(of=SliceSelectionOp, backend="polars")
class PolarsSliceSelectionOp(SliceSelectionOp, PhysicalOp):
    @classmethod
    def carries_index(cls, op: SliceSelectionOp) -> bool:
        return True

    def process(self, mode: str, inputs: list):
        return _slice(self, inputs)

    def process_indexed(self, mode: str, inputs: list):
        plain = [unwrap(value) for value in inputs]
        if not self._tracks_index:
            return _slice(self, plain), None
        rows = _slice(self, [row_numbers(len(plain[0])), *plain[1:]])
        return _slice(self, plain), take_index(polars_index_of(inputs[0]), rows)


# --- SampleSelectionOp ----------------------------------------------------------

@physical_impl(of=SampleSelectionOp, backend="pandas")
class PandasSampleSelectionOp(SampleSelectionOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        kwargs = _resolve_kwargs({
            "n": self.n, "frac": self.frac, "replace": self.with_replacement,
            "weights": self.weights, "random_state": self.seed,
            "ignore_index": self.ignore_index,
        }, inputs)
        return inputs[0].sample(**kwargs)


def get_polars_sample_kwargs(op: SampleSelectionOp, inputs: list) -> dict:
    """``DataFrame.sample`` kwargs for ``op``'s fields, resolved against ``inputs``."""
    n, frac, with_replacement, seed = (
        _resolve_operand(v, inputs) for v in (op.n, op.frac, op.with_replacement, op.seed))
    if n is not None and frac is not None:
        raise ValueError("PolarsSampleSelectionOp: provide n or frac exclusively, not both")
    # Sampled rows come back in random order; polars keeps input order unless
    # asked to shuffle.
    kwargs = dict(shuffle=True, with_replacement=with_replacement)
    if n is not None:
        kwargs["n"] = n
    if frac is not None:
        # Round the count like pandas does, rather than polars' own fraction.
        kwargs["n"] = round(frac * len(inputs[0]))
    if seed is not None:
        kwargs["seed"] = seed
    return kwargs


@physical_impl(of=SampleSelectionOp, backend="polars")
class PolarsSampleSelectionOp(SampleSelectionOp, PhysicalOp):
    """Unweighted samples with an integer (or no) seed."""

    @classmethod
    def supports(cls, op: SampleSelectionOp, ctx) -> bool:
        return op.weights is None and (op.seed is None or isinstance(op.seed, Integral))

    def process(self, mode: str, inputs: list):
        return inputs[0].sample(**get_polars_sample_kwargs(self, inputs))


# --- DropDuplicatesSelectionOp --------------------------------------------------

_PANDAS_KEEP = {"first": "first", "last": "last", "none": False}


@physical_impl(of=DropDuplicatesSelectionOp, backend="pandas")
class PandasDropDuplicatesSelectionOp(DropDuplicatesSelectionOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        kwargs = {"keep": _PANDAS_KEEP[self.keep], "ignore_index": self.ignore_index}
        if self.subset is not None:  # a Series takes no subset
            kwargs["subset"] = self.subset
        return inputs[0].drop_duplicates(**_resolve_kwargs(kwargs, inputs))


@physical_impl(of=DropDuplicatesSelectionOp, backend="polars")
class PolarsDropDuplicatesSelectionOp(DropDuplicatesSelectionOp, PhysicalOp):
    @classmethod
    def supports(cls, op: DropDuplicatesSelectionOp, ctx) -> bool:
        return op.output_type is OutputType.FRAME

    @classmethod
    def carries_index(cls, op: DropDuplicatesSelectionOp) -> bool:
        return True

    def process(self, mode: str, inputs: list):
        subset = _resolve_operand(self.subset, inputs)
        return inputs[0].unique(subset=subset, keep=self.keep, maintain_order=True)

    def process_indexed(self, mode: str, inputs: list):
        plain = [unwrap(value) for value in inputs]
        if not self._tracks_index:
            return self.process(mode, plain), None
        frame = plain[0]
        # The rows compare on the frame's own columns, not their positions.
        subset = _resolve_operand(self.subset, plain)
        subset = frame.columns if subset is None else subset
        result, rows = with_row_positions(frame, lambda numbered, _: numbered.unique(
            subset=subset, keep=self.keep, maintain_order=True))
        return result, take_index(polars_index_of(inputs[0]), rows)


@physical_impl(of=DropDuplicatesSelectionOp, backend="polars")
class PolarsSeriesDropDuplicatesSelectionOp(DropDuplicatesSelectionOp, PhysicalOp):
    _KEEP_MASK = {"first": "is_first_distinct", "last": "is_last_distinct", "none": "is_unique"}

    @classmethod
    def supports(cls, op: DropDuplicatesSelectionOp, ctx) -> bool:
        return op.output_type is OutputType.SERIES

    @classmethod
    def carries_index(cls, op: DropDuplicatesSelectionOp) -> bool:
        return True

    def process(self, mode: str, inputs: list):
        return inputs[0].filter(self._mask(inputs[0]))

    def process_indexed(self, mode: str, inputs: list):
        return _filter_indexed(self, inputs, self._mask(unwrap(inputs[0])))

    def _mask(self, series):
        return getattr(series, self._KEEP_MASK[self.keep])()


# --- DropMissingSelectionOp -----------------------------------------------------

@physical_impl(of=DropMissingSelectionOp, backend="pandas")
class PandasDropMissingSelectionOp(DropMissingSelectionOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        kwargs = {"ignore_index": self.ignore_index}
        if self.subset is not None:  # a Series takes neither subset nor thresh
            kwargs["subset"] = self.subset
        if self.min_present is not None:
            kwargs["thresh"] = self.min_present
        return inputs[0].dropna(**_resolve_kwargs(kwargs, inputs))


def _present_count_predicate(subset, min_present):
    """Row predicate: enough of the ``subset`` columns hold a value.

    NaN counts as missing in float columns, as it does in pandas.
    """
    from polars import selectors as cs

    selected = cs.all() if subset is None else cs.by_name(subset)
    present = ((selected & cs.float()).fill_nan(None).is_not_null(),
               (selected - cs.float()).is_not_null())
    if min_present is None:
        return pl.all_horizontal(*present)
    return pl.sum_horizontal(*(expr.cast(pl.UInt32) for expr in present)) >= min_present


@physical_impl(of=DropMissingSelectionOp, backend="polars")
class PolarsDropMissingSelectionOp(DropMissingSelectionOp, PhysicalOp):
    @classmethod
    def supports(cls, op: DropMissingSelectionOp, ctx) -> bool:
        return op.output_type is OutputType.FRAME

    @classmethod
    def carries_index(cls, op: DropMissingSelectionOp) -> bool:
        return True

    def process(self, mode: str, inputs: list):
        return inputs[0].filter(self._predicate(inputs))

    def process_indexed(self, mode: str, inputs: list):
        plain = [unwrap(value) for value in inputs]
        return _filter_indexed(self, inputs, _mask(plain[0], self._predicate(plain)))

    def _predicate(self, inputs):
        subset, min_present = (_resolve_operand(v, inputs)
                               for v in (self.subset, self.min_present))
        return _present_count_predicate(subset, min_present)


@physical_impl(of=DropMissingSelectionOp, backend="polars")
class PolarsSeriesDropMissingSelectionOp(DropMissingSelectionOp, PhysicalOp):
    @classmethod
    def supports(cls, op: DropMissingSelectionOp, ctx) -> bool:
        return op.output_type is OutputType.SERIES

    @classmethod
    def carries_index(cls, op: DropMissingSelectionOp) -> bool:
        return True

    def process(self, mode: str, inputs: list):
        return inputs[0].drop_nulls().drop_nans()

    def process_indexed(self, mode: str, inputs: list):
        series = unwrap(inputs[0])
        missing = series.is_null()
        if series.dtype.is_float():
            missing = missing | series.is_nan()
        return _filter_indexed(self, inputs, ~missing.fill_null(True))


# --- DropLabelsSelectionOp ------------------------------------------------------

@physical_impl(of=DropLabelsSelectionOp, backend="pandas")
class PandasDropLabelsSelectionOp(DropLabelsSelectionOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        kwargs = _resolve_kwargs(
            {"index": self.labels, "level": self.level, "errors": self.errors}, inputs)
        return inputs[0].drop(**kwargs)
