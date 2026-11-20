"""Physical implementations of ``AggregateOp`` (grouped and whole-object).

The implementations subclass the logical ``AggregateOp``:

- Polars translates ``AggExpr`` nodes into expressions passed to
  ``group_by(...).agg(...)`` or ``select(...)``. A computed aggregation such as
  ``SUM(a * b)`` remains one expression tree.
- The pandas grouped path materializes each aggregation's row-wise child in a
  working frame, then reduces it by column.
"""
from __future__ import annotations

import pandas as pd
import polars as pl

from stratum.optimizer.logical._aggregation_ops import AggregateOp
from stratum.optimizer.logical._base import OutputType
from stratum.optimizer.logical._column_expr import AggExpr, AllCols, Col
from stratum.optimizer.physical._expressions import (
    EvalContext, refuse_unsupported_exprs, to_pandas, to_polars)
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl


@physical_impl(of=AggregateOp, backend="pandas")
class PandasAggregateOp(AggregateOp, PhysicalOp):
    def process(self, inputs: list):
        ctx = EvalContext.of(inputs, self.mode)
        if not self.grouped:
            return self._reduce_whole(ctx)
        return self._reduce_grouped(ctx)

    def _reduce_whole(self, ctx: EvalContext):
        """A bare reduction: ``df.sum()`` / ``series.mean()``."""
        if self._is_wildcard_only() or self.output_type is OutputType.SCALAR:
            _, agg = self.aggregations[0]
            return to_pandas(agg, ctx)
        if self.labelled():
            # A dict or labels on a Series: every entry reduces the same Series,
            # so it is computed once, and its name also names the result, as in
            # pandas.
            _, first = self.aggregations[0]
            series = to_pandas(first.child, ctx)
            on_series = EvalContext.of([series], self.mode)
            return pd.Series({name: to_pandas(AggExpr(agg.func, AllCols(), agg.params), on_series)
                              for name, agg in self.aggregations}, name=series.name)
        results = {self.entry_name(i): to_pandas(agg, ctx)
                   for i, (_, agg) in enumerate(self.aggregations)}
        if self.output_type is OutputType.SERIES:
            return pd.Series(results)
        return results

    def _reduce_grouped(self, ctx: EvalContext):
        keys = [to_pandas(expr, ctx) for expr in self.grouping]
        # A level-based grouping is carried by `keys`, so `level` is not passed.
        options = dict(sort=self.sort_keys, dropna=self.drop_null_keys,
                       observed=self.observed)
        if self._is_wildcard_only():
            # `.agg(func)` on the grouped frame keeps pandas' own column naming
            # and its exclusion of the grouping keys.
            _, agg = self.aggregations[0]
            grouped = ctx.frame.groupby(keys, as_index=not self.keys_as_columns,
                                        **options)
            return getattr(grouped, agg.func)(**agg.params)
        # Materialise each entry's row-wise child, then reduce column by column.
        columns: dict = {}
        for index, (_, agg) in enumerate(self.aggregations):
            columns.setdefault(agg.child, f"_child{index}")
        materialized = {child: to_pandas(child, ctx) for child in columns}
        work = pd.DataFrame({name: materialized[child]
                             for child, name in columns.items()})
        grouped = work.groupby(keys, as_index=True, **options)
        out = {}
        for index, (_, agg) in enumerate(self.aggregations):
            series = grouped[columns[agg.child]]
            reduced = getattr(series, agg.func)(**agg.params)
            if self.sort_categories and isinstance(reduced.index, pd.CategoricalIndex):
                reduced = reduced.sort_index()
            out[self.entry_name(index)] = reduced
        if self.output_type is OutputType.SERIES and len(out) == 1:
            # A single-column result keeps its name; `grouped[...]` carries the
            # internal working-frame name, so rename it back.
            name, series = next(iter(out.items()))
            explicit, agg = self.aggregations[0]
            if explicit is None and not isinstance(agg.child, Col):
                name = getattr(materialized[agg.child], "name", name)
            return series.rename(name)
        result = pd.DataFrame(out)
        if not self.keys_as_columns:
            return result
        # pandas omits a grouping key when a reduction already produces that
        # label, rather than creating duplicate output columns.
        levels = [i for i, name in enumerate(result.index.names)
                  if name not in result.columns]
        if levels:
            result = result.reset_index(level=levels)
        return result.reset_index(drop=True)

    def _is_wildcard_only(self) -> bool:
        """A single unnamed entry over every column, i.e. a plain ``.agg(func)``."""
        return (len(self.aggregations) == 1
                and self.aggregations[0][0] is None
                and isinstance(self.aggregations[0][1].child, AllCols))

# The reductions Polars evaluates as pandas does, per input dtype, as used by
# `_dtype_supports_polars_agg`. Outside them the backends disagree: a different
# value, an error on one side only, or a different error.
#
# Reductions that compare, pick or count values but do no arithmetic on them.
_POLARS_ORDERED_COUNT_FUNCS = frozenset(
    {"min", "max", "count", "nunique", "first", "last", "size"})
# A duration can also be summed and averaged.
_POLARS_DURATION_FUNCS = _POLARS_ORDERED_COUNT_FUNCS | {"sum", "mean", "median"}
# A categorical has no min/max: pandas orders it by the declared categories (and
# raises for an unordered one), Polars does not.
_POLARS_META_FUNCS = frozenset({"count", "nunique", "first", "last", "size"})
# Any other dtype (list, struct, binary, object): only counting rows agrees.
_POLARS_COUNT_ONLY_FUNCS = frozenset({"count", "size"})


def _dtype_supports_polars_agg(func: str, dtype: pl.DataType) -> bool:
    """Whether Polars' ``func`` over a ``dtype`` column gives pandas' result.

    ``False`` binds the pandas impl instead, which also lets pandas raise its own
    error for a reduction it rejects.
    """
    if dtype == pl.Unknown:
        # The schema could not resolve the dtype, so there is nothing to check.
        return True
    if isinstance(dtype, pl.Decimal):
        # Polars computes these over decimals where pandas raises or disagrees.
        return func not in {"quantile", "skew", "kurt"}
    if dtype.is_numeric():
        return True
    if dtype == pl.Boolean:
        # Both reject a boolean quantile, but with different errors.
        return func != "quantile"
    if dtype in (pl.String, pl.Date, pl.Datetime, pl.Time):
        # No arithmetic: pandas concatenates strings on `sum` where Polars
        # raises, and Polars returns null for a string `mean` where pandas raises.
        return func in _POLARS_ORDERED_COUNT_FUNCS
    if dtype == pl.Duration:
        return func in _POLARS_DURATION_FUNCS
    if dtype == pl.Null:
        # A column with no values to type: only counting is defined.
        return func in {"count", "nunique", "size"}
    if isinstance(dtype, (pl.Categorical, pl.Enum)):
        return func in _POLARS_META_FUNCS
    return func in _POLARS_COUNT_ONLY_FUNCS


def _normalize_float_nan(value):
    """``value`` with float NaN turned into null, since pandas skips NaN as missing."""
    if isinstance(value, pl.DataFrame):
        return value.with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None))
    if isinstance(value, pl.Series) and value.dtype in (pl.Float32, pl.Float64):
        return value.fill_nan(None)
    return value


@physical_impl(of=AggregateOp, backend="polars")
class PolarsAggregateOp(AggregateOp, PhysicalOp):

    @classmethod
    def supports(cls, op: AggregateOp, ctx) -> bool:
        """A grouped aggregation into a frame, or a single scalar reduction.

        The grouped path calls ``group_by`` on the source, which ``pl.Series``
        lacks, and always returns a frame. The ungrouped path runs ``select``,
        which returns a one-row frame: that is a scalar only for one entry over a
        series (``df["v"].sum()``, which :meth:`process` wraps into a frame) or
        over a computed expression (``(df["a"] * df["b"]).sum()``). A wildcard
        over a frame (``df.sum()``) is a series indexed by column name, which this
        path cannot build. Unsupported shapes use another implementation selected
        during planning.
        """
        if not op.inputs:
            return False
        if op.grouped:
            if op.output_type is not OutputType.FRAME:
                return False
            if op.inputs[0].output_type is OutputType.SERIES:
                return False
        else:
            if op.output_type is not OutputType.SCALAR or len(op.aggregations) != 1:
                return False
            if (isinstance(op.aggregations[0][1].child, AllCols)
                    and op.inputs[0].output_type is not OutputType.SERIES):
                return False
        # Polars has no pandas index labels or unobserved categorical groups.
        if op.observed is False or op.level is not None:
            return False
        if not refuse_unsupported_exprs(
                op, [*(agg for _, agg in op.aggregations), *op.grouping]):
            return False
        return cls._schema_supports(op)

    @classmethod
    def _schema_supports(cls, op: AggregateOp) -> bool:
        """Whether every entry's input dtypes reduce on Polars as on pandas.

        Read off the source's plan-time schema, per :func:`_dtype_supports_polars_agg`.
        A source without a Polars schema is not checked, and neither is an entry
        that reads a graph-fed operand, whose dtype is only known at run time.
        """
        if op.grouped and op.counts_group_rows():
            # Counting the rows per group reads no values.
            return True
        schema = op.inputs[0].output_schema
        if not isinstance(schema, pl.Schema):
            return True
        key_names = ({g.name for g in op.grouping if isinstance(g, Col)}
                     if op.grouped else set())
        for _, agg in op.aggregations:
            if agg.func == "size":
                continue
            if isinstance(agg.child, AllCols):
                # A wildcard reduces every column but the grouping keys.
                dtypes = [dt for col, dt in schema.items() if col not in key_names]
            elif isinstance(agg.child, Col):
                dtypes = [schema[agg.child.name]] if agg.child.name in schema else []
            elif not any(True for _ in agg.child.iter_operand_refs()):
                # A computed child over the source alone: resolve its dtype from
                # the schema, without data.
                try:
                    expr = to_polars(agg.child, EvalContext(None, [], "fit_transform"))
                    dtypes = pl.LazyFrame(schema=schema).select(expr).collect_schema().dtypes()
                except Exception as exc:  # noqa: BLE001 - schema mismatch falls back to pandas
                    op._unsupported_reason = str(exc)
                    return False
            else:
                dtypes = []
            for dtype in dtypes:
                if not _dtype_supports_polars_agg(agg.func, dtype):
                    op._unsupported_reason = (
                        f"polars {agg.func!r} semantics differ for {dtype}")
                    return False
        return True

    def process(self, inputs: list):
        # Polars selects over a DataFrame, so Series scalar sources wrap into
        # a 1-column DataFrame. Float NaNs become null to match pandas skipna.
        primary = inputs[0].to_frame() if isinstance(inputs[0], pl.Series) else inputs[0]
        frame = _normalize_float_nan(primary)
        leaf_inputs = [_normalize_float_nan(x) for x in inputs[1:]]
        ctx = EvalContext.of([frame, *leaf_inputs], self.mode)
        if self.grouped and self.counts_group_rows():
            # The rows per group, not a count per column; pandas names it "size".
            exprs = [pl.len().alias("size")]
        else:
            exprs = [self._entry_expr(index, agg, ctx)
                     for index, (_, agg) in enumerate(self.aggregations)]
        if not self.grouped:
            result = ctx.frame.select(exprs)
            if self.output_type is OutputType.SCALAR:
                value = result.item()
                # A numeric reduction over no values (empty or all null) is null
                # in Polars and NaN in pandas. pandas' NA for nullable and
                # Arrow-backed input is not reproduced.
                if value is None and (result.dtypes[0].is_numeric()
                                      or result.dtypes[0] == pl.Boolean):
                    return float("nan")
                return value
            return result
        keys = [to_polars(expr, ctx) for expr in self.grouping]
        # pandas sorts the group keys by default; polars neither sorts nor
        # preserves order unless asked, so both cases are made explicit.
        sort = self.sort_keys
        frame = ctx.frame
        if self.drop_null_keys:
            frame = frame.filter(pl.all_horizontal([key.is_not_null() for key in keys]))
        result = frame.group_by(keys, maintain_order=not sort).agg(exprs)
        if sort:
            result = result.sort(result.columns[:len(keys)], nulls_last=True)
        return result

    def _entry_expr(self, index: int, agg, ctx: EvalContext):
        expr = to_polars(agg, ctx)
        # A wildcard keeps one output per source column, so it must not be
        # collapsed under a single alias.
        if not isinstance(agg.child, AllCols):
            expr = expr.alias(self.entry_name(index))
        return expr
