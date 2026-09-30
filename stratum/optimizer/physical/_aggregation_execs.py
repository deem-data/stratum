"""Physical implementations of ``AggregateOp`` (grouped and whole-object).

Same-shape backend-variant family: the concrete impls subclass the logical
``AggregateOp``.

The two backends split the work differently, which is the point of keeping the
logical op expression-based. Polars consumes an ``AggExpr`` directly -- one
expression list handed to ``group_by(...).agg(...)`` or ``select(...)``, so a
computed aggregation like ``SUM(a * b)`` stays a single kernel. Pandas has no
expression language, so the grouped path materialises each entry's row-wise child
into a working frame first and then reduces it per column.
"""
from __future__ import annotations

import pandas as pd
import polars as pl

from stratum.optimizer.logical._aggregation_ops import AggregateOp
from stratum.optimizer.logical._base import OutputType
from stratum.optimizer.logical._column_expr import AllCols, Col, EvalContext
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl


class AggregateExec(AggregateOp, PhysicalOp):
    """Physical base: shares the evaluation context and the output naming rule."""

    def _ctx(self, inputs: list, mode: str) -> EvalContext:
        return EvalContext(frame=inputs[0], inputs=inputs, mode=mode)


@physical_impl(of=AggregateOp, backend="pandas")
class PandasAggregateOp(AggregateExec):
    def process(self, mode: str, inputs: list):
        # A Series aggregate has no Polars implementation yet. The greedy
        # selector may therefore bind this pandas fallback after a Polars
        # projection; normalise that one input at the backend boundary.
        if isinstance(inputs[0], pl.Series):
            inputs = [inputs[0].to_pandas(), *inputs[1:]]
        ctx = self._ctx(inputs, mode)
        if not self.grouped:
            return self._reduce_whole(ctx)
        return self._reduce_grouped(ctx)

    def _reduce_whole(self, ctx: EvalContext):
        """A bare reduction: ``df.sum()`` / ``series.mean()``."""
        if self._is_wildcard_only():
            _, agg = self.aggregations[0]
            return agg.to_pandas(ctx)
        results = {self.entry_name(i): agg.to_pandas(ctx)
                   for i, (_, agg) in enumerate(self.aggregations)}
        if self.output_type is OutputType.SERIES:
            return pd.Series(results)
        return results

    def _reduce_grouped(self, ctx: EvalContext):
        keys = [expr.to_pandas(ctx) for expr in self.grouping]
        options = dict(self.options)
        sort_categories = options.pop("sort_categories", False)
        options.pop("level", None)  # a level-based grouping is carried by `keys`
        if self._is_wildcard_only():
            # `.agg(func)` on the grouped frame keeps pandas' own column naming
            # and its exclusion of the grouping keys.
            _, agg = self.aggregations[0]
            grouped = ctx.frame.groupby(keys, **options)
            return getattr(grouped, agg.func)(**agg.params)
        # Materialise each entry's row-wise child, then reduce column by column.
        columns: dict = {}
        for index, (_, agg) in enumerate(self.aggregations):
            columns.setdefault(agg.child, f"_child{index}")
        materialized = {child: child.to_pandas(ctx) for child in columns}
        work = pd.DataFrame({name: materialized[child]
                             for child, name in columns.items()})
        as_index = options.pop("as_index", True)
        grouped = work.groupby(keys, as_index=True, **options)
        out = {}
        for index, (_, agg) in enumerate(self.aggregations):
            series = grouped[columns[agg.child]]
            reduced = getattr(series, agg.func)(**agg.params)
            if sort_categories and isinstance(reduced.index, pd.CategoricalIndex):
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
        if as_index:
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

@physical_impl(of=AggregateOp, backend="polars")
class PolarsAggregateOp(AggregateExec):

    @classmethod
    def supports(cls, op: AggregateOp, ctx) -> bool:
        """Refuse a series source: polars has no expression context for one.

        Both paths below start from ``select``/``group_by`` on the source, and a
        ``pl.Series`` has neither. Reducing it directly would return a Python
        value rather than an expression, so the aggregation would silently stop
        being one kernel. Refusing here turns that into a plan-time decision.
        """
        if op.output_type is not OutputType.FRAME:
            return False
        if op.inputs and op.inputs[0].output_type is OutputType.SERIES:
            return False
        # Polars has no pandas index labels or unobserved categorical groups.
        if op.options.get("observed") is False or op.options.get("level") is not None:
            return False
        return all(agg.supports_polars() for _, agg in op.aggregations)

    def process(self, mode: str, inputs: list):
        # pandas treats floating NaN as missing for grouping and reductions.
        frame = inputs[0].with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None))
        ctx = self._ctx([frame, *inputs[1:]], mode)
        exprs = []
        for index, (name, agg) in enumerate(self.aggregations):
            expr = agg.to_polars(ctx)
            # A wildcard keeps one output per source column, so it must not be
            # collapsed under a single alias.
            if not isinstance(agg.child, AllCols):
                expr = expr.alias(self.entry_name(index))
            exprs.append(expr)
        if not self.grouped:
            return ctx.frame.select(exprs)
        keys = [expr.to_polars(ctx) for expr in self.grouping]
        # pandas sorts the group keys by default; polars neither sorts nor
        # preserves order unless asked, so both cases are made explicit.
        sort = self.options.get("sort", True)
        frame = ctx.frame
        if self.options.get("dropna", True):
            frame = frame.filter(pl.all_horizontal([key.is_not_null() for key in keys]))
        result = frame.group_by(keys, maintain_order=not sort).agg(exprs)
        if sort:
            result = result.sort(result.columns[:len(keys)], nulls_last=True)
        return result
