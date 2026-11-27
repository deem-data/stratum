"""Physical implementations of ``AggregateOp`` (grouped and whole-object).

The implementations subclass the logical ``AggregateOp``:

- Polars translates ``AggExpr`` nodes into expressions passed to
  ``group_by(...).agg(...)`` or ``select(...)``. A computed aggregation such as
  ``SUM(a * b)`` remains one expression tree.
- A scalar aggregate over a numeric Series reduces its input directly in Polars.
- The pandas grouped path materializes each aggregation's row-wise child in a
  working frame, then reduces it by column.
"""
from __future__ import annotations

import operator

import numpy as np
import pandas as pd
import polars as pl

from stratum.optimizer.logical._aggregation_ops import AggregateOp
from stratum.optimizer.logical._base import OutputType
from stratum.optimizer.logical._column_expr import AllCols, BinOpExpr, Col, OperandLeaf
from stratum.optimizer.logical._map_ops import ExpressionMapOp
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp
from stratum.optimizer.logical._selection_ops import SelectionOp
from stratum.optimizer.logical._source_ops import InMemoryFrameOp
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
        if self._is_wildcard_only():
            _, agg = self.aggregations[0]
            return to_pandas(agg, ctx)
        if isinstance(ctx.frame, pd.Series) and self.labelled():
            return ctx.frame.agg({name: agg.func for name, agg in self.aggregations})
        if (self.output_type is OutputType.SERIES and self.aggregations
                and all(name is None and isinstance(agg.child, Col) and not agg.params
                        for name, agg in self.aggregations)):
            # pandas resolves dictionary keys against the column index, including duplicates.
            return ctx.frame.agg({agg.child.name: agg.func for _, agg in self.aggregations})
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


def _projected_dtype(op):
    """Supported pandas dtype of a projected column, including after row selections."""
    if (not isinstance(op, ColumnProjectionOp) or op.output_type is not OutputType.SERIES
            or not op.inputs):
        return None
    source = op.inputs[0]
    while isinstance(source, SelectionOp) and source.inputs:
        source = source.inputs[0]
    if not isinstance(source, InMemoryFrameOp) or not isinstance(source.data, pd.DataFrame):
        return None
    columns = source.data.columns
    if columns.nlevels != 1 or not columns.is_unique or op.key not in columns:
        return None
    # The logical schema collapses NumPy, Arrow and pandas nullable dtypes.
    # Their scalar types and their handling of valid NaNs differ.
    dtype = source.data[op.key].dtype
    width = getattr(dtype, "itemsize", None)
    if (dtype.kind in "iu" and width in (1, 2, 4, 8)) or (dtype.kind == "f" and width == 8):
        return dtype
    return None


def _scalar_input_dtype(source):
    """Infer the input dtype used to select the accumulator and result scalar types."""
    if isinstance(source, ExpressionMapOp):
        expr = source.expr
        if (not isinstance(expr, BinOpExpr)
                or expr.op not in (operator.add, operator.sub, operator.mul, operator.truediv)
                or not all(isinstance(operand, OperandLeaf) for operand in expr.children())):
            return None
        left = _projected_dtype(source.inputs[expr.left.ref.k])
        right = _projected_dtype(source.inputs[expr.right.ref.k])
        return left if left is not None and left.kind == "f" and left == right else None
    return _projected_dtype(source)


@physical_impl(of=AggregateOp, backend="polars", input_format="series", output_format="scalar")
class PolarsScalarAggregateOp(AggregateOp, PhysicalOp):
    """Reduce a numeric Series with pandas' accumulator and scalar result types."""

    @classmethod
    def supports(cls, op: AggregateOp, ctx=None) -> bool:
        if (op.grouped or op.output_type is not OutputType.SCALAR or len(op.inputs) != 1
                or len(op.aggregations) != 1):
            return False
        name, agg = op.aggregations[0]
        if name is not None or not isinstance(agg.child, AllCols) or agg.params:
            return False
        source = op.inputs[0]
        return (agg.func in ("sum", "min") and source.output_type is OutputType.SERIES
                and _scalar_input_dtype(source) is not None)

    def on_impl_selected(self, ctx):
        self._dtype = _scalar_input_dtype(self.inputs[0])
        self._arithmetic = isinstance(self.inputs[0], ExpressionMapOp)

    def _reduce(self, values):
        """Lower the reduction over a Polars Series or a fused expression."""
        if self._arithmetic:
            # pandas arithmetic masks NaNs, including on Arrow and nullable floats.
            values = values.fill_nan(None)
        if self.aggregations[0][1].func == "min":
            return values.min()
        if self._dtype.kind in "iu":
            values = values.cast(pl.UInt64 if self._dtype.kind == "u" else pl.Int64)
        return values.sum()

    def _as_scalar(self, value, length):
        if value is None:
            if isinstance(self._dtype, pd.api.extensions.ExtensionDtype):
                return pd.NA
            # An empty NumPy-backed Series returns a Python float NaN.
            return np.float64(np.nan) if length else float("nan")
        if isinstance(self._dtype, pd.ArrowDtype):
            return value
        if self.aggregations[0][1].func == "min":
            return self._dtype.type(value)
        if self._dtype.kind == "f":
            if isinstance(self._dtype, pd.Float64Dtype) and np.isnan(value):
                return pd.NA
            return np.float64(value)
        return np.uint64(value) if self._dtype.kind == "u" else np.int64(value)

    def process(self, inputs: list):
        series = inputs[0]
        # A valid NaN in a projected nullable float column propagates to pd.NA.
        if (self.aggregations[0][1].func == "min" and isinstance(self._dtype, pd.Float64Dtype)
                and not self._arithmetic and series.is_nan().any()):
            return pd.NA
        return self._as_scalar(self._reduce(series), len(series))


class PolarsFusedScalarAggregateOp(PolarsScalarAggregateOp):
    """Evaluate a row-wise expression and its scalar reduction in one frame select."""

    def process(self, inputs: list):
        _, agg = self.aggregations[0]
        values = to_polars(agg.child, EvalContext.of(inputs, self.mode))
        frame = inputs[0]
        return self._as_scalar(frame.select(self._reduce(values)).item(), frame.height)


@physical_impl(of=AggregateOp, backend="polars")
class PolarsAggregateOp(AggregateOp, PhysicalOp):

    @classmethod
    def supports(cls, op: AggregateOp, ctx) -> bool:
        """Require a frame source and result for expression-based aggregation.

        These paths call ``select`` or ``group_by`` on the source, neither of
        which is available on ``pl.Series``. Unsupported shapes use another
        implementation selected during planning.
        """
        if op.output_type is not OutputType.FRAME:
            return False
        if op.inputs and op.inputs[0].output_type is OutputType.SERIES:
            return False
        # Polars has no pandas index labels or unobserved categorical groups.
        if op.observed is False or op.level is not None:
            return False
        return refuse_unsupported_exprs(
            op, [*(agg for _, agg in op.aggregations), *op.grouping])

    def process(self, inputs: list):
        # pandas treats floating NaN as missing for grouping and reductions.
        frame = inputs[0].with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None))
        ctx = EvalContext.of([frame, *inputs[1:]], self.mode)
        if self.grouped and self.counts_group_rows():
            # The rows per group, not a count per column; pandas names it "size".
            exprs = [pl.len().alias("size")]
        else:
            exprs = [self._entry_expr(index, agg, ctx)
                     for index, (_, agg) in enumerate(self.aggregations)]
        if not self.grouped:
            return ctx.frame.select(exprs)
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
