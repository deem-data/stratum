from stratum.optimizer.logical._base import IndexEffect
from stratum.optimizer.logical._ops import (OperandRef, OutputType, is_frame_like, BaseEstimatorOp, BinOp, UnaryOp, CallOp, ChoiceOp, GetAttrOp, GetItemOp,
                                       MethodCallOp, Op, TransformerOp, ValueOp, substitute_op)
from pandas import DataFrame
from polars import DataFrame as PolarsDataFrame
from skrub import SelectCols
from skrub.selectors._base import All
import operator
from stratum.optimizer.logical import _schema
import pandas as pd
import numpy as np

# Per-category frame ops. These are imported here both because the
# `extract_dataframe_op` dispatcher below references them and so that existing
# `from ..._dataframe_ops import X` import sites keep working (re-export hub).
from stratum.optimizer.logical._source_ops import (
    FileReadOp, InMemoryFrameOp, LoadNumpyOp, ReadCSVOp, ReadParquetOp, SourceOp, make_read_op,
    try_make_read_op)
from stratum.optimizer.logical._selection_ops import (
    SelectionOp, MaskSelectionOp, DropMissingSelectionOp, DropDuplicatesSelectionOp,
    SliceSelectionOp, SampleSelectionOp, DropLabelsSelectionOp, _SELECTION_METHODS,
    make_selection_op, is_mask_selection, make_mask_selection_op)
from stratum.optimizer.logical._aggregation_ops import (
    AggregateOp, _AGG_METHODS, _AGG_FUNCS, _is_groupby_op,
    _is_aggregation, _is_value_counts, _extract_grouping, _extract_aggregations,
    make_aggregate_op, make_value_counts_ops)
from stratum.optimizer.logical._sort_ops import SortOp, make_sort_op
from stratum.optimizer.logical._projection_ops import (
    ColumnProjectionOp, ColumnSelectorOp, ProjectionOp, DropOp,
    drop_targets_columns, make_column_projection_op, make_column_selector_op,
    make_drop_op)
from stratum.optimizer.logical._metadata_ops import (
    MetadataOp, RenameOp, ResetIndexOp, make_rename_op, make_reset_index_op)
from stratum.optimizer.logical._map_ops import (
    MapOp, ApplyUDFOp, AssignMapOp, AssignOp, AttributeReadOp, DatetimeConversionOp,
    DatetimeFieldOp,
    StringMethodOp, make_assign_map_op, make_datetime_conversion_op,
    make_frame_get_attr, make_string_method_op)
from stratum.optimizer.logical._column_methods import (
    COLUMN_METHODS, ColumnMethodOp, IsnaOp, NotnaOp, make_column_method_op)
from stratum.optimizer.logical._join_ops import (
    JoinOp, FilteringJoinOp, make_join_op, _make_chained_join_op)
from stratum.optimizer.logical._split_ops import SplitOp, SplitOutput, add_splitting_op


class ConcatOp(Op):
    """Concatenates frames: ``axis=0`` stacks their rows, ``axis=1`` places their
    columns side by side.

    ``operands`` are the ``OperandRef``s of the frames, in order; one input edge
    may feed several of them. ``axis`` is ``0`` or ``1`` (extraction normalizes
    pandas' names for them), or an ``OperandRef`` when graph-fed.

    Like skrub's ``.skb.concat`` it also stacks numpy arrays. Extraction turns a
    concat whose first operand is known to be an array into an
    :class:`ArrayConcatOp`. When the first operand's kind is unknown at plan time
    (an opaque call can return either), the frame impls stack arrays at run time.
    """
    logical_family = "Concat"
    fields = ["operands", "axis"]
    # Stacked rows keep their inputs' labels, repeats and all.
    index_effect = IndexEffect.NEW

    def __init__(self, operands: list = None, axis=0, inputs: list[Op] = None,
                 outputs: list[Op] = None):
        # No name: the "Concat" family label alone renders cleanly (avoids the
        # redundant "Concat(CONCAT)").
        super().__init__(name="", inputs=inputs, outputs=outputs)
        self.operands = list(operands or [])
        self.axis = axis
        self.output_type = OutputType.FRAME

    @property
    def aligns_by_label(self):
        # Stacking rows pairs nothing; placing columns side by side aligns rows.
        return False if self.axis == 0 else None

    def propagate_output_schema(self):
        """See :func:`_schema.concat_schemas` for the per-axis rules."""
        if isinstance(self.axis, OperandRef):
            # The axis picks the dtype rule, so a graph-fed axis is unknown.
            self.output_schema = None
            return
        self.output_schema = _schema.concat_schemas(
            [self.inputs[ref.k].output_schema for ref in self.operands], self.axis)


class ArrayConcatOp(ConcatOp):
    """Joins numpy arrays along ``axis`` (``np.concatenate``): a concat whose first
    operand is known at plan time to be an array."""

    def __init__(self, operands: list = None, axis=0, inputs: list[Op] = None,
                 outputs: list[Op] = None):
        super().__init__(operands=operands, axis=axis, inputs=inputs, outputs=outputs)
        self.output_type = OutputType.MATRIX

    def propagate_output_schema(self):
        """An array has no columns."""
        self.output_schema = None


#: pandas' names for the two concat axes.
_CONCAT_AXES = {0: 0, "index": 0, "rows": 0, 1: 1, "columns": 1}


def _is_array(op: Op) -> bool:
    """Whether ``op`` is known at plan time to produce a numpy array."""
    return (op.output_type is OutputType.MATRIX
            or isinstance(op, ValueOp) and isinstance(op.value, np.ndarray))


def make_concat_op(op: ConcatOp) -> ConcatOp | None:
    """Normalize the axis of a recorded concat and pick its variant.

    The first operand decides, as in skrub: a known array makes an
    :class:`ArrayConcatOp`. Returns ``None`` when there is nothing to change.
    """
    axis = op.axis
    if not isinstance(axis, OperandRef):
        axis = _CONCAT_AXES.get(axis, axis)
    is_array = bool(op.operands) and _is_array(op.inputs[op.operands[0].k])
    normalized = axis == op.axis and type(axis) is type(op.axis)
    if type(op) is not ConcatOp or (normalized and not is_array):
        return None
    cls = ArrayConcatOp if is_array else ConcatOp
    return substitute_op(op, cls(operands=op.operands, axis=axis))


# The accessors whose ``[...]`` takes one indexer per axis, so a tuple key is a
# (rows, columns) pair rather than a single (MultiIndex) label.
_ROW_COL_ACCESSORS = ("loc", "iloc")


# pandas' scalar comparison methods have the same result as the corresponding
# operators. Graph-fed arguments are excluded: Series.eq(other_series) aligns
# indexes, whereas a row-local ColumnExpr works positionally.
_SCALAR_COMPARISONS = {
    "eq": operator.eq, "ne": operator.ne,
    "lt": operator.lt, "le": operator.le,
    "gt": operator.gt, "ge": operator.ge,
}


def _make_scalar_comparison(op: MethodCallOp) -> BinOp | None:
    if (op.method_name not in _SCALAR_COMPARISONS
            or op.inputs[0].output_type is not OutputType.SERIES
            or len(op.args or ()) != 1 or op.kwargs):
        return None
    value = op.args[0]
    if (isinstance(value, OperandRef) or isinstance(value, (pd.Series, pd.DataFrame,
                                                           np.ndarray, list, tuple, dict))
            or (not isinstance(value, str) and hasattr(value, "__len__"))):
        return None
    # A series compared to a scalar is a series.
    return substitute_op(op, BinOp(_SCALAR_COMPARISONS[op.method_name], OperandRef(0), value),
                         keep_kind=True)


def _opaque_method_output_type(op: MethodCallOp) -> OutputType | None:
    """Infer only the container kind of pandas methods we leave as calls.

    This does not claim a backend-neutral implementation. It lets a later
    operation be extracted while the original method still executes verbatim.
    Each listed method has a stable return kind for both Series and DataFrame.
    """
    source = op.inputs[0].output_type
    keeps_kind, _ = MethodCallOp.OPAQUE_METHODS.get(op.method_name, (False, False))
    if keeps_kind:
        return source
    if op.method_name == "to_frame" and source is OutputType.SERIES:
        return OutputType.FRAME
    if op.method_name == "reset_index":
        # Series.reset_index(drop=False) widens to a DataFrame. The drop=True
        # case is handled above by make_reset_index_op.
        return OutputType.FRAME
    return None


def _is_receiver(value) -> bool:
    """Whether a call argument is the call's frame-world operand 0."""
    return isinstance(value, OperandRef) and value.k == 0


def _is_row_col_indexer(op: GetItemOp) -> bool:
    """Whether ``op`` indexes through ``.loc``/``.iloc``, i.e. one axis per key element.

    The accessor survives extraction as the ``AttributeReadOp`` feeding this
    GetItem, so the container tells the two tuple-key meanings apart: ``df.loc[a, b]``
    indexes rows by ``a`` and columns by ``b``, while a bare ``df[("a", "b")]`` is one
    MultiIndex column label.
    """
    container = op.inputs[0]
    return (isinstance(container, AttributeReadOp) and bool(container.attr_name)
            and container.attr_name[-1] in _ROW_COL_ACCESSORS)


def _getitem_output_type(op: GetItemOp) -> OutputType:
    """Infer the output type of a ``GetItemOp`` whose container is frame-like.

    Indexing into a SERIES yields a SERIES. For a FRAME: ``df["col"]`` selects a
    single column -> SERIES; ``df[["a", "b"]]`` selects a sub-frame -> FRAME;
    ``df[mask]`` / ``df[label_series]`` (a graph-fed key, i.e. an
    :class:`OperandRef`) or a slice selects rows -> FRAME.

    A two-axis ``df.loc[rows, cols]`` / ``df.iloc[rows, cols]`` restricts rows *and*
    columns, so its column indexer decides the kind: a single label or position
    extracts one column -> SERIES; a list, slice or mask keeps a frame -> FRAME.
    """
    container = op.inputs[0]
    if container.output_type is OutputType.SERIES:
        return OutputType.SERIES
    # container is a FRAME (the only other frame-like type reaching here).
    if isinstance(op.key, str):
        return OutputType.SERIES
    if isinstance(op.key, tuple) and len(op.key) == 2 and _is_row_col_indexer(op):
        return (OutputType.SERIES if isinstance(op.key[1], (str, int))
                else OutputType.FRAME)
    # list/tuple of columns, an OperandRef mask, or a slice -> FRAME.
    return OutputType.FRAME


def _is_column_projection(op: GetItemOp) -> bool:
    """Whether a ``df[key]`` GetItem selects columns by literal name(s).

    True only when the container is a ``FRAME`` and the key is a literal column
    label (``str``) or list of labels (``list[str]``). A graph-fed key (an
    :class:`OperandRef`), a slice, an integer/positional key, a boolean-mask list
    or a ``SERIES`` container all fall through to the general ``GetItemOp``.
    """
    if op.inputs[0].output_type is not OutputType.FRAME:
        return False
    key = op.key
    if isinstance(key, str):
        return True
    return isinstance(key, list) and all(isinstance(k, str) for k in key)


#: ``pd.isna(x)`` / ``pd.notna(x)`` as the column method they are equivalent to.
#: ``pd.isnull`` and ``pd.notnull`` are the same function objects, so they map
#: here too.
_MISSING_VALUE_FUNCS = {pd.isna: IsnaOp, pd.notna: NotnaOp}


def _make_missing_value_method(op: CallOp) -> ColumnMethodOp | None:
    """``pd.isna(x)`` / ``pd.notna(x)`` as ``x.isna()`` / ``x.notna()``.

    Only for a lone graph-fed Series operand, positional or ``obj=``: on a scalar
    or a list the function returns a bool or an array instead, and the method is
    only modelled on a Series, as it is for the method call.
    """
    if len(op.args or ()) == 1 and not op.kwargs:
        operand = op.args[0]
    elif not op.args and set(op.kwargs or {}) == {"obj"}:
        operand = op.kwargs["obj"]
    else:
        return None
    if not isinstance(operand, OperandRef) or len(op.inputs) != 1:
        return None
    if op.inputs[operand.k].output_type is not OutputType.SERIES:
        return None
    # The operand is the only input, so it is the receiver whose kind is kept.
    return substitute_op(op, _MISSING_VALUE_FUNCS[op.func](), keep_kind=True)


def extract_dataframe_op(op: Op, root: Op, selection_op = True, map_op = True,
                         column_projection = True) -> tuple[Op, bool]:
    new_op = None
    if isinstance(op, ConcatOp):
        # Concat is built when the skrub graph is converted; its operands' kinds
        # are only known here.
        new_op = make_concat_op(op)

    elif len(op.inputs) == 0:
        # DataSource detection (directly passed dataframe)
        if isinstance(op, ValueOp) and (isinstance(op.value, DataFrame) or isinstance(op.value, PolarsDataFrame)):
            new_op = InMemoryFrameOp(data=op.value)
            new_op.outputs = op.outputs
        else:
            # DataSource detection (read operation with a literal path):
            # `skrub.deferred(pd.read_csv)("data.csv")` inlines the path into the
            # call, so the read has no operands at all.
            new_op = try_make_read_op(op)

    # DataSource detection (read operation): the input is not frame-world data --
    # a raw value (path / variable), or a numpy MATRIX left to the numeric path.
    elif not is_frame_like(op.inputs[0]):
        new_op = try_make_read_op(op)

    # input is frame-world data (a frame or a series): this is a dataframe op
    else:
        if isinstance(op, CallOp):
            # Datetime conversion detection
            if op.func is pd.to_datetime:
                new_op = make_datetime_conversion_op(op)
                if new_op is None and _is_receiver(op.args[0] if op.args
                                                   else (op.kwargs or {}).get("arg")):
                    # An opaque conversion of a column or a frame (the assembly
                    # form) still yields a Series.
                    op.output_type = OutputType.SERIES
            elif op.func in (pd.isna, pd.notna):
                new_op = _make_missing_value_method(op)

        elif isinstance(op, MethodCallOp):
            if isinstance(op.inputs[0], AttributeReadOp) and op.inputs[0].attr_name == ["str"]:
                # `col.str.<method>(...)` -> fuse the .str accessor and the method
                # call into a single StringMethodOp (a SERIES-typed projection). An
                # enclosing `df[...]` then sees a mask and folds the chain into a
                # StrExpr predicate, matching the StringMethodOp directly.
                new_op = make_string_method_op(op)
            elif op.method_name in _SCALAR_COMPARISONS:
                new_op = _make_scalar_comparison(op)
            elif op.method_name in COLUMN_METHODS:
                new_op = make_column_method_op(op)
            elif op.method_name == "groupby":
                # Leave groupby as-is; mark it as a dataframe op so the following
                # aggregation call is visited and can fuse with it.
                op.output_type = OutputType.FRAME
            elif _is_aggregation(op):
                new_op = make_aggregate_op(op)
            elif _is_value_counts(op):
                # Expands to an aggregation plus a sort, and returns the sort;
                # the caller rewires this op's consumers onto whatever comes back.
                new_op = make_value_counts_ops(op)
            elif op.method_name == "reset_index":
                new_op = make_reset_index_op(op)
            elif op.method_name == "sort_values":
                new_op = make_sort_op(op)
            elif op.method_name == "rename":
                new_op = make_rename_op(op)
            elif op.method_name == "drop":
                # Columns are a projection, rows (the pandas default) a selection.
                # An unknown axis, both axes at once, or a "column" drop on a
                # series (a no-op in pandas) stays an opaque call.
                drops_columns = drop_targets_columns(op.kwargs)
                if drops_columns is False:
                    new_op = make_selection_op(op)
                elif drops_columns and op.inputs[0].output_type is OutputType.FRAME:
                    new_op = make_drop_op(op)
            elif op.method_name == "apply":
                # apply on a column yields a column, on a frame yields a frame:
                # keep the input's kind (MapOp defaults to FRAME).
                new_op = substitute_op(op, ApplyUDFOp(args=op.args, kwargs=op.kwargs),
                                       keep_kind=True)
            elif op.method_name in ["assign"]:
                if map_op:
                    new_op = make_assign_map_op(op)
                if new_op is None:
                    # Not foldable (positional args / sequence-valued constant
                    # kwargs): keep the opaque assign.
                    new_op = substitute_op(op, AssignOp(args=op.args, kwargs=op.kwargs))
            elif op.method_name in ["join", "merge"]:
                new_op = make_join_op(op)
            elif op.method_name in _SELECTION_METHODS:
                new_op = make_selection_op(op)

            if new_op is None and op.output_type is OutputType.UNKNOWN:
                inferred = _opaque_method_output_type(op)
                if inferred is not None:
                    op.output_type = inferred

        # GetAttr Fusing and conversion to GetAttrDataframeOp
        elif isinstance(op, GetAttrOp):
            new_op = make_frame_get_attr(op)

        # Projection: BinOp/UnaryOp over tabular data -> same tabular kind as its
        # operand (e.g. `df["a"] > 7` is a SERIES, `df + 1` is a FRAME, `~mask` is
        # a SERIES).
        elif isinstance(op, (BinOp, UnaryOp)):
            op.output_type = op.inputs[0].output_type

        # GetItem: a mask selection df[bool_series] folds into a MaskSelectionOp;
        # otherwise it is a column projection (SERIES) / sub-frame / row selection.
        elif isinstance(op, GetItemOp):
            if is_mask_selection(op):
                op.is_filter = True
                if selection_op:
                    new_op = make_mask_selection_op(op)
            elif column_projection and _is_column_projection(op):
                # A literal column selection df["a"] / df[["a", "b"]] on a frame
                # becomes a dedicated ColumnProjectionOp; GetItem stays the
                # fallback for masks, graph-fed keys, slices and positional keys.
                new_op = make_column_projection_op(op)

            if new_op is None:
                op.output_type = _getitem_output_type(op)

        elif isinstance(op, BaseEstimatorOp):
            op.output_type = OutputType.FRAME
            # skrub implements `skb.select(cols)` as an Apply of its SelectCols
            # transformer; surface the selector as a dedicated ColumnSelectorOp.
            # Only while the Apply hands it every column: restricted by `cols` or
            # `exclude_cols`, skrub passes the other columns through untouched.
            if (isinstance(op, TransformerOp) and isinstance(op.estimator, SelectCols)
                    and not op.param_refs and not isinstance(op.y, OperandRef)
                    and isinstance(op.cols, All) and op.exclude_cols is None):
                new_op = make_column_selector_op(op)

        elif isinstance(op, ChoiceOp):
            # Propagate a shared frame type across all outcomes; mixed kinds fall
            # back to FRAME.
            if all(is_frame_like(outcome) for outcome in op.inputs):
                types = {outcome.output_type for outcome in op.inputs}
                op.output_type = types.pop() if len(types) == 1 else OutputType.FRAME

    if new_op is None:
        return root, False
    else:
        op.replace_input_of_outputs(new_op)
        if root is op:
            root = new_op
    return root, True

