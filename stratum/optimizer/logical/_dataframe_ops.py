from stratum.optimizer.logical._base import _resolve_operand
from stratum.optimizer.logical._ops import (
    OperandRef, OutputType, is_frame_like, BaseEstimatorOp, BinOp, UnaryOp, CallOp, ChoiceOp,
    GetAttrOp, GetItemOp, MethodCallOp, Op, TransformerOp, ValueOp, substitute_op)
from pandas import DataFrame
from polars import DataFrame as PolarsDataFrame
from skrub import SelectCols
from skrub.selectors._base import All
import operator
from stratum.optimizer.logical import _schema
import pandas as pd
import polars as pl
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
    make_selection_op)
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
    MetadataOp, RenameOp, RowLabelsOp, make_rename_op, make_reset_index_op,
    make_set_axis_op, make_set_index_op, make_to_frame_op)
from stratum.optimizer.logical._map_ops import (
    MapOp, ApplyUDFOp, ExpressionMapOp, AttributeReadOp, DatetimeConversionOp,
    DatetimeFieldOp,
    make_datetime_conversion_op, make_expression_map_op, ufunc_name, make_frame_get_attr)
from stratum.optimizer.logical._column_expr import (
    AstypeExpr, ClipExpr, FillnaExpr, IsinExpr, IsnaExpr, MapValuesExpr, NotnaExpr,
    OperandLeaf, StrExpr, WhereExpr)
from stratum.optimizer.logical._combine_ops import (
    AssignOp, CombineOp, ConcatOp, ArrayConcatOp, make_concat_op)
from stratum.optimizer.logical._join_ops import (
    JoinOp, FilteringJoinOp, make_join_op, _make_chained_join_op)
from stratum.optimizer.logical._split_ops import SplitOp, SplitOutput, add_splitting_op


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


_SERIES_REDUCTIONS = frozenset({
    "all", "any", "count", "max", "mean", "median", "min", "nunique", "prod", "sem",
    "std", "sum", "var",
})


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
    if (op.method_name in _SERIES_REDUCTIONS and source is OutputType.SERIES
            and not isinstance(op.inputs[0], AttributeReadOp)):
        return OutputType.SCALAR
    if op.method_name == "to_frame" and source is OutputType.SERIES:
        return OutputType.FRAME
    if op.method_name == "reset_index":
        # Series.reset_index(drop=False) widens to a DataFrame. The drop=True
        # case is handled above by make_reset_index_op.
        return OutputType.FRAME
    return None


def _is_receiver(value) -> bool:
    """Whether a call argument references the dataframe or series at input 0."""
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

    SERIES inputs currently retain SERIES as an approximation. For FRAME inputs,
    ``df["col"]`` yields SERIES; column lists, graph-fed keys, and slices retain
    FRAME.

    For two-axis ``df.loc[rows, cols]`` / ``df.iloc[rows, cols]``, this rule uses
    only the column indexer: a single label or position selects SERIES; a list,
    slice or mask selects FRAME. Scalar row indexers are not yet accounted for.
    """
    # TODO: Infer scalar results for single-element Series indexing and
    # two-axis scalar indexing such as df.iloc[0, 0].
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
_MISSING_VALUE_FUNCS = {pd.isna: IsnaExpr, pd.notna: NotnaExpr}


def _make_missing_value_method(op: CallOp) -> ExpressionMapOp | None:
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
    expr = _MISSING_VALUE_FUNCS[op.func](OperandLeaf(operand))
    return substitute_op(op, ExpressionMapOp(expr), keep_kind=True)


# --- Extraction: recorded pandas call -> column method fields -------------------
#
# Each parser takes the call bound against pandas' signature and whether the
# receiver is a DataFrame (a method can mean something else there: ``fillna``
# with a dict fills per column on a frame, per index label on a Series). It
# returns the expression's fields, or ``None`` to keep the call opaque.

def _is_ref_or_scalar(value) -> bool:
    if isinstance(value, OperandRef):
        return True
    if isinstance(value, (pd.Series, pd.DataFrame, pl.Series, pl.DataFrame)):
        return False
    return isinstance(value, str) or not hasattr(value, "__len__")


def _astype_fields(call: dict, frame: bool) -> dict | None:
    dtype = call.get("dtype")
    if (call.keys() != {"dtype"} or isinstance(dtype, OperandRef)
            or _schema.normalize_dtype(dtype) is None):
        return None
    return dict(dtype=dtype)


def _fillna_fields(call: dict, frame: bool) -> dict | None:
    value = call.get("value")
    if call.keys() != {"value"}:
        return None
    if frame:
        # A plain dict of column name -> scalar fill value.
        if type(value) is not dict or not all(
                isinstance(k, str) and v is not None and not pd.isna(v)
                and _is_ref_or_scalar(v)
                for k, v in value.items() if not isinstance(v, OperandRef)):
            return None
    elif not _is_ref_or_scalar(value):
        return None
    return dict(value=value)


def _clip_fields(call: dict, frame: bool) -> dict | None:
    if (frame or not call or call.keys() - {"lower", "upper"}
            or not all(_is_ref_or_scalar(v) for v in call.values())):
        return None
    return dict(lower=call.get("lower"), upper=call.get("upper"))


def _where_fields(call: dict, frame: bool) -> dict | None:
    # Without `other`, pandas fills NaN, which not every polars dtype can hold.
    if (frame or call.keys() != {"cond", "other"}
            or not all(_is_ref_or_scalar(v) for v in call.values())):
        return None
    return dict(cond=call["cond"], other=call["other"])


def _isin_fields(call: dict, frame: bool) -> dict | None:
    values = call.get("values")
    if frame or call.keys() != {"values"} or not (
            isinstance(values, OperandRef)
            or isinstance(values, (list, tuple, set, frozenset, pd.Index, np.ndarray))):
        return None
    return dict(values=values)


def _no_fields(call: dict, frame: bool) -> dict | None:
    return None if frame or call else {}


def _map_fields(call: dict, frame: bool) -> dict | None:
    """A plain dict of scalar keys and values: a lookup that misses to NaN.

    Refused: functions and Series (arbitrary code / index alignment), dict
    subclasses (which may define ``__missing__``), missing-valued keys, and keys
    or values that mix types beyond int/float, whose pandas result dtype
    Polars would not reproduce.
    """
    if frame:
        return None
    # pandas 3 renamed `arg` to `func`; the old name still arrives as a keyword.
    given = {**call.pop("kwargs", {}), **call}
    names = given.keys() & {"func", "arg"}
    if len(names) != 1 or given.keys() - {"func", "arg", "na_action"}:
        return None
    mapping, na_action = given[names.pop()], given.get("na_action")
    if type(mapping) is not dict or na_action not in (None, "ignore"):
        return None

    def homogeneous(values):
        types = {type(v) for v in values if v is not None}
        return len(types) <= 1 or types <= {int, float}

    if not (all(type(k) in (str, int, float) and not pd.isna(k) for k in mapping)
            and all(v is None or type(v) in (str, int, float, bool)
                    for v in mapping.values())
            and homogeneous(mapping) and homogeneous(mapping.values())):
        return None
    return dict(mapping=mapping, na_action=na_action)


# pandas method -> (expression class, parser of the bound call)
COLUMN_METHODS = {
    "astype": (AstypeExpr, _astype_fields),
    "fillna": (FillnaExpr, _fillna_fields),
    "clip": (ClipExpr, _clip_fields),
    "where": (WhereExpr, _where_fields),
    "isin": (IsinExpr, _isin_fields),
    "isna": (IsnaExpr, _no_fields),
    "isnull": (IsnaExpr, _no_fields),
    "notna": (NotnaExpr, _no_fields),
    "notnull": (NotnaExpr, _no_fields),
    "map": (MapValuesExpr, _map_fields),
}


def make_column_method_op(op: MethodCallOp) -> ExpressionMapOp | None:
    """Parse a supported method into a Map containing one method expression.

    Returns ``None`` (the call stays opaque) when the call is not one we model.
    """
    cls, parse = COLUMN_METHODS[op.method_name]
    call = op.bind_pandas()
    if call is None:
        return None
    fields = parse(call, op.inputs[0].output_type is OutputType.FRAME)
    if fields is None:
        return None
    operands = [OperandLeaf(OperandRef(k)) for k in range(len(op.inputs))]
    expr = cls(operands[0], **_resolve_operand(fields, operands))
    return substitute_op(op, ExpressionMapOp(expr), keep_kind=True)


def make_string_method_op(op: MethodCallOp) -> ExpressionMapOp | None:
    """Extract ``col.str.<method>(...)`` as an expression Map over the column.

    Input 0 changes from the ``.str`` accessor to its source column. Remaining
    input positions are preserved as ``OperandLeaf`` references in ``params``.
    Arguments are bound against pandas' signature; invalid calls stay opaque.
    A shared accessor is detached after its last consumer has been replaced.
    """
    if not hasattr(pd.Series.str, op.method_name):
        return None
    params = op.bind_pandas(owner=pd.Series.str)
    if params is None:
        return None
    accessor = op.inputs[0]
    column = accessor.inputs[0]
    operands = [OperandLeaf(OperandRef(k)) for k in range(len(op.inputs))]
    expr = StrExpr(operands[0], op.method_name, _resolve_operand(params, operands))
    new_op = ExpressionMapOp(expr, inputs=[column, *op.inputs[1:]],
                             outputs=list(op.outputs))
    # TODO: Infer output kind by method and arguments; methods such as
    # str.get_dummies() return a frame, and str.cat() can return a scalar.
    new_op.output_type = accessor.output_type

    # Rewire every operand except the accessor: the args feed the new op, and the
    # column now feeds it directly (in place of feeding the accessor).
    for in_ in op.inputs[1:]:
        in_.replace_output(op, new_op)
    column.add_output(new_op)
    accessor.outputs = [o for o in accessor.outputs if o is not op]
    if not accessor.outputs:
        accessor.detach_and_prune()
    return new_op


def extract_dataframe_op(op: Op, root: Op, map_op = True,
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

    # DataSource detection (read operation): the input is not a dataframe or series --
    # a raw value (path / variable), or a numpy MATRIX left to the numeric path.
    elif not is_frame_like(op.inputs[0]):
        new_op = try_make_read_op(op)

    # A dataframe or series input selects the dataframe extraction path.
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
            elif ufunc_name(op) is not None:
                op.output_type = op.inputs[0].output_type

        elif isinstance(op, MethodCallOp):
            if isinstance(op.inputs[0], AttributeReadOp) and op.inputs[0].attr_name == ["str"]:
                # Replace the accessor and call with a Map containing StrExpr.
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
            elif op.method_name == "set_index":
                new_op = make_set_index_op(op)
            elif op.method_name == "set_axis":
                new_op = make_set_axis_op(op)
            elif op.method_name == "to_frame":
                new_op = make_to_frame_op(op)
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
                # TODO: Infer apply's output kind from its arguments and function;
                # DataFrame.apply can produce a Series, and Series.apply a frame.
                new_op = substitute_op(op, ApplyUDFOp(args=op.args, kwargs=op.kwargs),
                                       keep_kind=True)
            elif op.method_name == "assign" and not op.args:
                new_op = substitute_op(op, AssignOp(kwargs=op.kwargs))
            elif op.method_name in ["join", "merge"]:
                new_op = make_join_op(op)
            elif op.method_name in _SELECTION_METHODS:
                new_op = make_selection_op(op)

            if new_op is None and op.output_type is OutputType.UNKNOWN:
                inferred = _opaque_method_output_type(op)
                if inferred is not None:
                    op.output_type = inferred

        # Fuse attribute chains and extract supported dataframe metadata reads.
        elif isinstance(op, GetAttrOp):
            new_op = make_frame_get_attr(op)

        # Elementwise BinOp/UnaryOp over tabular data -> same tabular kind as its
        # operand (e.g. `df["a"] > 7` is a SERIES, `df + 1` is a FRAME, `~mask` is
        # a SERIES).
        elif isinstance(op, (BinOp, UnaryOp)):
            op.output_type = op.inputs[0].output_type

        # Masked indexing keeps its container kind here; a later pass folds masks.
        elif isinstance(op, GetItemOp):
            if column_projection and _is_column_projection(op):
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

    if map_op:
        mapped = make_expression_map_op(new_op if new_op is not None else op)
        if mapped is not None:
            new_op = mapped

    if new_op is None:
        return root, False
    else:
        op.replace_input_of_outputs(new_op)
        if root is op:
            root = new_op
    return root, True

