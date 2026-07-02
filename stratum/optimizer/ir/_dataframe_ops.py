from stratum.optimizer.ir._ops import (OperandRef, OutputType, is_frame_like, BaseEstimatorOp, BinOp, UnaryOp, CallOp, ChoiceOp, GetAttrOp, GetItemOp,
                                       MethodCallOp, Op, ValueOp)
from pandas import DataFrame
from polars import DataFrame as PolarsDataFrame
import pandas as pd
import polars as pl
import numpy as np
from stratum._config import FLAGS

# Per-category frame ops. These are imported here both because the
# `extract_dataframe_op` dispatcher below references them and so that existing
# `from ..._dataframe_ops import X` import sites keep working (re-export hub).
from stratum.optimizer.ir._source_ops import DataSourceOp, make_read_op
from stratum.optimizer.ir._selection_ops import (
    SelectionKind, SelectionOp, _SELECTION_METHODS, make_selection_op,
    is_mask_selection, make_mask_selection_op)
from stratum.optimizer.ir._aggregation_ops import (
    AggregateOp, GroupedDataframeOp, _AGG_METHODS, _AGG_FUNCS, _is_groupby_op,
    _is_aggregation, _extract_grouping, _extract_aggregations, make_aggregate_op)
from stratum.optimizer.ir._projection_ops import (
    MetadataOp, ProjectionOp, DropOp, ApplyUDFOp, AssignOp, DatetimeConversionOp,
    GetAttrProjectionOp, StringMethodOp, make_datetime_conversion_op,
    make_frame_get_attr, make_string_method_op)
from stratum.optimizer.ir._join_ops import (
    JoinOp, _MERGE_POSITIONAL, _JOIN_POSITIONAL, _JOIN_OP_FIELDS, make_join_op,
    _make_chained_join_op)
from stratum.optimizer.ir._split_ops import SplitOp, SplitOutput, add_splitting_op


class ConcatOp(Op):
    fields = ["first", "others", "axis"] # Add more if needed

    axis_map = {
        0: "diagonal_relaxed",
        1: "horizontal",
    }
    def __init__(self, first, others: list, axis):
        super().__init__(name="CONCAT", is_X=False, is_y=False)
        # first/others entries/axis are OperandRefs when graph-fed, else constants.
        self.first = first
        self.others = list(others)
        self.axis = axis
        self.output_type = OutputType.FRAME

    def process(self, mode: str, inputs: list):
        first = inputs[self.first.k] if isinstance(self.first, OperandRef) else self.first
        others = [inputs[o.k] if isinstance(o, OperandRef) else o for o in self.others]
        axis = inputs[self.axis.k] if isinstance(self.axis, OperandRef) else self.axis
        if FLAGS.force_polars:
            return pl.concat([first, *others], how=self.axis_map[axis])
        else:
            return pd.concat([first, *others], axis=axis)


def _getitem_output_type(op: GetItemOp) -> OutputType:
    """Infer the output type of a ``GetItemOp`` whose container is frame-like.

    Indexing into a SERIES yields a SERIES. For a FRAME: ``df["col"]`` selects a
    single column -> SERIES; ``df[["a", "b"]]`` selects a sub-frame -> FRAME;
    ``df[mask]`` / ``df[label_series]`` (a graph-fed key, i.e. an
    :class:`OperandRef`) or a slice selects rows -> FRAME.
    """
    container = op.inputs[0]
    if container.output_type is OutputType.SERIES:
        return OutputType.SERIES
    # container is a FRAME (the only other frame-like type reaching here).
    if isinstance(op.key, str):
        return OutputType.SERIES
    # list/tuple of columns, an OperandRef mask, or a slice -> FRAME.
    return OutputType.FRAME


def extract_dataframe_op(op: Op, root: Op, selection_op = True) -> tuple[Op, bool]:
    new_op = None
    # DataSource detection (directly passed dataframe)
    if len(op.inputs) == 0:
        if isinstance(op, ValueOp) and (isinstance(op.value, DataFrame) or isinstance(op.value, PolarsDataFrame)):
            new_op = DataSourceOp(data=op.value)
            new_op.outputs = op.outputs

    # DataSource detection (read operation): the input is not frame-world data --
    # a raw value (path / variable), or a numpy MATRIX left to the numeric path.
    elif not is_frame_like(op.inputs[0]):
        if isinstance(op, CallOp):
            if op.func is pd.read_csv:
                new_op = make_read_op(op)

            elif op.func is pd.read_parquet:
                new_op = make_read_op(op, "parquet")

            elif op.func is np.load:
                new_op = make_read_op(op, "npy")

    # input is frame-world data (a frame or a series): this is a dataframe op
    else:
        if isinstance(op, CallOp):
            # Datetime conversion detection
            if op.func is pd.to_datetime:
                new_op = make_datetime_conversion_op(op)

        elif isinstance(op, MethodCallOp):
            if isinstance(op.inputs[0], GetAttrProjectionOp) and op.inputs[0].attr_name == ["str"]:
                # `col.str.<method>(...)` -> fuse the .str accessor and the method
                # call into a single StringMethodOp (a SERIES-typed projection). An
                # enclosing `df[...]` then sees a mask and folds the chain into a
                # StrExpr predicate, matching the StringMethodOp directly.
                new_op = make_string_method_op(op)
            elif op.method_name == "groupby":
                # Leave groupby as-is; mark it as a dataframe op so the following
                # aggregation call is visited and can fuse with it.
                op.output_type = OutputType.FRAME
            elif _is_aggregation(op):
                new_op = make_aggregate_op(op)
            elif op.method_name in ["rename"]:
                new_op = MetadataOp(func=op.method_name, args=op.args, kwargs=op.kwargs, inputs=op.inputs,
                                    outputs=op.outputs)
                op.replace_output_of_inputs(new_op)
            elif op.method_name == "drop":
                new_op = DropOp(args=op.args, kwargs=op.kwargs, inputs=op.inputs, outputs=op.outputs)
                op.replace_output_of_inputs(new_op)
            elif op.method_name == "apply":
                new_op = ApplyUDFOp(args=op.args, kwargs=op.kwargs, inputs=op.inputs, outputs=op.outputs)
                # apply on a column yields a column, on a frame yields a frame:
                # keep the input's kind (ProjectionOp defaults to FRAME).
                new_op.output_type = op.inputs[0].output_type
                op.replace_output_of_inputs(new_op)
            elif op.method_name in ["assign"]:
                new_op = AssignOp(args=op.args, kwargs=op.kwargs, inputs=op.inputs, outputs=op.outputs)
                op.replace_output_of_inputs(new_op)
            elif op.method_name in ["join", "merge"]:
                new_op = make_join_op(op)
            elif op.method_name in _SELECTION_METHODS:
                new_op = make_selection_op(op)

        # GetAttr Fusing and conversion to GetAttrDataframeOp
        elif isinstance(op, GetAttrOp):
            new_op = make_frame_get_attr(new_op, op)

        # Projection: BinOp/UnaryOp over tabular data -> same tabular kind as its
        # operand (e.g. `df["a"] > 7` is a SERIES, `df + 1` is a FRAME, `~mask` is
        # a SERIES).
        elif isinstance(op, (BinOp, UnaryOp)):
            op.output_type = op.inputs[0].output_type

        # GetItem: a mask selection df[bool_series] folds into a SelectionOp;
        # otherwise it is a column projection (SERIES) / sub-frame / row selection.
        elif isinstance(op, GetItemOp):
            if is_mask_selection(op):
                op.is_filter = True
                if selection_op:
                    new_op = make_mask_selection_op(op)

            if new_op is None:
                op.output_type = _getitem_output_type(op)

        elif isinstance(op, BaseEstimatorOp):
            op.output_type = OutputType.FRAME

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


def group_dataframe_ops(root: Op) -> Op:
    return root
