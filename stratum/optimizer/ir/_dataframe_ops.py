from typing import Callable
from collections.abc import Sequence
from stratum.optimizer.ir._ops import (OperandRef, BaseEstimatorOp, BinOp, CallOp, ChoiceOp, GetAttrOp, GetItemOp,
                                       MethodCallOp, Op, ValueOp, VariableOp,_resolve_args, _resolve_kwargs)
from pandas import DataFrame
import pandas as pd
import polars as pl
import numpy as np
from stratum.optimizer._op_utils import topological_iterator
from stratum._config import FLAGS
from stratum.utils._utils import start_time, log_time
from skrub._data_ops._data_ops import DataOp
import logging
from numpy import sin, cos
logger = logging.getLogger(__name__)

class DataSourceOp(Op):
    def __init__(self, data: DataFrame = None, file_path: str = None, _format: str = None,
                 read_args: tuple | list = None, read_kwargs: dict = None, is_X=False, is_y=False, outputs: list[Op] = None, inputs: list[Op] = None):
        if outputs is None:
            outputs = []
        super().__init__(name="Frame" if data is not None else f"read_{_format}", is_X=is_X, is_y=is_y, outputs=outputs, inputs=inputs)
        if read_kwargs is not None:
            self.check_kwargs(read_kwargs)
        self.data = data
        self.format = _format
        self.file_path = file_path
        self.read_args = read_args
        self.read_kwargs = read_kwargs
        self.is_dataframe_op = format != "npy"

    def process(self, mode: str, environment: dict, inputs: list):
        if self.data is not None:
            if FLAGS.force_polars:
                return pl.DataFrame(self.data)
            else:
                return self.data
        else:
            file_path = inputs[self.file_path.k] if isinstance(self.file_path, OperandRef) else self.file_path
            read_args = _resolve_args(self.read_args, inputs) if self.read_args else []
            read_kwargs = _resolve_kwargs(self.read_kwargs, inputs) if self.read_kwargs else {}
            if FLAGS.force_polars:
                return pl.read_csv(file_path, *read_args, **read_kwargs)
            else:
                if self.format == "csv":
                    return pd.read_csv(file_path, *read_args, **read_kwargs)
                elif self.format == "npy":
                    return np.load(file_path, *read_args, **read_kwargs)
                else:
                    raise ValueError(f"Unsupported format: {self.format}")

    def clone(self):
        raise ValueError(f"We should not clone DataSourceOp objects.")

class JoinOp(Op):
    fields = ["how", "left_on", "right_on", "left_index", "right_index", "suffixes"]

    def __init__(
        self,
        how: str = "inner",
        left_on: str | list[str] | None = None,
        right_on: str | list[str] | None = None,
        left_index: bool = False,
        right_index: bool = False,
        suffixes: Sequence[str] = ("_x", "_y"),
        inputs: list[Op] | None = None,
        outputs: list[Op] | None = None,
    ):
        super().__init__(name="", inputs=inputs, outputs=outputs)
        self.how = how
        self.left_on = left_on
        self.right_on = right_on
        self.left_index = left_index
        self.right_index = right_index
        self.suffixes = suffixes
        self.is_dataframe_op = True

    def process(self, mode: str, environment: dict, inputs: list):
        if len(inputs) != 2:
            raise ValueError(f"JoinOp expects exactly 2 inputs (left and right dataframes), got {len(inputs)}.")
        left_df = inputs[0]
        right_df = inputs[1]

        if FLAGS.force_polars:
            raise NotImplementedError("JoinOp Polars backend is not implemented yet.")
        else:
            return left_df.merge(
                right_df,
                left_on=self.left_on,
                right_on=self.right_on,
                how=self.how,
                suffixes=self.suffixes,
                left_index=self.left_index,
                right_index=self.right_index
            )

class AggregateOp(Op):
    """Fused ``groupby(...).agg(...)`` operation.

    Captures a ``DataFrame.groupby(by)`` followed by a single aggregation call
    (e.g. ``.agg("mean")``, ``.sum()``, ``.mean()``, ``.count()``) as one op.
    Both the direct methods and ``.agg(spec)`` are normalized to ``aggregations``
    so ``grouped.agg(aggregations)`` reproduces the original result.
    """
    fields = ["grouping_attributes", "aggregations", "groupby_kwargs"]

    def __init__(self, grouping_attributes: str | list[str] | OperandRef,
                 aggregations: str | list[str] | dict | OperandRef,
                 groupby_kwargs: dict | None = None,
                 inputs: list[Op] | None = None, outputs: list[Op] | None = None):
        super().__init__(name="", inputs=inputs, outputs=outputs)
        self.grouping_attributes = grouping_attributes
        self.aggregations = aggregations
        self.groupby_kwargs = groupby_kwargs or {}
        self.is_dataframe_op = True

    def __str__(self):
        return f"AggregateOp(by={self.grouping_attributes}, agg={self.aggregations}) [df]"

    def process(self, mode: str, environment: dict, inputs: list):
        _obj = inputs[0]
        grouping = inputs[self.grouping_attributes.k] if isinstance(self.grouping_attributes, OperandRef) else self.grouping_attributes
        aggregations = inputs[self.aggregations.k] if isinstance(self.aggregations, OperandRef) else self.aggregations
        if FLAGS.force_polars:
            raise NotImplementedError("AggregateOp Polars backend is not implemented yet.")
        return _obj.groupby(grouping, **self.groupby_kwargs).agg(aggregations)

class MetadataOp(Op):
    fields = ["func", "args", "kwargs"]

    def __init__(self, func: str, args: tuple | list = None, kwargs: dict = None, inputs: list[Op] = None, outputs: list[Op] = None, is_X=False, is_y=False):
        super().__init__(name=func.upper(), is_X=is_X, is_y=is_y, inputs=inputs, outputs=outputs)
        if kwargs is not None:
            self.check_kwargs(kwargs)
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.is_dataframe_op = True

    def process(self, mode: str, environment: dict, inputs: list):
        _obj = inputs[0]
        _args = _resolve_args(self.args, inputs)
        _kwargs = _resolve_kwargs(self.kwargs, inputs)
        if FLAGS.force_polars:
            if "columns" in _kwargs:
                _args.append(_kwargs["columns"])
            return getattr(_obj, self.func)(*_args)
        else:
            return getattr(_obj, self.func)(*_args, **_kwargs)

class ProjectionOp(Op):
    fields = ["func", "method", "args", "kwargs", "columns"]

    def __init__(self, func: Callable | None = None, method: str | None = None,
        args: tuple | list = None, kwargs: dict = None,
        inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None):
        if func is not None and method is not None:
            raise ValueError("`func` and `method` are mutually exclusive; set exactly one (or neither for subclasses that override `process`).")
        if method is not None:
            name = method.upper()
        elif func is not None:
            name = func.__name__.upper()
        else:
            name = ""
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        if kwargs is not None:
            self.check_kwargs(kwargs)
        self.func = func
        self.method = method
        self.args = args
        self.columns = columns
        self.kwargs = kwargs
        self.is_dataframe_op = True

    def _extract_args_and_kwargs(self, inputs: list):
        """Extract and process arguments and kwargs from inputs."""
        # The object is the implicit primary operand (index 0). For func-based ops
        # the first positional arg is that object slot, so skip it here.
        _obj = inputs[0]
        args = self.args[1:] if self.func is not None else self.args
        _args = _resolve_args(args, inputs)
        _kwargs = _resolve_kwargs(self.kwargs, inputs)
        return _obj, _args, _kwargs

    def process(self, mode: str, environment: dict, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        if self.method is not None:
            if FLAGS.force_polars:
                raise ValueError(f"Unsupported method: {self.method}")
            return getattr(_obj, self.method)(*_args, **_kwargs)
        if self.func is not None:
            return self.func(_obj, *_args, **_kwargs)
        raise TypeError("ProjectionOp requires either `func` or `method` to be set.")

class DropOp(ProjectionOp):
    fields = ["args", "kwargs", "columns"]
    def __init__(self, args: tuple | list = (), kwargs: dict = {},
        inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None):
        super().__init__(args=args, kwargs=kwargs, inputs=inputs, outputs=outputs, columns=columns)

    def process(self, mode: str, environment: dict, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)

        if FLAGS.force_polars:
            if "columns" in _kwargs:
                _args.append(_kwargs["columns"])
            if "ignore_errors" in _kwargs:
                _args.append(_kwargs["ignore_errors"] == "raise")
            return _obj.drop(*_args)
        else:
            return _obj.drop(*_args, **_kwargs)

class ApplyUDFOp(ProjectionOp):
    fields = ["args", "kwargs", "columns"]
    def __init__(self, args: tuple | list = (), kwargs: dict = {},
        inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None):
        super().__init__(args=args, kwargs=kwargs, inputs=inputs, outputs=outputs, columns=columns)

    def process(self, mode: str, environment: dict, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)

        n_cols = None
        if self.columns:
            _obj = _obj[self.columns]
            if type(self.columns) == str:
                n_cols = 1
            else:
                n_cols = len(self.columns)

        if FLAGS.force_polars:
            if isinstance(_obj, pl.Series):
                n_cols = 1
            if n_cols == 1:
                if _args[0] == sin:
                    logger.debug("Rewrite UDF sin to polars sin")
                    return _obj.sin()
                elif _args[0] == cos:
                    logger.debug("Rewrite UDF cos to polars cos")
                    return _obj.cos()
                else:
                    return _obj.map_elements(*_args, **_kwargs)
            else:
                return _obj.map_rows(*_args, **_kwargs)
        else:
            return _obj.apply(*_args, **_kwargs)

class AssignOp(ProjectionOp):
    def __init__(self, args: tuple | list = (), kwargs: dict = {},
        inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None):
        super().__init__(args=args, kwargs=kwargs, inputs=inputs, outputs=outputs, columns=columns)

    def process(self, mode: str, environment: dict, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        if FLAGS.force_polars:
            checked_kwargs = {}
            for k, v in _kwargs.items():
                if isinstance(v, OperandRef):
                    raise NotImplementedError("Is not yet suppoerted, please report this issue")
                elif isinstance(v, pd.Series) or isinstance(v, pd.DataFrame):
                    logger.warning(f"Converting pandas object to polars object for column {k}")
                    checked_kwargs[k] = pl.from_pandas(v)
                else:
                    checked_kwargs[k] = v
            return _obj.with_columns(*_args, **checked_kwargs)
        else:
            return _obj.assign(*_args, **_kwargs)

class DatetimeConversionOp(ProjectionOp):
    def __init__(self, args: tuple | list = (), kwargs: dict = {},
        inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None):
        super().__init__(args=args, inputs=inputs, outputs=outputs, columns=columns)
        self.strict = kwargs.get("errors", "raise") == "raise"

    def process(self, mode: str, environment: dict, inputs: list):
        if FLAGS.force_polars:
            return inputs[0].str.to_datetime(*self.args, strict=self.strict)
        else:
            return pd.to_datetime(inputs[0], *self.args, errors="raise" if self.strict else "coerce")

class GetAttrProjectionOp(Op):
    fields = ["attr_name"]

    # NOTE: Polars and Pandas differ in semantics for some datetime attributes:
    #   - dayofweek: Pandas uses Monday=0, Polars weekday() uses Monday=1 (ISO 8601)
    #   - dayofyear: Pandas is 1-indexed, Polars ordinal_day() is also 1-indexed (same)
    POLARS_ATTR_NAME_MAP = {"dayofweek": "weekday","dayofyear": "ordinal_day"}

    def __init__(self, attr_name: list[str] | str = None, inputs: list[Op] = None, outputs: list[Op] = None):
        if attr_name is None:
            self.attr_name = []
        elif isinstance(attr_name, str):
            self.attr_name = [attr_name]
        else:
            self.attr_name = attr_name
        attr_name_str = ".".join(self.attr_name) if self.attr_name else '?'
        super().__init__(name=attr_name_str)
        self.inputs = inputs
        self.outputs = outputs
        self.is_dataframe_op = True

    def __str__(self):
        attr_name = ".".join(self.attr_name)
        return f"GetAttrProjectionOp({attr_name}) [df]"

    def process(self, mode: str, environment: dict, inputs: list):
        result = inputs[0]
        tmp = result
        if FLAGS.force_polars:
            for attr in self.attr_name:
                attr = self.POLARS_ATTR_NAME_MAP.get(attr, attr)

                # TODO find better way to handle this
                if attr == "is_month_end":
                    return result.dt.month_end() == result

                # polars implements dt.day as a method, not an attribute
                # use getattr to handle both attributes and methods
                tmp = getattr(tmp, attr)
            return tmp()
        else:
            for attr in self.attr_name:
                tmp = getattr(tmp, attr)
            return tmp

class GroupedDataframeOp(Op):
    def __init__(self, ops: list[Op]):
        super().__init__(name="GROUPED_DATAFRAME", is_X=False, is_y=False)
        self.ops = ops
        self.is_dataframe_op = True

    def process(self, mode: str, environment: dict, inputs: list):  # pragma: no cover
        # TODO: GroupedDataframeOp is experimental and not integrated yet.
        # Needs proper refactoring to collect sub-op inputs from the pool.
        raise NotImplementedError("GroupedDataframeOp is not integrated yet.")

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
        self.is_dataframe_op = True

    def process(self, mode: str, environment: dict, inputs: list):
        first = inputs[self.first.k] if isinstance(self.first, OperandRef) else self.first
        others = [inputs[o.k] if isinstance(o, OperandRef) else o for o in self.others]
        axis = inputs[self.axis.k] if isinstance(self.axis, OperandRef) else self.axis
        if FLAGS.force_polars:
            return pl.concat([first, *others], how=self.axis_map[axis])
        else:
            return pd.concat([first, *others], axis=axis)


class SplitOp(Op):
    def __init__(self, inputs: list[Op]=None, outputs: list[Op]=None):
        super().__init__(name="Train/Test", is_X=False, is_y=False, inputs=inputs, outputs=outputs)
        self.is_split_op = True
        self.is_dataframe_op = True
        self.indices = None

    def process(self, mode: str, environment: dict, inputs: list):
        # we need to handle both pandas and polars dfs
        x = inputs[0]
        y = inputs[1]
        if isinstance(x, pd.DataFrame):
            return (x.iloc[self.indices], y.iloc[self.indices])
        elif isinstance(x, pl.DataFrame):
            return (x[self.indices], y[self.indices])
        elif isinstance(x, np.ndarray):
            return (x[self.indices], y[self.indices])
        else:
            raise ValueError(f"Unsupported dataframe type: {type(x)}")

class SplitOutput(Op):
    def __init__(self, inputs: list[Op]=None, outputs: list[Op]=None, is_x = True, ):
        name = "X" if is_x else "y"
        super().__init__(name=name, is_X=False, is_y=False, inputs=inputs, outputs=outputs)
        self.is_x = is_x
        self.is_dataframe_op = True

    def process(self, mode: str, environment: dict, inputs: list):
        if self.is_x:
            return inputs[0][0]
        else:
            return inputs[0][1]

def add_splitting_op(root: Op) -> Op:
    start = start_time()
    x_op = None
    y_op = None
    for op in topological_iterator(root):
        if op.is_X:
            x_op = op
        if op.is_y:
            y_op = op
        if x_op and y_op:

            split_out_x = SplitOutput(outputs=x_op.outputs)
            x_op.replace_input_of_outputs(split_out_x)
            split_out_y = SplitOutput(outputs=y_op.outputs, is_x=False)
            y_op.replace_input_of_outputs(split_out_y)
            split_op = SplitOp(inputs=[x_op, y_op], outputs=[split_out_x, split_out_y])
            split_out_x.inputs = [split_op]
            split_out_y.inputs = [split_op]
            x_op.outputs = [split_op]
            y_op.outputs = [split_op]
            break
    log_time("splitting took", start)
    return root

def extract_dataframe_op(op: Op, root: Op) -> tuple[Op, bool]:
    new_op = None
    # DataSource detection (directly passed dataframe)
    if len(op.inputs) == 0:
        if isinstance(op, ValueOp) and isinstance(op.value, DataFrame):
            new_op = DataSourceOp(data=op.value)
            new_op.outputs = op.outputs

    # DataSource detection (read operation)
    elif not op.inputs[0].is_dataframe_op:
        if isinstance(op, CallOp):
            if op.func is pd.read_csv:
                new_op = make_read_op(op)
            
            elif op.func is np.load:
                new_op = make_read_op(op, "npy")

    # input is a dataframe op
    else:
        if isinstance(op, CallOp):
            # Datetime conversion detection
            if op.func is pd.to_datetime:
                new_op = make_datetime_conversion_op(op)

        elif isinstance(op, MethodCallOp):
            if op.method_name == "groupby":
                # Leave groupby as-is; mark it as a dataframe op so the following
                # aggregation call is visited and can fuse with it.
                op.is_dataframe_op = True
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
                op.replace_output_of_inputs(new_op)
            elif op.method_name in ["assign"]:
                new_op = AssignOp(args=op.args, kwargs=op.kwargs, inputs=op.inputs, outputs=op.outputs)
                op.replace_output_of_inputs(new_op)
            elif op.method_name in ["join", "merge"]:
                new_op = make_join_op(op)

        # GetAttr Fusing and conversion to GetAttrDataframeOp
        elif isinstance(op, GetAttrOp) and op.inputs[0].is_dataframe_op:
            new_op = make_frame_get_attr(new_op, op)

        # Projection: BinOp detection
        elif isinstance(op, BinOp) and op.inputs[0].is_dataframe_op:
            op.is_dataframe_op = True

        # mark as dataframe op
        elif isinstance(op, GetItemOp) or isinstance(op, BaseEstimatorOp):
            op.is_dataframe_op = True

        elif isinstance(op, ChoiceOp):
            # check if all outcomes are dataframe ops
            if all(outcome.is_dataframe_op for outcome in op.inputs):
                op.is_dataframe_op = True

    if new_op is None:
        return root, False
    else:
        op.replace_input_of_outputs(new_op)
        if root is op:
            root = new_op
    return root, True


# Aggregation methods callable directly on a groupby (no .agg wrapper needed).
_AGG_METHODS = {"sum", "mean", "count", "min", "max", "median", "std", "var",
                "first", "last", "prod", "size", "nunique", "sem"}
# Generic aggregation entrypoints that take the aggregation spec as an argument.
_AGG_FUNCS = {"agg", "aggregate"}


def _is_groupby_op(op: Op) -> bool:
    return isinstance(op, MethodCallOp) and op.method_name == "groupby"


def _is_aggregation(op: MethodCallOp) -> bool:
    """True for a `groupby(...).<agg>()` pair that can fuse into an AggregateOp.

    Requires the aggregation to consume a `groupby` op directly (no GetItem or
    other op in between) and that groupby to have a single consumer.
    """
    if not op.inputs or not _is_groupby_op(op.inputs[0]):
        return False
    if len(op.inputs[0].outputs) != 1:
        return False
    if _extract_grouping(op.inputs[0]) is None:
        return False
    if op.method_name in _AGG_METHODS:
        return True
    # `.agg(spec)` / `.aggregate(spec)`: only the positional-spec form is supported.
    return op.method_name in _AGG_FUNCS and bool(op.args)


def _extract_grouping(groupby_op: MethodCallOp) -> str | list[str] | OperandRef:
    if groupby_op.args:
        return groupby_op.args[0]
    if groupby_op.kwargs and "by" in groupby_op.kwargs:
        return groupby_op.kwargs["by"]
    return None


def _extract_aggregations(op: MethodCallOp) -> str | list[str] | OperandRef:
    if op.method_name in _AGG_FUNCS:
        return op.args[0]
    # direct method such as .mean()/.sum()/.count() -> normalize to its name
    return op.method_name


def make_aggregate_op(op: MethodCallOp) -> AggregateOp:
    """Fuse `groupby(by).agg(...)` (or `.sum()/.mean()/...`) into an AggregateOp."""
    groupby_op = op.inputs[0]
    df = groupby_op.inputs[0]

    grouping_attributes = _extract_grouping(groupby_op)
    aggregations = _extract_aggregations(op)

    # Inputs in resolution order: the frame, then any placeholder operands of the
    # grouping key, then any placeholder operands of the aggregation spec.
    inputs = [df] + list(groupby_op.inputs[1:]) + list(op.inputs[1:])

    # OperandRefs in aggregations index into op.inputs. After prepending
    # groupby_op.inputs[1:], those refs need to shift by that slice's length.
    offset = len(groupby_op.inputs) - 1
    if isinstance(aggregations, OperandRef):
        aggregations = OperandRef(aggregations.k + offset)

    # All groupby kwargs except 'by', which is captured in grouping_attributes.
    groupby_kwargs = {k: v for k, v in (groupby_op.kwargs or {}).items() if k != "by"}

    new_op = AggregateOp(
        grouping_attributes=grouping_attributes,
        aggregations=aggregations,
        groupby_kwargs=groupby_kwargs,
        inputs=inputs,
        outputs=op.outputs,
    )
    # Bypass the now-orphaned groupby op: rewire the frame and grouping-key
    # producers, plus any aggregation-arg producers, to feed the new op.
    groupby_op.replace_output_of_inputs(new_op)
    for extra in op.inputs[1:]:
        extra.replace_output(op, new_op)
    groupby_op.outputs.remove(op)
    return new_op


def make_datetime_conversion_op(op: CallOp) -> DatetimeConversionOp:
    # arg[0] is the input
    if len(op.args) > 1:
        args = op.args[1:]
    else:
        args = ()

    new_op = DatetimeConversionOp(args=args, kwargs=op.kwargs, inputs=op.inputs, outputs=op.outputs)
    op.replace_output_of_inputs(new_op)
    return new_op


def make_read_op(op: CallOp, format: str = "csv") -> DataSourceOp:
    # assume all inputs are ValueOps or VariableOps
    assert all(isinstance(arg, ValueOp) or isinstance(arg, VariableOp) for arg in op.inputs), "All inputs must be ValueOps or VariableOps"
    # Rebuild a fresh, renumbered inputs list keeping only VariableOps as edges;
    # ValueOp operands are inlined as their constant value.
    inputs = []
    index = {}  # id(input op) -> new operand index

    def keep(input_op):
        i = index.get(id(input_op))
        if i is None:
            i = len(inputs)
            inputs.append(input_op)
            index[id(input_op)] = i
        return OperandRef(i)

    def convert(value):
        if isinstance(value, OperandRef):
            actual_input_op = op.inputs[value.k]
            if isinstance(actual_input_op, VariableOp):
                return keep(actual_input_op)
            return actual_input_op.value
        return value

    args = [convert(a) for a in op.args]
    kwargs = {k: convert(v) for k, v in op.kwargs.items()}
    new_op = DataSourceOp(file_path=args[0], _format=format, read_args=args[1:], read_kwargs=kwargs, inputs=inputs, outputs=op.outputs)
    for in_ in inputs:
        in_.replace_output(op, new_op)
    return new_op


_MERGE_POSITIONAL = ["how", "on", "left_on", "right_on",
                    "left_index", "right_index", "sort", "suffixes"]
_JOIN_POSITIONAL = ["on", "how", "lsuffix", "rsuffix", "sort"]
_JOIN_OP_FIELDS = {"how", "left_on", "right_on", "left_index", "right_index", "suffixes"}

def make_join_op(op: MethodCallOp) -> JoinOp:
    # First positional arg is the right/other df; it's already in op.inputs
    pos_args = op.args[1:] if op.args else ()
    pos_names = _MERGE_POSITIONAL if op.method_name == "merge" else _JOIN_POSITIONAL

    params = dict(zip(pos_names, pos_args))
    if op.kwargs:
        params.update(op.kwargs)
    params.pop("other", None)

    other_arg = op.args[0] if op.args else None
    if other_arg is None and op.kwargs:
        other_arg = op.kwargs.get("other")

    if isinstance(other_arg, (list, tuple)):
        # Compare by operand index: a frame used twice de-duplicates to one input
        # edge (the same OperandRef.k), which the chained-join unrolling can't handle.
        keys = [x.k if isinstance(x, OperandRef) else id(x) for x in other_arg]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "Duplicate right-hand frames in chained joins are not supported."
            )

    is_chained = (
        op.method_name == "join"
        and (isinstance(other_arg, (list, tuple)) or len(op.inputs) > 2)
    )

    if op.method_name == "join":
        # pandas .join() defaults to how="left" and matches against right's index.
        params.setdefault("how", "left")
        if is_chained:
            # Chained joins are always index-based on every link.
            params["left_index"] = True
            params["right_index"] = True
            params.pop("on", None)
            params.pop("left_on", None)
            params.pop("right_on", None)
        elif "on" in params:
            params["left_on"] = params.pop("on")
            params["left_index"] = False
            params["right_index"] = True
        else:
            params["left_index"] = True
            params["right_index"] = True
        # join uses lsuffix/rsuffix instead of suffixes; both default to "" in pandas.
        if "lsuffix" in params or "rsuffix" in params:
            params["suffixes"] = (params.pop("lsuffix", ""), params.pop("rsuffix", ""))
        params.setdefault("suffixes", ("", ""))
    else:
        # merge's `on` applies to both sides when left_on/right_on are unset.
        if "on" in params and "left_on" not in params and "right_on" not in params:
            shared = params.pop("on")
            params["left_on"] = shared
            params["right_on"] = shared
        else:
            params.pop("on", None)
        params.setdefault("suffixes", ("_x", "_y"))

    if params.pop("sort", False):
        raise NotImplementedError(
            "sort=True is not supported by JoinOp."
        )

    unsupported = [
        k for k in params
        if k not in _JOIN_OP_FIELDS and k not in ("right", "other")
    ]
    if unsupported:
        raise NotImplementedError(
            f"Unsupported arguments for {op.method_name}(): {', '.join(sorted(unsupported))}"
        )

    join_kwargs = {k: v for k, v in params.items() if k in _JOIN_OP_FIELDS}

    if is_chained:
        return _make_chained_join_op(op, join_kwargs)

    new_op = JoinOp(**join_kwargs, inputs=op.inputs, outputs=op.outputs)
    op.replace_output_of_inputs(new_op)
    return new_op

def _make_chained_join_op(op: MethodCallOp, join_kwargs: dict) -> JoinOp:
    """Unroll df1.join([df2, df3, ...]) into a chain of binary JoinOps."""
    dfs = op.inputs
    prev = dfs[0]
    final_join = None
    n_links = len(dfs) - 1
    for i, right in enumerate(dfs[1:]):
        is_last = i == n_links - 1
        join = JoinOp(**join_kwargs, inputs=[prev, right],
                      outputs=op.outputs if is_last else [])
        right.replace_output(op, join)
        if final_join is not None:
            final_join.outputs = [join]
        else:
            prev.replace_output(op, join)
        prev = join
        final_join = join
    return final_join

def make_frame_get_attr(new_op: GetAttrProjectionOp, op: GetAttrOp) -> GetAttrProjectionOp:
    input_ = op.inputs[0]
    if isinstance(input_, GetAttrProjectionOp):
        # Fuse chained GetAttr operations
        concat_attr_name = input_.attr_name.copy()
        attr_to_add = op.attr_name if isinstance(op.attr_name, list) else [op.attr_name]
        concat_attr_name.extend(attr_to_add)

        new_input = input_.inputs[0]
        new_op = GetAttrProjectionOp(attr_name=concat_attr_name, inputs=[new_input], outputs=op.outputs)

        if len(input_.outputs) > 1:
            input_.outputs.remove(op)
            new_input.add_output(new_op)
        else:
            new_input.replace_output(input_, new_op)

    else:
        # Convert single GetAttrOp to GetAttrDataframeOp
        attr_name = op.attr_name if isinstance(op.attr_name, list) else [op.attr_name]
        new_op = GetAttrProjectionOp(attr_name=attr_name, inputs=op.inputs, outputs=op.outputs)
        op.replace_output_of_inputs(new_op)
    return new_op


def group_dataframe_ops(root: Op) -> Op:
    return root
