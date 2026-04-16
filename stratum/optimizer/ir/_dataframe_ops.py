from stratum.optimizer.ir._ops import (DATA_OP_PLACEHOLDER, BaseEstimatorOp, BinOp, CallOp, GetAttrOp, GetItemOp,
                                       MethodCallOp, Op, ValueOp, VariableOp,_resolve_args, _resolve_kwargs)
from pandas import DataFrame
import pandas as pd
import polars as pl
from stratum.optimizer._op_utils import topological_iterator
from stratum._config import FLAGS
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
        self.is_dataframe_op = True

    def process(self, mode: str, environment: dict, inputs: list):
        if self.data is not None:
            if FLAGS.force_polars:
                return pl.DataFrame(self.data)
            else:
                return self.data
        else:
            file_path = inputs[0] if self.file_path is DATA_OP_PLACEHOLDER else self.file_path
            if FLAGS.force_polars:
                return pl.read_csv(file_path, *self.read_args, **self.read_kwargs)
            else:
                return pd.read_csv(file_path, *self.read_args, **self.read_kwargs)

    def clone(self):
        raise ValueError(f"We should not clone DataSourceOp objects.")

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
        input_iter = iter(inputs)
        _obj = next(input_iter)
        _args = _resolve_args(self.args, input_iter)
        _kwargs = _resolve_kwargs(self.kwargs, input_iter)
        if FLAGS.force_polars:
            if "columns" in _kwargs:
                _args.append(_kwargs["columns"])
            return getattr(_obj, self.func)(*_args)
        else:
            return getattr(_obj, self.func)(*_args, **_kwargs)

class ProjectionOp(Op):
    fields = ["func", "is_method", "args", "kwargs", "columns"]
    
    def __init__(self, func="", is_method: bool = True, args: tuple | list = None, kwargs: dict = None,
        inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None):
        super().__init__(name=func.upper() if is_method else f"{func.__name__.upper()}", inputs=inputs, outputs=outputs)
        if kwargs is not None:
            self.check_kwargs(kwargs)
        self.is_method = is_method
        self.func = func
        self.args = args
        self.columns = columns
        self.kwargs = kwargs
        self.is_dataframe_op = True

    def _extract_args_and_kwargs(self, inputs: list):
        """Extract and process arguments and kwargs from inputs."""
        input_iter, args_iter = iter(inputs), iter(self.args)
        _obj = next(input_iter)
        if not self.is_method:
            next(args_iter)
        _args = _resolve_args(args_iter, input_iter)
        _kwargs = _resolve_kwargs(self.kwargs, input_iter)
        return _obj, _args, _kwargs

    def process(self, mode: str, environment: dict, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        if self.is_method:
            if FLAGS.force_polars:
                raise ValueError(f"Unsupported method: {self.func}")
            else:
                return getattr(_obj, self.func)(*_args, **_kwargs)
        else:
            return self.func(_obj, *_args, **_kwargs)

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
                if v is DATA_OP_PLACEHOLDER:
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
    def __init__(self, first: Op, others: list[Op], axis: int):
        super().__init__(name="CONCAT", is_X=False, is_y=False)
        self.first = DATA_OP_PLACEHOLDER if isinstance(first, DataOp) else first
        self.others = [DATA_OP_PLACEHOLDER if isinstance(other, DataOp) else other for other in others]
        self.axis = DATA_OP_PLACEHOLDER if isinstance(axis, DataOp) else axis
        self.is_dataframe_op = True

    def process(self, mode: str, environment: dict, inputs: list):
        input_iter = iter(inputs)
        first = next(input_iter) if self.first is DATA_OP_PLACEHOLDER else self.first
        others = [next(input_iter) if other is DATA_OP_PLACEHOLDER else other for other in self.others]
        axis = next(input_iter) if self.axis is DATA_OP_PLACEHOLDER else self.axis
        if FLAGS.force_polars:
            return pl.concat([first, *others], how=self.axis_map[axis])
        else:
            return pd.concat([first, *others], axis=axis)


def rewrite_fuse_get_item_ops(op: Op) -> Op:
    pass
    # obj = op.inputs[0]
    # fuse GetItem and Projection
    # if isinstance(obj, GetItemOp) and not obj.is_X and not obj.is_y:
    #     op.inputs[0] = op.inputs[0].inputs[0]
    #     new_op = ProjectionOp(func=op.method_name, args=op.args, kwargs=op.kwargs, inputs=op.inputs, outputs=op.outputs, columns=obj.key, is_X=op.is_X, is_y=op.is_y)
    #     # remove GetItem --> Projection connection
    #     obj.outputs.remove(op)
    #     # check if GetItem has other outputs, if not remove op
    #     if len(obj.outputs) == 0:
    #         obj.replace_output_of_inputs(new_op)
    #     else:
    #         # append new_op to all GetItem's inputs
    #         for in_ in obj.inputs:
    #             in_.add_output(new_op)
    #     # set the output of all new op's inputs correctly
    #     for in_ in new_op.inputs[1:]:
    #         in_.replace_output(op, new_op)

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
                new_op = make_read_op(new_op, op)

    # input is a dataframe op
    else:
        if isinstance(op, CallOp):
            # Datetime conversion detection
            if op.func is pd.to_datetime:
                new_op = make_datetime_conversion_op(new_op, op)

        elif isinstance(op, MethodCallOp):
            if op.method_name in ["rename"]:
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

        # GetAttr Fusing and conversion to GetAttrDataframeOp
        elif isinstance(op, GetAttrOp) and op.inputs[0].is_dataframe_op:
            new_op = make_frame_get_attr(new_op, op)

        # Projection: BinOp detection
        elif isinstance(op, BinOp) and op.inputs[0].is_dataframe_op:
            op.is_dataframe_op = True

        # mark as dataframe op
        elif isinstance(op, GetItemOp) or isinstance(op, BaseEstimatorOp):
            op.is_dataframe_op = True

    if new_op is None:
        return root, False
    else:
        op.replace_input_of_outputs(new_op)
        if root is op:
            root = new_op
    return root, True


def make_datetime_conversion_op(new_op: DatetimeConversionOp, op: CallOp) -> DatetimeConversionOp:
    # arg[0] is the input
    if len(op.args) > 1:
        args = op.args[1:]
    else:
        args = ()

    new_op = DatetimeConversionOp(args=args, kwargs=op.kwargs, inputs=op.inputs, outputs=op.outputs)
    op.replace_output_of_inputs(new_op)
    return new_op


def make_read_op(new_op: DataSourceOp, op: CallOp) -> DataSourceOp:
    input_iter = iter(op.inputs)
    # assume all inputs are ValueOps
    assert all(isinstance(arg, ValueOp) or isinstance(arg, VariableOp) for arg in op.inputs), "All inputs must be ValueOps or VariableOps"
    inputs = []
    args = []
    for arg in op.args:
        if arg is DATA_OP_PLACEHOLDER:
            actual_input_op = next(input_iter)
            if isinstance(actual_input_op, VariableOp):
                args.append(DATA_OP_PLACEHOLDER)
                inputs.append(actual_input_op)
            else:
                args.append(actual_input_op.value)
        else:
            args.append(arg)
    kwargs = {}
    for k, v in op.kwargs.items():
        if v is DATA_OP_PLACEHOLDER:
            actual_input_op = next(input_iter)
            if isinstance(actual_input_op, VariableOp):
                kwargs[k] = DATA_OP_PLACEHOLDER
                inputs.append(actual_input_op)
            else:
                kwargs[k] = actual_input_op.value
        else:
            kwargs[k] = v
    new_op = DataSourceOp(file_path=args[0], _format="csv", read_args=args[1:], read_kwargs=kwargs, inputs=inputs, outputs=op.outputs)
    for in_ in inputs:
        in_.replace_output(op, new_op)
    return new_op


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