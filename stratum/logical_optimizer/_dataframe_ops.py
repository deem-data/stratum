from stratum.logical_optimizer._ops import DATA_OP_PLACEHOLDER, BinOp, CallOp, GetAttrOp, GetItemOp, MethodCallOp, Op, ValueOp
from pandas import DataFrame
import pandas as pd
from stratum.logical_optimizer._op_utils import topological_iterator


class DataSourceOp(Op):
    def __init__(self, data: DataFrame = None, file_path: str = None, _format: str = None,
                 read_args: tuple | list = None, read_kwargs: dict = None, is_X=False, is_y=False, outputs: list[Op] = None):
        if outputs is None:
            outputs = []
        super().__init__(name="Frame" if data is not None else f"read_{_format}", is_X=is_X, is_y=is_y, outputs=outputs, inputs=None)
        if read_kwargs is not None:
            self.check_kwargs(read_kwargs)
        self.data = data
        self.format = _format
        self.file_path = file_path
        self.read_args = read_args
        self.read_kwargs = read_kwargs
        self.is_dataframe_op = True

    def process(self, mode: str, environment: dict):
        self.intermediate = self.data if self.data is not None else pd.read_csv(self.file_path, *self.read_args, **self.read_kwargs)

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

    def process(self, mode: str, environment: dict):
        iter_ins = iter(self.inputs)
        _obj = next(iter_ins).intermediate
        _args = [next(iter_ins).intermediate if arg is DATA_OP_PLACEHOLDER else arg for arg in self.args]
        _kwargs = {k: next(iter_ins).intermediate if v is DATA_OP_PLACEHOLDER else v for k, v in self.kwargs.items()}
        self.intermediate = getattr(_obj, self.func)(*_args, **_kwargs)

class ProjectionOp(Op):
    fields = ["func", "args", "kwargs", "columns"]
    
    def __init__(self, func, is_method: bool = True, args: tuple | list = None, kwargs: dict = None, 
        inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None, is_X=False, is_y=False):
        super().__init__(name=func.upper() if is_method else f"{func.__name__.upper()}", is_X=is_X, is_y=is_y, inputs=inputs, outputs=outputs)
        if kwargs is not None:
            self.check_kwargs(kwargs)
        self.is_method = is_method
        self.func = func
        self.args = args
        self.columns = columns
        self.kwargs = kwargs
        self.is_dataframe_op = True

    def process(self, mode: str, environment: dict):
        iter_ins, args_iter = iter(self.inputs), iter(self.args)
        _obj = next(iter_ins).intermediate
        if not self.is_method:
            next(args_iter)
        _args = [next(iter_ins).intermediate if arg is DATA_OP_PLACEHOLDER else arg for arg in args_iter]
        _kwargs = {k: next(iter_ins).intermediate if v is DATA_OP_PLACEHOLDER else v for k, v in self.kwargs.items()}
        if self.columns:
            _obj = _obj[self.columns]
        self.intermediate = getattr(_obj, self.func)(*_args, **_kwargs) if self.is_method else self.func(_obj, *_args, **_kwargs)


class SplitOp(Op):
    def __init__(self, inputs: list[Op]=None, outputs: list[Op]=None):
        super().__init__(name="Train/Test", is_X=False, is_y=False, inputs=inputs, outputs=outputs)
        self.is_split_op = True
        self.is_dataframe_op = True
        self.indices = None

    def process(self, mode: str, environment: dict):
        self.intermediate = (self.inputs[0].intermediate.iloc[self.indices], self.inputs[1].intermediate.iloc[self.indices])

class SplitOutput(Op):
    def __init__(self, inputs: list[Op]=None, outputs: list[Op]=None, is_x = True, ):
        name = "X" if is_x else "y"
        super().__init__(name=name, is_X=False, is_y=False, inputs=inputs, outputs=outputs)
        self.is_x = is_x
        self.is_dataframe_op = True

    def process(self, mode: str, environment: dict):
        if self.is_x:
            self.intermediate = self.inputs[0].intermediate[0]
        else:
            self.intermediate = self.inputs[0].intermediate[1]

def add_splitting_op(sink: Op) -> Op:
    x_op = None
    y_op = None
    for op in topological_iterator(sink):
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
    return sink

def rewrite_dataframe_ops(sink: Op) -> Op:
    """ Rewrite the dataframe ops in the dag to the new dataframe ops."""
    for op in topological_iterator(sink):
        new_op = None

        # DataSource detection
        if isinstance(op, CallOp):
            if op.func is pd.read_csv:
                input_iter = iter(op.inputs)
                # assume all inputs are ValueOps
                assert all(isinstance(arg, ValueOp) for arg in op.inputs), "All inputs must be ValueOps"
                args = [next(input_iter).value if arg is DATA_OP_PLACEHOLDER else arg for arg in op.args]
                kwargs = {k: next(input_iter).value if v is DATA_OP_PLACEHOLDER else v for k, v in op.kwargs.items()}
                new_op = DataSourceOp(file_path=args[0], _format = "csv", read_args=args[1:], read_kwargs=kwargs, is_X=op.is_X, is_y=op.is_y, outputs=op.outputs)
                op.replace_input_of_outputs(new_op)
            elif op.inputs[0].is_dataframe_op and op.func is pd.to_datetime:
                new_op = ProjectionOp(func=op.func, is_method=False, args=op.args, kwargs=op.kwargs, inputs=op.inputs, outputs=op.outputs, is_X=op.is_X, is_y=op.is_y)
                op.replace_output_of_inputs(new_op)
                op.replace_input_of_outputs(new_op)
        elif isinstance(op, ValueOp) and isinstance(op.value, DataFrame):
            new_op = DataSourceOp(data=op.value, is_X=op.is_X, is_y=op.is_y, outputs=op.outputs)
            op.replace_input_of_outputs(new_op)

        # Projection detection
        elif isinstance(op, MethodCallOp) and op.inputs[0].is_dataframe_op:
            if op.method_name in ["rename"]:
                new_op = MetadataOp(func=op.method_name, args=op.args, kwargs=op.kwargs, inputs=op.inputs, outputs=op.outputs, is_X=op.is_X, is_y=op.is_y)
                op.replace_output_of_inputs(new_op)
                op.replace_input_of_outputs(new_op)
            elif op.method_name in ["drop", "apply", "assign"]:
                obj = op.inputs[0]
                # fuse GetItem and Projection
                if isinstance(obj, GetItemOp) and not obj.is_X and not obj.is_y:
                    op.inputs[0] = op.inputs[0].inputs[0]
                    new_op = ProjectionOp(func=op.method_name, args=op.args, kwargs=op.kwargs, inputs=op.inputs, outputs=op.outputs, columns=obj.key, is_X=op.is_X, is_y=op.is_y)
                    # remove GetItem --> Projection connection
                    obj.outputs.remove(op)
                    # check if GetItem has other outputs, if not remove op
                    if len(obj.outputs) == 0:
                        obj.replace_output_of_inputs(new_op)
                    else:
                        # append new_op to all GetItem's inputs
                        for in_ in obj.inputs:
                            in_.add_output(new_op)
                    # set the output of all new op's inputs correctly
                    for in_ in new_op.inputs[1:]:
                        in_.replace_output(op, new_op)
                    
                else:
                    new_op = ProjectionOp(func=op.method_name, args=op.args, kwargs=op.kwargs, inputs=op.inputs, outputs=op.outputs, is_X=op.is_X, is_y=op.is_y)
                    op.replace_output_of_inputs(new_op)
                op.replace_input_of_outputs(new_op)

        # mark as dataframe op
        elif isinstance(op, GetItemOp) and op.inputs[0].is_dataframe_op:
            op.is_dataframe_op = True

        # GetAttr Fusing
        elif isinstance(op, GetAttrOp) and op.inputs[0].is_dataframe_op:
            input_ = op.inputs[0]
            op.is_dataframe_op = True
            if isinstance(input_, GetAttrOp):
                concat_attr_name = input_.attr_name.copy()
                concat_attr_name.append(op.attr_name)
                op.attr_name = concat_attr_name
                new_input = input_.inputs[0]
                op.inputs[0] = new_input
                if len(input_.outputs) > 1:
                    input_.outputs.remove(op)
                    new_input.add_output(op)
                else:
                    new_input.replace_output(input_, op)
                    del input_
                
            else:
                op.attr_name = [op.attr_name]

        # Projection: BinOp detection
        elif isinstance(op, BinOp) and op.inputs[0].is_dataframe_op:
            op.is_dataframe_op = True

        if sink is op and new_op is not None:
            sink = new_op

    return sink