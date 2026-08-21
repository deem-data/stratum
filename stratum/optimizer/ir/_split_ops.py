from stratum.optimizer.ir._ops import OutputType, Op
from stratum.optimizer._op_utils import topological_iterator
from stratum.utils._utils import start_time, log_time
import pandas as pd
import polars as pl
import numpy as np


class SplitOp(Op):
    logical_family = "Split"

    def __init__(self, inputs: list[Op]=None, outputs: list[Op]=None):
        super().__init__(name="Train/Test", is_X=False, is_y=False, inputs=inputs, outputs=outputs)
        self.is_split_op = True
        self.output_type = OutputType.FRAME
        self.indices = None

    def propagate_output_schema(self):
        """A split is a structural ``(X, y)`` fan-out, not a single frame: it has
        no schema of its own. The consuming :class:`SplitOutput`s read their
        schema from this op's inputs directly."""
        self.output_schema = None

    def process(self, mode: str, inputs: list):
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
        self.output_type = OutputType.FRAME

    def propagate_output_schema(self):
        # Subsetting rows keeps the columns: take the schema of the matching
        # SplitOp input (inputs[0] = X, inputs[1] = y, per add_splitting_op).
        split_op = self.inputs[0]
        src = split_op.inputs[0 if self.is_x else 1]
        self.output_schema = src.output_schema

    def process(self, mode: str, inputs: list):
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
