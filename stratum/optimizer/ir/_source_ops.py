from stratum.optimizer.ir._ops import (OperandRef, Op, ValueOp, VariableOp, CallOp,
                                       _resolve_args, _resolve_kwargs)
from stratum.optimizer.ir._ops import OutputType
from pandas import DataFrame
import pandas as pd
import polars as pl
import numpy as np
from stratum._config import FLAGS

def rechunk_pl_frame(df, rows_per_chunk = 128_000):
    n = len(df)
    if rows_per_chunk <= 0 or n <= rows_per_chunk:
        return df
    parts = [df.slice(i, rows_per_chunk) for i in range(0, n, rows_per_chunk)]
    return pl.concat(parts, rechunk=False)

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
        # A directly-passed DataFrame or a csv read is a FRAME; np.load yields an
        # ndarray, so an npy source is a MATRIX.
        self.output_type = OutputType.MATRIX if _format == "npy" else OutputType.FRAME

    def process(self, mode: str, inputs: list):
        if self.data is not None:
            if FLAGS.force_polars:
                out = pl.DataFrame(self.data)
                return rechunk_pl_frame(out) if FLAGS.rechunk else out
            else:
                return self.data
        else:
            file_path = inputs[self.file_path.k] if isinstance(self.file_path, OperandRef) else self.file_path
            read_args = _resolve_args(self.read_args, inputs) if self.read_args else []
            read_kwargs = _resolve_kwargs(self.read_kwargs, inputs) if self.read_kwargs else {}
            if FLAGS.force_polars:
                if self.format == "parquet":
                    return pl.read_parquet(file_path, *read_args, **read_kwargs)
                return pl.read_csv(file_path, *read_args, **read_kwargs)
            else:
                if self.format == "csv":
                    return pd.read_csv(file_path, *read_args, **read_kwargs)
                elif self.format == "parquet":
                    return pd.read_parquet(file_path, *read_args, **read_kwargs)
                elif self.format == "npy":
                    return np.load(file_path, *read_args, **read_kwargs)
                else:
                    raise ValueError(f"Unsupported format: {self.format}")

    def clone(self):
        raise ValueError(f"We should not clone DataSourceOp objects.")


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
