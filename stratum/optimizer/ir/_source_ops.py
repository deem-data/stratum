from stratum.optimizer.ir._ops import OperandRef, Op, OutputType, ValueOp, VariableOp, CallOp
from pandas import DataFrame
import numpy as np
import pandas as pd


class DataSourceOp(Op):
    """Logical data source: an already-materialised frame or a file read.

    Pure plan-time data -- it carries what to read (or the frame itself) but has
    no ``process``: lowering always rewrites it into a physical source op
    (``ReadCSV``/``ReadParquet``/``InMemoryFrame``/``NumpyLoad`` in
    ``physical/_source_execs.py``), whose selected backend impl does the work.
    """
    logical_family = "Source"

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


# Reader functions recognised as data sources, paired with the source format they
# produce. Matched by identity (as `op.func is pd.read_csv` was), so a callable
# with an exotic `__eq__`/`__hash__` cannot confuse the lookup.
_READ_FORMATS = (
    (pd.read_csv, "csv"),
    (pd.read_parquet, "parquet"),
    (np.load, "npy"),
)


def try_make_read_op(op: Op) -> DataSourceOp | None:
    """Rewrite a call to a supported reader into a :class:`DataSourceOp`.

    Covers both spellings of a read step, which differ only in whether the path
    reaches the call as an operand or as a plain literal::

        X.skb.apply_func(pd.read_csv)        # == skrub.deferred(pd.read_csv)(X)
        skrub.deferred(pd.read_csv)(path)    # path is a literal -> no operands

    Returns ``None`` (leaving a plain ``CallOp``) when ``op`` is not a call to a
    known reader, or when the path is not the first positional argument -- e.g.
    ``skrub.deferred(pd.read_csv)(filepath_or_buffer=path)``, whose keyword name
    differs per reader.
    """
    if not isinstance(op, CallOp):
        return None
    fmt = next((fmt for func, fmt in _READ_FORMATS if op.func is func), None)
    if fmt is None or not op.args:
        return None
    return make_read_op(op, fmt)
