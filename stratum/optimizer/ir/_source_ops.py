from stratum.optimizer.ir._ops import OperandRef, Op, OutputType, ValueOp, VariableOp, CallOp
from stratum.optimizer.ir import _schema
from pandas import DataFrame
import numpy as np
import pandas as pd
import polars as pl
import logging

logger = logging.getLogger(__name__)


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

    def propagate_output_schema(self):
        """Source schema: read from the in-memory frame, or the file header.

        Column *names* are authoritative; *dtypes* are only kept when they are
        statically certain. A pandas ``object`` column has no column-level element
        type (it must be inferred by scanning values, which can be confidently
        wrong -- e.g. ints in early rows, strings later), and a CSV column's dtype
        needs a full-file scan, so in both cases we keep the name but mark the
        dtype ``Unknown``. Falls back to the unknown schema (``None``) when even
        the names can't be read statically (a graph-fed path, an npy matrix, or an
        unreadable file)."""
        if self.data is not None:
            if isinstance(self.data, pl.DataFrame):
                # polars frames already carry exact dtypes.
                self.output_schema = self.data.schema
            else:
                # head(0) converts names + typed dtypes without copying data or
                # risking a mixed-object conversion error. An object column is
                # element-typed only by scanning, so keep its name with an Unknown
                # dtype rather than a guessed one. An exotic/extension dtype can
                # still make the conversion fail, so fall back to unknown then.
                try:
                    schema = pl.from_pandas(self.data.head(0)).schema
                    self.output_schema = pl.Schema({
                        name: (_schema.UNKNOWN_DTYPE if pd.api.types.is_object_dtype(self.data[name]) else dt)
                        for name, dt in schema.items()
                    })
                except Exception:
                    logger.debug("Could not derive schema for in-memory frame; falling back to unknown.")
                    self.output_schema = None
            return

        if isinstance(self.file_path, OperandRef) or self.format != "csv":
            self.output_schema = None
            return
        try:
            # Read only the header (no rows): names are exact, dtypes need a full
            # scan to be safe, so leave them Unknown.
            names = pl.read_csv(self.file_path, n_rows=0).columns
            self.output_schema = pl.Schema({name: _schema.UNKNOWN_DTYPE for name in names})
        except Exception:
            logger.debug("Could not derive schema for %s; falling back to unknown.", self.file_path)
            self.output_schema = None

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
