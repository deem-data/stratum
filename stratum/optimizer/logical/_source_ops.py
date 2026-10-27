"""Data sources (the Source family): where data enters the plan.

:class:`InMemoryFrameOp` wraps a frame passed in as a value; the read ops stand
for a recorded ``pd.read_csv``, ``pd.read_parquet`` or ``np.load`` call.
Extraction binds the call against the reader's signature and keeps the common
options as fields. They carry pandas' names and values, since extraction assumes
the pipeline reads with pandas. An option without a field goes to
``pandas_options``, which only the pandas impl can run. The physical impls in
``physical/_source_execs.py`` translate the fields per library.
"""
from stratum.optimizer.logical._base import IndexEffect, has_default_labels
from stratum.optimizer.logical._ops import OperandRef, Op, OutputType, ValueOp, VariableOp, CallOp
from stratum.frontend._pandas_calls import bind_pandas_function
from stratum.optimizer.logical import _schema
import inspect
import numpy as np
import pandas as pd
import polars as pl
import logging

logger = logging.getLogger(__name__)

#: fsspec protocols served by gcsfs.
_GCS_PROTOCOLS = ("gs://", "gcs://")


def _is_gcs_path(file_path) -> bool:
    return isinstance(file_path, str) and file_path.lower().startswith(_GCS_PROTOCOLS)


class SourceOp(Op):
    """Abstract base of the source family. Impls register on the subclasses."""
    logical_family = "Source"
    # A reader numbers the rows it reads 0..n-1.
    index_effect = IndexEffect.DEFAULT


def _path_name(file_path) -> str | None:
    """A read is named by its path; a graph-fed one is only known at run time."""
    if file_path is None or isinstance(file_path, OperandRef):
        return None
    return str(file_path)


class InMemoryFrameOp(SourceOp):
    """A frame passed to the plan as a value, pandas or polars."""
    fields = ["data"]

    @property
    def index_effect(self):
        return IndexEffect.DEFAULT if has_default_labels(self.data) else IndexEffect.NEW

    def __init__(self, data=None, inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name="Frame", inputs=inputs, outputs=outputs)
        self.data = data
        self.output_type = OutputType.FRAME

    def propagate_output_schema(self):
        """Read off the frame itself."""
        self.output_schema = _schema.schema_of_frame(self.data)


class FileReadOp(SourceOp):
    """Abstract base of the pandas reads.

    ``file_path`` is a literal path, or an ``OperandRef`` to the ``VariableOp``
    that feeds it at run time. ``OPTIONS`` lists the reader options a subclass
    has a field for, each with the value that means it was not given.
    """
    OPTIONS: dict = {}

    def __init__(self, file_path=None, inputs: list[Op] = None, outputs: list[Op] = None,
                 pandas_options: dict | None = None, **options):
        super().__init__(name=_path_name(file_path), inputs=inputs, outputs=outputs)
        self.file_path = file_path
        unknown = sorted(options.keys() - self.OPTIONS.keys())
        if unknown:
            raise TypeError(f"{type(self).__name__} has no fields {unknown}")
        for name, unset in self.OPTIONS.items():
            setattr(self, name, options.get(name, unset))
        self.pandas_options = dict(pandas_options or {})
        self.output_type = OutputType.FRAME

    def given_options(self) -> dict:
        """The option fields given a value."""
        return {name: getattr(self, name) for name, unset in self.OPTIONS.items()
                if not _equals(getattr(self, name), unset)}


def _equals(value, default) -> bool:
    """Whether a field value is its default; graph-fed values never are."""
    if isinstance(value, OperandRef):
        return False
    try:
        return bool(value == default)
    except (TypeError, ValueError):  # array-likes compare elementwise
        return False


class ReadCSVOp(FileReadOp):
    """``pd.read_csv``. ``sep`` also holds a ``delimiter``; ``None`` (``"infer"``
    for ``header``) means the option was not given."""
    OPTIONS = {
        "usecols": None, "sep": None, "header": "infer", "names": None, "nrows": None,
        "skiprows": None, "na_values": None, "comment": None, "quotechar": None,
        "encoding": None, "low_memory": None, "storage_options": None,
    }
    fields = ["file_path", *OPTIONS, "pandas_options"]

    def propagate_output_schema(self):
        """The names in the file header, dtypes Unknown; unknown when they can't
        be read statically (graph-fed path) or only at a cost the optimizer
        shouldn't pay (a GCS path: the header probe is a network round-trip at
        plan time)."""
        if isinstance(self.file_path, OperandRef) or _is_gcs_path(self.file_path):
            self.output_schema = None
        else:
            self.output_schema = self._csv_header_schema()

    def _csv_header_schema(self):
        """Column names from the CSV header (``nrows=0``), dtypes left Unknown.

        Must pass this op's own options: several of them (``sep``, ``header``,
        ``names``, ``usecols``, ``index_col``, ...) change the resulting column
        set, so ignoring them would yield a confidently wrong schema rather than
        an unknown one. That is also why the probe is pandas and not
        ``pl.read_csv`` -- polars renames every one of those options and has no
        ``index_col`` at all, so translating would silently drift.

        Names are not necessarily strings (``header=None`` yields integer labels);
        the ``_schema`` helpers already fall back to unknown on those.
        """
        read_kwargs = pandas_read_kwargs(self)
        # A graph-fed option is still an OperandRef at plan time, so its value --
        # and with it the column set -- isn't knowable here.
        if any(isinstance(v, OperandRef) for v in read_kwargs.values()):
            return None
        read_kwargs.pop("nrows", None)
        try:
            names = pd.read_csv(self.file_path, nrows=0, **read_kwargs).columns
        except Exception:
            logger.debug("Could not derive schema for %s; falling back to unknown.", self.file_path)
            return None
        return pl.Schema({name: _schema.UNKNOWN_DTYPE for name in names})


class ReadParquetOp(FileReadOp):
    """``pd.read_parquet``. ``filters`` is pyarrow's: a list of ``(column, op,
    value)`` tuples, or a list of such lists. ``None`` means not given."""
    OPTIONS = {"columns": None, "filters": None, "storage_options": None}
    fields = ["file_path", *OPTIONS, "pandas_options"]


class LoadNumpyOp(SourceOp):
    """``np.load``: an ndarray (a MATRIX), not a frame. ``options`` are
    ``np.load``'s keyword arguments."""
    fields = ["file_path", "options"]

    def __init__(self, file_path=None, inputs: list[Op] = None, outputs: list[Op] = None,
                 options: dict | None = None):
        super().__init__(name=_path_name(file_path), inputs=inputs, outputs=outputs)
        self.file_path = file_path
        self.options = dict(options or {})
        self.output_type = OutputType.MATRIX


def pandas_read_kwargs(op: FileReadOp) -> dict:
    """The keyword arguments of the pandas reader call ``op`` stands for."""
    return {**op.given_options(), **op.pandas_options}


# --- Extraction: recorded read call -> source fields ---------------------------

# Readers recognised as data sources, with the op each becomes. Matched by
# identity (as `op.func is pd.read_csv` was), so a callable with an exotic
# `__eq__`/`__hash__` cannot confuse the lookup.
_READERS = (
    (pd.read_csv, ReadCSVOp),
    (pd.read_parquet, ReadParquetOp),
    (np.load, LoadNumpyOp),
)


def make_read_op(op: CallOp) -> FileReadOp | LoadNumpyOp | None:
    """Parse a reader call into its source op.

    ``op``'s inputs are ``ValueOp``s, whose values are inlined, and
    ``VariableOp``s, which stay graph inputs. Returns ``None`` (the call stays a
    ``CallOp``) when the reader would reject the arguments.
    """
    # assume all inputs are ValueOps or VariableOps
    assert all(isinstance(arg, ValueOp) or isinstance(arg, VariableOp) for arg in op.inputs), "All inputs must be ValueOps or VariableOps"
    reader, cls = next((func, cls) for func, cls in _READERS if op.func is func)
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

    call = bind_pandas_function(reader, [convert(a) for a in op.args],
                                {k: convert(v) for k, v in op.kwargs.items()})
    if call is None:
        return None
    file_path = call.pop(next(iter(inspect.signature(reader).parameters)))
    if cls is LoadNumpyOp:
        new_op = LoadNumpyOp(file_path=file_path, options=call,
                             inputs=inputs, outputs=op.outputs)
    else:
        new_op = cls(file_path=file_path, **_split_options(cls, call),
                     inputs=inputs, outputs=op.outputs)
    for in_ in inputs:
        in_.replace_output(op, new_op)
    return new_op


def _split_options(cls, call: dict) -> dict:
    """The bound reader call's options as ``cls``'s fields, plus ``pandas_options``."""
    options, other = {}, dict(call.pop("kwargs", {}))
    for name, value in call.items():
        option = "sep" if name == "delimiter" else name
        # A second spelling of an option (sep next to delimiter) stays pandas',
        # which rejects it, and so does a separator pandas is to sniff.
        if option not in cls.OPTIONS or option in options or (option == "sep" and value is None):
            other[name] = value
        else:
            options[option] = value
    return {**options, "pandas_options": other}


def try_make_read_op(op: Op) -> FileReadOp | LoadNumpyOp | None:
    """Rewrite a call to a supported reader into its source op.

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
    if not any(op.func is func for func, _ in _READERS) or not op.args:
        return None
    return make_read_op(op)
