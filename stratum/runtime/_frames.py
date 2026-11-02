"""Runtime values: a frame or a series, with what stratum knows about it.

A :class:`StratumFrame` or :class:`StratumSeries` wraps a pandas or polars value.
Its :attr:`~StratumValue.index` is the pandas index the value has in the pandas
plan: a pandas value's own, and for a polars value a :class:`PolarsIndex` held
beside it, never in its columns, or ``None`` for ``0..n-1``.

The scheduler wraps and unwraps values around ``process``, so kernels see plain
frames; only the ops that handle a polars value's index see the wrapper (see
``PhysicalOp.process_indexed``). A value with nothing to carry beyond itself (a
pandas value, a polars one whose index is not tracked) stays plain: implementation
selection tracks a polars index only where an op uses it.

The library conversions live here too, as the values they convert are runtime
values (``ConversionOp`` runs them).
"""
from __future__ import annotations

import pandas as pd
import polars as pl


def to_polars(value):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return pl.from_pandas(value)
    return value


def to_pandas(value):
    """polars -> pandas via Arrow defaults: dates as ``datetime.date``, strings as ``str``."""
    if isinstance(value, pl.Series):
        return to_pandas(value.to_frame()).iloc[:, 0]
    if isinstance(value, pl.DataFrame):
        frame = value.to_arrow().to_pandas()
        strings = [name for name, dtype in value.schema.items() if dtype == pl.String]
        return frame.astype({name: "str" for name in strings}) if strings else frame
    return value


class PolarsIndex:
    """The pandas index of a polars value.

    Held as pandas' own index (a pandas value converted to polars), or as polars
    columns of the index levels' names (an aggregation's group keys), which
    convert to it as a frame's columns would: the pandas form is built from them
    when first asked for, the polars form is not built from pandas.
    """

    __slots__ = ("_pandas", "_columns")

    def __init__(self, pandas: pd.Index | None = None, columns: pl.DataFrame | None = None):
        if pandas is None and columns is None:
            raise ValueError("a PolarsIndex holds a pandas index or polars columns")
        self._pandas = pandas
        self._columns = columns

    @classmethod
    def of(cls, index: pd.Index) -> PolarsIndex | None:
        """``index`` as a polars value's, or ``None`` where it is ``0..n-1``."""
        if isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1 \
                and index.name is None:
            return None
        return cls(pandas=index)

    @property
    def pandas(self) -> pd.Index | None:
        """The pandas index, if it is held so (see :meth:`to_pandas`)."""
        return self._pandas

    @property
    def columns(self) -> pl.DataFrame | None:
        """The index levels as polars columns, if it is held so."""
        return self._columns

    def __len__(self) -> int:
        return len(self._pandas) if self._pandas is not None else self._columns.height

    def to_pandas(self) -> pd.Index:
        if self._pandas is None:
            frame = to_pandas(self._columns)
            self._pandas = frame.set_index(list(frame.columns)).index
        return self._pandas

    def take(self, rows: pl.Series) -> PolarsIndex:
        """The index of the rows at positions ``rows``, in their order."""
        return PolarsIndex(
            None if self._pandas is None else self._pandas.take(rows.to_numpy()),
            None if self._columns is None else self._columns[rows])


def take_index(index: PolarsIndex | None, rows: pl.Series) -> PolarsIndex:
    """``index`` of the rows at positions ``rows``; ``None`` reads as ``0..n-1``."""
    if index is None:
        return PolarsIndex(pandas=pd.Index(rows.to_numpy().astype("int64")))
    return index.take(rows)


def row_numbers(height: int) -> pl.Series:
    """``0..height-1``: run a row-moving kernel on them to find where rows go."""
    return pl.int_range(0, height, dtype=pl.UInt32, eager=True)


def unused_name(columns) -> str:
    """A column name none of ``columns`` has, for a column a kernel adds and drops."""
    name = "__stratum_rows"
    while name in columns:
        name += "_"
    return name


def with_row_positions(frame: pl.DataFrame, kernel):
    """``kernel(frame)`` and the input positions of its rows, in its order.

    ``kernel`` takes the frame with a column of row numbers beside the others,
    and that column's name; it must read none but the frame's own columns.
    """
    name = unused_name(frame.columns)
    result = kernel(frame.with_row_index(name), name)
    return result.drop(name), result.get_column(name)


class StratumValue:
    """A frame or a series, with its pandas index. See the module docstring."""

    __slots__ = ("data", "polars_index")

    def __init__(self, data, polars_index: PolarsIndex | None = None):
        if isinstance(data, StratumValue):
            raise TypeError("a StratumValue does not wrap another")
        if polars_index is not None and not _is_polars(data):
            raise TypeError("only a polars value holds its index beside it")
        self.data = data
        #: A polars value's pandas index, or ``None`` for ``0..n-1``.
        self.polars_index = polars_index

    @property
    def index(self) -> pd.Index:
        """The pandas index."""
        if not _is_polars(self.data):
            return self.data.index
        if self.polars_index is None:
            return pd.RangeIndex(len(self.data))
        return self.polars_index.to_pandas()

    def __repr__(self) -> str:
        held = "" if self.polars_index is None else ", with its index"
        return f"{type(self).__name__}({type(self.data).__module__.split('.')[0]}{held})"


class StratumFrame(StratumValue):
    __slots__ = ()


class StratumSeries(StratumValue):
    __slots__ = ()


def _is_polars(value) -> bool:
    return isinstance(value, (pl.DataFrame, pl.Series))


def wrap(value, polars_index: PolarsIndex | None = None):
    """``value`` as the scheduler holds it: wrapped where it carries an index of
    its own beside it, else plain."""
    if polars_index is None or isinstance(value, StratumValue) or not _is_polars(value):
        return value
    cls = StratumSeries if isinstance(value, pl.Series) else StratumFrame
    return cls(value, polars_index)


def unwrap(value):
    """The plain frame or series of a wrapped value; any other value as is."""
    return value.data if isinstance(value, StratumValue) else value


def polars_index_of(value) -> PolarsIndex | None:
    """The index held beside a polars value, ``None`` where it is ``0..n-1``."""
    return value.polars_index if isinstance(value, StratumValue) else None


def as_result(value):
    """A pool value as the plan returns it: plain, a polars frame with an index
    held in polars columns leading them, as pandas' ``reset_index`` would."""
    index = polars_index_of(value)
    data = unwrap(value)
    if index is None or index.columns is None or not isinstance(data, pl.DataFrame):
        return data
    return pl.concat([index.columns, data], how="horizontal")


def execute(op, mode: str, inputs: list):
    """``op``'s output and the index its polars value holds, given pool values:
    ``process`` on plain values, unless ``op`` handles a polars value's index
    (``PhysicalOp.process_indexed``)."""
    if op._indexed:
        return op.process_indexed(mode, inputs)
    return op.process(mode=mode, inputs=[unwrap(value) for value in inputs]), None
