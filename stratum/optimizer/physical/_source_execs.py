"""Physical source operators: the concrete read / in-memory-frame impls.

Lowering turns a logical :class:`~stratum.optimizer.ir._source_ops.DataSourceOp`
into an *abstract* source op (``ReadCSV``, ``ReadParquet``, ``InMemoryFrame``, or
the already-concrete ``NumpyLoad``). Implementation selection then swaps each
abstract op to one of the backend-specific concrete classes registered below via
``@physical_impl`` (``PandasReadCSV`` / ``PolarsReadCSV`` / ...). The concrete
``process`` methods contain no ``force_polars`` / ``rechunk`` branch -- the
backend and rechunk decision were fixed at plan time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from stratum.optimizer.ir._base import (OperandRef, OutputType, _resolve_args,
                                        _resolve_kwargs)
from stratum.optimizer.ir._source_ops import DataSourceOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._lowering import lowering_rule
from stratum.optimizer.physical._registry import physical_impl


def rechunk_pl_frame(df, rows_per_chunk=128_000):
    n = len(df)
    if rows_per_chunk <= 0 or n <= rows_per_chunk:
        return df
    parts = [df.slice(i, rows_per_chunk) for i in range(0, n, rows_per_chunk)]
    return pl.concat(parts, rechunk=False)


class FileReadOp(PhysicalOp):
    """Abstract read-from-file source. Concrete subclasses pick the backend reader."""
    is_abstract = True
    format: str | None = None

    def __init__(self, file_path=None, read_args=None, read_kwargs=None):
        # No name: the concrete class (PandasReadCSV, ...) already encodes backend
        # and format, so a "read_csv" name would just double it in the plan.
        super().__init__(name=file_path if file_path else None)
        # file_path is an OperandRef when graph-fed (e.g. a variable), else a literal.
        self.file_path = file_path
        self.read_args = read_args
        self.read_kwargs = read_kwargs
        self.output_type = OutputType.FRAME

    def _resolve(self, inputs):
        """Resolve the (possibly graph-fed) path and read args/kwargs from inputs."""
        file_path = inputs[self.file_path.k] if isinstance(self.file_path, OperandRef) else self.file_path
        read_args = _resolve_args(self.read_args, inputs) if self.read_args else []
        read_kwargs = _resolve_kwargs(self.read_kwargs, inputs) if self.read_kwargs else {}
        return file_path, read_args, read_kwargs


class ReadCSV(FileReadOp):
    is_abstract = True
    format = "csv"


@physical_impl(of=ReadCSV, backend="pandas", input_format="value", output_format="frame")
class PandasReadCSV(ReadCSV):
    is_abstract = False

    def process(self, mode: str, inputs: list):
        file_path, read_args, read_kwargs = self._resolve(inputs)
        return pd.read_csv(file_path, *read_args, **read_kwargs)


@physical_impl(of=ReadCSV, backend="polars", input_format="value", output_format="frame")
class PolarsReadCSV(ReadCSV):
    is_abstract = False

    def process(self, mode: str, inputs: list):
        file_path, read_args, read_kwargs = self._resolve(inputs)
        return pl.read_csv(file_path, *read_args, **read_kwargs)


class ReadParquet(FileReadOp):
    is_abstract = True
    format = "parquet"


@physical_impl(of=ReadParquet, backend="pandas", input_format="value", output_format="frame")
class PandasReadParquet(ReadParquet):
    is_abstract = False

    def process(self, mode: str, inputs: list):
        file_path, read_args, read_kwargs = self._resolve(inputs)
        return pd.read_parquet(file_path, *read_args, **read_kwargs)


@physical_impl(of=ReadParquet, backend="polars", input_format="value", output_format="frame")
class PolarsReadParquet(ReadParquet):
    is_abstract = False

    def process(self, mode: str, inputs: list):
        file_path, read_args, read_kwargs = self._resolve(inputs)
        return pl.read_parquet(file_path, *read_args, **read_kwargs)


class NumpyLoad(FileReadOp):
    """``np.load`` source. Produces an ndarray (MATRIX), so it is backend-agnostic:
    a single concrete impl serves both frame backends."""
    is_abstract = False
    format = "npy"

    def __init__(self, file_path=None, read_args=None, read_kwargs=None):
        super().__init__(file_path, read_args, read_kwargs)
        self.output_type = OutputType.MATRIX

    def process(self, mode: str, inputs: list):
        file_path, read_args, read_kwargs = self._resolve(inputs)
        return np.load(file_path, *read_args, **read_kwargs)


# Registered as its own implementation so the default registry includes it;
# selection is a no-op (the class is already concrete and backend-agnostic).
physical_impl(of=NumpyLoad, backend="numpy", input_format="value",
              output_format="matrix")(NumpyLoad)


class InMemoryFrame(PhysicalOp):
    """Abstract source wrapping an already-materialised dataframe."""
    is_abstract = True

    def __init__(self, data=None):
        # No name: the concrete class (PandasInMemoryFrame, ...) already says it all.
        super().__init__()
        self.data = data
        self.output_type = OutputType.FRAME


@physical_impl(of=InMemoryFrame, backend="pandas", input_format="value", output_format="frame")
class PandasInMemoryFrame(InMemoryFrame):
    is_abstract = False

    def process(self, mode: str, inputs: list):
        return self.data


@physical_impl(of=InMemoryFrame, backend="polars", input_format="value", output_format="frame")
class PolarsInMemoryFrame(InMemoryFrame):
    is_abstract = False

    def on_impl_selected(self, ctx) -> None:
        # Fold the rechunk decision into instance state at plan time.
        self.rechunk = ctx.rechunk

    def process(self, mode: str, inputs: list):
        out = pl.DataFrame(self.data)
        return rechunk_pl_frame(out) if getattr(self, "rechunk", True) else out


@lowering_rule(DataSourceOp)
def lower_data_source(op: DataSourceOp, ctx):
    """Lower a logical ``DataSourceOp`` to the matching abstract physical source."""
    if op.data is not None:
        return InMemoryFrame(data=op.data)
    fmt = op.format
    if fmt == "csv":
        return ReadCSV(file_path=op.file_path, read_args=op.read_args, read_kwargs=op.read_kwargs)
    if fmt == "parquet":
        return ReadParquet(file_path=op.file_path, read_args=op.read_args, read_kwargs=op.read_kwargs)
    if fmt == "npy":
        return NumpyLoad(file_path=op.file_path, read_args=op.read_args, read_kwargs=op.read_kwargs)
    raise ValueError(f"Unsupported source format for lowering: {fmt!r}")
