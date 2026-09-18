"""Physical source operators: the concrete read / in-memory-frame impls.

Lowering turns a logical :class:`~stratum.optimizer.logical._source_ops.DataSourceOp`
into an *abstract* source op (``ReadCSV``, ``ReadParquet``, ``InMemoryFrame``, or
the already-concrete ``NumpyLoad``). Implementation selection then swaps each
abstract op to one of the backend-specific concrete classes registered below via
``@physical_impl`` (``PandasReadCSV`` / ``PolarsReadCSV`` / ...). The concrete
``process`` methods contain no backend or ``rechunk`` branch -- both decisions
were fixed at plan time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from stratum.optimizer.logical._base import (OperandRef, OutputType, _resolve_args,
                                        _resolve_kwargs)
from stratum.optimizer.logical._source_ops import DataSourceOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._lowering import lowering_rule
from stratum.optimizer.physical._registry import physical_impl


def rechunk_pl_frame(df, rows_per_chunk=128_000):
    n = len(df)
    if rows_per_chunk <= 0 or n <= rows_per_chunk:
        return df
    parts = [df.slice(i, rows_per_chunk) for i in range(0, n, rows_per_chunk)]
    return pl.concat(parts, rechunk=False)


# --- pandas -> polars read options --------------------------------------------
# ``read_args`` / ``read_kwargs`` always arrive in *pandas* spelling: the
# ``_READ_FORMATS`` table in ``logical/_source_ops.py`` recognises only
# ``pd.read_csv`` / ``pd.read_parquet`` / ``np.load``, so a source op can only
# ever have been built from a pandas call. The polars readers rename most of
# those options, take different values for some, and have no counterpart at all
# for others -- forwarding them verbatim is what made ``PolarsReadCSV`` reject a
# plain ``usecols``.
#
# Only exact-meaning renames are listed below. An option whose polars
# counterpart takes different *values* (``dtype`` vs ``schema_overrides``, whose
# type objects are not interchangeable) or has no counterpart at all
# (``index_col``) is rejected, not approximated: a read option silently dropped
# or mis-mapped yields a quietly wrong frame, which is a far worse failure than
# a plan that refuses to compile.
#
# Rejecting is also why this is not a ``supports()`` check. Declining the op
# would let selection fall back to ``PandasReadCSV`` inside an otherwise-polars
# plan, feeding a pandas frame into polars operators -- a later, stranger crash
# than the one raised here.

#: pandas CSV option -> polars CSV option, same meaning, same value.
_POLARS_CSV_RENAMES = {
    "sep": "separator",
    "delimiter": "separator",
    "usecols": "columns",
    "names": "new_columns",
    "nrows": "n_rows",
    "skiprows": "skip_rows",
    "na_values": "null_values",
    "comment": "comment_prefix",
    "quotechar": "quote_char",
    "low_memory": "low_memory",
    "storage_options": "storage_options",
}

#: pandas parquet option -> polars parquet option. polars takes no positional
#: options at all, hence the ``read_args`` guard in ``_translate_read_options``.
_POLARS_PARQUET_RENAMES = {
    "columns": "columns",
    "storage_options": "storage_options",
}


def _translate_header(value):
    """pandas ``header`` -> polars ``has_header``.

    pandas takes a row *index* (0 = the first row is the header, ``None`` = no
    header); polars takes a bool. Only the two spellings that mean "first row"
    and "no header" carry over -- ``header=2`` (skip two rows, then read a
    header) has no polars equivalent.
    """
    if value is None:
        return "has_header", False
    if value == 0:
        return "has_header", True
    raise ValueError(
        f"header={value!r} has no polars equivalent (polars' has_header is a "
        f"bool: only header=0 and header=None translate).")


def _translate_encoding(value):
    """pandas ``encoding`` -> polars ``encoding``.

    Same option name, much narrower domain: polars reads ``utf8`` or
    ``utf8-lossy`` only, so every other codec pandas accepts is a rejection.
    """
    if value is None or str(value).lower().replace("-", "") in ("utf8",):
        return "encoding", "utf8"
    if str(value).lower() == "utf8-lossy":
        return "encoding", "utf8-lossy"
    raise ValueError(
        f"encoding={value!r} has no polars equivalent (polars reads 'utf8' or "
        f"'utf8-lossy' only).")


#: pandas option -> callable(value) -> (polars option, polars value), for the
#: options that need the value rewritten and not just the name.
_POLARS_CSV_VALUE_TRANSLATORS = {
    "header": _translate_header,
    "encoding": _translate_encoding,
}


def _check_read_options(read_args, read_kwargs, renames, translators, reader):
    """Reject read options that have no polars counterpart, by name.

    Which options a source carries is known when the plan is built, so an
    untranslatable one should fail the plan rather than the run -- the same
    reason the backend itself is bound at plan time. Values can still be
    ``OperandRef``s at that point, so the value *translators* run later, in
    ``process``.
    """
    if read_args:
        raise ValueError(
            f"{reader} cannot take positional read options {list(read_args)!r}: "
            f"the polars readers accept only the path positionally, so a "
            f"pandas positional argument would bind to a different option.")
    unsupported = sorted(name for name in (read_kwargs or {})
                         if name not in renames and name not in translators)
    if unsupported:
        raise ValueError(
            f"{reader} has no polars equivalent for read option(s) "
            f"{unsupported}. Read this source with the pandas backend, or drop "
            f"the option.")


def _translate_read_options(read_args, read_kwargs, renames, translators, reader):
    """Rewrite pandas-spelled read options into their polars spelling.

    Re-runs :func:`_check_read_options` so a graph-fed source op reaching this
    with an option the plan-time check never saw still fails loudly.
    """
    _check_read_options(read_args, read_kwargs, renames, translators, reader)
    translated = {}
    for name, value in (read_kwargs or {}).items():
        if name in translators:
            new_name, new_value = translators[name](value)
            translated[new_name] = new_value
        else:
            translated[renames[name]] = value
    return translated


class FileReadOp(PhysicalOp):
    """Abstract read-from-file source. Concrete subclasses pick the backend reader."""
    is_abstract = True
    format: str | None = None

    def __init__(self, file_path=None, read_args=None, read_kwargs=None):
        # No name beyond the path: the concrete class (PandasReadCSV, ...) already
        # encodes backend and format, so a "read_csv" name would just double it in
        # the plan. A graph-fed path is only known at runtime, so it is not a name.
        super().__init__(name=str(file_path) if file_path is not None
                              and not isinstance(file_path, OperandRef) else None)
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

    _RENAMES = _POLARS_CSV_RENAMES
    _VALUE_TRANSLATORS = _POLARS_CSV_VALUE_TRANSLATORS

    def on_impl_selected(self, ctx) -> None:
        _check_read_options(self.read_args, self.read_kwargs, self._RENAMES,
                            self._VALUE_TRANSLATORS, type(self).__name__)

    def process(self, mode: str, inputs: list):
        file_path, read_args, read_kwargs = self._resolve(inputs)
        read_kwargs = _translate_read_options(
            read_args, read_kwargs, self._RENAMES, self._VALUE_TRANSLATORS,
            type(self).__name__)
        return pl.read_csv(file_path, **read_kwargs)


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

    _RENAMES = _POLARS_PARQUET_RENAMES
    _VALUE_TRANSLATORS = {}

    def on_impl_selected(self, ctx) -> None:
        _check_read_options(self.read_args, self.read_kwargs, self._RENAMES,
                            self._VALUE_TRANSLATORS, type(self).__name__)

    def process(self, mode: str, inputs: list):
        file_path, read_args, read_kwargs = self._resolve(inputs)
        read_kwargs = _translate_read_options(
            read_args, read_kwargs, self._RENAMES, self._VALUE_TRANSLATORS,
            type(self).__name__)
        return pl.read_parquet(file_path, **read_kwargs)


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
