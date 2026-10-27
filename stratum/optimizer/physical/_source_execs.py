"""Physical source operators: one impl per backend for each logical source op.

Implementation selection swaps a logical source op (``ReadCSVOp``,
``ReadParquetOp``, ``LoadNumpyOp``, ``InMemoryFrameOp``) to one of the concrete
classes registered below via ``@physical_impl`` (``PandasReadCSV`` /
``PolarsReadCSV`` / ...). The concrete ``process`` methods contain no backend or
``rechunk`` branch -- both decisions were fixed at plan time.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import polars as pl

from stratum.optimizer.logical._base import OperandRef, _resolve_kwargs, _resolve_operand
from stratum.optimizer.logical._source_ops import (
    FileReadOp, InMemoryFrameOp, LoadNumpyOp, ReadCSVOp, ReadParquetOp, pandas_read_kwargs)
from stratum.optimizer._op_utils import _iter_operand_refs
from stratum.optimizer.physical._conversion import to_pandas
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl


def rechunk_pl_frame(df, rows_per_chunk=128_000):
    n = len(df)
    if rows_per_chunk <= 0 or n <= rows_per_chunk:
        return df
    parts = [df.slice(i, rows_per_chunk) for i in range(0, n, rows_per_chunk)]
    return pl.concat(parts, rechunk=False)


# --- read options -> polars --------------------------------------------------
# A read op's fields are pandas' options: extraction only recognises
# ``pd.read_csv`` / ``pd.read_parquet`` / ``np.load``. The polars readers name
# most of those options differently, take different values for some, and have
# no counterpart at all for the ``pandas_options``.
#
# Each option field has a translator below: ``value -> (polars option, polars
# value)``, raising ValueError for a value with no polars spelling of the same
# meaning. Such a value is refused in ``supports()``, not approximated: a read
# option silently dropped or mis-mapped yields a quietly wrong frame. An option
# with no field (``index_col``, ``dtype``, ...) is refused outright. The pandas
# impl reads the source instead, and a polars plan converts its frame (#227).


#: fsspec (s3fs) credential key -> polars object-store key, same value.
_OBJECT_STORE_KEYS = {
    "key": "aws_access_key_id",
    "secret": "aws_secret_access_key",
    "endpoint_url": "aws_endpoint_url",
}


def _translate_storage_options(value):
    """pandas ``storage_options`` (fsspec) -> polars ``storage_options`` (object store).

    pandas hands these to fsspec (s3fs, gcsfs); polars reads cloud paths with
    its own object store, whose keys differ. An untranslated fsspec key is not an
    error there: it is ignored, the request goes out unauthenticated, and the
    read fails later with a 401. The common credential spellings are mapped;
    every other key is passed on unchanged, so options already written in
    object-store spelling keep working.
    """
    if value is None:
        return "storage_options", None
    translated = {}
    for name, option in value.items():
        if name == "token":  # gcsfs
            if isinstance(option, dict):
                translated["service_account_key"] = json.dumps(option)
            elif option == "anon":
                translated["skip_signature"] = "true"
            elif option not in (None, "google_default", "cloud"):
                translated["service_account"] = str(option)
            # else: application default credentials, the object store's default
        elif name == "anon":  # s3fs
            if option:
                translated["skip_signature"] = "true"
        else:
            translated[_OBJECT_STORE_KEYS.get(name, name)] = option
    return "storage_options", translated


def _polars_cloud_path(file_path):
    """Spell a cloud dataset directory the way polars lists it.

    pyarrow treats ``gs://bucket/table`` as a dataset directory when no object
    has that exact name; polars only lists a prefix that ends in ``/`` and
    otherwise asks for the single object, which does not exist. A cloud path
    without a file extension or glob is taken to be such a directory.
    """
    if (not isinstance(file_path, str) or "://" not in file_path
            or file_path.startswith("file://") or file_path.endswith("/")
            or any(ch in file_path for ch in "*?[")):
        return file_path
    if "." in file_path.rsplit("/", 1)[-1]:
        return file_path
    return file_path + "/"


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


def _translate_usecols(value):
    """pandas ``usecols`` -> polars ``columns``: names or positions, repeats dropped
    as pandas does. pandas keeps the file's column order whatever order they are
    given in, polars the order given; :class:`PolarsReadCSV` restores pandas'.
    A callable has no polars spelling."""
    if value is None:
        return "columns", None
    if callable(value) or isinstance(value, str) or not isinstance(value, (list, tuple, set)):
        raise ValueError(f"usecols={value!r} has no polars equivalent: only a list of "
                         f"names or of positions translates.")
    columns = list(dict.fromkeys(value))
    if not (all(isinstance(c, str) for c in columns) or all(_is_count(c) for c in columns)):
        raise ValueError(f"usecols={value!r} has no polars equivalent: it must hold "
                         f"only names or only positions.")
    return "columns", columns


def _translate_sep(value):
    """pandas ``sep`` -> polars ``separator``. pandas reads a longer separator as a
    regular expression; polars takes a single byte."""
    return "separator", _single_byte("sep", value)


def _translate_quotechar(value):
    return "quote_char", _single_byte("quotechar", value)


def _single_byte(option: str, value) -> str:
    if isinstance(value, str) and len(value.encode()) == 1:
        return value
    raise ValueError(f"{option}={value!r} has no polars equivalent (polars takes a "
                     f"single-byte character).")


def _translate_names(value):
    """pandas ``names`` -> polars ``new_columns``, for distinct string names. Their
    count must match the file's width, which :class:`PolarsReadCSV` checks."""
    if value is None:
        return "new_columns", None
    if (not isinstance(value, (list, tuple)) or not all(isinstance(n, str) for n in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"names={value!r} has no polars equivalent (polars takes "
                         f"distinct string names).")
    return "new_columns", list(value)


def _translate_nrows(value):
    if value is None:
        return "n_rows", None
    return "n_rows", _count("nrows", value)


def _translate_skiprows(value):
    """pandas ``skiprows`` -> polars ``skip_rows``. pandas also takes a list of
    line numbers or a callable; polars only a count of leading lines."""
    if value is None:
        return "skip_rows", 0
    return "skip_rows", _count("skiprows", value)


def _is_count(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value >= 0


def _count(option: str, value) -> int:
    if not _is_count(value):
        raise ValueError(f"{option}={value!r} has no polars equivalent (polars takes a "
                         f"row count).")
    return int(value)


def _translate_na_values(value):
    """pandas ``na_values`` -> polars ``null_values``.

    Only strings that do not read as a number carry over: pandas also matches a
    number against the parsed value (``"1"`` matches ``1.0``), and polars only the
    exact string. Per column, polars takes a single string.
    """
    if value is None:
        return "null_values", None
    if isinstance(value, dict):
        per_column = {}
        for column, values in value.items():
            values = [values] if isinstance(values, str) else values
            if (not isinstance(column, str) or not isinstance(values, (list, tuple, set))
                    or len(values) != 1):
                raise ValueError(f"na_values={value!r} has no polars equivalent (polars "
                                 f"takes one string per column name).")
            per_column[column] = _null_string(value, next(iter(values)))
        return "null_values", per_column
    if isinstance(value, str):
        return "null_values", _null_string(value, value)
    if isinstance(value, (list, tuple, set)):
        return "null_values", [_null_string(value, v) for v in value]
    raise ValueError(f"na_values={value!r} has no polars equivalent.")


def _null_string(na_values, value) -> str:
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            return value
    raise ValueError(f"na_values={na_values!r} has no polars equivalent: pandas also "
                     f"matches {value!r} as a number, polars only as that exact string.")


def _translate_comment(value):
    """pandas ``comment`` also drops the rest of a line after it; polars'
    ``comment_prefix`` only skips whole lines."""
    if value is None:
        return "comment_prefix", None
    raise ValueError(f"comment={value!r} has no polars equivalent: pandas also drops "
                     f"the rest of a line after it, polars only skips whole lines.")


def _translate_low_memory(value):
    # Neither library's low_memory changes the values polars reads.
    return "low_memory", value


def _translate_parquet_columns(value):
    """pandas reads parquet columns by name only; polars also by position."""
    if value is None:
        return "columns", None
    if (isinstance(value, str) or not isinstance(value, (list, tuple, set))
            or not all(isinstance(c, str) for c in value)):
        raise ValueError(f"columns={value!r} has no polars equivalent: only a list of "
                         f"names translates.")
    return "columns", list(value)


#: pyarrow ``filters`` operator -> polars predicate on (column expr, value).
_PYARROW_FILTER_OPS = {
    "=": lambda col, v: col == v,
    "==": lambda col, v: col == v,
    "!=": lambda col, v: col != v,
    "<": lambda col, v: col < v,
    ">": lambda col, v: col > v,
    "<=": lambda col, v: col <= v,
    ">=": lambda col, v: col >= v,
    "in": lambda col, v: col.is_in(list(v)),
    "not in": lambda col, v: ~col.is_in(list(v)),
}


def _translate_filters(value):
    """pandas ``filters`` (pyarrow DNF) -> one polars predicate expression.

    pyarrow accepts a list of ``(column, op, value)`` tuples, ANDed, or a list of
    such lists, ORed -- told apart the way pyarrow does, by whether the first
    element's first item is a column name. The predicate filters rows just like
    pyarrow does; :class:`PolarsReadParquet` applies it to a ``scan_parquet``,
    so hive partitions and row groups it excludes are never read. A
    ``pyarrow.compute.Expression`` filter has no polars spelling and is rejected.
    """
    if value is None:
        return "filters", None
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(
            f"filters={value!r} has no polars equivalent: only pyarrow's "
            f"list-of-tuples (DNF) form translates.")
    disjuncts = [value] if isinstance(value[0][0], str) else value
    predicate = None
    for conjunct in disjuncts:
        term = None
        for clause in conjunct:
            if len(clause) != 3 or clause[1] not in _PYARROW_FILTER_OPS:
                raise ValueError(
                    f"filter {clause!r} has no polars equivalent: expected "
                    f"(column, op, value) with op in {sorted(_PYARROW_FILTER_OPS)}.")
            column, op, operand = clause
            expr = _PYARROW_FILTER_OPS[op](pl.col(column), operand)
            term = expr if term is None else term & expr
        predicate = term if predicate is None else predicate | term
    return "filters", predicate


#: ReadCSVOp option -> its translator.
_POLARS_CSV_TRANSLATORS = {
    "usecols": _translate_usecols,
    "sep": _translate_sep,
    "header": _translate_header,
    "names": _translate_names,
    "nrows": _translate_nrows,
    "skiprows": _translate_skiprows,
    "na_values": _translate_na_values,
    "comment": _translate_comment,
    "quotechar": _translate_quotechar,
    "encoding": _translate_encoding,
    "low_memory": _translate_low_memory,
    "storage_options": _translate_storage_options,
}

#: ReadParquetOp option -> its translator.
_POLARS_PARQUET_TRANSLATORS = {
    "columns": _translate_parquet_columns,
    "filters": _translate_filters,
    "storage_options": _translate_storage_options,
}


def _polars_refusal(op: FileReadOp, translators: dict) -> str | None:
    """Why polars cannot read ``op`` exactly, or ``None``.

    Which options a source carries is known when the plan is built, and so is
    every value that is not graph-fed. A graph-fed value is translated in
    ``process``, which raises if it has no polars spelling.
    """
    if op.pandas_options:
        return f"no polars equivalent for read option(s) {sorted(op.pandas_options)}"
    for name, value in op.given_options().items():
        if next(_iter_operand_refs(value), None) is None:
            try:
                translators[name](value)
            except ValueError as exc:
                return str(exc)
    return None


def _refuse(op: FileReadOp, reason: str | None) -> bool:
    """``supports()``: whether polars reads ``op``, recording why not."""
    if reason is not None:
        op._unsupported_reason = reason
        return False
    return True


def _polars_read_kwargs(op: FileReadOp, inputs: list, translators: dict) -> dict:
    """``op``'s given options, resolved against ``inputs``, in polars' spelling."""
    return dict(translators[name](value)
                for name, value in _resolve_kwargs(op.given_options(), inputs).items())


def _csv_shape_refusal(op: ReadCSVOp) -> str | None:
    """Why polars cannot read ``op``'s columns as pandas would, or ``None``."""
    if op.usecols is not None and op.names is not None:
        return ("usecols with names has no polars equivalent: pandas selects among the "
                "given names, polars among the file's")
    if op.names is None or isinstance(op.names, OperandRef):
        return None
    width = _csv_width(op)
    if width is not None and width != len(op.names):
        return (f"names has {len(op.names)} entries for {width} columns: pandas then "
                f"makes an index or pads, polars only reads an exact fit")
    return None


def _csv_width(op: ReadCSVOp) -> int | None:
    """The number of fields in the first row ``op`` reads, from the file at plan
    time. ``None`` when only the run knows it: a graph-fed path or parsing option,
    a remote file, or one that cannot be read yet."""
    parsing = {name: value for name, value in op.given_options().items()
               if name in ("sep", "skiprows", "quotechar", "encoding")}
    if (not isinstance(op.file_path, (str, os.PathLike)) or "://" in str(op.file_path)
            or next(_iter_operand_refs(parsing), None) is not None):
        return None
    try:
        kwargs = dict(_POLARS_CSV_TRANSLATORS[name](value) for name, value in parsing.items())
        return pl.read_csv(op.file_path, has_header=False, n_rows=0, **kwargs).width
    except Exception:  # noqa: BLE001 -- unknown here; polars checks it when the read runs
        return None


def _read_csv_in_file_order(file_path, kwargs: dict):
    """``pl.read_csv`` with a ``columns`` projection kept in the file's column order,
    as pandas keeps ``usecols``. The header reaches ``with_column_names`` during
    the same read, so a remote file is fetched once."""
    header = []

    def keep_header(names):
        header.extend(names)
        return names

    frame = pl.read_csv(file_path, with_column_names=keep_header, **kwargs)
    selected = set(frame.columns)
    return frame.select([name for name in header if name in selected])


# --- ReadCSVOp ------------------------------------------------------------------

@physical_impl(of=ReadCSVOp, backend="pandas", input_format="value", output_format="frame")
class PandasReadCSV(ReadCSVOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        return pd.read_csv(_resolve_operand(self.file_path, inputs),
                           **_resolve_kwargs(pandas_read_kwargs(self), inputs))


@physical_impl(of=ReadCSVOp, backend="polars", input_format="value", output_format="frame")
class PolarsReadCSV(ReadCSVOp, PhysicalOp):
    @classmethod
    def supports(cls, op: ReadCSVOp, ctx) -> bool:
        return _refuse(op, _polars_refusal(op, _POLARS_CSV_TRANSLATORS)
                       or _csv_shape_refusal(op))

    def process(self, mode: str, inputs: list):
        translated = _polars_read_kwargs(self, inputs, _POLARS_CSV_TRANSLATORS)
        # pandas infers header=None when names are supplied. Polars otherwise
        # consumes the first data row as a header, even with new_columns set.
        if self.names is not None and self.header == "infer":
            translated["has_header"] = False
        file_path = _resolve_operand(self.file_path, inputs)
        if translated.get("columns") is None:
            return pl.read_csv(file_path, **translated)
        return _read_csv_in_file_order(file_path, translated)


# --- ReadParquetOp --------------------------------------------------------------

@physical_impl(of=ReadParquetOp, backend="pandas", input_format="value", output_format="frame")
class PandasReadParquet(ReadParquetOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        return pd.read_parquet(_resolve_operand(self.file_path, inputs),
                               **_resolve_kwargs(pandas_read_kwargs(self), inputs))


@physical_impl(of=ReadParquetOp, backend="polars", input_format="value", output_format="frame")
class PolarsReadParquet(ReadParquetOp, PhysicalOp):
    @classmethod
    def supports(cls, op: ReadParquetOp, ctx) -> bool:
        return _refuse(op, _polars_refusal(op, _POLARS_PARQUET_TRANSLATORS))

    def process(self, mode: str, inputs: list):
        read_kwargs = _polars_read_kwargs(self, inputs, _POLARS_PARQUET_TRANSLATORS)
        file_path = _polars_cloud_path(_resolve_operand(self.file_path, inputs))
        predicate = read_kwargs.pop("filters", None)
        if predicate is None:
            return pl.read_parquet(file_path, **read_kwargs)
        # Filter before projecting: like pyarrow, a filter may name a column
        # (e.g. the hive partition key) that `columns` leaves out.
        columns = read_kwargs.pop("columns", None)
        frame = pl.scan_parquet(file_path, **read_kwargs).filter(predicate)
        if columns is not None:
            frame = frame.select(columns)
        return frame.collect()


# --- LoadNumpyOp ----------------------------------------------------------------

@physical_impl(of=LoadNumpyOp, backend="numpy", input_format="value", output_format="matrix")
class NumpyLoad(LoadNumpyOp, PhysicalOp):
    """``np.load`` yields an ndarray, so one impl serves both frame backends."""

    def process(self, mode: str, inputs: list):
        return np.load(_resolve_operand(self.file_path, inputs),
                       **_resolve_kwargs(self.options, inputs))


# --- InMemoryFrameOp ------------------------------------------------------------

@physical_impl(of=InMemoryFrameOp, backend="pandas", input_format="value", output_format="frame")
class PandasInMemoryFrame(InMemoryFrameOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        # The frame passed in may be a polars one; this impl produces pandas.
        return to_pandas(self.data)


@physical_impl(of=InMemoryFrameOp, backend="polars", input_format="value", output_format="frame")
class PolarsInMemoryFrame(InMemoryFrameOp, PhysicalOp):
    def on_impl_selected(self, ctx) -> None:
        # Fold the rechunk decision into instance state at plan time.
        self.rechunk = ctx.rechunk

    def process(self, mode: str, inputs: list):
        out = pl.DataFrame(self.data)
        return rechunk_pl_frame(out) if getattr(self, "rechunk", True) else out
