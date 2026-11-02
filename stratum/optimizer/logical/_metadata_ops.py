"""Metadata operators: describe or relabel a frame's structure, never its values.

Set apart from projections (which select columns) and maps (which compute
column values): ``df.rename(...)`` and the row relabellings (``reset_index``,
``set_index``, ``set_axis``) keep every row and every value and only relabel
them, while ``df.shape`` and ``df.columns`` read the structure without touching
a value. The output kind is per op: a relabelling returns the relabelled frame,
a read returns the structure it describes.

Extraction parses the recorded pandas call into backend-neutral fields; the
physical impls in ``physical/_metadata_execs.py`` translate them per library. A
call the factories cannot model returns ``None`` and stays an opaque method call
that runs on pandas as recorded.
"""
from __future__ import annotations

import pandas as pd

from stratum.optimizer.logical._base import IndexEffect
from stratum.optimizer.logical._index_ops import IndexAccessOp
from stratum.optimizer.logical._ops import MethodCallOp, Op, OperandRef, OutputType, substitute_op
from stratum.optimizer.logical import _schema


class MetadataOp(Op):
    """Abstract base of the metadata family. Impls register on the subclasses.

    Each subclass sets its own ``output_type``.
    """
    logical_family = "Metadata"
    # A structure read is not a frame; a relabelling says how it relabels.
    index_effect = IndexEffect.FRESH
    # A relabelling passes the index on (``rename``) or drops it; a structure
    # read does not look at it. Neither pairs rows.
    reads_index = False
    aligns_by_label = False
    cacheable = True


class RenameOp(MetadataOp):
    """Relabels columns and/or index rows, and may turn a series into a frame.

    ``columns`` / ``index``: a mapping from old to new label, or a function
    applied to every label; ``None`` leaves that axis alone.
    ``errors``: ``"raise"`` when a mapped label is absent, ``"ignore"`` to skip it.
    ``label``: ``(label,)``, the new label of a series' only column (its name,
    which may be ``None``); ``None`` keeps it.
    ``as_frame``: the series becomes a one-column frame (``to_frame``). Without
    a new label, pandas labels the column by the series' name, or ``0`` when it
    has none.
    """
    fields = ["columns", "index", "errors", "label", "as_frame"]

    @property
    def index_effect(self):
        # Relabelled rows keep their place, not their labels.
        return IndexEffect.KEEP if self.index is None else IndexEffect.NEW

    def __init__(self, columns=None, index=None, errors="ignore", label: tuple | None = None,
                 as_frame: bool = False, inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name="to_frame" if as_frame else "rename", inputs=inputs,
                         outputs=outputs)
        self.output_type = OutputType.FRAME
        self.columns = columns
        self.index = index
        self.errors = errors
        self.label = label
        self.as_frame = as_frame

    def propagate_output_schema(self):
        """Remaps the column names, or relabels a series' one column; an
        index-only rename, or a bare ``to_frame``, passes them through."""
        source = self.inputs[0].output_schema
        if self.label is not None:
            # A series' schema is its one column, under its name.
            label = self.label[0]
            self.output_schema = (
                _schema.rename_columns(source, {next(iter(source)): label})
                if _schema.is_known(source) and len(source) == 1 and isinstance(label, str)
                else None)
        else:
            self.output_schema = (source if self.columns is None
                                  else _schema.rename_columns(source, self.columns))


class RowLabelsOp(MetadataOp):
    """Replaces the row labels, discarding the old ones; rows and values stay.

    The new labels are, by the fields set (at most one):

    * neither: ``0..n-1`` (``reset_index(drop=True)``);
    * ``keys``: the named columns, which leave the frame (``set_index(keys)``);
    * ``labels``: another input's values, row by row (``set_axis(labels)``),
      which must be as many as the rows.

    Libraries whose rows carry no labels can only take ``0..n-1``, a no-op there.
    """
    fields = ["keys", "labels"]
    # Pairs rows by position, whatever the labels of its inputs.
    aligns_by_label = False

    def __init__(self, keys: tuple | None = None, labels: OperandRef | None = None,
                 inputs: list[Op] = None, outputs: list[Op] = None):
        if keys is not None and labels is not None:
            raise ValueError("RowLabelsOp takes keys or labels, not both")
        name = ("reset_index" if keys is None and labels is None
                else "set_index" if keys is not None else "set_axis")
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        self.output_type = OutputType.FRAME
        self.keys = None if keys is None else tuple(keys)
        self.labels = labels

    @property
    def index_effect(self):
        return IndexEffect.DEFAULT if self.keys is None and self.labels is None else IndexEffect.FRESH

    @property
    def labels_source(self) -> Op | None:
        """The op whose labels this one takes over, row by row: ``x`` for
        ``set_axis(x.index)``. Index lineage follows it instead of starting anew."""
        if not isinstance(self.labels, OperandRef):
            return None
        source = self.inputs[self.labels.k]
        return source.inputs[0] if isinstance(source, IndexAccessOp) else None

    def propagate_output_schema(self):
        """The old labels are dropped, not promoted to a column; ``keys`` leave."""
        source = self.inputs[0].output_schema if self.inputs else None
        self.output_schema = (source if self.keys is None
                              else _schema.drop_columns(source, list(self.keys)))


class ShapeOp(MetadataOp):
    """Reads ``.shape``: ``(n_rows, n_cols)`` of a frame, ``(n,)`` of a series.

    A tuple, not a frame, so it carries no frame library and puts no library
    constraint on its producer.
    """
    fields = []

    def __init__(self, inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name="shape", inputs=inputs, outputs=outputs)
        self.output_type = OutputType.SCALAR


class ColumnNamesOp(MetadataOp):
    """Reads a frame's ``.columns`` as a pandas ``Index`` on every backend."""
    fields = []

    def __init__(self, inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name="columns", inputs=inputs, outputs=outputs)
        self.output_type = OutputType.UNKNOWN


# --- Extraction: recorded pandas call -> metadata fields -----------------------

def make_rename_op(op: MethodCallOp) -> RenameOp | None:
    """Parse ``DataFrame.rename``, or ``Series.rename(label)``, into a :class:`RenameOp`.

    The mapper's axis follows pandas: a positional ``mapper`` renames the index
    unless ``axis`` says columns. ``inplace``, ``level`` and a graph-fed ``axis``
    stay opaque.

    On a frame, every label argument must be dict-like or a function. A scalar
    sets a Series' name, which a receiver typed as a series gets
    (:func:`_make_series_rename_op`); on one typed as a frame it stays opaque, as
    the static kind can be off (a grouped ``size()`` is typed as a frame but
    returns a Series).
    """
    if op.inputs[0].output_type is OutputType.SERIES:
        return _make_series_rename_op(op)
    if op.inputs[0].output_type is not OutputType.FRAME:
        return None
    call = op.bind_pandas(owner=pd.DataFrame)
    if call is None:
        return None
    if (call.get("inplace", False) is not False or call.get("level") is not None
            or "copy" in call):
        return None
    axis, mapper = call.get("axis"), call.get("mapper")
    index, columns = call.get("index"), call.get("columns")
    if mapper is not None:
        if index is not None or columns is not None:
            return None  # pandas rejects a mapper next to index=/columns=
        if axis in (None, 0, "index"):
            index = mapper
        elif axis in (1, "columns"):
            columns = mapper
        else:
            return None  # graph-fed or invalid axis
    elif axis is not None:
        return None  # pandas rejects an axis next to index=/columns=
    if index is None and columns is None:
        return None
    if not all(_is_label_map(m) for m in (index, columns) if m is not None):
        return None
    return substitute_op(op, RenameOp(columns=columns, index=index,
                                      errors=call.get("errors", "ignore")))


#: Labels a series' column takes here: polars names are strings, and anything
#: else hashable (a tuple, a float) is rare enough to leave opaque.
_COLUMN_LABEL_TYPES = (str, int)


def _make_series_rename_op(op: MethodCallOp) -> RenameOp | None:
    """``Series.rename(label)``: a scalar sets the series' name. A mapping or a
    function relabels its index rows instead, which stays opaque, as do the
    other arguments."""
    call = op.bind_pandas(owner=pd.Series)
    if call is None or set(call) != {"index"}:
        return None
    label = call["index"]
    if type(label) not in _COLUMN_LABEL_TYPES:
        return None
    return substitute_op(op, RenameOp(label=(label,)), keep_kind=True)


def make_to_frame_op(op: MethodCallOp) -> RenameOp | None:
    """Parse ``Series.to_frame(name)`` into a :class:`RenameOp` that makes a frame.

    It relabels the series' only column, if a name is given (``None`` included:
    pandas then labels it ``None``, not ``0``), and changes no value or row.
    """
    if op.inputs[0].output_type is not OutputType.SERIES:
        return None
    call = op.bind_pandas(owner=pd.Series)
    if call is None or set(call) - {"name"}:
        return None
    if "name" not in call:
        return substitute_op(op, RenameOp(as_frame=True))
    label = call["name"]
    if label is not None and type(label) not in _COLUMN_LABEL_TYPES:
        return None
    return substitute_op(op, RenameOp(label=(label,), as_frame=True))


def _is_label_map(value) -> bool:
    """A mapping or function over labels; anything else (a scalar name, a
    graph-fed value) is not a rename we can model."""
    return pd.api.types.is_dict_like(value) or callable(value)


def make_reset_index_op(op: MethodCallOp) -> RowLabelsOp | None:
    """Parse ``reset_index(drop=True)`` into a :class:`RowLabelsOp`.

    Only the dropping form. ``drop=False`` promotes the index to a column and so
    changes the schema, which we cannot describe until schema propagation exists,
    and it is also pandas' default, so it is refused rather than guessed at.
    """
    kwargs = dict(op.kwargs or {})
    if op.args or kwargs.pop("drop", False) is not True or kwargs:
        return None
    # reset_index keeps the container kind: a series stays a series.
    return substitute_op(op, RowLabelsOp(), keep_kind=True)


def make_set_index_op(op: MethodCallOp) -> RowLabelsOp | None:
    """Parse ``DataFrame.set_index(keys)`` into a :class:`RowLabelsOp`.

    Only column names, moved out of the frame: ``drop=False``, ``append``,
    ``verify_integrity``, ``inplace`` and keys that are arrays or another op's
    values stay opaque.
    """
    if op.inputs[0].output_type is not OutputType.FRAME:
        return None
    call = op.bind_pandas(owner=pd.DataFrame)
    if call is None or set(call) - {"keys", "drop", "append"}:
        return None
    if call.get("drop", True) is not True or call.get("append", False) is not False:
        return None
    keys = call["keys"]
    keys = [keys] if isinstance(keys, str) else keys
    if (not isinstance(keys, list) or not keys or len(set(keys)) != len(keys)
            or not all(isinstance(k, str) for k in keys)):
        return None
    return substitute_op(op, RowLabelsOp(keys=tuple(keys)))


def make_set_axis_op(op: MethodCallOp) -> RowLabelsOp | None:
    """Parse ``set_axis(labels)`` along the rows into a :class:`RowLabelsOp`.

    Only labels another op computes (``other.index``, a column): literal labels,
    the column axis and ``copy`` stay opaque.
    """
    call = op.bind_pandas()
    if call is None or set(call) - {"labels", "axis"}:
        return None
    if call.get("axis", 0) not in (0, "index", "rows") or not isinstance(
            call["labels"], OperandRef) or call["labels"].k == 0:
        return None
    return substitute_op(op, RowLabelsOp(labels=call["labels"]), keep_kind=True)


def make_structure_read_op(op) -> MetadataOp | None:
    """Parse a ``.shape`` or ``.columns`` read into its :class:`MetadataOp`.

    ``op`` is the recorded attribute read. ``.columns`` is only a frame's: a
    series has none, so that read stays an opaque attribute.
    """
    kind = op.inputs[0].output_type
    if op.attr_name in ("shape", ["shape"]):
        cls = ShapeOp
    elif op.attr_name in ("columns", ["columns"]) and kind is OutputType.FRAME:
        cls = ColumnNamesOp
    else:
        return None
    return substitute_op(op, cls())
