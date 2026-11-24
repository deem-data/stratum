"""Metadata operators read structure or change labels and container kind.

- Relabeling preserves row order and retained column values.
- ``set_index`` moves key columns into row labels and removes those columns.
- ``to_frame`` converts a Series to a one-column DataFrame.
- ``shape`` and ``columns`` return structural metadata.

Unsupported calls remain opaque pandas method calls.
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
    index_effect = IndexEffect.NEW
    cacheable = True

    def clone(self):
        cloned = super().clone()
        cloned.output_type = self.output_type
        return cloned


class RenameOp(MetadataOp):
    """Rename axis labels or a Series, optionally returning a one-column frame.

    - ``columns`` / ``index``: a label mapping or function; ``None`` preserves
      that axis.
    - ``errors``: ``"raise"`` for absent mapped labels, ``"ignore"`` to skip them.
    - ``label``: ``(name,)`` sets the Series name, including ``(None,)`` to clear
      it; ``None`` preserves the name.
    - ``as_frame``: return a DataFrame. Without an explicit label, use the
      Series name, or ``0`` if the Series is unnamed.
    """
    fields = ["columns", "index", "errors", "label", "as_frame"]

    @property
    def index_effect(self):
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
        """Propagate column labels; non-string Series labels yield unknown schema."""
        source = self.inputs[0].output_schema
        if self.label is not None:
            label = self.label[0]
            self.output_schema = (
                _schema.rename_columns(source, {next(iter(source)): label})
                if _schema.is_known(source) and len(source) == 1 and isinstance(label, str)
                else None)
        else:
            self.output_schema = (source if self.columns is None
                                  else _schema.rename_columns(source, self.columns))


class RowLabelsOp(MetadataOp):
    """Replace row labels, preserving row order and the remaining column values.

    Set at most one of ``keys`` and ``labels``:

    - Neither: assign ``0..n-1`` (``reset_index(drop=True)``).
    - ``keys``: use the named columns and remove them from the frame
      (``set_index(keys, drop=True)``).
    - ``labels``: use another input's values by position (``set_axis(labels)``).
      The label count must equal the row count.
    """
    fields = ["keys", "labels"]
    # Assign labels by position, independently of the inputs' existing labels.
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
        return IndexEffect.DEFAULT if self.keys is None and self.labels is None else IndexEffect.NEW

    @property
    def labels_source(self) -> Op | None:
        """Return ``x`` for ``set_axis(x.index)``, otherwise ``None``.

        Index lineage follows ``x`` because its label order is preserved.
        """
        if not isinstance(self.labels, OperandRef):
            return None
        source = self.inputs[self.labels.k]
        return source.inputs[0] if isinstance(source, IndexAccessOp) else None

    def propagate_output_schema(self):
        """Preserve the input schema, excluding columns moved into row labels."""
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

    - A positional DataFrame mapper targets the index unless ``axis`` selects
      columns. Mappers must be dict-like or callable.
    - In-place changes, index levels, ``copy``, and a computed ``axis`` remain
      opaque.
    - Scalar names require a receiver typed as Series. A grouped ``size()``
      can be typed as DataFrame while returning a Series; its scalar rename
      remains opaque.
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


# Other hashable names, such as tuples and floats, remain opaque.
_COLUMN_LABEL_TYPES = (str, int)


def _make_series_rename_op(op: MethodCallOp) -> RenameOp | None:
    """Extract a string or integer Series name.

    Index mappings, callables, and additional arguments remain opaque.
    """
    call = op.bind_pandas(owner=pd.Series)
    if call is None or set(call) != {"index"}:
        return None
    label = call["index"]
    if type(label) not in _COLUMN_LABEL_TYPES:
        return None
    return substitute_op(op, RenameOp(label=(label,)), keep_kind=True)


def make_to_frame_op(op: MethodCallOp) -> RenameOp | None:
    """Extract ``Series.to_frame`` with an optional string, integer, or None name.

    Row order and values are preserved. Explicit ``None`` names the column
    ``None``; omitting the name uses the Series name, or ``0`` if unnamed.
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

    Only ``drop=True`` is modeled. The default ``drop=False`` promotes index
    levels to columns, whose names and types are not tracked by this operator.
    """
    kwargs = dict(op.kwargs or {})
    if op.args or kwargs.pop("drop", False) is not True or kwargs:
        return None
    # reset_index keeps the container kind: a series stays a series.
    return substitute_op(op, RowLabelsOp(), keep_kind=True)


def make_set_index_op(op: MethodCallOp) -> RowLabelsOp | None:
    """Parse ``DataFrame.set_index(keys)`` into a :class:`RowLabelsOp`.

    Keys must be a string or a nonempty list of distinct strings. Only
    ``drop=True`` and ``append=False`` are supported; other arguments remain
    opaque.
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
    if (not isinstance(keys, list) or not keys
            or not all(isinstance(k, str) for k in keys) or len(set(keys)) != len(keys)):
        return None
    return substitute_op(op, RowLabelsOp(keys=tuple(keys)))


def make_set_axis_op(op: MethodCallOp) -> RowLabelsOp | None:
    """Parse ``set_axis(labels)`` along the rows into a :class:`RowLabelsOp`.

    Labels must come from another input, such as ``other.index`` or a column.
    Literal labels, column-axis changes, and ``copy`` remain opaque.
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
