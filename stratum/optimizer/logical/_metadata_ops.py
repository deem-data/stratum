"""Metadata operators: describe or relabel a frame's structure, never its values.

Set apart from projections (which select columns) and maps (which compute
column values): ``df.rename(...)`` and ``df.reset_index(drop=True)`` keep every
row and every value and only relabel them, while ``df.shape`` and
``df.columns`` read the structure without touching a value. The output kind is
per op: a relabelling returns the relabelled frame, a read returns the
structure it describes.

Extraction parses the recorded pandas call into backend-neutral fields; the
physical impls in ``physical/_metadata_execs.py`` translate them per library. A
call the factories cannot model returns ``None`` and stays an opaque method call
that runs on pandas as recorded.
"""
from __future__ import annotations

import pandas as pd

from stratum.optimizer.logical._base import IndexEffect
from stratum.optimizer.logical._ops import MethodCallOp, Op, OutputType, substitute_op
from stratum.optimizer.logical import _schema


class MetadataOp(Op):
    """Abstract base of the metadata family. Impls register on the subclasses.

    Each subclass sets its own ``output_type``.
    """
    logical_family = "Metadata"
    # A structure read is not a frame; a relabelling says how it relabels.
    index_effect = IndexEffect.NEW


class RenameOp(MetadataOp):
    """Relabels columns and/or index rows.

    ``columns`` / ``index``: a mapping from old to new label, or a function
    applied to every label; ``None`` leaves that axis alone.
    ``errors``: ``"raise"`` when a mapped label is absent, ``"ignore"`` to skip it.
    """
    fields = ["columns", "index", "errors"]
    index_effect = IndexEffect.KEEP

    def __init__(self, columns=None, index=None, errors="ignore",
                 inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name="rename", inputs=inputs, outputs=outputs)
        self.output_type = OutputType.FRAME
        self.columns = columns
        self.index = index
        self.errors = errors

    def propagate_output_schema(self):
        """Remaps the column names; an index-only rename passes them through."""
        source = self.inputs[0].output_schema
        self.output_schema = (source if self.columns is None
                              else _schema.rename_columns(source, self.columns))


class ResetIndexOp(MetadataOp):
    """Replaces the row labels with ``0..n-1``, discarding the old ones.

    Libraries whose rows carry no labels have nothing to do.
    """
    fields = []
    index_effect = IndexEffect.DEFAULT

    def __init__(self, inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name="reset_index", inputs=inputs, outputs=outputs)
        self.output_type = OutputType.FRAME

    def propagate_output_schema(self):
        """The old labels are dropped, not promoted to a column."""
        self.output_schema = self.inputs[0].output_schema if self.inputs else None


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
    """Parse ``DataFrame.rename`` into a :class:`RenameOp`.

    The mapper's axis follows pandas: a positional ``mapper`` renames the index
    unless ``axis`` says columns. ``Series.rename`` (which also sets the series
    name), ``inplace``, ``level`` and a graph-fed ``axis`` stay opaque.

    Every label argument must be dict-like or a function. A scalar sets a
    Series' name, which this op does not model; checking the argument itself
    keeps that call opaque even where the receiver's static kind is off (a
    grouped ``size()`` is typed as a frame but returns a Series).
    """
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


def _is_label_map(value) -> bool:
    """A mapping or function over labels; anything else (a scalar name, a
    graph-fed value) is not a rename we can model."""
    return pd.api.types.is_dict_like(value) or callable(value)


def make_reset_index_op(op: MethodCallOp) -> ResetIndexOp | None:
    """Parse ``reset_index(drop=True)`` into a :class:`ResetIndexOp`.

    Only the dropping form. ``drop=False`` promotes the index to a column and so
    changes the schema, which we cannot describe until schema propagation exists,
    and it is also pandas' default, so it is refused rather than guessed at.
    """
    kwargs = dict(op.kwargs or {})
    if op.args or kwargs.pop("drop", False) is not True or kwargs:
        return None
    # reset_index keeps the container kind: a series stays a series.
    return substitute_op(op, ResetIndexOp(), keep_kind=True)


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
