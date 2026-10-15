"""Metadata operators: change row or column *labels*, never values.

Set apart from projections (which select columns) and maps (which compute
column values): ``df.rename(...)`` and ``df.reset_index(drop=True)`` keep every
row and every value and only relabel them.

Extraction parses the recorded pandas call into backend-neutral fields; the
physical impls in ``physical/_metadata_execs.py`` translate them per library. A
call the factories cannot model returns ``None`` and stays an opaque method call
that runs on pandas as recorded.
"""
from __future__ import annotations

import inspect

import pandas as pd

from stratum.optimizer.logical._ops import MethodCallOp, Op, OutputType
from stratum.optimizer.logical import _schema


class MetadataOp(Op):
    """Abstract base of the metadata family. Impls register on the subclasses."""
    logical_family = "Metadata"

    def __init__(self, name: str = "", inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        self.output_type = OutputType.FRAME


class RenameOp(MetadataOp):
    """Relabels columns and/or index rows.

    ``columns`` / ``index``: a mapping from old to new label, or a function
    applied to every label; ``None`` leaves that axis alone.
    ``errors``: ``"raise"`` when a mapped label is absent, ``"ignore"`` to skip it.
    """
    fields = ["columns", "index", "errors"]

    def __init__(self, columns=None, index=None, errors="ignore",
                 inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name="rename", inputs=inputs, outputs=outputs)
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

    def __init__(self, inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name="reset_index", inputs=inputs, outputs=outputs)

    def propagate_output_schema(self):
        """The old labels are dropped, not promoted to a column."""
        self.output_schema = self.inputs[0].output_schema if self.inputs else None


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
    try:
        call = inspect.signature(pd.DataFrame.rename).bind(
            None, *(op.args or ()), **(op.kwargs or {})).arguments
    except TypeError:
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
    new_op = RenameOp(columns=columns, index=index, errors=call.get("errors", "ignore"),
                      inputs=op.inputs, outputs=op.outputs)
    op.replace_output_of_inputs(new_op)
    return new_op


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
    new_op = ResetIndexOp(inputs=op.inputs, outputs=op.outputs)
    # reset_index keeps the container kind: a series stays a series.
    new_op.output_type = op.inputs[0].output_type
    op.replace_output_of_inputs(new_op)
    return new_op
