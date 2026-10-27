"""Projection operators: select columns, never modify them.

A projection keeps the rows and the values of its input and only restricts (or
reorders) the column set: ``df[["a", "b"]]``, ``skb.select(...)``,
``df.drop(columns=...)``. Anything that changes column values or adds columns is
a map (:mod:`stratum.optimizer.logical._map_ops`).
"""
import pandas as pd

from stratum.optimizer.logical._base import IndexEffect
from stratum.optimizer.logical._ops import OutputType, MethodCallOp, Op, TransformerOp, substitute_op
from stratum.optimizer.logical import _schema


class ProjectionOp(Op):
    """Base for projection operators: keeps rows and values, selects columns."""
    logical_family = "Projection"
    index_effect = IndexEffect.KEEP

    def __init__(self, name: str = "", inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        self.output_type = OutputType.FRAME


def drop_targets_columns(kwargs: dict | None) -> bool | None:
    """Which axis a ``drop(...)`` call removes labels from: True for columns,
    False for rows (index labels), None when that isn't statically known.

    pandas defaults to ``axis=0``, i.e. dropping *rows* by index label.
    ``index=`` / ``columns=`` name the axis themselves; naming both drops along
    both axes, which is neither a projection nor a selection, and pandas rejects
    either one next to an ``axis``, so those calls are left to raise as written.
    ``axis`` is keyword-only on ``DataFrame.drop``, so it can never arrive
    positionally.
    """
    kwargs = kwargs or {}
    by_index = kwargs.get("index") is not None
    by_columns = kwargs.get("columns") is not None
    if by_index or by_columns:
        if (by_index and by_columns) or "axis" in kwargs:
            return None
        return by_columns
    axis = kwargs.get("axis", 0)
    if axis in (0, "index"):
        return False
    if axis in (1, "columns"):
        return True
    return None  # e.g. a graph-fed axis


class DropOp(ProjectionOp):
    """Removes the named columns.

    ``columns``: a column label or a list of them (graph-fed labels arrive as an
    ``OperandRef``). ``level``: the level of a MultiIndex column axis the labels
    belong to, ``None`` for a flat one. ``errors``: ``"raise"`` when a label is
    absent, ``"ignore"`` to skip it.

    Only the statically column-wise form of ``drop`` becomes a ``DropOp`` (see
    :func:`drop_targets_columns`). The pandas default (``axis=0``) drops *rows*
    by index label, which extraction turns into a ``DropLabelsSelectionOp``.
    """
    fields = ["columns", "level", "errors"]

    def __init__(self, columns=None, level=None, errors="raise",
                 inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(inputs=inputs, outputs=outputs)
        self.columns = columns
        self.level = level
        self.errors = errors

    def propagate_output_schema(self):
        """Removes the named columns from the input schema."""
        self.output_schema = _schema.drop_columns(self.inputs[0].output_schema,
                                                  self.columns)


def make_drop_op(op: MethodCallOp) -> DropOp | None:
    """Parse a column-wise ``DataFrame.drop`` into a :class:`DropOp`.

    Extraction has already decided the axis (:func:`drop_targets_columns`), so the
    labels are either ``columns=`` or ``labels=`` next to a column ``axis``.
    ``inplace`` and calls pandas would reject stay opaque.
    """
    call = op.bind_pandas(owner=pd.DataFrame)
    if call is None or call.get("inplace", False) is not False:
        return None
    columns, labels = call.get("columns"), call.get("labels")
    if columns is not None and labels is not None:
        return None  # pandas rejects labels next to columns=
    columns = columns if columns is not None else labels
    if columns is None:
        return None  # pandas requires something to drop
    return substitute_op(op, DropOp(columns=columns, level=call.get("level"),
                                    errors=call.get("errors", "raise")))


class ColumnSelectorOp(ProjectionOp):
    """A column selection by (deferred) skrub selector: keeps rows, restricts columns.

    Produced from ``skb.select(cols)``. Matches ``SelectCols`` semantics: the selector resolves against the schema at
    fit time and the *stored* column list is reused at predict time (by the
    physical impls, see ``physical/_projection_execs.py``).
    """
    fields = ["selector"]

    def __init__(self, selector, inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=f"select[{selector!r}]", inputs=inputs, outputs=outputs)
        self.selector = selector


def make_column_selector_op(op: TransformerOp) -> ColumnSelectorOp:
    """Rewrite a ``TransformerOp`` wrapping skrub's ``SelectCols`` into a
    :class:`ColumnSelectorOp` carrying the selector itself."""
    return substitute_op(op, ColumnSelectorOp(selector=op.estimator.cols))


class ColumnProjectionOp(ProjectionOp):
    """Column selection by literal name(s): ``df[key]`` where ``key`` is a
    column label or list of labels.

    The specialised, always-column form of a ``df[...]`` indexing: produced from
    a :class:`~stratum.optimizer.logical._ops.GetItemOp` whose container is a
    ``FRAME`` and whose key is a literal ``str`` (a single column -> ``SERIES``)
    or ``list[str]`` (a sub-frame -> ``FRAME``). Unlike :class:`ColumnSelectorOp`
    (``skb.select``, a deferred skrub selector resolved against the schema), the
    columns are given verbatim, so no fit-time resolution is needed. The general
    ``GetItemOp`` stays the fallback for masks, graph-fed keys, slices and
    positional indexing.
    """
    fields = ["key"]

    def __init__(self, key, inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=f"cols[{key!r}]", inputs=inputs, outputs=outputs)
        self.key = key
        # A single label extracts a column (SERIES); a list selects a sub-frame.
        self.output_type = (OutputType.SERIES if isinstance(key, str)
                            else OutputType.FRAME)

    def propagate_output_schema(self):
        """A literal projection restricts the schema to its key, in order."""
        self.output_schema = _schema.project_columns(self.inputs[0].output_schema, self.key)


def make_column_projection_op(op) -> ColumnProjectionOp:
    """Rewrite a column-selecting ``df[key]`` :class:`GetItemOp` (a literal
    ``str`` / ``list[str]`` key on a frame) into a :class:`ColumnProjectionOp`."""
    return substitute_op(op, ColumnProjectionOp(key=op.key))
