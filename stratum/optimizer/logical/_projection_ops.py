"""Projection operators: select columns, never modify them.

A projection keeps the rows and the values of its input and only restricts (or
reorders) the column set: ``df[["a", "b"]]``, ``skb.select(...)``,
``df.drop(columns=...)``. Anything that changes column values or adds columns is
a map (:mod:`stratum.optimizer.logical._map_ops`).
"""
from skrub.selectors._base import make_selector
from stratum.optimizer.logical._ops import (OutputType, MethodCallOp, Op, TransformerOp,
                                            _resolve_args, _resolve_kwargs)
from stratum.optimizer.logical import _schema


def resolve_selector_columns(frame, selector) -> list[str]:
    """Resolve a skrub selector (or column name / list of names) against ``frame``.

    Returns the concrete column-name list. Selectors are *deferred*: which columns
    match (e.g. ``numeric()``) depends on the data, so resolution can only happen
    once a frame with a schema is available. skrub's dispatch handles both pandas
    and polars frames.

    # TODO with schema propagation we can resolute the column name list at compile time
    """
    return make_selector(selector).expand(frame)

class MetadataOp(Op):
    fields = ["func", "args", "kwargs"]

    def __init__(self, func: str, args: tuple | list = None, kwargs: dict = None, inputs: list[Op] = None, outputs: list[Op] = None, is_X=False, is_y=False):
        super().__init__(name=func.upper(), is_X=is_X, is_y=is_y, inputs=inputs, outputs=outputs)
        if kwargs is not None:
            self.check_kwargs(kwargs)
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.output_type = OutputType.FRAME

    def propagate_output_schema(self):
        """Only ``rename`` is modelled.

        A rename aimed at the columns (``columns=``, or a mapper with
        ``axis=1``/``"columns"``) remaps them. One aimed at the index -- ``index=``,
        or a positional mapper under pandas' default ``axis=0`` -- leaves the
        column names untouched, so the schema passes straight through instead of
        being given up. ``axis`` is keyword-only on ``DataFrame.rename``, so only
        the mapper itself can arrive positionally.
        """
        if self.func != "rename":
            self.output_schema = None
            return
        kwargs = self.kwargs or {}
        args = self.args or ()
        axis = kwargs.get("axis")
        if "columns" in kwargs:
            mapping = kwargs["columns"]
        elif axis in (1, "columns"):
            mapping = kwargs.get("mapper", args[0] if args else None)
        elif axis is None or axis in (0, "index"):
            mapping = {}  # renames the index (or nothing): columns unchanged
        else:
            mapping = None  # e.g. a graph-fed axis: can't tell which way it renames
        self.output_schema = _schema.rename_columns(self.inputs[0].output_schema, mapping)


def make_reset_index_op(op: MethodCallOp) -> MetadataOp | None:
    """Fuse ``reset_index(drop=True)`` into a :class:`MetadataOp`.

    Only the dropping form. ``drop=False`` promotes the index to a column and so
    changes the schema, which we cannot describe until schema propagation exists,
    and it is also pandas' default, so it is refused rather than guessed at.
    """
    kwargs = dict(op.kwargs or {})
    if op.args or kwargs.pop("drop", False) is not True or kwargs:
        return None
    new_op = MetadataOp(func="reset_index", args=(), kwargs={"drop": True},
                        inputs=op.inputs, outputs=op.outputs)
    # reset_index keeps the container kind: a series stays a series.
    new_op.output_type = op.inputs[0].output_type
    op.replace_output_of_inputs(new_op)
    return new_op


class ProjectionOp(Op):
    """Base for projection operators: keeps rows and values, selects columns."""
    logical_family = "Projection"

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
    """``df.drop(...)`` of columns.

    Only the statically column-wise form becomes a ``DropOp`` (see
    :func:`drop_targets_columns`). The pandas default (``axis=0``) drops *rows*
    by index label, which extraction turns into a ``SelectionOp(DROP)``.
    """
    fields = ["args", "kwargs", "columns"]
    def __init__(self, args: tuple | list = (), kwargs: dict = {},
        inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None):
        super().__init__(inputs=inputs, outputs=outputs)
        if kwargs is not None:
            self.check_kwargs(kwargs)
        self.args = args
        self.kwargs = kwargs
        self.columns = columns

    def _extract_args_and_kwargs(self, inputs: list):
        """The frame (operand 0) and the drop's resolved arguments."""
        return inputs[0], _resolve_args(self.args, inputs), _resolve_kwargs(self.kwargs, inputs)

    def _dropped_columns(self):
        """Column labels being dropped, or ``None`` if not statically known."""
        kwargs = self.kwargs or {}
        for key in ("columns", "labels"):
            if kwargs.get(key) is not None:
                return kwargs[key]
        return self.args[0] if self.args else None

    def propagate_output_schema(self):
        """Removes the named columns from the input schema."""
        self.output_schema = _schema.drop_columns(self.inputs[0].output_schema,
                                                  self._dropped_columns())


def make_drop_op(op: MethodCallOp) -> DropOp:
    """Rewrite a column-wise ``df.drop(...)`` call into a :class:`DropOp`."""
    new_op = DropOp(args=op.args, kwargs=op.kwargs, inputs=op.inputs, outputs=op.outputs)
    op.replace_output_of_inputs(new_op)
    return new_op


class ColumnSelectorOp(ProjectionOp):
    """A column selection by (deferred) skrub selector: keeps rows, restricts columns.

    Produced from ``skb.select(cols)``. Matches ``SelectCols`` semantics: the selector resolves against the schema at
    fit time and the *stored* column list is reused at predict time.
    """
    fields = ["selector"]

    def __init__(self, selector, inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=f"select[{selector!r}]", inputs=inputs, outputs=outputs)
        self.selector = selector
        self.selected_columns = None


def make_column_selector_op(op: TransformerOp) -> ColumnSelectorOp:
    """Rewrite a ``TransformerOp`` wrapping skrub's ``SelectCols`` into a
    :class:`ColumnSelectorOp` carrying the selector itself."""
    new_op = ColumnSelectorOp(selector=op.estimator.cols,
                              inputs=op.inputs, outputs=op.outputs)
    op.replace_output_of_inputs(new_op)
    return new_op


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
    new_op = ColumnProjectionOp(key=op.key, inputs=op.inputs, outputs=op.outputs)
    op.replace_output_of_inputs(new_op)
    return new_op
