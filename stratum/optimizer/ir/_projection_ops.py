from typing import Callable
from pandas.api.types import is_dict_like, is_list_like
from skrub.selectors._base import make_selector
from stratum.optimizer.ir._ops import (OutputType, CallOp, GetAttrOp,
                                       MethodCallOp, Op, TransformerOp, _resolve_args, _resolve_kwargs)


def resolve_selector_columns(frame, selector) -> list[str]:
    """Resolve a skrub selector (or column name / list of names) against ``frame``.

    Returns the concrete column-name list. Selectors are *deferred*: which columns
    match (e.g. ``numeric()``) depends on the data, so resolution can only happen
    once a frame with a schema is available. skrub's dispatch handles both pandas
    and polars frames.

    # TODO with schema propagation we can resolute the column name list at compile time
    """
    return make_selector(selector).expand(frame)

# pandas ``.str.<method>`` name -> polars ``.str.<method>`` name, for the methods
# whose names differ between backends. Methods that match (contains, replace, ...)
# need no entry. A method absent from a backend's str namespace simply won't run
# there. Shared by :class:`StringMethodOp` and the column-expression ``StrExpr``.
STR_POLARS_METHODS = {
    "count": "count_matches",
    "lower": "to_lowercase",
    "upper": "to_uppercase",
    "startswith": "starts_with",
    "endswith": "ends_with",
    "len": "len_chars",
    "strip": "strip_chars",
    "lstrip": "strip_chars_start",
    "rstrip": "strip_chars_end",
}


_POLARS_DATETIME_KWARGS = frozenset({"format", "errors", "exact", "cache"})


def polars_datetime_kwargs(args, kwargs) -> dict | None:
    """Translate the pandas datetime options supported by Polars string parsing.

    ``None`` means the call must stay unfused and use the pandas compatibility
    path. Positional options are deliberately kept there because their ordering
    differs between ``pd.to_datetime`` and ``Expr.str.to_datetime``.
    """
    if args:
        return None
    kwargs = dict(kwargs or {})
    if not kwargs.keys() <= _POLARS_DATETIME_KWARGS:
        return None
    errors = kwargs.pop("errors", "raise")
    if errors not in ("raise", "coerce"):
        return None
    translated = {key: kwargs[key] for key in ("format", "exact", "cache")
                  if key in kwargs}
    translated["strict"] = errors == "raise"
    return translated


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
        self.has_schema = False
        self.schema = None
        self.infer_schema()

    def infer_schema(self):
        """Infer the schema produced by a dataframe metadata operation.

        For ``rename``, pandas changes column labels in either of these forms::

            frame.rename(mapper, axis=1)
            frame.rename(columns=mapper)

        ``mapper`` may be dict-like or callable. Renaming the index does not
        affect the column schema. A schema is represented as ``(name, dtype)``
        pairs, so only the names change and their order and dtypes are retained.
        """
        if not self.inputs or not getattr(self.inputs[0], "has_schema", False):
            return None

        input_schema = getattr(self.inputs[0], "schema", None)
        if input_schema is None:
            return None
        if self.func != "rename":
            return input_schema

        args = self.args or ()
        kwargs = self.kwargs or {}

        if "columns" in kwargs:
            mapper = kwargs["columns"]
        else:
            mapper = kwargs.get("mapper", args[0] if args else None)
            axis = kwargs.get("axis", args[1] if len(args) > 1 else 0)
            if axis not in (1, "columns"):
                self.schema = input_schema
                self.has_schema = True
                return

        if is_dict_like(mapper):
            rename = lambda column: mapper[column] if column in mapper else column
        elif callable(mapper):
            rename = mapper
        else:
            # Dynamic/unsupported mappers cannot be evaluated at plan time.
            return None

        self.schema = tuple((rename(column), dtype) for column, dtype in input_schema)
        self.has_schema = True


class ProjectionOp(Op):
    logical_family = "Projection"
    fields = ["func", "method", "args", "kwargs", "columns"]

    def __init__(self, func: Callable | None = None, method: str | None = None,
        args: tuple | list = None, kwargs: dict = None,
        inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None):
        if func is not None and method is not None:
            raise ValueError("`func` and `method` are mutually exclusive; set exactly one (or neither for subclasses that override `process`).")
        if method is not None:
            name = method.upper()
        elif func is not None:
            name = func.__name__.upper()
        else:
            name = ""
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        if kwargs is not None:
            self.check_kwargs(kwargs)
        self.func = func
        self.method = method
        self.args = args
        self.columns = columns
        self.kwargs = kwargs
        self.output_type = OutputType.FRAME

    def _extract_args_and_kwargs(self, inputs: list):
        """Extract and process arguments and kwargs from inputs."""
        # The object is the implicit primary operand (index 0). For func-based ops
        # the first positional arg is that object slot, so skip it here.
        _obj = inputs[0]
        args = self.args[1:] if self.func is not None else self.args
        _args = _resolve_args(args, inputs)
        _kwargs = _resolve_kwargs(self.kwargs, inputs)
        return _obj, _args, _kwargs

    # Execution lives in the physical impls (physical/_projection_execs.py);
    # the logical op only carries config + the backend-agnostic operand plumbing
    # (`_extract_args_and_kwargs`).


class DropOp(ProjectionOp):
    fields = ["args", "kwargs", "columns"]

    def __init__(self, args: tuple | list = (), kwargs: dict = None,
        inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None):
        super().__init__(args=args, kwargs=kwargs, inputs=inputs, outputs=outputs, columns=columns)
        self.schema = None
        self.has_schema = False
        self.infer_schema()

    def infer_schema(self):
        if not self.inputs or not getattr(self.inputs[0], "has_schema", False):
            return None
        input_schema = self.inputs[0].schema
        columns_to_remove = None

        args = self.args or ()
        kwargs = self.kwargs or {}

        if "columns" in kwargs:
            columns_to_remove = kwargs["columns"]
        else:
            labels = kwargs.get("labels", args[0] if args else None)
            axis = kwargs.get("axis", args[1] if len(args) > 1 else 0)
            if labels is not None and axis in (1, "columns"):
                columns_to_remove = labels

        if columns_to_remove is None:
            self.schema = input_schema
            self.has_schema = True
            return

        if not is_list_like(columns_to_remove):
            columns_to_remove = [columns_to_remove]

        self.schema = tuple(
            (column, dtype)
            for column, dtype in input_schema
            if column not in columns_to_remove
        )
        self.has_schema = True

class ColumnSelectorOp(Op):
    """A column selection by (deferred) skrub selector: keeps rows, restricts columns.

    Produced from ``skb.select(cols)``. Matches ``SelectCols`` semantics: the selector resolves against the schema at
    fit time and the *stored* column list is reused at predict time.
    """
    logical_family = "Projection"
    fields = ["selector"]

    def __init__(self, selector, inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=f"select[{selector!r}]", inputs=inputs, outputs=outputs)
        self.selector = selector
        self.selected_columns = None
        self.output_type = OutputType.FRAME


def make_column_selector_op(op: TransformerOp) -> ColumnSelectorOp:
    """Rewrite a ``TransformerOp`` wrapping skrub's ``SelectCols`` into a
    :class:`ColumnSelectorOp` carrying the selector itself."""
    new_op = ColumnSelectorOp(selector=op.estimator.cols,
                              inputs=op.inputs, outputs=op.outputs)
    op.replace_output_of_inputs(new_op)
    return new_op


class ColumnProjectionOp(Op):
    """Column selection by literal name(s): ``df[key]`` where ``key`` is a
    column label or list of labels.

    The specialised, always-column form of a ``df[...]`` indexing: produced from
    a :class:`~stratum.optimizer.ir._ops.GetItemOp` whose container is a
    ``FRAME`` and whose key is a literal ``str`` (a single column -> ``SERIES``)
    or ``list[str]`` (a sub-frame -> ``FRAME``). Unlike :class:`ColumnSelectorOp`
    (``skb.select``, a deferred skrub selector resolved against the schema), the
    columns are given verbatim, so no fit-time resolution is needed. The general
    ``GetItemOp`` stays the fallback for masks, graph-fed keys, slices and
    positional indexing.
    """
    logical_family = "Projection"
    fields = ["key"]

    def __init__(self, key, inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=f"cols[{key!r}]", inputs=inputs, outputs=outputs)
        self.key = key
        # A single label extracts a column (SERIES); a list selects a sub-frame.
        self.output_type = (OutputType.SERIES if isinstance(key, str)
                            else OutputType.FRAME)


def make_column_projection_op(op) -> ColumnProjectionOp:
    """Rewrite a column-selecting ``df[key]`` :class:`GetItemOp` (a literal
    ``str`` / ``list[str]`` key on a frame) into a :class:`ColumnProjectionOp`."""
    new_op = ColumnProjectionOp(key=op.key, inputs=op.inputs, outputs=op.outputs)
    op.replace_output_of_inputs(new_op)
    return new_op


class ApplyUDFOp(ProjectionOp):
    fields = ["args", "kwargs", "columns"]
    def __init__(self, args: tuple | list = (), kwargs: dict = {},
        inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None):
        super().__init__(args=args, kwargs=kwargs, inputs=inputs, outputs=outputs, columns=columns)


class AssignOp(ProjectionOp):
    def __init__(self, args: tuple | list = (), kwargs: dict = {},
        inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None):
        super().__init__(args=args, kwargs=kwargs, inputs=inputs, outputs=outputs, columns=columns)


class DatetimeConversionOp(ProjectionOp):
    def __init__(self, args: tuple | list = (), kwargs: dict = {},
        inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None):
        super().__init__(args=args, kwargs=dict(kwargs or {}), inputs=inputs,
                         outputs=outputs, columns=columns)


class StringMethodOp(ProjectionOp):
    """A ``col.str.<method>(...)`` accessor call on a (string) column expression.

    Produced by fusing ``GetAttrProjectionOp(["str"]) + MethodCallOp`` during frame
    extraction (see :func:`make_string_method_op`), so the ``.str`` accessor never
    survives as its own op. Making the str call a first-class projection lets
    selection folding match it directly -- lifting it into a
    :class:`~stratum.optimizer.ir._column_expr.StrExpr` predicate -- instead of
    re-discovering the accessor+call shape inside the mask folder.

    polars exposes the same call on its ``.str`` namespace, with a few methods
    renamed (see :data:`STR_POLARS_METHODS`).
    """
    fields = ["method", "args", "kwargs", "columns"]

    def __init__(self, method: str, args: tuple | list = (), kwargs: dict = None,
                 inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None):
        super().__init__(method=method, args=args, kwargs=kwargs or {},
                         inputs=inputs, outputs=outputs, columns=columns)


class GetAttrProjectionOp(Op):
    logical_family = "Projection"
    fields = ["attr_name"]

    # NOTE: Polars and Pandas differ in semantics for some datetime attributes:
    #   - dayofweek: Pandas uses Monday=0, Polars weekday() uses Monday=1 (ISO 8601)
    #   - dayofyear: Pandas is 1-indexed, Polars ordinal_day() is also 1-indexed (same)
    POLARS_ATTR_NAME_MAP = {"dayofweek": "weekday","dayofyear": "ordinal_day"}

    def __init__(self, attr_name: list[str] | str = None, inputs: list[Op] = None, outputs: list[Op] = None):
        if attr_name is None:
            self.attr_name = []
        elif isinstance(attr_name, str):
            self.attr_name = [attr_name]
        else:
            self.attr_name = attr_name
        attr_name_str = ".".join(self.attr_name) if self.attr_name else '?'
        super().__init__(name=attr_name_str)
        self.inputs = inputs
        self.outputs = outputs
        self.output_type = OutputType.FRAME


def make_datetime_conversion_op(op: CallOp) -> DatetimeConversionOp:
    # arg[0] is the input
    if len(op.args) > 1:
        args = op.args[1:]
    else:
        args = ()

    new_op = DatetimeConversionOp(args=args, kwargs=op.kwargs, inputs=op.inputs, outputs=op.outputs)
    # Converting a column yields a column and a frame yields a frame: keep the
    # input's kind (ProjectionOp defaults to FRAME).
    if op.inputs:
        new_op.output_type = op.inputs[0].output_type
    op.replace_output_of_inputs(new_op)
    return new_op


def make_frame_get_attr(new_op: GetAttrProjectionOp, op: GetAttrOp) -> GetAttrProjectionOp:
    input_ = op.inputs[0]
    if isinstance(input_, GetAttrProjectionOp):
        # Fuse chained GetAttr operations
        concat_attr_name = input_.attr_name.copy()
        attr_to_add = op.attr_name if isinstance(op.attr_name, list) else [op.attr_name]
        concat_attr_name.extend(attr_to_add)

        new_input = input_.inputs[0]
        new_op = GetAttrProjectionOp(attr_name=concat_attr_name, inputs=[new_input], outputs=op.outputs)
        # Attribute access (e.g. `.dt.year`, `.str...`) keeps the container's
        # tabular kind: a series stays a series, a frame stays a frame.
        new_op.output_type = new_input.output_type

        if len(input_.outputs) > 1:
            input_.outputs.remove(op)
            new_input.add_output(new_op)
        else:
            new_input.replace_output(input_, new_op)

    else:
        # Convert single GetAttrOp to GetAttrDataframeOp
        attr_name = op.attr_name if isinstance(op.attr_name, list) else [op.attr_name]
        new_op = GetAttrProjectionOp(attr_name=attr_name, inputs=op.inputs, outputs=op.outputs)
        new_op.output_type = input_.output_type
        op.replace_output_of_inputs(new_op)
    return new_op


def make_string_method_op(op: MethodCallOp) -> StringMethodOp:
    """Fuse ``col.str.<method>(...)`` into a single :class:`StringMethodOp`.

    ``op.inputs[0]`` is the ``GetAttrProjectionOp(["str"])`` accessor; the new op
    takes the *column* (the accessor's source) as its primary operand instead, so
    the accessor drops out of the graph. The method's ``args``/``kwargs`` (which may
    carry ``OperandRef``s into the remaining inputs) and those inputs are unchanged,
    so operand indices stay valid without renumbering.

    The accessor is only detached when this was its last consumer -- a ``.str``
    accessor shared by several method calls (not the common case) stays in the graph
    until the final call is fused.
    """
    accessor = op.inputs[0]
    column = accessor.inputs[0]
    new_op = StringMethodOp(method=op.method_name, args=op.args, kwargs=op.kwargs,
                            inputs=[column, *op.inputs[1:]], outputs=list(op.outputs))
    # `.str.<method>` keeps the column's tabular kind (a series stays a series).
    new_op.output_type = accessor.output_type

    # Rewire every operand except the accessor: the args feed the new op, and the
    # column now feeds it directly (in place of feeding the accessor).
    for in_ in op.inputs[1:]:
        in_.replace_output(op, new_op)
    column.add_output(new_op)
    accessor.outputs = [o for o in accessor.outputs if o is not op]
    if not accessor.outputs:
        column.outputs = [o for o in column.outputs if o is not accessor]
        accessor.inputs = []
    return new_op
