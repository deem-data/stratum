from typing import Callable
from skrub.selectors._base import make_selector
from stratum.optimizer.logical._ops import (OutputType, CallOp, GetAttrOp,
                                       MethodCallOp, Op, TransformerOp, _resolve_args, _resolve_kwargs)
from stratum.optimizer.logical import _schema
import polars as pl


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
    def __init__(self, args: tuple | list = (), kwargs: dict = {},
        inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None):
        super().__init__(args=args, kwargs=kwargs, inputs=inputs, outputs=outputs, columns=columns)

    def _drops_columns(self):
        """Whether this drop works across the columns: True, False for a row drop,
        or None when the axis isn't statically known.

        pandas defaults to ``axis=0``, i.e. dropping *rows* by index label, which
        leaves the column set untouched. ``axis`` is keyword-only on
        ``DataFrame.drop``, so it can never arrive positionally in ``args``.
        """
        kwargs = self.kwargs or {}
        if "columns" in kwargs:
            return True
        if "index" in kwargs and "axis" not in kwargs:
            return False
        axis = kwargs.get("axis", 0)
        if axis in (0, "index"):
            return False
        if axis in (1, "columns"):
            return True
        return None  # e.g. a graph-fed axis

    def _dropped_columns(self):
        """Column labels being dropped, or ``None`` if not statically known."""
        kwargs = self.kwargs or {}
        if "columns" in kwargs:
            return kwargs["columns"]
        return self.args[0] if self.args else None

    def propagate_output_schema(self):
        """``drop(columns=...)`` / ``axis=1`` removes the named columns; a row drop
        (the default ``axis=0``, or ``index=``) leaves every column in place."""
        drops_columns = self._drops_columns()
        schema = self.inputs[0].output_schema
        if drops_columns is None:
            self.output_schema = None
        elif drops_columns:
            self.output_schema = _schema.drop_columns(schema, self._dropped_columns())
        else:
            self.output_schema = schema


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
    a :class:`~stratum.optimizer.logical._ops.GetItemOp` whose container is a
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

    def propagate_output_schema(self):
        """A literal projection restricts the schema to its key, in order."""
        self.output_schema = _schema.project_columns(self.inputs[0].output_schema, self.key)


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

    def propagate_output_schema(self):
        """``df.assign(a=..., b=...)`` adds/overwrites the kwargs-named columns,
        Unknown-typed since the dtype follows the assigned expression."""
        new_columns = list(self.kwargs.keys()) if self.kwargs else []
        self.output_schema = _schema.add_columns(self.inputs[0].output_schema, new_columns)


class DatetimeConversionOp(ProjectionOp):
    def __init__(self, args: tuple | list = (), kwargs: dict = {},
        inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None):
        super().__init__(args=args, kwargs=dict(kwargs or {}), inputs=inputs,
                         outputs=outputs, columns=columns)

    def propagate_output_schema(self):
        """Converting a column keeps its name and leaves the dtype Unknown.

        The Unknown dtype is not just imprecision: the result is a Datetime whose
        *unit* is the backend's (pandas gives ``datetime64[ns]``, polars
        microseconds), the backend is only fixed during physical planning, and
        ``pl.Schema`` rejects a unit-less ``pl.Datetime``, so "Datetime of unknown
        unit" has no representation here.

        Only a column operand is column-preserving. Given a *frame*,
        ``pd.to_datetime`` is the assembly form: it reads year/month/day columns
        and fans them in to one unnamed Series (any other frame raises), so the
        input's names are not the output's.
        """
        if self.inputs[0].output_type is not OutputType.SERIES:
            self.output_schema = None
            return
        self.output_schema = _schema.cast_columns(self.inputs[0].output_schema)


class StringMethodOp(ProjectionOp):
    """A ``col.str.<method>(...)`` accessor call on a (string) column expression.

    Produced by fusing ``GetAttrProjectionOp(["str"]) + MethodCallOp`` during frame
    extraction (see :func:`make_string_method_op`), so the ``.str`` accessor never
    survives as its own op. Making the str call a first-class projection lets
    selection folding match it directly -- lifting it into a
    :class:`~stratum.optimizer.logical._column_expr.StrExpr` predicate -- instead of
    re-discovering the accessor+call shape inside the mask folder.

    polars exposes the same call on its ``.str`` namespace, with a few methods
    renamed (see :data:`STR_POLARS_METHODS`).
    """
    fields = ["method", "args", "kwargs", "columns"]

    def __init__(self, method: str, args: tuple | list = (), kwargs: dict = None,
                 inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None):
        super().__init__(method=method, args=args, kwargs=kwargs or {},
                         inputs=inputs, outputs=outputs, columns=columns)

    # `.str` methods that map one input element to one output element, so the call
    # keeps its operand's columns and only changes their dtype. Extraction fuses
    # *any* `.str.<method>` into this op, so this must be an allow-list: `extract`,
    # `extractall`, `get_dummies`, `partition` and `rpartition` return a frame of
    # new columns, `split`/`rsplit` do the same under `expand=True`, and `cat()`
    # collapses the column to a single string. Every member below was checked to
    # return a Series.
    ONE_TO_ONE_METHODS = frozenset({
        # fused into polars expressions, see STR_POLARS_METHODS
        "count", "lower", "upper", "startswith", "endswith", "len",
        "strip", "lstrip", "rstrip",
        # scalar on the pandas compatibility path
        "capitalize", "casefold", "contains", "find", "fullmatch", "match",
        "pad", "removeprefix", "removesuffix", "repeat", "replace", "rfind",
        "slice", "swapcase", "title", "zfill",
        "isalnum", "isalpha", "isdecimal", "isdigit", "islower", "isnumeric",
        "isspace", "istitle", "isupper",
    })

    def propagate_output_schema(self):
        """A one-to-one method keeps the column names, dtype Unknown (per-method,
        and the backends disagree on some). A method that fans the column out into
        several is unknown; see :data:`ONE_TO_ONE_METHODS`."""
        if self.method in self.ONE_TO_ONE_METHODS:
            self.output_schema = _schema.cast_columns(self.inputs[0].output_schema)
        else:
            self.output_schema = None


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

    # Accessor attributes whose `<ns>.<attr>` projection keeps the input's columns
    # and only changes their dtype, keyed by namespace. `make_frame_get_attr` wraps
    # *any* attribute of a frame-like input, so this must be an allow-list: `.T`
    # transposes, `.values` is an ndarray, `.shape` a tuple, and
    # `.iloc`/`.loc`/`.index`/`.columns` are not column-preserving either. Only
    # `.dt` reaches here (`.str` is fused into a `StringMethodOp` during
    # extraction), and the mask folder gates on the same set (see
    # `_column_expr._is_foldable`).
    #
    # The gate is per *attribute*, not per namespace: `.dt.components` and
    # `.dt.isocalendar` fan one column out into several (7 and 3 respectively), so
    # a namespace-level allow-list would report the input's single column for them.
    # Listed below are the `.dt` properties that yield a Series; a `.dt` *method*
    # is deliberately absent, since the schema of a bound method is meaningless
    # (the enclosing call has no rule and stays unknown).
    ELEMENTWISE_ACCESSORS = {
        "dt": frozenset({
            "date", "day", "day_of_week", "day_of_year", "dayofweek", "dayofyear",
            "days", "days_in_month", "daysinmonth", "hour", "is_leap_year",
            "is_month_end", "is_month_start", "is_quarter_end", "is_quarter_start",
            "is_year_end", "is_year_start", "microsecond", "microseconds",
            "minute", "month", "nanosecond", "nanoseconds", "quarter", "second",
            "seconds", "time", "timetz", "weekday", "year",
        }),
    }

    def propagate_output_schema(self):
        """An accessor projection (``.dt.year``) keeps the column names, dtype
        Unknown. Anything else, including a ``.dt`` attribute that fans out into
        several columns, is unknown; see :data:`ELEMENTWISE_ACCESSORS`."""
        allowed = (self.ELEMENTWISE_ACCESSORS.get(self.attr_name[0], ())
                   if len(self.attr_name) == 2 else ())
        if self.attr_name and self.attr_name[-1] in allowed:
            self.output_schema = _schema.cast_columns(self.inputs[0].output_schema)
        else:
            self.output_schema = None


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


def _get_attr_output_type(attr_name: list, container: Op) -> OutputType:
    """The kind an attribute access produces.

    ``.index`` is a sequence of labels whatever it is read off, so it is a SERIES
    even when the container is a frame. There is no ``Index`` member in the
    lattice and it would have no readers: every consumer we have treats an index
    as a sequence of values. Every other accessor (``.dt.year``, ``.str``, ...)
    keeps the container's own kind.
    """
    if attr_name == ["index"]:
        return OutputType.SERIES
    return container.output_type


def make_frame_get_attr(new_op: GetAttrProjectionOp, op: GetAttrOp) -> GetAttrProjectionOp:
    input_ = op.inputs[0]
    if isinstance(input_, GetAttrProjectionOp):
        # Fuse chained GetAttr operations
        concat_attr_name = input_.attr_name.copy()
        attr_to_add = op.attr_name if isinstance(op.attr_name, list) else [op.attr_name]
        concat_attr_name.extend(attr_to_add)

        new_input = input_.inputs[0]
        new_op = GetAttrProjectionOp(attr_name=concat_attr_name, inputs=[new_input], outputs=op.outputs)
        new_op.output_type = _get_attr_output_type(concat_attr_name, new_input)

        if len(input_.outputs) > 1:
            input_.outputs.remove(op)
            new_input.add_output(new_op)
        else:
            new_input.replace_output(input_, new_op)

    else:
        # Convert single GetAttrOp to GetAttrDataframeOp
        attr_name = op.attr_name if isinstance(op.attr_name, list) else [op.attr_name]
        new_op = GetAttrProjectionOp(attr_name=attr_name, inputs=op.inputs, outputs=op.outputs)
        new_op.output_type = _get_attr_output_type(attr_name, input_)
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
