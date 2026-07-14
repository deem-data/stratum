from typing import Callable
from skrub.selectors._base import make_selector
from stratum.optimizer.ir._ops import (OperandRef, OutputType, CallOp, GetAttrOp,
                                       MethodCallOp, Op, TransformerOp, _resolve_args, _resolve_kwargs)
from stratum._config import FLAGS
import pandas as pd
import polars as pl
from numpy import sin, cos
import logging
logger = logging.getLogger(__name__)


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


def _to_datetime_via_pandas(obj, args, kwargs):
    """Preserve pandas datetime semantics for options Polars cannot express."""
    name = getattr(obj, "name", None)
    if isinstance(obj, (pl.Series, pl.DataFrame)):
        obj = obj.to_pandas()
    result = pd.to_datetime(obj, *args, **kwargs)
    if isinstance(result, (pd.Series, pd.DataFrame)):
        return pl.from_pandas(result)
    if isinstance(result, pd.Index):
        return pl.Series(name, result)
    return result


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

    def process(self, mode: str, inputs: list):
        _obj = inputs[0]
        _args = _resolve_args(self.args, inputs)
        _kwargs = _resolve_kwargs(self.kwargs, inputs)
        if FLAGS.force_polars:
            if "columns" in _kwargs:
                _args.append(_kwargs["columns"])
            return getattr(_obj, self.func)(*_args)
        else:
            return getattr(_obj, self.func)(*_args, **_kwargs)


class ProjectionOp(Op):
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

    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        if self.method is not None:
            if FLAGS.force_polars:
                raise ValueError(f"Unsupported method: {self.method}")
            return getattr(_obj, self.method)(*_args, **_kwargs)
        if self.func is not None:
            return self.func(_obj, *_args, **_kwargs)
        raise TypeError("ProjectionOp requires either `func` or `method` to be set.")


class DropOp(ProjectionOp):
    fields = ["args", "kwargs", "columns"]
    def __init__(self, args: tuple | list = (), kwargs: dict = {},
        inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None):
        super().__init__(args=args, kwargs=kwargs, inputs=inputs, outputs=outputs, columns=columns)

    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)

        if FLAGS.force_polars:
            if "columns" in _kwargs:
                _args.append(_kwargs["columns"])
            if "ignore_errors" in _kwargs:
                _args.append(_kwargs["ignore_errors"] == "raise")
            return _obj.drop(*_args)
        else:
            return _obj.drop(*_args, **_kwargs)


class ColumnSelectorOp(Op):
    """A column selection by (deferred) skrub selector: keeps rows, restricts columns.

    Produced from ``skb.select(cols)``. Matches ``SelectCols`` semantics: the selector resolves against the schema at
    fit time and the *stored* column list is reused at predict time.
    """
    fields = ["selector"]

    def __init__(self, selector, inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=f"select[{selector!r}]", inputs=inputs, outputs=outputs)
        self.selector = selector
        self.selected_columns = None
        self.output_type = OutputType.FRAME

    def process(self, mode: str, inputs: list):
        frame = inputs[0]
        if mode == "fit_transform":
            self.selected_columns = resolve_selector_columns(frame, self.selector)
        elif self.selected_columns is None:
            raise RuntimeError(
                f"{self} was asked to transform before the selector was resolved; "
                f"run fit_transform first.")
        if FLAGS.force_polars:
            return frame.select(self.selected_columns)
        return frame[self.selected_columns]


def make_column_selector_op(op: TransformerOp) -> ColumnSelectorOp:
    """Rewrite a ``TransformerOp`` wrapping skrub's ``SelectCols`` into a
    :class:`ColumnSelectorOp` carrying the selector itself."""
    new_op = ColumnSelectorOp(selector=op.estimator.cols,
                              inputs=op.inputs, outputs=op.outputs)
    op.replace_output_of_inputs(new_op)
    return new_op


class ApplyUDFOp(ProjectionOp):
    fields = ["args", "kwargs", "columns"]
    def __init__(self, args: tuple | list = (), kwargs: dict = {},
        inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None):
        super().__init__(args=args, kwargs=kwargs, inputs=inputs, outputs=outputs, columns=columns)

    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)

        n_cols = None
        if self.columns:
            _obj = _obj[self.columns]
            if type(self.columns) == str:
                n_cols = 1
            else:
                n_cols = len(self.columns)

        if FLAGS.force_polars:
            if isinstance(_obj, pl.Series):
                n_cols = 1
            if n_cols == 1:
                if _args[0] == sin:
                    logger.debug("Rewrite UDF sin to polars sin")
                    return _obj.sin()
                elif _args[0] == cos:
                    logger.debug("Rewrite UDF cos to polars cos")
                    return _obj.cos()
                else:
                    return _obj.map_elements(*_args, **_kwargs)
            else:
                return _obj.map_rows(*_args, **_kwargs)
        else:
            return _obj.apply(*_args, **_kwargs)


class AssignOp(ProjectionOp):
    def __init__(self, args: tuple | list = (), kwargs: dict = {},
        inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None):
        super().__init__(args=args, kwargs=kwargs, inputs=inputs, outputs=outputs, columns=columns)

    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        if FLAGS.force_polars:
            checked_kwargs = {}
            for k, v in _kwargs.items():
                if isinstance(v, OperandRef):
                    raise NotImplementedError("Is not yet suppoerted, please report this issue")
                elif isinstance(v, pd.Series) or isinstance(v, pd.DataFrame):
                    logger.warning(f"Converting pandas object to polars object for column {k}")
                    checked_kwargs[k] = pl.from_pandas(v)
                elif isinstance(v, list):
                    checked_kwargs[k] = pl.Series(v)
                else:
                    checked_kwargs[k] = v
            return _obj.with_columns(*_args, **checked_kwargs)
        else:
            return _obj.assign(*_args, **_kwargs)


class DatetimeConversionOp(ProjectionOp):
    def __init__(self, args: tuple | list = (), kwargs: dict = {},
        inputs: list[Op] = None, outputs: list[Op] = None, columns: list[str] = None):
        super().__init__(args=args, kwargs=dict(kwargs or {}), inputs=inputs,
                         outputs=outputs, columns=columns)

    def process(self, mode: str, inputs: list):
        if FLAGS.force_polars:
            translated = polars_datetime_kwargs(self.args, self.kwargs)
            if translated is not None:
                # TODO: Support already-datetime and numeric operands natively;
                # the Polars string namespace only accepts string input.
                return inputs[0].str.to_datetime(**translated)
            return _to_datetime_via_pandas(inputs[0], self.args, self.kwargs)
        return pd.to_datetime(inputs[0], *self.args, **self.kwargs)


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

    def process(self, mode: str, inputs: list):
        _obj, _args, _kwargs = self._extract_args_and_kwargs(inputs)
        name = self.method
        if FLAGS.force_polars:
            name = STR_POLARS_METHODS.get(name, name)
        return getattr(_obj.str, name)(*_args, **_kwargs)


class GetAttrProjectionOp(Op):
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

    def __str__(self):
        attr_name = ".".join(self.attr_name)
        return f"GetAttrProjectionOp({attr_name}) [df]"

    def process(self, mode: str, inputs: list):
        result = inputs[0]
        tmp = result
        if FLAGS.force_polars:
            for attr in self.attr_name:
                attr = self.POLARS_ATTR_NAME_MAP.get(attr, attr)

                # TODO find better way to handle this
                if attr == "is_month_end":
                    return result.dt.month_end() == result

                # polars implements dt.day as a method, not an attribute
                # use getattr to handle both attributes and methods
                tmp = getattr(tmp, attr)
            if len(self.attr_name) == 2:
                return tmp()
            else:
                return tmp
        else:
            for attr in self.attr_name:
                tmp = getattr(tmp, attr)
            return tmp


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
