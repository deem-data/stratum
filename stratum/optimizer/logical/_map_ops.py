"""Column-map operators (MapOp).

A map keeps the rows of its source and computes columns from them: it changes
column values or dtypes (``.str``/``.dt`` accessors, ``pd.to_datetime``, a UDF,
``astype``), or adds columns to the frame (``df.assign``), possibly from other
intermediates. This sets maps apart from projections
(:mod:`stratum.optimizer.logical._projection_ops`), which only select columns
and never modify them.

Two kinds of map:

* :class:`AssignMapOp` computes its columns from backend-agnostic
  :class:`~stratum.optimizer.logical._column_expr.ColumnExpr` trees. The grammar
  is restricted to natively-lazy computations (arithmetic, boolean logic,
  ``.str``/``.dt`` accessors, datetime parsing); on polars all entries compile
  into one ``with_columns`` kernel. Anything outside the grammar stays in the
  graph and feeds the map through an ``OperandLeaf`` input. :class:`SeriesMapOp`
  is the same for one series that no assign takes in.
* Call-based maps (:class:`CallMapOp` and its subclasses) replay one recorded
  pandas call; :class:`StringMethodOp`, :class:`DatetimeFieldOp`,
  :class:`DatetimeConversionOp` and the ``ColumnMethodOp`` family hold one parsed
  call. They are what the folder lifts
  into ``ColumnExpr`` nodes, and the fallback where it cannot.
"""
from __future__ import annotations

import pandas as pd
import polars as pl

from stratum.optimizer.logical._base import IndexEffect, Uncacheable
from stratum.optimizer.logical._ops import (CallOp, GetAttrOp, GetItemOp, MethodCallOp, Op,
                                            OperandRef, OutputType, ValueOp, substitute_op)
from stratum.optimizer.logical._column_expr import (
    UFUNCS, BinOpExpr, Col, ColumnExpr, Const, OperandLeaf)
from stratum.optimizer.logical._index_ops import IndexAccessOp
from stratum.optimizer.logical._metadata_ops import make_structure_read_op
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp
from stratum.frontend._pandas_calls import bind_pandas_function
from stratum.optimizer.logical import _schema


class MapOp(Op):
    """Base for column-map operators: row-preserving, computes column values."""
    logical_family = "Map"
    index_effect = IndexEffect.KEEP

    def __init__(self, name: str = "", inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        self.output_type = OutputType.FRAME


class CallMapOp(MapOp):
    """Base for the maps that replay a recorded pandas call on their receiver.

    ``args``/``kwargs`` are the call's arguments after the receiver, which is
    operand 0. Only the subclasses are built; each says which call it replays.
    """
    fields = ["args", "kwargs"]

    def __init__(self, args: tuple | list = (), kwargs: dict | None = None,
                 inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(inputs=inputs, outputs=outputs)
        if kwargs is not None:
            self.check_kwargs(kwargs)
        self.args = args
        self.kwargs = kwargs


class ApplyUDFOp(CallMapOp):
    """``df.apply(func, ...)`` or ``series.apply(func, ...)``."""


class AssignOp(CallMapOp):
    """``df.assign(...)`` that does not fold into an :class:`AssignMapOp`."""
    cacheable = True

    def propagate_output_schema(self):
        """``df.assign(a=..., b=...)`` adds/overwrites the kwargs-named columns,
        Unknown-typed since the dtype follows the assigned expression."""
        new_columns = list(self.kwargs.keys()) if self.kwargs else []
        self.output_schema = _schema.add_columns(self.inputs[0].output_schema, new_columns)


class DatetimeConversionOp(MapOp):
    """Parses a column of strings into datetimes (``pd.to_datetime``).

    ``format``: the ``strftime`` pattern, ``None`` to infer it. ``exact``: whether
    the pattern must match the whole string (``None``: pandas' default).
    ``cache``: reuse the result for repeated strings. ``errors``: ``"raise"`` on
    an unparseable string, or ``"coerce"`` it to a missing value. Each may be an
    ``OperandRef`` when graph-fed.
    """
    fields = ["format", "exact", "cache", "errors"]
    cacheable = True

    ERRORS = ("raise", "coerce")

    def __init__(self, format=None, exact=None, cache=True, errors="raise",
                 inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name="TO_DATETIME", inputs=inputs, outputs=outputs)
        if not isinstance(errors, OperandRef) and errors not in self.ERRORS:
            raise ValueError(f"errors must be one of {self.ERRORS}, got {errors!r}")
        self.format = format
        self.exact = exact
        self.cache = cache
        self.errors = errors

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
        input's names are not the output's. Extraction leaves that form opaque.
        """
        if self.inputs[0].output_type is not OutputType.SERIES:
            self.output_schema = None
            return
        self.output_schema = _schema.cast_columns(self.inputs[0].output_schema)


class StringMethodOp(MapOp):
    """A ``col.str.<method>(...)`` accessor call on a (string) column expression.

    Produced by fusing ``AttributeReadOp(["str"]) + MethodCallOp`` during frame
    extraction (see :func:`make_string_method_op`), so the ``.str`` accessor never
    survives as its own op. Making the str call a first-class map lets
    selection folding match it directly -- lifting it into a
    :class:`~stratum.optimizer.logical._column_expr.StrExpr` predicate -- instead of
    re-discovering the accessor+call shape inside the mask folder.

    ``method`` keeps pandas' name and ``params`` holds the arguments passed, by
    pandas' parameter names, in the signature's order; a graph-fed one is an
    ``OperandRef``. The polars impl translates both (see
    ``physical/_polars_translate.py``).
    """
    fields = ["method", "params"]
    cacheable = True

    def __init__(self, method: str = None, params: dict = None,
                 inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=method.upper() if method is not None else "",
                         inputs=inputs, outputs=outputs)
        self.method = method
        self.params = dict(params or {})

    # `.str` methods that map one input element to one output element, so the call
    # keeps its operand's columns and only changes their dtype. Extraction fuses
    # *any* `.str.<method>` into this op, so this must be an allow-list: `extract`,
    # `extractall`, `get_dummies`, `partition` and `rpartition` return a frame of
    # new columns, `split`/`rsplit` do the same under `expand=True`, and `cat()`
    # collapses the column to a single string. Every member below was checked to
    # return a Series.
    ONE_TO_ONE_METHODS = frozenset({
        "count", "lower", "upper", "startswith", "endswith", "len",
        "strip", "lstrip", "rstrip", "capitalize", "casefold", "contains", "find", "fullmatch", "match",
        "pad", "removeprefix", "removesuffix", "repeat", "replace", "rfind",
        "slice", "swapcase", "title", "zfill",
        "isalnum", "isalpha", "isdecimal", "isdigit", "islower", "isnumeric",
        "isspace", "istitle", "isupper",
    })

    def lineage_config(self):
        if self.method not in self.ONE_TO_ONE_METHODS:
            raise Uncacheable(f"str.{self.method} is not a known string method")
        return super().lineage_config()

    def propagate_output_schema(self):
        """A one-to-one method keeps the column names, dtype Unknown (per-method,
        and the backends disagree on some). A method that fans the column out into
        several is unknown; see :data:`ONE_TO_ONE_METHODS`."""
        if self.method in self.ONE_TO_ONE_METHODS:
            self.output_schema = _schema.cast_columns(self.inputs[0].output_schema)
        else:
            self.output_schema = None


class DatetimeFieldOp(MapOp):
    """Reads one field of each datetime, ``col.dt.<field>`` (``year``, ``hour``, ...).

    ``field`` keeps pandas' name and meaning; the polars impl translates it (see
    ``physical/_polars_translate.py``).
    """
    fields = ["field"]
    cacheable = True

    # The `.dt` properties that yield one value per datetime, so the read keeps
    # the input's columns and only changes their dtype. Extraction sees *any*
    # attribute read, so this is an allow-list: `.dt.components` and
    # `.dt.isocalendar` fan one column out into several (7 and 3 respectively),
    # `.dt.tz` and `.dt.freq` are a single value, and a `.dt` method read without
    # its call is a bound method. Those stay an AttributeReadOp.
    FIELDS = frozenset({
        "date", "day", "day_of_week", "day_of_year", "dayofweek", "dayofyear",
        "days", "days_in_month", "daysinmonth", "hour", "is_leap_year",
        "is_month_end", "is_month_start", "is_quarter_end", "is_quarter_start",
        "is_year_end", "is_year_start", "microsecond", "microseconds",
        "minute", "month", "nanosecond", "nanoseconds", "quarter", "second",
        "seconds", "time", "timetz", "weekday", "year",
    })

    def __init__(self, field: str = None, inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=f"dt.{field}", inputs=inputs, outputs=outputs)
        self.field = field

    def propagate_output_schema(self):
        """A field read keeps the column names, dtype Unknown."""
        self.output_schema = _schema.cast_columns(self.inputs[0].output_schema)


class AttributeReadOp(Op):
    """An attribute read off a frame or column that the optimizer does not model:
    ``.loc``, ``.T``, ``.values``, the ``.str``/``.dt``/``.cat`` namespaces, a ``.dt``
    property that is not one value per datetime, ...

    ``attr_name`` is the chain of names read, e.g. ``["cat", "codes"]``. Opaque and
    pandas-only: it reads pandas' attributes as recorded. Not a map, since most of
    these change the shape (``.T``) or leave the frame world (``.values``).
    """
    logical_family = "Attribute"
    fields = ["attr_name"]
    # `.T` turns the columns into rows and `.values` drops the labels.
    index_effect = IndexEffect.NEW

    def __init__(self, attr_name: list[str] = None, inputs: list[Op] = None,
                 outputs: list[Op] = None):
        self.attr_name = list(attr_name or [])
        super().__init__(name=".".join(self.attr_name) or "?", inputs=inputs, outputs=outputs)


class AssignMapOp(MapOp):
    """``df.assign(...)`` with each assigned column folded to a ``ColumnExpr``.

    ``entries`` maps a new column name to its series-valued expression; input
    columns pass through unchanged.
    """
    fields = ["entries"]
    cacheable = True

    def __init__(self, entries: dict[str, ColumnExpr],
                 inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=f"assign: {', '.join(entries)}",
                         inputs=inputs, outputs=outputs)
        self.entries = entries

    def propagate_output_schema(self):
        """``df.assign(...)`` folded to a map: adds/overwrites the entry names,
        Unknown-typed, and passes the source columns through."""
        self.output_schema = _schema.add_columns(self.inputs[0].output_schema,
                                                 list(self.entries))


#: The name of a series fed in from elsewhere, which only its values tell.
UNKNOWN_NAME = object()
#: What a scalar operand contributes to a series name: nothing.
_NO_NAME = object()


class SeriesMapOp(MapOp):
    """A series computed row by row from columns of one frame (not ``Series.map``).

    ``expr`` is the folded :class:`~stratum.optimizer.logical._column_expr.ColumnExpr`
    over the source frame, operand 0: ``df["a"] * 2 + df["b"]``,
    ``df["s"].str.upper()``, ``df["a"].fillna(0)``. What it cannot fold feeds it
    through an ``OperandLeaf``: a scalar (``df["a"].mean()``), another series map
    of the same frame, or a series of other rows, which pandas aligns by label.
    """
    fields = ["expr"]
    cacheable = True

    def __init__(self, expr: ColumnExpr, inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name="series_map", inputs=inputs, outputs=outputs)
        self.output_type = OutputType.SERIES
        self.expr = expr

    @property
    def index_effect(self):
        """The source's rows, unless a series of other rows joins them by label."""
        return (IndexEffect.KEEP if all(self.reads_source_rows(leaf.ref.k)
                                        for leaf in self._leaves())
                else IndexEffect.COMMON)

    def reads_source_rows(self, k: int) -> bool:
        """Whether operand ``k`` pairs with the source's rows by position as well
        as by label: a scalar, or a series map of the same rows."""
        if k == 0:
            return False  # the whole frame, as a value of its own
        leaf = self.inputs[k]
        return leaf.output_type is OutputType.SCALAR or (
            isinstance(leaf, SeriesMapOp) and leaf.inputs[0] is self.inputs[0]
            and leaf.index_effect is IndexEffect.KEEP)

    def result_name(self):
        """The name pandas gives the result, or :data:`UNKNOWN_NAME`.

        A column keeps its name through every method and through arithmetic with
        a scalar; two series combined keep it only when both have it, else the
        result has none (``None``).
        """
        return self._name_of(self.expr)

    def _name_of(self, expr):
        if isinstance(expr, Col):
            return expr.name
        if isinstance(expr, Const):
            return _NO_NAME
        if isinstance(expr, OperandLeaf):
            k = expr.ref.k
            if k and self.inputs[k].output_type is OutputType.SCALAR:
                return _NO_NAME
            return self.inputs[k].result_name() if self.reads_source_rows(k) else UNKNOWN_NAME
        if isinstance(expr, BinOpExpr):
            left, right = self._name_of(expr.left), self._name_of(expr.right)
            if left is _NO_NAME:
                return right
            if right is _NO_NAME:
                return left
            if left is UNKNOWN_NAME or right is UNKNOWN_NAME:
                return UNKNOWN_NAME
            return left if left == right else None
        # Every other node (a unary operator, a .str/.dt call, a column method)
        # keeps the name of the series it is called on.
        operand = getattr(expr, "operand", None)
        return self._name_of(operand) if isinstance(operand, ColumnExpr) else UNKNOWN_NAME

    def _leaves(self):
        stack = [self.expr]
        while stack:
            expr = stack.pop()
            if isinstance(expr, OperandLeaf):
                yield expr
            stack.extend(expr.children())

    def propagate_output_schema(self):
        """One column under the result's name, Unknown-typed; unknown when the
        name is not a string."""
        name = self.result_name()
        self.output_schema = (pl.Schema({name: _schema.UNKNOWN_DTYPE})
                              if isinstance(name, str) else None)


# --- Extraction: recorded calls -> call-based maps ----------------------------

def ufunc_name(op: Op) -> str | None:
    """The :data:`~stratum.optimizer.logical._column_expr.UFUNCS` name of a call
    ``np.<func>(x)`` on its one operand, with nothing else passed; else ``None``."""
    if not isinstance(op, CallOp) or op.kwargs or list(op.args or ()) != [OperandRef(0)]:
        return None
    return next((name for name, func in UFUNCS.items() if op.func is func), None)


#: The ``pd.to_datetime`` arguments a :class:`DatetimeConversionOp` holds.
_DATETIME_OPTIONS = frozenset({"format", "exact", "cache", "errors"})


def make_datetime_conversion_op(op: CallOp) -> DatetimeConversionOp | None:
    """Parse ``pd.to_datetime(column, ...)`` into a :class:`DatetimeConversionOp`.

    Returns ``None`` (the call stays opaque and runs on pandas as recorded) for
    anything else: options other than ``format``, ``exact``, ``cache`` and
    ``errors="raise"``/``"coerce"``, and a frame operand, which is the assembly
    form. An option's value may be graph-fed.
    """
    call = bind_pandas_function(pd.to_datetime, op.args, op.kwargs)
    if call is None:
        return None
    arg = call.pop("arg", None)
    if (not isinstance(arg, OperandRef) or arg.k != 0
            or op.inputs[0].output_type is not OutputType.SERIES
            or set(call) - _DATETIME_OPTIONS):
        return None
    errors = call.get("errors", "raise")
    if not isinstance(errors, OperandRef) and errors not in DatetimeConversionOp.ERRORS:
        return None
    return substitute_op(op, DatetimeConversionOp(
        format=call.get("format"), exact=call.get("exact"),
        cache=call.get("cache", True), errors=errors), keep_kind=True)


def make_frame_get_attr(op: GetAttrOp) -> Op:
    """Parse an attribute read off frame-world data.

    ``.index`` reads the row labels, ``.shape`` and ``.columns`` the structure, and
    ``.dt.<field>`` a datetime field. Any other read becomes an opaque
    :class:`AttributeReadOp`; a read off one extends its chain of names.
    """
    if op.attr_name == "index" or op.attr_name == ["index"]:
        return substitute_op(op, IndexAccessOp())
    structure_op = make_structure_read_op(op)
    if structure_op is not None:
        return structure_op
    input_ = op.inputs[0]
    attr_name = op.attr_name if isinstance(op.attr_name, list) else [op.attr_name]
    if not isinstance(input_, AttributeReadOp):
        new_op = substitute_op(op, AttributeReadOp(attr_name=attr_name))
        # An attribute read keeps the container's kind.
        new_op.output_type = input_.output_type
        return new_op

    # Fuse the chained read with the one it reads off.
    chain = [*input_.attr_name, *attr_name]
    source = input_.inputs[0]
    if len(chain) == 2 and chain[0] == "dt" and chain[1] in DatetimeFieldOp.FIELDS:
        new_op = DatetimeFieldOp(field=chain[1], inputs=[source], outputs=op.outputs)
    else:
        new_op = AttributeReadOp(attr_name=chain, inputs=[source], outputs=op.outputs)
    new_op.output_type = input_.output_type
    if len(input_.outputs) > 1:
        input_.outputs.remove(op)
        source.add_output(new_op)
    else:
        source.replace_output(input_, new_op)
    return new_op


def make_string_method_op(op: MethodCallOp) -> StringMethodOp | None:
    """Fuse ``col.str.<method>(...)`` into a single :class:`StringMethodOp`.

    ``op.inputs[0]`` is the ``AttributeReadOp(["str"])`` accessor; the new op
    takes the *column* (the accessor's source) as its primary operand instead, so
    the accessor drops out of the graph. The call is bound against pandas'
    ``.str`` method; its arguments (which may carry ``OperandRef``s into the
    remaining inputs) and those inputs are unchanged, so operand indices stay
    valid without renumbering. A call pandas would reject stays opaque.

    The accessor is only detached when this was its last consumer -- a ``.str``
    accessor shared by several method calls (not the common case) stays in the graph
    until the final call is fused.
    """
    if not hasattr(pd.Series.str, op.method_name):
        return None
    params = op.bind_pandas(owner=pd.Series.str)
    if params is None:
        return None
    accessor = op.inputs[0]
    column = accessor.inputs[0]
    new_op = StringMethodOp(method=op.method_name, params=params,
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
        accessor.detach_and_prune()
    return new_op


# --- Folding: assign subgraphs -> AssignMapOp ---------------------------------

def _is_scalar_constant(value) -> bool:
    """Whether an assign kwarg constant can fold to a ``Const`` entry.

    Scalars broadcast identically on every backend and fold; sequence-like
    values (lists, arrays, series) keep the ``AssignOp`` fallback.
    """
    if isinstance(value, str):
        return True
    return not hasattr(value, "__len__") and not isinstance(
        value, (pd.Series, pd.DataFrame, pl.Series, pl.DataFrame))


def make_assign_map_op(op: MethodCallOp) -> AssignMapOp | None:
    """Fold ``df.assign(**kwargs)`` into an :class:`AssignMapOp`.

    All graph-fed kwargs share one folder, so a producer feeding several columns
    folds once. Returns ``None`` for non-foldable calls (positional args or a
    sequence-valued constant kwarg), leaving the plain ``AssignOp`` in place.
    """
    # The folder absorbs the call-based maps defined above, so it imports this
    # module and has to be imported lazily here.
    from stratum.optimizer.logical._expr_folding import _Folder
    if op.args:
        return None
    kwargs = op.kwargs or {}
    if not kwargs:
        return None
    src = op.inputs[0]
    ref_names, roots, const_entries = [], [], {}
    for name, value in kwargs.items():
        if isinstance(value, OperandRef):
            ref_names.append(name)
            roots.append(op.inputs[value.k])
        elif _is_scalar_constant(value):
            const_entries[name] = Const(value)
        else:
            return None

    folder = _Folder(src)
    exprs = folder.fold_many(roots, root_consumer=op)
    # Preserve the kwargs' assignment order (later columns may overwrite earlier).
    entries = {name: (const_entries[name] if name in const_entries
                      else exprs[ref_names.index(name)])
               for name in kwargs}

    new_op = AssignMapOp(entries=entries,
                         inputs=[src, *folder.leaf_ops], outputs=list(op.outputs))
    folder.install(new_op, replaced=[op])
    return new_op


# --- Folding: standalone series chains -> SeriesMapOp ---------------------------

def fold_series_maps(ops: list[Op], root: Op) -> Op:
    """Fold each chain of row-local series ops left standing into a :class:`SeriesMapOp`.

    Runs once frame extraction is done: the chains an assign, a mask or a
    grouping absorbed are gone, so what is left stands alone. ``ops`` is the
    plan in topological order; consumers are visited first, so a chain folds
    from its last op. An op feeding two chains has a consumer outside each, so
    it stays an operand of both and becomes a series map of its own after them.
    Returns the root, which a fold may replace.
    """
    from stratum.optimizer.logical._expr_folding import _Folder
    probe = _Folder(src=None)
    absorbed: set[int] = set()
    for op in reversed(ops):
        if id(op) in absorbed or not _is_chain_op(op, probe):
            continue
        if op.outputs and all(_is_chain_op(out, probe) for out in op.outputs):
            continue  # a consumer's chain takes it in
        src = _chain_source(op, probe)
        if src is None:
            continue
        folder = _Folder(src)
        expr = folder.fold_many([op], root_consumer=None)[0]
        expr, leaf_ops = _inline_column_reads(expr, src, folder.leaf_ops)
        consumers = list(op.outputs)
        new_op = SeriesMapOp(expr, inputs=[src, *leaf_ops], outputs=consumers)
        for consumer in consumers:
            consumer.replace_input(op, new_op)
        folder.install(new_op, replaced=[])
        absorbed.update(id(node) for node in folder.absorbed)
        if root is op:
            root = new_op
    return root


def _is_chain_op(op: Op, probe) -> bool:
    """A series op the folder can take in that computes something: not a column
    read or a constant, which are the chain's leaves."""
    return (op.output_type is OutputType.SERIES and probe._is_foldable(op)
            and not isinstance(op, ValueOp))


def _column_read_of(op: Op) -> Op | None:
    """The frame ``op`` reads one column of by name, if it does."""
    if (isinstance(op, (GetItemOp, ColumnProjectionOp)) and isinstance(op.key, str)
            and op.inputs and op.inputs[0].output_type is OutputType.FRAME):
        return op.inputs[0]
    return None


def _chain_source(root: Op, probe) -> Op | None:
    """The frame most of the chain's columns are read from; the first one read
    on a tie. ``None`` when it reads no column."""
    counts: dict[int, list] = {}
    seen: set[int] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        frame = _column_read_of(node)
        if frame is not None:
            counts.setdefault(id(frame), [frame, 0])[1] += 1
        elif probe._is_foldable(node):
            # Reversed, so operands are visited left to right.
            stack.extend(reversed(probe._producer_ops(node)))
    if not counts:
        return None
    return max(counts.values(), key=lambda entry: entry[1])[0]


def _inline_column_reads(expr: ColumnExpr, src: Op, leaf_ops: list[Op]):
    """``expr`` with every operand that reads one column of ``src`` as a ``Col``.

    The folder keeps a column read that something else reads too as an operand;
    reading it again in the expression costs nothing and keeps it one of the
    source's columns, by name. Returns the expression and the remaining operands.
    """
    refs, kept = {}, []
    for k, leaf in enumerate(leaf_ops, start=1):
        if _column_read_of(leaf) is src:
            refs[k] = Col(leaf.key)
        else:
            kept.append(leaf)
            refs[k] = OperandLeaf(OperandRef(len(kept)))
    return _replace_leaves(expr, refs), kept


def _replace_leaves(expr: ColumnExpr, refs: dict) -> ColumnExpr:
    if isinstance(expr, OperandLeaf):
        return refs.get(expr.ref.k, expr)
    return expr.map_children(lambda child: _replace_leaves(child, refs))
