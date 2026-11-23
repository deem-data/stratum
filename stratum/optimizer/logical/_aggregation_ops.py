from typing import NamedTuple

import pandas as pd

from stratum.optimizer.logical._base import IndexEffect
from stratum.optimizer.logical._column_expr import AggExpr, AllCols, Col, OperandLeaf
from stratum.optimizer.logical._expr_folding import _Folder
from stratum.optimizer.logical._ops import (
    BinOp, UnaryOp, GetItemOp, OperandRef, OutputType, MethodCallOp, Op, ValueOp)
from stratum.optimizer.logical._sort_ops import SortOp
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp
from stratum.optimizer.logical import _schema


class AggregateOp(Op):
    """Aggregation over a frame or series, grouped or whole-object.

    Models both ``groupby(by).agg(...)`` and a bare reduction such as
    ``df.sum()``. The aggregation itself is a tuple of ``(name, AggExpr)``
    entries, so ``.sum()`` and ``.agg("sum")`` normalize to the same value and
    compare equal under CSE, and a computed aggregation like ``SUM(a * b)`` needs
    no separate map op. ``name`` is ``None`` when the output name comes from the
    entry's own expression (``Col("v")`` names it ``"v"``; :class:`AllCols` names
    one output per source column).

    ``grouping`` holds the grouping keys as expressions, which is what lets
    ``value_counts`` (group by the series' own values) and expression grouping
    (``groupby(df["d"].dt.year)``) share one representation.

    Grouping options describe result semantics. Execution strategy is chosen
    by the physical layer. Defaults follow pandas:

    * ``sort_keys``: order the groups by key, or else by first appearance.
    * ``drop_null_keys``: drop the groups whose key is missing.
    * ``observed``: for a categorical key, only the categories that occur form
      groups.
    * ``keys_as_columns``: the keys become output columns rather than the row
      labels (pandas' ``as_index=False``).
    * ``level``: the index level the grouping refers to, ``None`` for none.
    * ``sort_categories``: order categorical groups by their declared domain while
      leaving other groups in appearance order, as ``value_counts`` requires.

    Pure config -- execution is provided by the physical impls in
    ``physical/_aggregation_execs.py``, selected at plan time.
    """
    logical_family = "Aggregation"
    fields = ["grouped", "grouping", "aggregations", "sort_keys", "drop_null_keys",
              "observed", "keys_as_columns", "level", "sort_categories", "output_type"]
    cacheable = True

    def __init__(self, grouped: bool = False,
                 grouping: tuple = (),
                 aggregations: tuple = (),
                 sort_keys: bool = True,
                 drop_null_keys: bool = True,
                 observed: bool = True,
                 keys_as_columns: bool = False,
                 level=None,
                 sort_categories: bool = False,
                 output_type: OutputType | None = None,
                 inputs: list[Op] | None = None, outputs: list[Op] | None = None):
        # The reductions go in the name so the base helper renders them for both
        # the logical family and the bound physical impl.
        super().__init__(name=_render_name(grouping, aggregations),
                         inputs=inputs, outputs=outputs)
        self.grouped = grouped
        # Tuples, not lists: `clone_value` recurses tuples but passes a list
        # through by reference, and `config_key`/`remap_operand_refs` both walk
        # tuples down into the ColumnExprs inside.
        self.grouping = tuple(grouping)
        self.aggregations = tuple(aggregations)
        self.sort_keys = sort_keys
        self.drop_null_keys = drop_null_keys
        self.observed = observed
        self.keys_as_columns = keys_as_columns
        self.level = level
        self.sort_categories = sort_categories
        self.output_type = (output_type if output_type is not None
                            else self.infer_output_type())

    def infer_output_type(self, src_type: OutputType | None = None) -> OutputType:
        """The kind of value this aggregation produces.

        Distinguish FRAME and SERIES for downstream extraction, including
        ``_getitem_output_type`` and ``is_mask_selection``. A single unlabelled
        reduction of a series is a SCALAR. Labels widen a series result by one
        level, as in pandas: ``s.agg(total="sum")`` is a SERIES indexed by the
        labels, and ``gb["v"].agg(total="sum")`` is a FRAME with a column each.

        ``src_type`` overrides the kind read off ``inputs[0]``. Absorbing a column
        selection needs it: in ``groupby(k)["v"].sum()`` the frame underneath is a
        FRAME, but the column selection already narrowed the aggregation to a series.

        Fall back to FRAME when the source kind is unknown, matching
        ``SelectionOp``. This keeps downstream operators on the dataframe
        extraction path rather than the source-read path.
        """
        src = src_type
        if src is None:
            src = self.inputs[0].output_type if self.inputs else OutputType.UNKNOWN
        if self.grouped and self.keys_as_columns:
            return OutputType.FRAME
        if self.grouped and self.counts_group_rows():
            # One count per group, whatever columns the source has: pandas'
            # grouped `size()` is a series keyed by the groups.
            return OutputType.SERIES
        if src is OutputType.UNKNOWN:
            return OutputType.FRAME
        if not self.grouped:
            # A whole-object reduction drops one level: a frame collapses to one
            # value per column (a series), a series collapses to a scalar.
            if src is OutputType.FRAME:
                return OutputType.SERIES
            if src is OutputType.SERIES and self.labelled():
                # The labels become the index: s.agg(total="sum") is a series.
                return OutputType.SERIES
            if src is OutputType.SERIES and len(self.aggregations) == 1:
                return OutputType.SCALAR
            # Several reductions of one series: extraction never builds this.
            return OutputType.UNKNOWN
        # Grouped: a frame aggregation stays a frame, a series aggregation stays a
        # series unless several entries, or a label, widen it to a column each.
        if src is OutputType.SERIES and (self.labelled() or len(self.aggregations) > 1):
            return OutputType.FRAME
        return src

    @property
    def index_effect(self):
        # The groups label the rows, unless the keys become columns.
        return (IndexEffect.DEFAULT if self.grouped and self.keys_as_columns
                else IndexEffect.NEW)

    def labelled(self) -> bool:
        """Whether an entry carries an output label, as in ``s.agg(total="sum")``.

        Without a groupby only a Series is aggregated with labels, since named
        pairs on a DataFrame stay unfused.
        """
        return any(name is not None for name, _ in self.aggregations)

    def counts_group_rows(self) -> bool:
        """A single unnamed ``size`` over every column: the rows per group.

        Unlike other wildcard reductions it does not produce one value per
        column, so a grouped one is a series even over a frame.
        """
        if len(self.aggregations) != 1:
            return False
        name, agg = self.aggregations[0]
        return name is None and agg.func == "size" and isinstance(agg.child, AllCols)

    def entry_name(self, index: int):
        """Backend-independent name for a scalar aggregation output."""
        name, agg = self.aggregations[index]
        if name is not None:
            return name
        return agg.child.name if isinstance(agg.child, Col) else f"_agg{index}"

    def update_name(self):
        self.name = _render_name(self.grouping, self.aggregations)

    def lineage_config(self):
        return {**super().lineage_config(), "output_type": self.output_type.name}

    def propagate_output_schema(self):
        """See :func:`_schema.aggregate_schema` for which entries are knowable.

        Only a grouped aggregation has a column schema; a whole-object reduction
        yields a series indexed by column name, or a scalar.
        """
        if not self.grouped or not self.inputs:
            self.output_schema = None
            return
        self.output_schema = _schema.aggregate_schema(
            self.inputs[0].output_schema,
            _grouping_names(self.grouping),
            (None if any(isinstance(agg.child, AllCols)
                         for _, agg in self.aggregations) else
             [self.entry_name(i) for i in range(len(self.aggregations))]),
            not self.keys_as_columns,
        )


def _grouping_names(grouping) -> list[str] | None:
    """The grouping keys' column names, or ``None`` unless every key is a column."""
    if all(isinstance(key, Col) for key in grouping):
        return [key.name for key in grouping]
    return None


def _render_name(grouping, aggregations) -> str:
    aggs = ", ".join(f"{name}=" f"{agg!r}" if name else repr(agg)
                     for name, agg in aggregations)
    if not grouping:
        return aggs
    return f"by={', '.join(repr(g) for g in grouping)}, {aggs}"


# --- Source discovery -------------------------------------------------------


def _is_groupby_op(op: Op) -> bool:
    return isinstance(op, MethodCallOp) and op.method_name == "groupby"


class _Sources(NamedTuple):
    """What an aggregation call reads, as a discovery stage finds it.

    The aggregated object is what the method is called on: ``df``, ``df["v"]``,
    ``df.groupby(k)`` or ``df.groupby(k)["v"]`` in ``<object>.sum()``.
    :func:`_get_sources` describes an object read straight from the graph: a
    frame, a column selection or a groupby. :func:`_get_computed_sources` describes a
    computed Series, whose ``producer`` is folded into the aggregation's expression.
    """
    groupby_op: Op | None     # folded into the AggregateOp, None when ungrouped
    projection_op: Op | None  # folded too, only set alongside groupby_op
    src: Op                   # what the AggregateOp reads as inputs[0]
    columns: tuple | None     # projection_op's column names in order, else None
    is_series: bool           # the aggregated object is a Series or SeriesGroupBy
    producer: Op | None = None  # computed Series over `src`, None when nothing is computed

    @property
    def grouped(self) -> bool:
        return self.groupby_op is not None


def _selected_columns(selection) -> tuple | None:
    """Literal column names a ``[key]`` selection picks, grouped or not, if any."""
    if selection is None:
        return None
    key = getattr(selection, "key", None)
    if isinstance(key, str):
        return (key,)
    if isinstance(key, (list, tuple)) and all(isinstance(k, str) for k in key):
        return tuple(key)
    return None


def _get_sources(op: MethodCallOp) -> _Sources | None:
    """The producers feeding ``op``: what is folded into the aggregation, and what stays.

    A groupby, and a column selection on it, are folded into the aggregation, so
    each must have the aggregation as its only reader. A ``df[key]`` projection
    is not folded: it stays in the graph as the input, so other readers of it do
    not matter. ``None`` when a groupby or its selection cannot be folded, and
    when the aggregated object is anything else that is not a frame.
    """
    if not op.inputs:
        return None
    producer = op.inputs[0]

    # Case 1: the call reads a groupby directly. What the groupby was called on
    # decides whether it is a DataFrameGroupBy or a SeriesGroupBy:
    #   df.groupby("g").sum()             -> DataFrameGroupBy
    #   df["v"].groupby(df["g"]).sum()    -> SeriesGroupBy
    if _is_groupby_op(producer):
        if len(producer.outputs) != 1:
            # Another call also reads the groupby, so folding it would take that
            # call's input away:
            #   gb = df.groupby("g"), then gb.sum() and gb.mean()
            return None
        return _Sources(groupby_op=producer, projection_op=None, src=producer.inputs[0],
                        columns=None,
                        is_series=producer.inputs[0].output_type is OutputType.SERIES)

    if isinstance(producer, (ColumnProjectionOp, GetItemOp)) and producer.inputs:
        source, columns = producer.inputs[0], _selected_columns(producer)
        # A string key selects one column as a series, a list keeps a frame:
        # x["v"] is a Series (or SeriesGroupBy), x[["v"]] a DataFrame (or DataFrameGroupBy).
        is_series = isinstance(producer.key, str)

        # Case 2: a column selection on a groupby. Both fold into the AggregateOp.
        #   df.groupby("g")["v"].sum()           -> SeriesGroupBy
        #   df.groupby("g")[["v", "w"]].sum()    -> DataFrameGroupBy
        if _is_groupby_op(source):
            if columns is None or len(producer.outputs) != 1 or len(source.outputs) != 1:
                return None
            return _Sources(groupby_op=source, projection_op=producer, src=source.inputs[0],
                            columns=columns, is_series=is_series)

        # Case 3: an ungrouped column selection, which stays in the graph as the input.
        #   df["v"].sum()           -> Series
        #   df[["v", "w"]].sum()    -> DataFrame
        if columns is not None and source.output_type is OutputType.FRAME:
            return _Sources(groupby_op=None, projection_op=None, src=producer,
                            columns=None, is_series=is_series)

        # Not a column selection of a frame, such as a row mask, a slice or a
        # graph-fed key. Fall through: if it yields a frame, case 4 reduces it.
        #   df[mask]    df[1:3]    df[graph_fed_key]

    # Any other aggregated object that is not a frame is refused here. A computed
    # Series, as below, is recognised by `_get_computed_sources` instead.
    #   (df["a"] * df["b"]).sum()
    if producer.output_type is not OutputType.FRAME:
        return None

    # Case 4: an ungrouped frame.
    #   df.sum()    df[mask].sum()
    return _Sources(groupby_op=None, projection_op=None, src=producer,
                    columns=None, is_series=False)


def _expression_source(node):
    """Find the single frame anchoring a row-wise arithmetic expression.

    Cross-frame arithmetic may align indexes, so it cannot be absorbed into a
    row-local aggregate. The folder still checks external consumers before any
    expression node is removed.
    """
    if not isinstance(node, (BinOp, UnaryOp)):
        return None
    sources = {}
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, (ColumnProjectionOp, GetItemOp)):
            if not isinstance(current.key, str) or not current.inputs:
                return None
            frame = current.inputs[0]
            if frame.output_type is not OutputType.FRAME:
                return None
            sources[id(frame)] = frame
        elif isinstance(current, (BinOp, UnaryOp)):
            stack.extend(current.inputs)
        else:
            return None
    return next(iter(sources.values())) if len(sources) == 1 else None


def _get_computed_sources(op: MethodCallOp, sources: _Sources | None) -> _Sources | None:
    """The computed stage: ``op`` reduces a Series computed from the columns of one frame.

    ``sources`` is what :func:`_get_sources` found. It is needed because a groupby
    can sit between the call and the computed Series. The result reads the frame
    as ``src`` and keeps the arithmetic as ``producer``, which the folder turns
    into the aggregation's expression. Other kinds of computed Series, such as a
    string method's result, would be recognised here too. ``None`` when ``op``
    does not reduce a computed Series.
    """
    if sources is None:
        # (df["a"] * df["b"]).sum(): ungrouped computed Series.
        if not op.inputs:
            return None
        producer, groupby_op = op.inputs[0], None
    else:
        # Only a groupby with no column selection can sit on a computed Series.
        # Everything else _get_sources finds is a frame, or columns selected from one.
        #   (df["a"] * df["b"]).groupby(df["g"]).sum()
        if not (sources.grouped and sources.projection_op is None):
            return None
        producer, groupby_op = sources.src, sources.groupby_op
    if (producer.output_type is not OutputType.SERIES
            or not isinstance(producer, (BinOp, UnaryOp))):
        # Not arithmetic. What _get_sources found is used if it found anything, as
        # in the first call, otherwise the call stays unfused, as in the second:
        #   df["v"].groupby(df["g"]).sum()    df["s"].str.len().sum()
        return None
    src = _expression_source(producer)
    if src is None:
        # The arithmetic reads two frames, which may align by index, so it is
        # not folded:
        #   (df["a"] * other["b"]).sum()
        return None
    # (df["a"] * df["b"]).sum()                    -> src=df, producer=df["a"] * df["b"]
    # (df["a"] * df["b"]).groupby(df["g"]).sum()   -> the same, with groupby_op set
    # The folder then turns the producer into the entry's child: sum(Col("a") * Col("b")).
    return _Sources(groupby_op=groupby_op, projection_op=None, src=src, columns=None,
                    is_series=True, producer=producer)


# --- Grouping keys & options ------------------------------------------------


def _extract_grouping(groupby_op: MethodCallOp) -> str | list[str] | OperandRef:
    if groupby_op.args:
        return groupby_op.args[0]
    if groupby_op.kwargs and "by" in groupby_op.kwargs:
        return groupby_op.kwargs["by"]
    return None


def _grouping_options(groupby_op: MethodCallOp) -> dict | None:
    """The groupby call's options as :class:`AggregateOp` fields.

    ``None`` when pandas would reject the call or an option is graph-fed.
    ``group_keys`` only affects ``apply``, so it has no field.
    """
    call = groupby_op.bind_pandas()
    if call is None:
        return None
    options = {name: call.get(name, default) for name, default in (
        ("sort", True), ("dropna", True), ("observed", True), ("as_index", True),
        ("level", None))}
    if any(isinstance(value, OperandRef) for value in options.values()):
        return None
    return dict(sort_keys=options["sort"], drop_null_keys=options["dropna"],
                observed=options["observed"], keys_as_columns=not options["as_index"],
                level=options["level"])


def _fold_roots(sources: _Sources, op: MethodCallOp, folder: _Folder) -> tuple:
    """The computed Series and the grouping keys as expressions: ``(child, grouping)``.

    A literal ``groupby("g")`` gives ``Col("g")`` directly. A graph-fed key is an
    op subgraph, so it goes through the folder: ``groupby(df["g"])`` also lands on
    ``Col("g")`` instead of an opaque leaf, and ``groupby(df["d"].dt.year)``
    becomes a real ``DtExpr`` -- which is what makes expression grouping and
    literal grouping share one representation (and one CSE key).

    A computed ``producer`` is folded in the same pass, since shared column
    producers may belong to both subgraphs and must be counted in one discovery
    pass. ``child`` is ``None`` when nothing is computed.
    """
    keys = []
    if sources.grouped:
        by = _extract_grouping(sources.groupby_op)
        keys = by if isinstance(by, (list, tuple)) else [by]
    roots = ([sources.producer] if sources.producer is not None else []) + [
        sources.groupby_op.inputs[key.k] for key in keys if isinstance(key, OperandRef)]
    # The op whose read of a root does not block folding: the groupby when there
    # is one, else the call itself. Any other reader keeps the root as a leaf.
    consumer = sources.groupby_op if sources.grouped else op
    folded = iter(folder.fold_many(roots, root_consumer=consumer) if roots else ())
    child = next(folded) if sources.producer is not None else None
    grouping = tuple(_grouping_key_expr(sources.groupby_op.inputs[key.k], next(folded))
                     if isinstance(key, OperandRef) else Col(key)
                     for key in keys)
    return child, grouping


def _grouping_key_expr(root: Op, expr):
    # A string passed to groupby names a column. Scalar folding normally turns
    # ValueOp("g") into Const("g"), which Polars would group as one literal value.
    return Col(root.value) if isinstance(root, ValueOp) and isinstance(root.value, str) else expr


# --- Aggregation specs & call parsing ---------------------------------------

# Aggregation methods callable directly on a groupby, frame or series (no .agg
# wrapper needed).
_AGG_METHODS = {"sum", "mean", "count", "min", "max", "median", "std", "var",
                "first", "last", "prod", "size", "nunique", "sem"}
# Generic aggregation entrypoints that take the aggregation spec as an argument.
_AGG_FUNCS = {"agg", "aggregate"}
# `.agg(spec)` also spells its spec as a keyword, and that keyword is the spec
# itself rather than a parameter of the reduction, so it is read separately.
_AGG_SPEC_KWARG = "func"
# Treat these options as execution hints and omit them from the logical IR;
# the physical layer selects the execution strategy.
_EXECUTION_HINTS = {"engine", "engine_kwargs"}

# How `.agg(...)` spells its spec. Syntax only: what the keys mean depends on the
# aggregated object (a dict names input columns on a DataFrame, labels on a Series).
_FUNC = "func"      # "sum", ["sum", "mean"], or a direct .sum()
_DICT = "dict"      # {"v": "sum"}
_NAMED = "named"    # total=("v", "sum") / total=pd.NamedAgg("v", "sum")
_LABELS = "labels"  # total="sum" (bare keywords)

# The spellings fused for each kind of aggregated object, keyed by
# (grouped, is_series). pandas rejects every missing spelling except named pairs
# on a DataFrame, whose layout AggregateOp cannot build yet: one row per label
# and one column per input column. Either way the call stays and pandas runs it.
_SUPPORTED_FORMS = {
    (False, False): {_FUNC, _DICT},           # df.agg("sum"), df.agg({"v": "sum"})
    (False, True):  {_FUNC, _DICT, _LABELS},  # df["v"].sum(), df["v"].agg(total="sum")
    (True,  False): {_FUNC, _DICT, _NAMED},   # df.groupby(g)[[...]].agg(...)
    (True,  True):  {_FUNC, _LABELS},         # df.groupby(g)["v"].agg(total="sum")
}
# Reductions fused only under a groupby: without one, AggExpr's pandas evaluation
# disagrees with pandas. `df.size` is an int property and `df.agg("size")` counts
# per column, but AggExpr returns `obj.size`. pandas 3 removed DataFrame.first/last,
# but AggExpr returns `iloc[0]` / `iloc[-1]`.
_GROUPBY_ONLY_FUNCS = {"size", "first", "last"}


class _AggItem(NamedTuple):
    """One reduction an aggregation call declares, as written.

    Syntax only, like ``_AggCall.form``: what ``key`` means is decided later, by
    the form and the aggregated object. A ``_NAMED`` or ``_LABELS`` key is an
    output label, a ``_DICT`` key is an input column on a frame but an output
    label on a series. Only ``_NAMED`` names both a label and a ``column``.
    """
    key: str | None     # dict key or keyword label, None for a function spec
    column: str | None  # input column, set only by the _NAMED form
    func: object        # as written: a name, a list of names, a callable, ...
    params: dict        # reduction parameters, e.g. {"ddof": 0}


class _AggCall(NamedTuple):
    """An aggregation call as :func:`_extract_aggregations` parses it."""
    form: str     # _FUNC, _DICT, _NAMED or _LABELS: syntax, not meaning
    items: tuple  # (_AggItem, ...), in the order the call spells them
    multi: bool   # a list of funcs, as the spec or a dict value: several outputs per input


def _is_aggregation(op: MethodCallOp) -> bool:
    """True for an aggregation call: a direct reduction or ``.agg``/``.aggregate``.

    Only the method name is read. Whether the call can fuse is decided in one
    place, :func:`make_aggregate_op`, which returns ``None`` to refuse.
    """
    return op.method_name in _AGG_METHODS | _AGG_FUNCS


def _aggregation_params(op: MethodCallOp) -> dict | None:
    """Reduction parameters from the aggregation call, or ``None`` if unrepresentable.

    Both spellings put the reduction's parameters in kwargs: a direct
    ``.std(ddof=0)`` takes them itself, and ``.agg(spec, **kwargs)`` forwards them
    to the reduction. Discard execution hints so the physical layer can select
    the aggregation implementation.
    """
    if ((op.method_name in _AGG_FUNCS and len(op.args or ()) > 1)
            or (op.method_name not in _AGG_FUNCS and op.args)):
        # Direct methods and agg forward positional reduction parameters.
        # Until normalized by signature, preserve the original call.
        return None
    skip = _EXECUTION_HINTS | ({_AGG_SPEC_KWARG} if op.method_name in _AGG_FUNCS
                              else set())
    params = {k: v for k, v in (op.kwargs or {}).items() if k not in skip}
    if any(isinstance(v, OperandRef) for v in params.values()):
        # A graph-fed parameter isn't representable in the expression, the same
        # way `_Folder` keeps a graph-fed StringMethodOp arg as a leaf.
        return None
    return params


def _named_aggregation_spec(value) -> tuple | None:
    """``(column, reduction, params)`` for one named-aggregation value, if representable."""
    if isinstance(value, pd.NamedAgg) and not isinstance(value, tuple):
        # Positional reduction arguments have no place in an AggExpr.
        if value.args:
            return None
        value, params = (value.column, value.aggfunc), dict(value.kwargs)
    else:
        params = {}
    if (not isinstance(value, tuple) or len(value) != 2
            or not all(isinstance(v, str) for v in value)):
        return None
    return value[0], value[1], params


def _extract_aggregations(op: MethodCallOp) -> _AggCall | None:
    """Normalize an aggregation call into an :class:`_AggCall`, or ``None``.

    Only the syntax is read here. What each item means for the aggregated object,
    and the refusal of ``multi`` calls, are decided by :func:`_aggregation_spec`.
    The examples below show each call's items as ``(key, column, func, params)``.
    """
    if op.method_name in _AGG_METHODS:
        # A direct reduction method, whose keywords are reduction parameters:
        #   obj.sum()       -> _FUNC  (None, None, "sum", {})
        #   obj.std(ddof=0) -> _FUNC  (None, None, "std", {"ddof": 0})
        params = _aggregation_params(op)
        if params is None:
            return None
        return _AggCall(form=_FUNC, multi=False, items=(
            _AggItem(key=None, column=None, func=op.method_name, params=params),))
    if op.method_name not in _AGG_FUNCS:
        return None
    spec = op.args[0] if op.args else (op.kwargs or {}).get(_AGG_SPEC_KWARG)
    if spec is None:
        # No positional or `func=` spec: the keywords themselves are output labels.
        kwargs = {k: v for k, v in (op.kwargs or {}).items() if k not in _EXECUTION_HINTS}
        if not kwargs or op.args:
            return None
        if all(isinstance(v, str) for v in kwargs.values()):
            # A label per reduction, with no column named:
            #   obj.agg(total="sum", average="mean")
            #     -> _LABELS  ("total", None, "sum", {}), ("average", None, "mean", {})
            return _AggCall(form=_LABELS, multi=False, items=tuple(
                _AggItem(key=k, column=None, func=v, params={}) for k, v in kwargs.items()))
        # A label per (column, reduction) pair, with the pair's own parameters:
        #   obj.agg(total=("v", "sum"))            -> _NAMED  ("total", "v", "sum", {})
        #   obj.agg(total=pd.NamedAgg("v", "sum")) -> _NAMED  ("total", "v", "sum", {})
        #   obj.agg(spread=pd.NamedAgg("v", "std", ddof=0))
        #     -> _NAMED  ("spread", "v", "std", {"ddof": 0})
        parsed = [(name, _named_aggregation_spec(value)) for name, value in kwargs.items()]
        if any(not isinstance(name, str) or pair is None for name, pair in parsed):
            return None
        return _AggCall(form=_NAMED, multi=False, items=tuple(
            _AggItem(key=name, column=column, func=func, params=params)
            for name, (column, func, params) in parsed))
    # A spec is present, so the remaining keywords are reduction parameters. pandas
    # forwards them to a named reduction but drops them for a dict (see below).
    params = _aggregation_params(op)
    if params is None:
        return None
    if isinstance(spec, str):
        # A reduction name:
        #   obj.agg("sum") / obj.agg(func="sum") -> _FUNC  (None, None, "sum", {})
        #   obj.agg("sum", min_count=1)          -> _FUNC  (None, None, "sum", {"min_count": 1})
        return _AggCall(form=_FUNC, multi=False, items=(
            _AggItem(key=None, column=None, func=spec, params=params),))
    if isinstance(spec, list):
        # Several reductions of one input, which _aggregation_spec refuses:
        #   obj.agg(["sum", "mean"])
        #     -> _FUNC, multi  (None, None, "sum", {}), (None, None, "mean", {})
        return _AggCall(form=_FUNC, multi=True, items=tuple(
            _AggItem(key=None, column=None, func=f, params=params) for f in spec))
    if isinstance(spec, dict) and spec:
        # A dict, whose keys are columns on a frame and labels on a series. pandas
        # drops the call's other keywords for a dict, so its items carry none.
        # Only `axis` keeps its meaning, and it has no spelling here:
        #   obj.agg({"v": "sum", "w": "mean"})
        #     -> _DICT  ("v", None, "sum", {}), ("w", None, "mean", {})
        #   obj.agg({"v": "std"}, ddof=0)   -> _DICT  ("v", None, "std", {})
        #   obj.agg({"v": ["sum", "mean"]}) -> _DICT, multi  ("v", None, ["sum", "mean"], {})
        if "axis" in params:
            return None
        multi = any(isinstance(f, list) for f in spec.values())
        return _AggCall(form=_DICT, multi=multi, items=tuple(
            _AggItem(key=k, column=None, func=f, params={}) for k, f in spec.items()))
    # A graph-fed spec, a callable, ...: not representable.
    return None


def _aggregation_entries(call: _AggCall, columns: tuple | None, is_series: bool) -> tuple:
    """Entries for a function spec: where no column selection was specified, i.e spec is a str/List[str]."""
    (item,) = call.items
    func, params = item.func, item.params
    if columns is None or (func == "size" and not is_series):
        # One reduction applied to every column. A frame-shaped selection does
        # not change a row count: pandas' `groupby(k)[["v"]].size()` is
        # `groupby(k).size()`. A single-column selection is kept, since it names
        # the resulting series.
        return ((None, AggExpr(func, AllCols(), params)),)
    # An absorbed `groupby(k)[cols]` selection names the columns, so the
    # wildcard is not needed.
    return tuple((None, AggExpr(func, Col(col), params)) for col in columns)


def _projected_aggregation_entries(call: _AggCall, columns: tuple | None) -> tuple | None:
    """Entries for a dict or named spec, where the aggregation call itself picks the input columns.

    ``None`` for a column or function that is not a name, or for a column the
    folded projection dropped.
    """
    entries = []
    for item in call.items:
        # agg(total=("v", "sum")) keeps its output label, while agg({"v": "sum"})
        # reads its key as the input column.
        label, column = (item.key, item.column) if call.form == _NAMED else (None, item.key)
        if not isinstance(column, str) or not isinstance(item.func, str):
            return None
        if columns is not None and column not in columns:
            # The folded selection dropped this column. Leaving the call unfused
            # lets pandas raise its KeyError:
            #   groupby("g")[["v"]].agg({"w": "sum"})
            return None
        entries.append((label, AggExpr(item.func, Col(column), item.params)))
    return tuple(entries)


def _labelled_aggregation_entries(call: _AggCall, columns: tuple | None) -> tuple | None:
    """Entries for a dict or labels on a Series, where every key labels a reduction of it.

    ``None`` for a key or function that is not a name.
    """
    # A folded groupby(k)["v"] selection names the Series, otherwise the Series
    # is the input itself.
    child = Col(columns[0]) if columns else AllCols()
    entries = []
    for item in call.items:
        if not isinstance(item.key, str) or not isinstance(item.func, str):
            return None
        entries.append((item.key, AggExpr(item.func, child, item.params)))
    return tuple(entries)


def _aggregation_spec(call: _AggCall, grouped: bool, is_series: bool, groupby_op: Op | None,
                      columns: tuple | None) -> tuple[dict, tuple] | None:
    """Validate ``call`` for the aggregated object: ``(grouping_options, entries)`` or ``None``."""
    if call.multi:
        # .agg(["sum", "mean"]) labels its columns with a MultiIndex, which
        # AggregateOp cannot build.
        return None
    if call.form not in _SUPPORTED_FORMS[grouped, is_series]:
        return None
    if grouped:
        # df.groupby("g").sum(): the groupby must name its keys, and its options
        # become AggregateOp fields.
        if _extract_grouping(groupby_op) is None:
            return None
        options = _grouping_options(groupby_op)
        if options is None:
            return None
    else:
        # df.sum(), df["v"].sum(): without a groupby, size, first and last mean
        # something else in pandas (see _GROUPBY_ONLY_FUNCS).
        if any(item.func in _GROUPBY_ONLY_FUNCS for item in call.items):
            return None
        options = {}
    try:
        if call.form == _FUNC:
            # .sum(), .agg("sum"), .agg(["sum", "mean"])
            entries = _aggregation_entries(call, columns, is_series)
        elif is_series:
            # On a Series every key is a label:
            #   df["v"].agg({"total": "sum"}), df.groupby("g")["v"].agg(total="sum")
            entries = _labelled_aggregation_entries(call, columns)
        else:
            # .agg({"v": "sum"}), .agg(total=("v", "sum")), agg(total=pd.NamedAgg("v","sum"))
            entries = _projected_aggregation_entries(call, columns)
    except NotImplementedError:
        # An unsupported reduction or parameter: leave the chain alone.
        return None
    if entries is None:
        return None
    return options, entries


# --- AggregateOp fusion -----------------------------------------------------


def make_aggregate_op(op: MethodCallOp) -> AggregateOp | None:
    """Fuse an aggregation call and what feeds it into one :class:`AggregateOp`, or ``None``.

    Parse the call, find the object it aggregates, validate the call for that
    object, then fold and build.
    """
    call = _extract_aggregations(op)
    if call is None:
        return None
    # Find the aggregated object. _get_sources finds one read straight from the
    # graph: a frame, a column selection or a groupby. _get_computed_sources then checks
    # for arithmetic beneath it, and its result replaces the first one when found.
    # Otherwise (df["a"] * df["b"]).groupby(df["g"]).sum() would be read as a
    # SeriesGroupBy over the product, and the product would stay outside the
    # aggregation.
    plain = _get_sources(op)
    sources = _get_computed_sources(op, plain) or plain
    if sources is None:
        return None

    spec = _aggregation_spec(call=call, grouped=sources.grouped, is_series=sources.is_series,
                             groupby_op=sources.groupby_op, columns=sources.columns)
    if spec is None:
        return None
    options, entries = spec

    folder = _Folder(sources.src)
    child, grouping = _fold_roots(sources, op, folder)
    # `_Folder` numbers its leaves from 1 with the source at 0, so this order
    # is fixed by the folder, not a choice. Refusing graph-fed reduction
    # parameters is what keeps the aggregation call from adding more.
    inputs = [sources.src, *folder.leaf_ops]
    if sources.producer is not None:
        # The entries reduce AllCols(), which stands for the aggregated object.
        # Here that object is the computed Series, so they must reduce its expression.
        if not sources.grouped and isinstance(child, OperandLeaf):
            # Another op also reads the Series, so the folder kept it as a leaf.
            # Without a groupby there are no keys to read from the frame, so the
            # Series becomes the only input and AllCols() reduces it directly:
            #   prod = df["v"] * df["w"]
            #   prod.sum() + prod.mean()
            inputs = [sources.producer]
        else:
            # AggExpr already accepted func and params in _aggregation_spec, and
            # child holds no aggregation, so rebuilding the entries cannot raise.
            #   (df["a"] * df["b"]).sum() -> sum(Col("a") * Col("b"))
            entries = tuple((name, AggExpr(agg.func, child, agg.params))
                            for name, agg in entries)

    new_op = AggregateOp(
        grouped=sources.grouped, grouping=grouping, aggregations=entries, **options,
        inputs=inputs, outputs=list(op.outputs),
    )
    # The output kind follows the aggregated object, not inputs[0]. In
    # groupby("g")["v"].sum() and (df["a"] * df["b"]).sum(), inputs[0] is a frame
    # but the aggregated object is a series, so the results are a series and a scalar.
    src_type = OutputType.SERIES if sources.is_series else OutputType.FRAME
    new_op.output_type = new_op.infer_output_type(src_type=src_type)
    folder.install(new_op, replaced=[sources.groupby_op, sources.projection_op, op])
    return new_op


# --- value_counts -------------------------------------------------------------

# `value_counts` options that map onto the aggregation. The rest have no spelling
# here: `normalize` is a ratio rather than a reduction, `bins` buckets the values
# first, and `subset` groups by some columns instead of all of them. A call using
# one of those is left unfused rather than mis-modelled.
_VALUE_COUNTS_OPTIONS = {"sort", "ascending", "dropna"}


def _reads_a_groupby(op: Op) -> bool:
    """Whether ``op`` consumes a groupby, directly or through a column selection.

    Unlike :func:`_get_sources` this asks only about the shape, with no
    single-consumer gate, because it is used to *refuse* rather than to fuse: a
    grouped call must be left alone whether or not the groupby is shared.
    """
    if not op.inputs:
        return False
    node = op.inputs[0]
    if _is_groupby_op(node):
        return True
    return (isinstance(node, (ColumnProjectionOp, GetItemOp))
            and bool(node.inputs) and _is_groupby_op(node.inputs[0]))


def _is_value_counts(op: Op) -> bool:
    """True for a ``value_counts()`` call this module can fuse.

    A *grouped* ``value_counts`` is a different computation (a second grouping
    level, counted and ordered within each group) and is left unfused, where the
    plain groupby path already handles it correctly.
    """
    return (isinstance(op, MethodCallOp) and op.method_name == "value_counts"
            and bool(op.inputs) and not op.args
            and op.inputs[0].output_type is OutputType.SERIES
            and not _reads_a_groupby(op)
            and set(op.kwargs or {}) <= _VALUE_COUNTS_OPTIONS
            and not any(isinstance(v, OperandRef)
                        for v in (op.kwargs or {}).values()))


def make_value_counts_ops(op: MethodCallOp) -> Op | None:
    """Fuse ``value_counts()`` into an aggregation, followed by a sort when it orders.

    No special case is needed in the op itself: ``value_counts`` is a grouped
    count keyed on the values being counted, and "the source itself" is already
    spelled in the grammar as ``OperandLeaf(OperandRef(0))`` -- the same leaf the
    folder produces for a node that *is* the source.

    The groupby deliberately runs unsorted with an explicit :class:`SortOp` after
    it, rather than reusing the groupby's own key sort. That is what reproduces
    pandas: ``value_counts`` leaves equally frequent values in order of first
    appearance, which a key-sorted groupby would replace with key order. Keeping
    the ordering as its own operator is also what lets a later rewrite drop it
    when the consumer never observes row order.

    Returns the op downstream consumers should read (the sort, or the aggregation
    when ``sort=False``), or ``None`` when the call cannot be represented.
    """
    if not _is_value_counts(op):
        return None
    kwargs = op.kwargs or {}
    src = op.inputs[0]

    values = OperandLeaf(OperandRef(0))
    agg = AggregateOp(
        grouped=True,
        grouping=(values,),
        aggregations=(("count", AggExpr("size", values)),),
        # Unsorted on purpose (see above); `dropna` carries straight over, since
        # groupby drops null groups under the same flag and default.
        sort_keys=False, drop_null_keys=kwargs.get("dropna", True),
        observed=False, sort_categories=True,
        inputs=[src],
    )
    # The supported Series value_counts form returns a Series of counts.
    agg.output_type = OutputType.SERIES
    op.replace_output_of_inputs(agg)

    if not kwargs.get("sort", True):
        agg.outputs = list(op.outputs)
        return agg
    # The counts are the series' own values, so the sort needs no key.
    sort = SortOp(ascending=kwargs.get("ascending", False),
                  inputs=[agg], outputs=list(op.outputs))
    sort.output_type = OutputType.SERIES
    agg.outputs = [sort]
    return sort
