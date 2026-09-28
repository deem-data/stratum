from stratum.optimizer.logical._column_expr import (
    AggExpr, AllCols, Col, OperandLeaf, _Folder)
from stratum.optimizer.logical._ops import (
    BinOp, UnaryOp, GetItemOp, OperandRef, OutputType, MethodCallOp, Op)
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

    ``options`` carries only the groupby options that change the *result*, e.g.
    ``sort`` (which fixes the output row order) and ``dropna``. How the
    aggregation runs is the physical layer's business and has no field here.
    ``sort_categories`` orders categorical groups by their declared domain while
    leaving other groups in appearance order, as required by ``value_counts``.

    Pure config -- execution is provided by the physical impls in
    ``physical/_aggregation_execs.py``, selected at plan time.
    """
    logical_family = "Aggregation"
    fields = ["grouped", "grouping", "aggregations", "options", "output_type"]

    def __init__(self, grouped: bool = False,
                 grouping: tuple = (),
                 aggregations: tuple = (),
                 options: dict | None = None,
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
        self.options = options or {}
        self.output_type = (output_type if output_type is not None
                            else self.infer_output_type())

    def infer_output_type(self, src_type: OutputType | None = None) -> OutputType:
        """The kind of value this aggregation produces.

        Deliberately coarse: what downstream extraction needs is whether the
        result is frame-world data at all, plus FRAME vs SERIES for the two
        rewrites that read it (``_getitem_output_type`` and
        ``is_mask_selection``). A scalar is ``UNKNOWN``, which is what the lattice
        already means by it; ``SCALAR`` has no readers, so nothing infers it.

        ``src_type`` overrides the kind read off ``inputs[0]``. Absorbing a column
        selection needs it: in ``groupby(k)["v"].sum()`` the frame underneath is a
        FRAME, but the selection already narrowed the aggregation to a series.

        Falls back to FRAME whenever the source kind isn't known yet, matching
        ``SelectionOp``'s conservative default -- staying inside frame-world is
        the safe direction, since dropping out of it silently reroutes every
        downstream op to the read/source path.
        """
        src = src_type
        if src is None:
            src = self.inputs[0].output_type if self.inputs else OutputType.UNKNOWN
        if self.grouped and self.options.get("as_index") is False:
            return OutputType.FRAME
        if src is OutputType.UNKNOWN:
            return OutputType.FRAME
        if not self.grouped:
            # A whole-object reduction drops one level: a frame collapses to one
            # value per column (a series), a series collapses to a scalar.
            return OutputType.SERIES if src is OutputType.FRAME else OutputType.UNKNOWN
        # Grouped: a frame aggregation stays a frame, a series aggregation stays a
        # series unless several entries widen it to a column each.
        if src is OutputType.SERIES and len(self.aggregations) > 1:
            return OutputType.FRAME
        return src

    def entry_name(self, index: int):
        """Backend-independent name for a scalar aggregation output."""
        name, agg = self.aggregations[index]
        if name is not None:
            return name
        return agg.child.name if isinstance(agg.child, Col) else f"_agg{index}"

    def update_name(self):
        self.name = _render_name(self.grouping, self.aggregations)

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
            self.options.get("as_index"),
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


class GroupedDataframeOp(Op):
    def __init__(self, ops: list[Op]):
        super().__init__(name="GROUPED_DATAFRAME", is_X=False, is_y=False)
        self.ops = ops
        self.output_type = OutputType.FRAME

    def process(self, mode: str, inputs: list):  # pragma: no cover
        # TODO: GroupedDataframeOp is experimental and not integrated yet.
        # Needs proper refactoring to collect sub-op inputs from the pool.
        raise NotImplementedError("GroupedDataframeOp is not integrated yet.")


# Aggregation methods callable directly on a groupby (no .agg wrapper needed).
_AGG_METHODS = {"sum", "mean", "count", "min", "max", "median", "std", "var",
                "first", "last", "prod", "size", "nunique", "sem"}
# Generic aggregation entrypoints that take the aggregation spec as an argument.
_AGG_FUNCS = {"agg", "aggregate"}
# `.agg(spec)` also spells its spec as a keyword, and that keyword is the spec
# itself rather than a parameter of the reduction, so it is read separately.
_AGG_SPEC_KWARG = "func"
# Groupby options that change the result and so belong in `AggregateOp.options`.
# `sort` fixes the output row order, `dropna`/`observed` decide which groups exist.
_GROUPBY_OPTIONS = {"sort", "dropna", "observed", "as_index", "level"}
# Options that only say *how* to run, never what the result is. The physical layer
# picks its own strategy, so these are discarded on the way in rather than
# refused: keeping them out of the logical IR must not cost us the rewrite.
_EXECUTION_HINTS = {"engine", "engine_kwargs"}


def _is_groupby_op(op: Op) -> bool:
    return isinstance(op, MethodCallOp) and op.method_name == "groupby"


def _grouped_source(op: MethodCallOp):
    """The groupby feeding ``op``, plus any column selection sitting between them.

    Returns ``(groupby_op, selection_op | None)`` or ``None``. A
    ``groupby(k)["v"]`` selection is absorbed into the aggregation entries rather
    than blocking the fusion, which is how ``groupby(k)["v"].sum()`` becomes a
    single op. Every op on the way must have exactly one consumer, otherwise
    absorbing it would drop a value someone else still reads.
    """
    if not op.inputs:
        return None
    node = op.inputs[0]
    if _is_groupby_op(node):
        return (node, None) if len(node.outputs) == 1 else None
    if not isinstance(node, (ColumnProjectionOp, GetItemOp)):
        return None
    if len(node.outputs) != 1 or not node.inputs:
        return None
    inner = node.inputs[0]
    if not _is_groupby_op(inner) or len(inner.outputs) != 1:
        return None
    return inner, node


def _selected_columns(selection) -> tuple | None:
    """Literal column names a ``groupby(...)[key]`` selection picks, if any."""
    if selection is None:
        return None
    key = getattr(selection, "key", None)
    if isinstance(key, str):
        return (key,)
    if isinstance(key, (list, tuple)) and all(isinstance(k, str) for k in key):
        return tuple(key)
    return None


def _is_aggregation(op: MethodCallOp) -> bool:
    """True for a `groupby(...).<agg>()` pair that can fuse into an AggregateOp.

    Requires the aggregation to consume a `groupby` op, optionally through a
    single column selection, and every op in that chain to have one consumer.
    """
    source = _grouped_source(op)
    if source is None:
        return False
    groupby_op, selection = source
    if _extract_grouping(groupby_op) is None:
        return False
    if selection is not None and _selected_columns(selection) is None:
        return False
    if op.method_name in _AGG_METHODS:
        return True
    # `.agg(spec)` / `.aggregate(spec)`, positionally or as `func=`; a call with
    # no spec at all is not an aggregation.
    return (op.method_name in _AGG_FUNCS
            and (_extract_aggregations(op) is not None
                 or _named_aggregation_entries(op) is not None))


def _named_aggregation_entries(op: MethodCallOp) -> tuple | None:
    """Recognise pandas ``agg(name=(column, reduction), ...)`` syntax.

    A positional dict whose values are tuples has different pandas semantics,
    so this path is deliberately limited to keyword named aggregation.
    """
    if op.method_name not in _AGG_FUNCS or op.args or "func" in (op.kwargs or {}):
        return None
    named = {k: v for k, v in (op.kwargs or {}).items()
             if k not in _EXECUTION_HINTS}
    if not named:
        return None
    entries = []
    try:
        for name, value in named.items():
            if (not isinstance(name, str) or not isinstance(value, tuple)
                    or len(value) != 2 or not all(isinstance(v, str) for v in value)):
                return None
            column, reduction = value
            entries.append((name, AggExpr(reduction, Col(column))))
    except NotImplementedError:
        return None
    return tuple(entries)


def _extract_grouping(groupby_op: MethodCallOp) -> str | list[str] | OperandRef:
    if groupby_op.args:
        return groupby_op.args[0]
    if groupby_op.kwargs and "by" in groupby_op.kwargs:
        return groupby_op.kwargs["by"]
    return None


def _extract_aggregations(op: MethodCallOp) -> str | list[str] | OperandRef | None:
    """The aggregation spec, or ``None`` when the call carries none.

    ``.agg("sum")`` and ``.agg(func="sum")`` are the same call, so both spellings
    have to reach the same spec; reading only the positional one would leave the
    keyword form unfused.
    """
    if op.method_name in _AGG_FUNCS:
        if op.args:
            return op.args[0]
        return (op.kwargs or {}).get(_AGG_SPEC_KWARG)
    # direct method such as .mean()/.sum()/.count() -> normalize to its name
    return op.method_name


def _grouping_exprs(groupby_op: MethodCallOp, folder: _Folder) -> tuple:
    """Grouping keys as expressions, folding graph-fed keys through ``folder``.

    A literal ``groupby("g")`` gives ``Col("g")`` directly. A graph-fed key is an
    op subgraph, so it goes through the folder: ``groupby(df["g"])`` also lands on
    ``Col("g")`` instead of an opaque leaf, and ``groupby(df["d"].dt.year)``
    becomes a real ``DtExpr`` -- which is what makes expression grouping and
    literal grouping share one representation (and one CSE key).
    """
    by = _extract_grouping(groupby_op)
    keys = by if isinstance(by, (list, tuple)) else [by]
    exprs = [None] * len(keys)
    roots, positions = [], []
    for index, key in enumerate(keys):
        if isinstance(key, OperandRef):
            roots.append(groupby_op.inputs[key.k])
            positions.append(index)
        else:
            exprs[index] = Col(key)
    if roots:
        for position, expr in zip(positions,
                                  folder.fold_many(roots, root_consumer=groupby_op)):
            exprs[position] = expr
    return tuple(exprs)


def _aggregation_params(op: MethodCallOp) -> dict | None:
    """Reduction parameters from the aggregation call, or ``None`` if unrepresentable.

    Both spellings put the reduction's parameters in kwargs: a direct
    ``.std(ddof=0)`` takes them itself, and ``.agg(spec, **kwargs)`` forwards them
    to the reduction. Execution hints are discarded rather than refused, since
    they cannot change the result and refusing them would block an otherwise
    optimizable pipeline.
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


def _aggregation_entries(spec, params: dict, columns: tuple | None = None) -> tuple | None:
    """Turn a pandas aggregation spec into ``(name, AggExpr)`` entries.

    Returns ``None`` for a spec the expression grammar cannot represent, so the
    caller leaves the chain unfused rather than mis-modelling it. That covers a
    graph-fed spec (the function is only known at runtime, so there is no
    canonical name to key on) and the multi-output forms (``agg(["sum", "mean"])``
    or ``agg({"v": ["sum", "mean"]})``), whose pandas result is MultiIndex-keyed.
    """
    try:
        if isinstance(spec, str):
            if columns is not None:
                # An absorbed `groupby(k)[cols]` selection names the columns, so
                # the wildcard is not needed.
                return tuple((None, AggExpr(spec, Col(col), params))
                             for col in columns)
            # One reduction applied to every column.
            return ((None, AggExpr(spec, AllCols(), params)),)
        if isinstance(spec, dict):
            if columns is not None:
                # A dict spec after a column selection would have to agree with
                # it; not worth modelling, so leave the chain alone.
                return None
            entries = []
            for col, func in spec.items():
                if not isinstance(col, str) or not isinstance(func, str):
                    return None
                entries.append((None, AggExpr(func, Col(col), params)))
            return tuple(entries)
    except NotImplementedError:
        # An unsupported reduction or parameter: leave the chain alone.
        return None
    return None


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


def make_aggregate_op(op: MethodCallOp) -> AggregateOp | None:
    """Fuse `groupby(by)[cols].agg(...)` (or `.sum()/.mean()/...`) into an AggregateOp."""
    source = _grouped_source(op)
    if source is None:
        return None
    groupby_op, selection = source
    df = groupby_op.inputs[0]

    columns = _selected_columns(selection)
    if selection is not None and columns is None:
        return None
    named_entries = _named_aggregation_entries(op)
    if named_entries is not None and selection is not None:
        return None
    if named_entries is not None:
        entries = named_entries
    else:
        params = _aggregation_params(op)
        if params is None:
            return None
        entries = _aggregation_entries(_extract_aggregations(op), params, columns)
    if entries is None:
        return None

    # Fold reduction and grouping roots together: shared column producers may
    # be internal to both cones and must be accounted for in one discovery pass.
    expression_source = _expression_source(df) if columns is None else None
    computed = expression_source is not None and all(
        isinstance(agg.child, AllCols) for _, agg in entries)
    source_kind = df.output_type
    if computed:
        original = df
        df = expression_source
        folder = _Folder(df)
        by = _extract_grouping(groupby_op)
        keys = by if isinstance(by, (list, tuple)) else [by]
        roots = [original] + [groupby_op.inputs[k.k] for k in keys
                              if isinstance(k, OperandRef)]
        folded = iter(folder.fold_many(roots, root_consumer=groupby_op))
        child = next(folded)
        grouping = tuple(next(folded) if isinstance(k, OperandRef) else Col(k)
                         for k in keys)
        entries = tuple((name, AggExpr(agg.func, child, agg.params))
                        for name, agg in entries)
        source_kind = OutputType.SERIES
    else:
        folder = _Folder(df)
        grouping = _grouping_exprs(groupby_op, folder)

    options = {k: v for k, v in (groupby_op.kwargs or {}).items()
               if k in _GROUPBY_OPTIONS}

    new_op = AggregateOp(
        grouped=True,
        grouping=grouping,
        aggregations=entries,
        options=options,
        # `_Folder` numbers its leaves from 1 with the source at 0, so this order
        # is fixed by the folder, not a choice. Refusing graph-fed reduction
        # parameters is what keeps the aggregation call from adding more.
        inputs=[df, *folder.leaf_ops],
        outputs=list(op.outputs),
    )
    # A selection already narrowed the aggregation to a series; the frame below it
    # still reads as a FRAME, so take the kind from the op being absorbed.
    if selection is not None or computed:
        new_op.output_type = new_op.infer_output_type(
            selection.output_type if selection is not None else source_kind)

    _detach_and_rewire(new_op, df, folder, replaced=[groupby_op, selection, op])
    return new_op


def _detach_and_rewire(new_op: AggregateOp, df: Op, folder: _Folder,
                       replaced: list) -> None:
    """Unlink the folded ops and point the surviving producers at ``new_op``.

    Mirrors ``make_mask_selection_op``: absorbed nodes leave the graph entirely,
    while the frame and every kept leaf feed the aggregation in place of the ops it
    replaced. Downstream consumers are rewired by the caller.
    """
    for node in folder.absorbed:
        for inp in node.inputs:
            inp.outputs = [o for o in inp.outputs if o is not node]
        node.inputs = []
        node.outputs = []
    replaced_ids = {id(node) for node in replaced if node is not None}
    for producer in (df, *folder.leaf_ops):
        producer.outputs = [o for o in producer.outputs
                            if id(o) not in replaced_ids]
        producer.add_output(new_op)
    for node in replaced:
        if node is not None and node is not new_op:
            node.inputs = []


# --- value_counts -------------------------------------------------------------

# `value_counts` options that map onto the aggregation. The rest have no spelling
# here: `normalize` is a ratio rather than a reduction, `bins` buckets the values
# first, and `subset` groups by some columns instead of all of them. A call using
# one of those is left unfused rather than mis-modelled.
_VALUE_COUNTS_OPTIONS = {"sort", "ascending", "dropna"}


def _reads_a_groupby(op: Op) -> bool:
    """Whether ``op`` consumes a groupby, directly or through a column selection.

    Unlike :func:`_grouped_source` this asks only about the shape, with no
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
        options={"sort": False, "dropna": kwargs.get("dropna", True),
                 "observed": False, "sort_categories": True},
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
