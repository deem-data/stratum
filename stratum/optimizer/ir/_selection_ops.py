from enum import Enum, auto
from stratum.optimizer.ir._ops import OperandRef, OutputType, GetItemOp, MethodCallOp, Op
from stratum.optimizer.ir._column_expr import ColumnExpr, fold_column_expr


class SelectionKind(Enum):
    """The kind of selection a :class:`SelectionOp` represents.

    ``MASK``/``QUERY`` carry a predicate; the rest map 1:1 to a frame method.
    """
    MASK = auto()
    QUERY = auto()
    DROPNA = auto()
    DROP_DUPLICATES = auto()
    HEAD = auto()
    TAIL = auto()
    SAMPLE = auto()


# Frame methods that are relational selections (restrict rows, keep columns).
_SELECTION_METHODS = {
    "dropna": SelectionKind.DROPNA,
    "drop_duplicates": SelectionKind.DROP_DUPLICATES,
    "head": SelectionKind.HEAD,
    "tail": SelectionKind.TAIL,
    "sample": SelectionKind.SAMPLE,
}
# Method name to call per backend for each method-based kind.
_SELECTION_PANDAS_METHOD = {v: k for k, v in _SELECTION_METHODS.items()}
_SELECTION_POLARS_METHOD = {
    SelectionKind.DROPNA: "drop_nulls",
    SelectionKind.DROP_DUPLICATES: "unique",
    SelectionKind.HEAD: "head",
    SelectionKind.TAIL: "tail",
    SelectionKind.SAMPLE: "sample",
}


class SelectionOp(Op):
    """A relational selection: restricts rows, keeps columns.

    Method-based kinds use ``args``/``kwargs``; ``MASK``/``QUERY`` use ``predicate``.
    """
    logical_family = "Selection"
    fields = ["kind", "args", "kwargs", "predicate"]

    def __init__(self, kind: SelectionKind, args: tuple | list = None, kwargs: dict = None,
                 predicate: ColumnExpr = None, inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=kind.name.lower(), inputs=inputs, outputs=outputs)
        if kwargs is not None:
            self.check_kwargs(kwargs)
        self.kind = kind
        self.args = args
        self.kwargs = kwargs
        self.predicate = predicate
        # A selection preserves its input's kind (a frame stays a frame, a series
        # stays a series); extraction overrides this with the propagated type.
        self.output_type = OutputType.FRAME

    def propagate_output_schema(self):
        """Selection restricts rows and keeps columns, so the schema is unchanged."""
        self.output_schema = self.inputs[0].output_schema if self.inputs else None

    # No custom __str__: the base IRNode renders ``<class>(<name>) [df]`` from
    # ``self.name`` (== the kind). While logical, it shows the ``Selection``
    # family; once lowered, ``_is_physical`` flips the class-name component to the
    # bound impl (PandasQuerySelectionOp, ...), so the plan reflects the selected
    # backend instead of a hardcoded name.


def make_selection_op(op: MethodCallOp) -> SelectionOp:
    """Convert a row-selection method call to a ``SelectionOp``."""
    new_op = SelectionOp(kind=_SELECTION_METHODS[op.method_name],
                         args=op.args, kwargs=op.kwargs,
                         inputs=op.inputs, outputs=op.outputs)
    new_op.output_type = op.inputs[0].output_type
    op.replace_output_of_inputs(new_op)
    return new_op


# --- Mask selection: df[bool_series] -> SelectionOp(MASK, predicate) ----------

def is_mask_selection(op: GetItemOp) -> bool:
    """Return whether ``op`` is ``df[series]`` (e.g. a boolean mask).

    Boolean-ness isn't tracked in the type lattice, so any series-keyed indexing
    of a frame counts.

    TODO: this misfires on a non-boolean series key (positional/label indexing,
    ``df[int_or_label_series]``), which is not a filter -- the polars fast path
    then calls ``.filter()`` on a non-boolean series. Gate on boolean dtype once
    the type lattice tracks it.
    """
    if not isinstance(op.key, OperandRef):
        return False
    container = op.inputs[0]
    key_op = op.inputs[op.key.k]
    return (container.output_type is OutputType.FRAME
            and key_op.output_type is OutputType.SERIES)


def make_mask_selection_op(op: GetItemOp) -> SelectionOp:
    """Fold ``df[mask]`` into a single ``SelectionOp(MASK, predicate)``.

    Nodes consumed only by the mask are absorbed into the predicate and detached; the
    rest stay in the graph as ``OperandLeaf`` inputs (``inputs = [src, *leaf_ops]``).
    """
    src = op.inputs[0]
    key_op = op.inputs[op.key.k]

    predicate, absorbed, leaf_ops = fold_column_expr(key_op, src, root_consumer=op)

    sel = SelectionOp(kind=SelectionKind.MASK, predicate=predicate,
                      inputs=[src, *leaf_ops], outputs=list(op.outputs))
    sel.output_type = OutputType.FRAME

    # Detach every absorbed op from the graph (remove it from its inputs' output
    # lists, then clear its edges).
    for node in absorbed:
        for inp in node.inputs:
            inp.outputs = [o for o in inp.outputs if o is not node]
        node.inputs = []
        node.outputs = []

    # The source and each kept leaf op now feed the selection in place of the mask
    # GetItem / the detached expression nodes they used to feed. Other (external)
    # consumers of a leaf op are left untouched. Downstream consumers of the mask
    # are rewired by the caller via ``op.replace_input_of_outputs(sel)``.
    for producer in (src, *leaf_ops):
        producer.outputs = [o for o in producer.outputs if o is not op]
        producer.add_output(sel)
    op.inputs = []
    return sel
