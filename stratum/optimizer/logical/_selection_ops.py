"""Relational row selections (SelectionOp family).

A selection restricts rows and keeps columns. Each variant is its own subclass
with backend-neutral fields; the physical impls in
``physical/_selection_execs.py`` translate those fields into each library's call.

Extraction is the only place that reads the recorded pandas call: the factories
below bind it against pandas' own signature and parse it into the variant's
fields. A call they cannot model (``inplace=True``, a column-wise ``axis``, a
graph-fed structural option, arguments pandas would reject) returns ``None``,
which leaves it an opaque method call that runs on pandas exactly as recorded.
"""
from __future__ import annotations

from stratum.optimizer.logical._base import IndexEffect
from stratum.optimizer.logical._ops import (
    FRAME_TYPES, OperandRef, OutputType, GetItemOp, MethodCallOp, Op, substitute_op)
from stratum.optimizer.logical._column_expr import ColumnExpr
from stratum.optimizer.logical._expr_folding import _Folder


class SelectionOp(Op):
    """Abstract base of the selection family: restricts rows, keeps columns.

    Never instantiated directly. Impls register on the concrete subclasses.
    """
    logical_family = "Selection"

    def __init__(self, name: str = "", inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        # A selection preserves its input's kind (a frame stays a frame, a series
        # stays a series); extraction overrides this with the propagated type.
        self.output_type = OutputType.FRAME

    def propagate_output_schema(self):
        """Rows go, columns stay: the source schema passes through unchanged."""
        # A selection with no source edge has nothing to pass through.
        self.output_schema = self.inputs[0].output_schema if self.inputs else None

    @property
    def index_effect(self):
        """The surviving rows keep their labels, unless they are renumbered."""
        if getattr(self, "ignore_index", False) is True:
            return IndexEffect.DEFAULT
        return IndexEffect.NEW

    # No custom __str__: the base IRNode renders ``<class>(<name>) [df]``. While
    # logical it shows the ``Selection`` family; once lowered, the bound impl
    # (PandasQuerySelectionOp, ...) replaces the class-name component.


class MaskSelectionOp(SelectionOp):
    """Keeps the rows where a boolean ``ColumnExpr`` predicate holds."""
    fields = ["predicate"]

    def __init__(self, predicate: ColumnExpr = None, inputs: list[Op] = None,
                 outputs: list[Op] = None):
        super().__init__(name="mask", inputs=inputs, outputs=outputs)
        self.predicate = predicate


class DropMissingSelectionOp(SelectionOp):
    """Drops rows with missing values (null, and NaN in float columns).

    ``subset``: the columns to inspect, ``None`` for all of them.
    ``min_present``: how many inspected values a row needs to be kept; ``None``
    means all of them. A series has a single value per row, so it is always
    ``None`` there.
    ``ignore_index``: renumber the surviving rows (meaningful only where rows
    carry index labels).
    """
    fields = ["subset", "min_present", "ignore_index"]

    def __init__(self, subset=None, min_present=None, ignore_index=False,
                 inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name="drop_missing", inputs=inputs, outputs=outputs)
        self.subset = subset
        self.min_present = min_present
        self.ignore_index = ignore_index


class DropDuplicatesSelectionOp(SelectionOp):
    """Drops repeated rows, comparing ``subset`` columns (``None``: all).

    ``keep``: which copy of a repeated row survives: ``"first"``, ``"last"``,
    or ``"none"`` to drop every copy.
    """
    fields = ["subset", "keep", "ignore_index"]

    KEEP = ("first", "last", "none")

    def __init__(self, subset=None, keep: str = "first", ignore_index=False,
                 inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name="drop_duplicates", inputs=inputs, outputs=outputs)
        if keep not in self.KEEP:
            raise ValueError(f"keep must be one of {self.KEEP}, got {keep!r}")
        self.subset = subset
        self.keep = keep
        self.ignore_index = ignore_index


class SliceSelectionOp(SelectionOp):
    """Keeps the first ``n`` rows, or the last ``n`` when ``from_end``.

    A negative ``n`` keeps all rows except ``|n|`` from the other end.
    """
    fields = ["n", "from_end"]

    @property
    def index_effect(self):
        # The first rows of 0..n-1 are 0..k-1; the last ones are not.
        return IndexEffect.NEW if self.from_end else IndexEffect.PREFIX

    def __init__(self, n=5, from_end: bool = False,
                 inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name="tail" if from_end else "head", inputs=inputs, outputs=outputs)
        self.n = n
        self.from_end = from_end


class SampleSelectionOp(SelectionOp):
    """Keeps a random sample of rows, in random order.

    ``n`` rows, or ``frac`` of them; one row when neither is given, and an error
    when both are. ``weights`` (a column name or a sequence) skews the draw;
    ``seed`` is an integer or a numpy random generator.
    """
    fields = ["n", "frac", "with_replacement", "weights", "seed", "ignore_index"]

    def __init__(self, n=None, frac=None, with_replacement=False, weights=None,
                 seed=None, ignore_index=False,
                 inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name="sample", inputs=inputs, outputs=outputs)
        self.n = n
        self.frac = frac
        self.with_replacement = with_replacement
        self.weights = weights
        self.seed = seed
        self.ignore_index = ignore_index


class DropLabelsSelectionOp(SelectionOp):
    """Drops the rows whose index label is in ``labels``.

    Only a library whose rows carry index labels can run this. ``level`` picks a
    level of a MultiIndex; ``errors="ignore"`` skips labels that are absent.
    """
    fields = ["labels", "level", "errors"]

    def __init__(self, labels=None, level=None, errors="raise",
                 inputs: list[Op] = None, outputs: list[Op] = None):
        super().__init__(name="drop_labels", inputs=inputs, outputs=outputs)
        self.labels = labels
        self.level = level
        self.errors = errors


# --- Extraction: recorded pandas call -> selection fields ----------------------

_ROW_AXES = (0, "index", "rows")


def _is_static(value) -> bool:
    return not isinstance(value, OperandRef)


def _not_inplace(call: dict) -> bool:
    """``inplace=True`` mutates the source and returns ``None``: not a selection."""
    return call.get("inplace", False) is False


def _drop_missing_fields(call: dict, series: bool) -> dict | None:
    if not _not_inplace(call) or call.get("axis", 0) not in _ROW_AXES:
        return None
    min_present = None
    if not series:
        how, thresh = call.get("how"), call.get("thresh")
        if "how" in call and "thresh" in call:
            return None  # pandas rejects both together
        if thresh is not None:
            min_present = thresh
        elif "how" in call:
            if how == "all":
                min_present = 1  # dropped only when nothing is present
            elif how != "any":
                return None  # graph-fed or invalid
    return dict(subset=call.get("subset"), min_present=min_present,
                ignore_index=call.get("ignore_index", False))


def _drop_duplicates_fields(call: dict, series: bool) -> dict | None:
    keep = call.get("keep", "first")
    if not _not_inplace(call) or not _is_static(keep):
        return None
    keep = {False: "none", "first": "first", "last": "last"}.get(keep)
    if keep is None:
        return None
    return dict(subset=call.get("subset"), keep=keep,
                ignore_index=call.get("ignore_index", False))


def _slice_fields(call: dict, from_end: bool) -> dict:
    return dict(n=call.get("n", 5), from_end=from_end)


def _sample_fields(call: dict, series: bool) -> dict | None:
    if call.get("axis") not in (None, *_ROW_AXES):
        return None  # column sampling, or a graph-fed axis
    return dict(n=call.get("n"), frac=call.get("frac"),
                with_replacement=call.get("replace", False),
                weights=call.get("weights"), seed=call.get("random_state"),
                ignore_index=call.get("ignore_index", False))


def _drop_labels_fields(call: dict, series: bool) -> dict | None:
    """Row form of ``drop``; extraction already decided the axis."""
    if not _not_inplace(call) or call.get("columns") is not None:
        return None
    if call.get("labels") is not None and call.get("index") is not None:
        return None  # pandas rejects both
    labels = call["index"] if call.get("index") is not None else call.get("labels")
    if labels is None:
        return None  # pandas requires something to drop
    return dict(labels=labels, level=call.get("level"),
                errors=call.get("errors", "raise"))


# pandas method -> (selection subclass, parser of the bound call)
_SELECTION_PARSERS = {
    "dropna": (DropMissingSelectionOp, _drop_missing_fields),
    "drop_duplicates": (DropDuplicatesSelectionOp, _drop_duplicates_fields),
    "head": (SliceSelectionOp, lambda call, series: _slice_fields(call, from_end=False)),
    "tail": (SliceSelectionOp, lambda call, series: _slice_fields(call, from_end=True)),
    "sample": (SampleSelectionOp, _sample_fields),
    "drop": (DropLabelsSelectionOp, _drop_labels_fields),
}
# Frame methods that are always row selections. ``drop`` is one only in its row
# form, which extraction decides from the axis.
_SELECTION_METHODS = frozenset(_SELECTION_PARSERS) - {"drop"}


def make_selection_op(op: MethodCallOp) -> SelectionOp | None:
    """Parse a row-selection method call into its ``SelectionOp`` subclass.

    Returns ``None`` (the call stays opaque) when the call is not a row
    selection we can model; see the module docstring.
    """
    cls, parse = _SELECTION_PARSERS[op.method_name]
    call = op.bind_pandas()
    if call is None:
        return None
    fields = parse(call, op.inputs[0].output_type is OutputType.SERIES)
    if fields is None:
        return None
    return substitute_op(op, cls(**fields), keep_kind=True)


# --- Mask selection: df[bool_series] -> MaskSelectionOp(predicate) -------------

def is_mask_selection(op: GetItemOp) -> bool:
    """Return whether ``op`` is ``container[series]`` (e.g. a boolean mask).

    Boolean-ness isn't tracked in the type lattice, so any series-keyed indexing
    of frame-world data counts. The container may itself be a series:
    ``counts[counts >= 3]`` is as much a filter as ``df[df["a"] >= 3]``, and both
    backends index a series by a mask the same way they index a frame.

    TODO: this misfires on a non-boolean series key (positional/label indexing,
    ``df[int_or_label_series]``), which is not a filter -- the polars fast path
    then calls ``.filter()`` on a non-boolean series. Accepting a series
    container widens the surface of that pre-existing hazard, since label
    indexing by a series is more idiomatic on a series than on a frame. Gate on
    boolean dtype once the type lattice tracks it.
    """
    if not isinstance(op.key, OperandRef):
        return False
    container = op.inputs[0]
    key_op = op.inputs[op.key.k]
    return (container.output_type in FRAME_TYPES
            and key_op.output_type is OutputType.SERIES)


def make_mask_selection_op(op: GetItemOp) -> MaskSelectionOp:
    """Fold ``df[mask]`` into a single ``MaskSelectionOp(predicate)``.

    Nodes consumed only by the mask are absorbed into the predicate and detached; the
    rest stay in the graph as ``OperandLeaf`` inputs (``inputs = [src, *leaf_ops]``).
    Downstream consumers of the mask are rewired by the caller via
    ``op.replace_input_of_outputs(sel)``.
    """
    src = op.inputs[0]
    folder = _Folder(src)
    predicate = folder.fold(op.inputs[op.key.k], root_consumer=op)

    sel = MaskSelectionOp(predicate=predicate, inputs=[src, *folder.leaf_ops],
                          outputs=list(op.outputs))
    # Filtering rows keeps the container's kind: a masked series is still a series.
    sel.output_type = src.output_type
    folder.install(sel, replaced=[op])
    return sel
