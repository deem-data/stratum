"""Shared IR foundation for both the logical and physical operator layers.

``IRNode`` holds everything the two layers have in common: the DAG edge
structure (``inputs``/``outputs`` and the wiring helpers), the buffer-lifetime
bookkeeping (``remove_after``), the split marker, and the ``process`` execution
hook. The logical layer subclasses it as :class:`~stratum.optimizer.ir._ops.Op`
and the physical layer as
:class:`~stratum.optimizer.physical._physical_ops.PhysicalOp`.

Keeping this base free of any logical- or physical-specific concept (no CSE
structure keys, no choice handling, no backend selection) is what lets the
two layers stay genuinely separate hierarchies instead of one inheriting the
other. Operand plumbing (:class:`OperandRef` and the resolve helpers) and the
output-kind lattice (:class:`OutputType`) live here too, because both layers
reference operands and carry an output kind.
"""
from __future__ import annotations
from enum import Enum, auto

from sklearn.base import BaseEstimator
from skrub._data_ops._data_ops import DataOp
from skrub.selectors._base import All


class OutputType(Enum):
    """The kind of value an :class:`IRNode` produces.

    Replaces the old boolean ``is_dataframe_op`` flag with a small lattice so that
    rewrites can distinguish a single column (``SERIES``) from a whole table
    (``FRAME``). Telling the two apart is what lets selection detection recognise
    ``df[mask]`` as a relational selection (the ``mask`` is a ``SERIES``).

    Boolean-ness is *not* a separate output type: a boolean mask is just a
    ``SERIES`` whose values happen to be booleans (a value-level property), so it
    is tracked separately rather than as its own enum member.

    ``UNKNOWN`` is the default (the op produces a non-tabular Python value, e.g. a
    scalar or an arbitrary object) and corresponds to the old ``is_dataframe_op =
    False``. ``MATRIX`` is ndarray-valued (e.g. ``np.load``) and is deliberately
    *not* a frame type: numpy data is handled by the numeric extraction path, not
    the dataframe path. (A ``VECTOR`` type will be added once we have an op that
    produces one -- e.g. a GetItem/aggregation on a MATRIX.)
    """
    UNKNOWN = auto()
    FRAME = auto()
    SERIES = auto()
    SCALAR = auto()
    MATRIX = auto()


# Output types that belong to the dataframe (pandas/polars) world. A frame and a
# series are both manipulated by the dataframe extraction path; a MATRIX (numpy)
# is not. Used to decide whether an op consumes already-produced frame data (a
# dataframe operation) or a leaf/raw value (a read/source).
FRAME_TYPES = frozenset({OutputType.FRAME, OutputType.SERIES})


def is_frame_like(op) -> bool:
    """True if ``op`` produces frame-world data (a frame or a series)."""
    return op.output_type in FRAME_TYPES


class OperandRef:
    """Explicit reference to the ``k``-th entry of an :class:`IRNode`'s ``inputs`` list.

    Replaces the old opaque ``DATA_OP_PLACEHOLDER`` sentinel. Instead of relying on
    the *order* in which placeholders are walked at runtime, an operand now carries
    the exact index of the input that fills it, so ``process()`` can resolve
    ``inputs[ref.k]`` directly and rewrites that reorder inputs are checkable.
    """
    __slots__ = ("k",)

    def __init__(self, k: int):
        self.k = k

    def __eq__(self, other):
        return isinstance(other, OperandRef) and other.k == self.k

    def __hash__(self):
        return hash(("OperandRef", self.k))

    def __str__(self):
        return f"${self.k}"

    def __repr__(self):
        return f"OperandRef({self.k})"


def _resolve_operand(value, inputs):
    """Recursively replace OperandRefs nested in value with values from inputs."""
    if isinstance(value, OperandRef):
        return inputs[value.k]
    if isinstance(value, tuple):
        return tuple(_resolve_operand(v, inputs) for v in value)
    if isinstance(value, list):
        return [_resolve_operand(v, inputs) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_operand(v, inputs) for k, v in value.items()}
    return value


def _resolve_args(args, inputs):
    """Replace OperandRefs in an args sequence with values from the inputs list."""
    return [_resolve_operand(a, inputs) for a in args]


def _resolve_kwargs(kwargs, inputs):
    """Replace OperandRefs in a kwargs dict with values from the inputs list."""
    return {k: _resolve_operand(v, inputs) for k, v in kwargs.items()}


def remap_operand_refs(value, mapping: dict):
    """Return ``value`` with every nested :class:`OperandRef` remapped through
    ``mapping`` (old input index -> new input index).

    Recurses tuples/lists/dicts and column-expression trees (anything exposing a
    ``remap_operand_refs`` method, e.g. a ``ColumnExpr`` predicate). This is the
    single walker shared by CSE edge de-duplication (``_op_cse``) and
    :meth:`IRNode._dedupe_input_refs`, so both renumber refs identically --
    including refs buried inside a ``ColumnExpr`` field.
    """
    if isinstance(value, OperandRef):
        return OperandRef(mapping[value.k])
    if isinstance(value, tuple):
        return tuple(remap_operand_refs(v, mapping) for v in value)
    if isinstance(value, list):
        return [remap_operand_refs(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: remap_operand_refs(v, mapping) for k, v in value.items()}
    if hasattr(value, "remap_operand_refs"):
        return value.remap_operand_refs(mapping)
    return value


# --- Structure keys for common-subexpression elimination -------------------
# `Op.structure_key()` returns a hashable value that is equal for two ops iff
# they are the same computation. Sentinel keys carry a leading marker string so
# they stay disjoint from real values. (CSE is a logical-layer concern, but the
# helpers live here so both layers can share the value-keying primitives.)
_ALL_SELECTOR_KEY = ("__all_selector__",)
# A graph-fed estimator hyper-parameter: its binding is captured by an op's
# `param_refs` plus the input ids, so the stale DataOp left in get_params() must
# not block two otherwise-equal estimators from merging.
_GRAPH_PARAM_KEY = ("__graph_param__",)


def config_key(value):
    """Turn a config-field value into a hashable, value-based key.

    OperandRefs and hashable scalars are kept by value (so equal operands and
    constants compare equal); containers are recursed into; estimators are keyed
    by type and parameters; unhashable leaves (DataFrames, arrays, ...) fall back
    to identity, which is conservative (distinct objects never compare equal).
    """
    if isinstance(value, OperandRef):
        return value
    if isinstance(value, All):
        return _ALL_SELECTOR_KEY
    if isinstance(value, BaseEstimator):
        return estimator_key(value)
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(config_key(v) for v in value))
    if isinstance(value, dict):
        return ("__dict__", frozenset((k, config_key(v)) for k, v in value.items()))
    if isinstance(value, (set, frozenset)):
        return ("__set__", frozenset(config_key(v) for v in value))
    try:
        hash(value)
    except TypeError:
        return ("__id__", id(value))
    return value


def estimator_key(est: BaseEstimator):
    """Structure key for an estimator, consistent with parameter-wise equality.

    Graph-fed parameters (still DataOps in ``get_params()``) are normalized to a
    constant marker: their binding is represented by the op's ``param_refs`` field
    and input ids, not by the estimator object itself.
    """
    items = []
    for k, v in est.get_params().items():
        items.append((k, _GRAPH_PARAM_KEY if isinstance(v, DataOp) else config_key(v)))
    return ("__estimator__", type(est), frozenset(items))


def clone_value(value):
    if isinstance(value, dict):
        return {k: clone_value(v) for k, v in value.items()}
    elif isinstance(value, tuple):
        return tuple(clone_value(el) for el in value)
    else:
        return value


class IRNode:
    """Structural base for every node in a logical or physical IR DAG.

    Carries the DAG edges, name/output-kind metadata, buffer-lifetime
    bookkeeping and the ``process`` execution hook -- everything shared by the
    logical and physical layers. Layer-specific behaviour (CSE keys, choice
    detection, backend selection) lives on the ``Op`` / ``PhysicalOp``
    subclasses, not here.
    """

    #: Display label for a logical op: the semantic family (SELECT, PROJECT, ...)
    #: or a stripped stratum-op name. When set, the plan shows this instead of the
    #: concrete class name -- but only for logical instances (see ``_is_physical``).
    #: Auto-generated skrub-compat wrappers leave this None and keep their class
    #: name.
    logical_family: str | None = None

    #: Physical impls multiply-inherit from their logical family op (e.g.
    #: ``PandasQuerySelectionOp(SelectionOp, PhysicalOp)``), so they would inherit
    #: the family's ``logical_family``. ``PhysicalOp`` overrides this to True --
    #: and sits ahead of ``IRNode`` in every impl's MRO -- so a lowered op falls
    #: back to its concrete class name, keeping the operator-selection choice
    #: visible in the plan.
    _is_physical = False

    def __init__(self, inputs=None, outputs=None, name=None, is_X=False, is_y=False):
        self.name = name
        self.outputs = outputs if outputs is not None else []
        self.inputs = inputs if inputs is not None else []
        self.is_X = is_X
        self.is_y = is_y
        self.output_type = OutputType.UNKNOWN
        self.is_split_op = False
        self.was_cloned = False
        self.remove_after: list[IRNode] = []
        self.output_schema = None

    def to_str_helper(self):
        class_name = (self.__class__.__name__ if self._is_physical
                      else self.logical_family or self.__class__.__name__)
        is_df = " [df]" if self.output_type is OutputType.FRAME else ""
        name = f"({self.name})" if self.name and len(self.name) > 0 else ""
        # truncate name if it is too long
        if len(name) > 50:
            name = name[:50] + "..."
        return class_name, name, is_df

    def __str__(self):
        return "".join(self.to_str_helper())

    def __repr__(self):
        class_name, name, is_df = self.to_str_helper()
        return f"{class_name}{name}[cloned={self.was_cloned}, id={id(self)}{is_df}]"

    def update_name(self):
        pass

    def has_outputs(self) -> bool:
        return self.outputs is not None and len(self.outputs) > 0

    @property
    def num_input_operands(self) -> int:
        return len(self.inputs)

    def _check_dup_in_inputs(self, input: "IRNode") -> int | None:
        """Return the input index if already present, otherwise None."""
        for i, in_ in enumerate(self.inputs):
            if in_ is input:
                return i
        return None

    def _check_dup_in_outputs(self, output: "IRNode") -> int | None:
        """Return the output index if already present, otherwise None."""
        for i, out_ in enumerate(self.outputs):
            if out_ is output:
                return i
        return None

    def add_output(self, output: "IRNode") -> int:
        """Add an output edge, de-duplicating. Returns the output index."""
        idx = self._check_dup_in_outputs(output)
        if idx is not None:
            return idx
        self.outputs.append(output)
        return len(self.outputs) - 1

    def add_input(self, input: "IRNode") -> int:
        """Add an input edge, de-duplicating. Returns the operand index of `input`."""
        idx = self._check_dup_in_inputs(input)
        if idx is not None:
            return idx
        self.inputs.append(input)
        return len(self.inputs) - 1

    def _dedupe_input_refs(self, old_ref, new_ref):
        """Remove input slot old_ref and redirect its OperandRefs to new_ref.

        ``old_ref`` is always the higher of the two slots (the duplicate we drop)
        and ``new_ref`` the surviving lower slot. Refs pointing at old_ref are
        redirected to new_ref; refs pointing after old_ref shift left by one. The
        renumbering goes through the shared :func:`remap_operand_refs` walker so
        refs buried in a ``ColumnExpr`` field (e.g. a SelectionOp predicate) are
        remapped too, not just refs in plain tuples/lists/dicts.
        """
        n = len(self.inputs)
        self.inputs.pop(old_ref)
        mapping = {k: (new_ref if k == old_ref else k - 1 if k > old_ref else k)
                   for k in range(n)}
        for field in getattr(type(self), "fields", []):
            value = getattr(self, field)
            new_value = remap_operand_refs(value, mapping)
            if new_value is not value:
                setattr(self, field, new_value)

    def consumes_inputs_positionally(self) -> bool:
        """Whether this op addresses its inputs by position rather than OperandRef.

        Such ops (ChoiceOp outcomes, ImplOp's cached ``operand_index``) must never
        have two input slots collapsed into one by edge de-duplication -- their
        slots are kept distinct instead. Shared with ``_op_cse._can_merge`` so the
        two dedup paths treat the same op types as un-collapsible.
        """
        return False

    def replace_input(self, old_input: "IRNode", new_input: "IRNode"):
        """Replace an input edge, deduplicating OperandRef-based inputs when needed.

        If replacing old_input with new_input would create a duplicate input, keep
        the leftmost slot and remap OperandRefs away from the removed slot. Ops that
        consume inputs positionally keep the duplicate slot (a plain swap instead).
        """
        i = self._check_dup_in_inputs(old_input)
        if i is None:
            raise ValueError(f"Input {old_input} not found in {self.__class__.__name__}.")
        idx = self._check_dup_in_inputs(new_input)
        if idx is not None and idx != i and not self.consumes_inputs_positionally():
            if idx < i:
                self._dedupe_input_refs(old_ref=i, new_ref=idx)
            else:
                self.inputs[i] = new_input
                self._dedupe_input_refs(old_ref=idx, new_ref=i)
            return
        self.inputs[i] = new_input

    def replace_input_of_outputs(self, new_input):
        for out in self.outputs:
            out.replace_input(self, new_input)

    def replace_output(self, old_output: "IRNode", new_output: "IRNode"):
        i = self._check_dup_in_outputs(old_output)
        if i is None:
            raise ValueError(f"Output {old_output} not found in {self.__class__.__name__}.")
        self.outputs[i] = new_output

    def replace_output_of_inputs(self, new_output):
        for in_ in self.inputs:
            in_.replace_output(self, new_output)

    def clone(self):
        if getattr(self.__class__, "fields", None) is None:
            raise NotImplementedError(f"Cloning of {self.__class__.__name__} objects is not implemented yet. Please implement it.")
        args, atts = self.__class__.fields, self.__dict__.items()
        fields = {k: clone_value(v) for k, v in atts if k in args}
        new_op = self.__class__(**fields)
        new_op.was_cloned = True
        return new_op

    def process(self, mode: str, inputs: list):
        raise NotImplementedError(f"Processing of {self.__class__.__name__} objects is not implemented yet. Please implement it.")

    def check_kwargs(self, kwargs):
        if not isinstance(kwargs, dict):
            raise TypeError(
                f"The `{self}'s kwargs` should be a dict of named arguments. Got an object of type"
                f" {type(kwargs).__name__!r} instead: {kwargs!r}"
            )
