"""Shared IR foundation for both the logical and physical operator layers.

``IRNode`` defines the shared DAG structure (``inputs``/``outputs`` and edge
updates), lifetime metadata (``remove_after``/``release_state_after``),
the split marker, and the ``process`` execution
hook. The logical layer subclasses it as :class:`~stratum.optimizer.logical._ops.Op`
and the physical layer as
:class:`~stratum.optimizer.physical._physical_ops.PhysicalOp`.

Logical and physical operators share this base while defining their own
structure keys, choice handling, and backend selection. Operand references
(:class:`OperandRef`), resolution helpers, and output categories
(:class:`OutputType`) are shared by both layers.
"""
from __future__ import annotations
import copy
from enum import Enum, auto
from typing import Any, NamedTuple

from sklearn.base import BaseEstimator
from skrub._data_ops._data_ops import DataOp
from skrub.selectors._base import All


def rebuild_dict(original: dict, items) -> dict:
    """A dict with ``items`` and the same type as ``original``.

    The operand helpers rebuild nested containers to swap operands for values.
    Preserve subclass behaviour, such as ``defaultdict.__missing__`` supplying
    defaults for absent keys in ``Series.map``.
    """
    if type(original) is dict:
        return dict(items)
    rebuilt = copy.copy(original)
    rebuilt.clear()
    for key, value in items:  # not update(): Counter.update adds counts
        rebuilt[key] = value
    return rebuilt


class OutputType(Enum):
    """The kind of value an :class:`IRNode` produces.

    - ``FRAME`` and ``SERIES`` distinguish tables from columns, allowing
      selection detection to recognize a Series-valued mask in ``df[mask]``.
      Boolean dtype is tracked separately from the output category.
    - ``SCALAR`` identifies scalar or aggregate metadata results.
    - ``MATRIX`` identifies ndarray results handled by numeric extraction.
    - ``UNKNOWN`` is the default when no more specific category is known.
    """
    UNKNOWN = auto()
    FRAME = auto()
    SERIES = auto()
    SCALAR = auto()
    MATRIX = auto()


class IndexEffect(Enum):
    """How an op's output row labels relate to its inputs' (on pandas).

    Index lineage (``physical/_conversion.py``) follows these to tell when two
    frames share row labels, so that pairing their rows by position (polars)
    gives what aligning them by label (pandas) does.
    """
    #: Labels of its own; nothing is assumed about them.
    NEW = auto()
    #: Its first input's rows, in order, with their labels.
    KEEP = auto()
    #: Elementwise over its inputs: their labels, if they all share them.
    COMMON = auto()
    #: A fresh ``0..n-1`` RangeIndex.
    DEFAULT = auto()
    #: The first rows of its first input: a prefix of ``0..n-1`` stays one.
    PREFIX = auto()


def has_default_labels(data) -> bool:
    """Whether ``data``'s rows carry a fresh ``0..n-1`` RangeIndex (a polars frame
    has no labels, which reads the same)."""
    import pandas as pd
    import polars as pl
    if isinstance(data, (pl.DataFrame, pl.Series)):
        return True
    index = getattr(data, "index", None)
    return isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1


# Output types handled by dataframe extraction (pandas/polars). A frame and a
# series are both manipulated by the dataframe extraction path; a MATRIX (numpy)
# is not. Used to decide whether an op consumes already-produced frame data (a
# dataframe operation) or a leaf/raw value (a read/source).
FRAME_TYPES = frozenset({OutputType.FRAME, OutputType.SERIES})


def is_frame_like(op) -> bool:
    """True if ``op`` produces a dataframe or a series."""
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
        return rebuild_dict(value, ((k, _resolve_operand(v, inputs)) for k, v in value.items()))
    if isinstance(value, slice):
        return slice(*(_resolve_operand(b, inputs) for b in (value.start, value.stop, value.step)))
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

    Recurses tuples/lists/dicts/slices and column-expression trees (anything exposing a
    ``remap_operand_refs`` method, e.g. a ``ColumnExpr`` predicate). This is the
    single walker shared by CSE edge de-duplication
    (``rewrites._common_subexpression_elimination``) and
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
        return rebuild_dict(value, ((k, remap_operand_refs(v, mapping)) for k, v in value.items()))
    if isinstance(value, slice):
        return slice(*(remap_operand_refs(b, mapping)
                       for b in (value.start, value.stop, value.step)))
    if hasattr(value, "remap_operand_refs"):
        return value.remap_operand_refs(mapping)
    return value


# --- Structure keys for common-subexpression elimination -------------------
# `Op.structure_key()` identifies computations eligible for CSE. Sentinel keys
# use marker strings to distinguish their roles. (CSE is a logical-layer concern, but the
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
        # The type is part of the key: a defaultdict is not the same argument
        # as a plain dict with the same items.
        # TODO: Preserve item order for configurations such as AssignMapOp.entries;
        # different insertion orders can produce different output column orders.
        return ("__dict__", type(value),
                frozenset((k, config_key(v)) for k, v in value.items()))
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


class Uncacheable(Exception):
    """An operator or its configuration cannot be given a safe lineage key."""


#: Default evaluation mode for fitting and stateless preprocessing.
FITTING_MODE = "fit_transform"


class Fitted(NamedTuple):
    """Data output and fitted state returned by a fitting operator.

    ``output`` supplies data consumers; a predictor's fit returns ``None``.
    ``state`` supplies response operators. The scheduler stores live outputs
    in the buffer pool and live fitted states in the state store.
    """
    output: Any
    state: Any
    #: Estimated state size in bytes; None delegates measurement to the scheduler.
    size: int | None = None


def clone_value(value):
    if isinstance(value, dict):
        return rebuild_dict(value, ((k, clone_value(v)) for k, v in value.items()))
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

    Execution borrows inputs and fitted states read-only, including nested
    objects. An operator that needs to mutate them must copy them itself before
    writing. Produced outputs and states may be shared with other consumers and
    the reuse cache, so the operator must not mutate them after returning either.
    """

    #: Display label for a logical op: the semantic family (SELECT, PROJECT, ...)
    #: or a stripped stratum-op name. When set, the plan shows this instead of the
    #: concrete class name -- but only for logical instances (see ``_is_physical``).
    #: Auto-generated skrub-compat wrappers leave this None and keep their class
    #: name.
    logical_family: str | None = None

    #: How the output's row labels relate to the inputs'; see :class:`IndexEffect`.
    #: A class whose effect depends on its fields defines a property instead.
    index_effect = IndexEffect.NEW

    #: How this op pairs the rows of several frame inputs on pandas: ``True`` by
    #: label, ``False`` by position or not at all, ``None`` by label wherever the
    #: labels differ (pandas' alignment of elementwise operands).
    aligns_by_label = None

    #: Physical impls multiply-inherit from their logical family op (e.g.
    #: ``PandasQuerySelectionOp(SelectionOp, PhysicalOp)``), so they would inherit
    #: the family's ``logical_family``. ``PhysicalOp`` overrides this to True --
    #: and sits ahead of ``IRNode`` in every impl's MRO -- so a lowered op falls
    #: back to its concrete class name, keeping the operator-selection choice
    #: visible in the plan.
    _is_physical = False

    #: Default mode before response operators are constructed during planning.
    fitting_mode = FITTING_MODE

    def __init__(self, inputs=None, outputs=None, name=None, is_X=False, is_y=False):
        self.name = name
        self.outputs = outputs if outputs is not None else []
        self.inputs = inputs if inputs is not None else []
        self.is_X = is_X
        self.is_y = is_y
        self.output_type = OutputType.UNKNOWN
        # Derived output schema (a polars.Schema, or None for "unknown"), filled in
        # by the schema-propagation pass. See logical/_schema.py.
        self.output_schema = None
        self.is_split_op = False
        self.was_cloned = False
        self.remove_after: list[IRNode] = []
        self.release_state_after: list[IRNode] = []
        #: Train/test copy membership; None for work outside the copied branches.
        self.train: bool | None = None
        self.state_ref: OperandRef | None = None
        #: Bound during planning; execution does not resolve or override it.
        self.mode: str = self.fitting_mode
        #: Set by reuse planning: admit this result under its lineage key.
        self.cache_key: str | None = None
        #: A retained cached value replaces execution; the op then has no inputs.
        #: See physical.rewrites._reuse_planning.CacheHit for the one-time
        #: handoff to the scheduler.
        self.cache_hit = None

    @property
    def produces_state(self) -> bool:
        """Whether execution returns data and fitted state in a :class:`Fitted`."""
        return self.has_state and self.mode in ("fit", FITTING_MODE)

    def reads_state(self, k: int) -> bool:
        """Whether input ``k`` is a fitted-state dependency."""
        return self.state_ref is not None and self.state_ref.k == k

    def has_data_reader(self) -> bool:
        """Whether this operator has a data-output consumer."""
        return any(in_op is self and not out.reads_state(k)
                   for out in self.outputs for k, in_op in enumerate(out.inputs))

    def has_state_reader(self) -> bool:
        """Whether this operator has a fitted-state consumer."""
        return any(out.reads_state(k) and in_op is self
                   for out in self.outputs for k, in_op in enumerate(out.inputs))

    def propagate_output_schema(self):
        """Default rule for the schema-propagation pass: the output is unknown.

        Ops that can statically determine their columns override this; see
        :mod:`stratum.optimizer.logical._schema` for the algebra they use and for the
        families that deliberately keep this default. Defined on ``IRNode`` so that
        physical nodes support the pass too (a physical source op is a
        ``PhysicalOp``, not an ``Op``).
        """
        self.output_schema = None

    #: Whether the operator produces fitted state for subsequent application.
    #: Planning creates response operators with explicit fitted-state inputs.
    has_state = False

    #: Opt-in contract: the output is a deterministic function of the inputs and
    #: lineage_config(). Generic calls and unknown operators remain uncacheable.
    cacheable = False

    def lineage_config(self):
        """Configuration naming this op's output, apart from its inputs.

        Cacheable ops return their declared ``fields`` as plain values. Operators
        needing a different encoding override this method; unsupported values are
        rejected by the lineage encoder. The configuration must include everything
        that can change the output. Raises :class:`Uncacheable` for ops that have
        not opted in.
        """
        if not self.cacheable:
            raise Uncacheable(f"{type(self).__name__} is not a cacheable operator")
        return {name: getattr(self, name) for name in type(self).fields}

    def to_str_helper(self):
        class_name = (self.__class__.__name__ if self._is_physical
                      else self.logical_family or self.__class__.__name__)
        if self.has_state:
            class_name = f"{class_name}_{self.mode}"
        if self.train is not None:
            class_name += "_train" if self.train else "_test"
        is_df = " [df]" if not self._is_physical and self.output_type is OutputType.FRAME else ""
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
        if self.state_ref is not None:
            self.state_ref = remap_operand_refs(self.state_ref, mapping)

    def consumes_inputs_positionally(self) -> bool:
        """Whether this op addresses its inputs by position rather than OperandRef.

        Such ops (ChoiceOp outcomes, ImplOp's cached ``operand_index``) must never
        have two input slots collapsed into one by edge de-duplication -- their
        slots are kept distinct instead. Shared with
        ``rewrites._common_subexpression_elimination._can_merge`` so the two
        dedup paths treat the same op types as un-collapsible.
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

    def detach(self) -> None:
        """Take this node out of the graph: drop it from its producers' outputs and
        clear its own edges. Its consumers are the caller's to rewire, before or after.

        Producers left without a consumer stay as they are, which is what a rewrite
        wants while it is about to hand them a new consumer; see ``detach_and_prune``.
        """
        for in_ in self.inputs:
            in_.outputs = [out for out in in_.outputs if out is not self]
        self.inputs = []
        self.outputs = []

    def detach_and_prune(self, keep: "IRNode | None" = None) -> None:
        """Detach this node and recursively prune producers with no consumers.

        Remove reverse edges from unreachable producers so topological traversal
        sees only the remaining DAG. Reconnect surviving producers before calling
        this method; a temporarily disconnected producer would otherwise be pruned.
        ``keep`` preserves a producer that becomes the new root without a consumer.
        """
        producers = list(dict.fromkeys(self.inputs))
        self.detach()
        for in_ in producers:
            if not in_.outputs and in_ is not keep:
                in_.detach_and_prune(keep)

    def clone(self):
        if getattr(self.__class__, "fields", None) is None:
            raise NotImplementedError(f"Cloning of {self.__class__.__name__} objects is not implemented yet. Please implement it.")
        args, atts = self.__class__.fields, self.__dict__.items()
        fields = {k: clone_value(v) for k, v in atts if k in args}
        new_op = self.__class__(**fields)
        new_op.was_cloned = True
        return new_op

    def process(self, mode: str, inputs: list):
        """Execute the bound mode with borrowed data and fitted-state inputs."""
        raise NotImplementedError(f"Processing of {self.__class__.__name__} objects is not implemented yet. Please implement it.")

    def check_kwargs(self, kwargs):
        if not isinstance(kwargs, dict):
            raise TypeError(
                f"The `{self}'s kwargs` should be a dict of named arguments. Got an object of type"
                f" {type(kwargs).__name__!r} instead: {kwargs!r}"
            )
