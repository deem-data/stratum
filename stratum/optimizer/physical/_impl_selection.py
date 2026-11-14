"""Bind operators to registered physical implementations.

For each operator, filter registry candidates through ``supports``, apply the
selection policy, and bind the selected class in place. ``on_impl_selected``
initializes its configuration. Execution dispatches through ``op.process``.

Ops without candidates keep their own ``process``. An op with candidates but
no bound impl and no ``process`` fails at plan time.

Frame libraries are propagated along edges; mismatches require a ``ConversionOp``
(see ``_conversion``).

TODO: Before supporting reselection or moving selection after train/test
unrolling, account for fitted-state compatibility, adapted estimator prototypes
and conversions inserted by an earlier selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from stratum.optimizer.logical._base import FRAME_TYPES, IRNode
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.physical._predictor_execs import RandomForestOp
from stratum.optimizer.physical._registry import (PhysicalImpl, PhysicalRegistry,
                                                  get_default_physical_registry)
from stratum.optimizer.physical._conversion import (ConversionOp, has_default_index,
                                                    index_origin)
from stratum.optimizer.physical._representation import (
    FrameLib, RepresentationError, accepts_input, check_frame_representations,
    input_libs, output_lib)
from stratum.optimizer._op_utils import topological_iterator
from stratum.utils._utils import start_time, log_time

import logging
logger = logging.getLogger(__name__)


class ImplementationSelector:
    """Selection policy for supported implementations of ``op``.

    ``candidates`` is already ``supports``-filtered. Returning ``None`` leaves the
    op unbound (valid only for non-abstract ops, which run their own ``process``).
    """

    def choose(self, op: IRNode, candidates: list[PhysicalImpl],
               ctx: PlanContext) -> PhysicalImpl | None:
        raise NotImplementedError


def _find_preferred_implementation(
    candidates: list[PhysicalImpl], preferred: tuple[str, ...]
) -> PhysicalImpl | None:
    """Return the first supported candidate in an explicit implementation order."""
    for implementation_name in preferred:
        for impl in candidates:
            if impl.implementation_name == implementation_name:
                return impl
    return None


# Internal debug flag. Exact remains registered and can become greedy's
# primary forest implementation without changing operator or registry code.
_GREEDY_RF_EXACT_FIRST = False # Set true to select rf_exact Rust implementation
_GREEDY_RF_PREFERENCES = (
    ("rf_exact", "rf_hist", "sklearn_rf")
    if _GREEDY_RF_EXACT_FIRST
    else ("rf_hist", "rf_exact", "sklearn_rf")
)


class PreferenceSelector(ImplementationSelector):
    """Choose an implementation from fixed preference lists.

    Selection is local to one operator. Candidates have
    already been filtered through ``supports(op, ctx)`` by :func:`bind_op`, so
    this policy only ranks the supported implementations: first by the
    implementation names listed for the op's type, then by backend, and in
    registration order when none of the preferred backends is available.
    Subclasses set the two lists.
    """

    #: Backends in order of preference.
    _PREFERRED_BACKENDS: tuple[str, ...] = ()
    #: op type -> implementation names in order of preference, ahead of backends.
    _OP_PREFERENCES: dict = {}

    def choose(self, op: IRNode, candidates: list[PhysicalImpl],
               ctx: PlanContext) -> PhysicalImpl | None:
        if not candidates:
            return None
        preferred = self._OP_PREFERENCES.get(type(op), ())
        implementation = _find_preferred_implementation(candidates, preferred)
        if implementation is not None:
            return implementation
        for backend_name in self._PREFERRED_BACKENDS:
            for impl in candidates:
                if impl.backend_name == backend_name:
                    return impl
        return candidates[0]


class DefaultImplementationSelector(PreferenceSelector):
    """Choose implementations using the stable default backend preference."""

    _PREFERRED_BACKENDS = ("pandas", "sklearn-skrub", "numpy")
    _OP_PREFERENCES = {
        RandomForestOp: ("sklearn_rf",),
    }


class GreedyImplementationSelector(PreferenceSelector):
    """Apply operator-specific preferences, then prefer Rust, Stratum and Polars.

    Backend preferences apply when no operator-specific preference matches.
    Selection uses fixed preferences rather than cost estimates. It does not
    consider neighboring operators or conversion costs. The Stratum backend
    combines Python orchestration with Rust kernels.
    """

    _PREFERRED_BACKENDS = (
        "rust", "stratum", "polars", "numpy", "sklearn-skrub", "pandas"
    )
    _OP_PREFERENCES = {
        RandomForestOp: _GREEDY_RF_PREFERENCES,
    }


_IMPLEMENTATION_SELECTOR_FACTORIES: dict[str, type[ImplementationSelector]] = {
    "default": DefaultImplementationSelector,
    "greedy": GreedyImplementationSelector,
}


def get_implementation_selector(mode: str) -> ImplementationSelector:
    """Construct the implementation-selection policy registered under ``mode``."""
    try:
        selector_type = _IMPLEMENTATION_SELECTOR_FACTORIES[mode]
    except KeyError as exc:
        raise ValueError(
            f"Unknown implementation selector {mode!r}; available selectors "
            f"are {sorted(_IMPLEMENTATION_SELECTOR_FACTORIES)}.") from exc
    return selector_type()


class FlagBasedSelector(ImplementationSelector):
    """Constrain frame operators to a specified dataframe backend.

    Prefer a supported Rust kernel when ``ctx.prefer_rust`` is enabled, then an
    implementation matching :attr:`backend`, then a backend-independent
    implementation (sklearn/skrub estimators or NumPy sources).

    The backend constraint is selector state. Return ``None`` when no candidate
    matches; unresolved abstract operators are rejected before execution.

    Used for testing through ``OptConfig(selector=...)`` and excluded from
    :data:`_IMPLEMENTATION_SELECTOR_FACTORIES`. The public ``"greedy"`` policy
    includes Polars in its backend preferences.
    """

    #: Backends whose impls run regardless of the chosen frame backend.
    _BACKEND_AGNOSTIC = ("sklearn-skrub", "numpy")

    def __init__(self, backend: str = "pandas"):
        self.backend = backend

    def __repr__(self) -> str:
        return f"{type(self).__name__}(backend={self.backend!r})"

    def choose(self, op: IRNode, candidates: list[PhysicalImpl],
               ctx: PlanContext) -> PhysicalImpl | None:
        if not candidates:
            return None
        if ctx.prefer_rust:
            for impl in candidates:
                if impl.backend_name == "rust":
                    return impl
        for impl in candidates:
            if impl.backend_name == self.backend:
                return impl
        for impl in candidates:
            if impl.backend_name in self._BACKEND_AGNOSTIC:
                return impl
        return None


def bind_op(op: IRNode, ctx: PlanContext,
            registry: PhysicalRegistry | None = None,
            selector: ImplementationSelector | None = None) -> IRNode:
    """Resolve a single op to a concrete implementation and bind it in place.

    Filter registered candidates through ``supports(op, ctx)``, apply the selector,
    and assign the selected ``op.__class__`` without changing node identity.
    Initialize the physical implementation with ``on_impl_selected(ctx)``.

    Operators without registered candidates remain unchanged. When candidates
    exist and the operator requires an implementation, reject the plan if none
    is selected. Return ``op``.

    No conversions are inserted; see :func:`select_implementations`.
    """
    if registry is None:
        registry = get_default_physical_registry()
    if selector is None:
        selector = get_implementation_selector(ctx.implementation_selector)
    impl = _choose_impl(op, ctx, registry, selector, pandas_required=False)
    _bind(op, impl, ctx, output_lib(_bound_class(op, impl), op, input_libs(op)))
    return op


def _impl_frame_lib(impl: PhysicalImpl) -> FrameLib | None:
    return getattr(impl.impl_class, "frame_lib", None) if impl.impl_class else None


def _choose_impl(op: IRNode, ctx: PlanContext, registry: PhysicalRegistry,
                 selector: ImplementationSelector,
                 pandas_required: bool) -> PhysicalImpl | None:
    """Choose an impl for ``op`` without binding it; records refusals on ``op``.

    ``pandas_required``: a consumer needs ``op``'s pandas index; polars excluded.
    """
    registered = registry.candidates_for(type(op))
    candidates, refusals = [], []
    for c in registered:
        if not c.supports(op, ctx):
            reason = getattr(op, "_unsupported_reason", None)
            op._unsupported_reason = None
            if reason:
                refusals.append((c.backend_name, reason))
            continue
        if pandas_required and _impl_frame_lib(c) is FrameLib.POLARS:
            refusals.append((c.backend_name, "a pandas consumer needs the index labels"))
            continue
        candidates.append(c)
    op._refusals = refusals
    impl = selector.choose(op, candidates, ctx)
    if impl is None and registered and _needs_an_impl(op):
        raise NotImplementedError(_no_impl_message(op, selector, registered, candidates))
    return impl


def _bound_class(op: IRNode, impl: PhysicalImpl | None):
    """Class of ``op`` after binding ``impl``."""
    return impl.impl_class if impl is not None and impl.impl_class else type(op)


def _bind(op: IRNode, impl: PhysicalImpl | None, ctx: PlanContext, lib) -> None:
    op._frame_lib = lib
    if impl is None:
        return
    logger.debug(f"Selected {impl.backend_name} implementation for {op}")
    if impl.impl_class is not None and impl.impl_class is not type(op):
        op.__class__ = impl.impl_class # late-binding
    if isinstance(op, PhysicalOp):
        op.on_impl_selected(ctx)


def _no_impl_message(op: IRNode, selector: ImplementationSelector,
                     registered, candidates) -> str:
    backends = sorted({c.backend_name for c in registered})
    if not candidates and getattr(op, "_refusals", None):
        reason = "every registered backend refused it: " + "; ".join(
            f"{backend}: {why}" for backend, why in op._refusals)
    elif not candidates:
        reason = ("every registered backend refused it in supports(), which is how "
                  "an operator whose shape a backend cannot express opts out")
    else:
        reason = (f"the selector matched none of the supported backends "
                  f"{sorted({c.backend_name for c in candidates})} against "
                  f"{selector!r}")
    return (f"No implementation was bound for {op}: {reason}. Registered backends "
            f"for {type(op).__name__} are {backends}.")


def _needs_an_impl(op: IRNode) -> bool:
    """Whether ``op`` inherits the unimplemented ``IRNode.process`` method.

    Such operators require a selected implementation. Operators with their own
    ``process`` method can retain it when no registered implementation is selected.
    """
    return type(op).process is IRNode.process


def select_implementations(root: IRNode, ctx: PlanContext,
                           registry: PhysicalRegistry | None = None,
                           selector: ImplementationSelector | None = None) -> IRNode:
    """Resolve every op with registered candidates to a concrete implementation.

    Operators without candidates retain their existing ``process`` method.
    Binding preserves node identity, including the returned ``root``.

    Insert ``ConversionOp`` nodes for library mismatches (#227). Where Polars -> pandas
    would lose index labels, the producer is required to run on pandas;
    selection iterates to a fixpoint over that set.
    """
    start = start_time()
    if registry is None:
        registry = get_default_physical_registry()
    if selector is None:
        selector = get_implementation_selector(ctx.implementation_selector)

    ops = list(topological_iterator(root))
    pandas_required: set[int] = set()
    while True:
        plan = _plan_selection(ops, ctx, registry, selector, pandas_required)
        if plan.new_pandas_required <= pandas_required:
            break
        pandas_required |= plan.new_pandas_required
    if plan.conflicts:
        raise RepresentationError(
            "The plan needs pandas index labels that a polars producer cannot "
            "provide, and no pandas implementation can take its place:\n  "
            + "\n  ".join(plan.conflicts))
    for op in ops:
        _bind(op, plan.impls[id(op)], ctx, plan.libs[id(op)])
    _insert_conversions(plan.conversions)
    log_time("implementation selection took", start)
    _assert_no_abstract_ops(root, selector)
    check_frame_representations(list(topological_iterator(root)))
    return root


@dataclass
class _SelectionPlan:
    impls: dict = field(default_factory=dict)        # id(op) -> PhysicalImpl | None
    libs: dict = field(default_factory=dict)         # id(op) -> output FrameLib | None
    conversions: dict = field(default_factory=dict)  # (producer, target) -> [(consumer, k)]
    new_pandas_required: set = field(default_factory=set)  # ids that must run on pandas
    conflicts: list = field(default_factory=list)          # unsatisfiable edges


def _plan_selection(ops, ctx, registry, selector,
                    pandas_required: set[int]) -> _SelectionPlan:
    """One selection pass: impls, edge libraries, conversions, pandas requirements."""
    plan = _SelectionPlan()
    index_memo: dict = {}
    for op in ops:
        impl = _choose_impl(op, ctx, registry, selector,
                            pandas_required=id(op) in pandas_required)
        bound_cls = _bound_class(op, impl)
        if (impl is not None and _impl_frame_lib(impl) is FrameLib.POLARS
                and _requires_label_alignment(op, index_memo)):
            # Positional pairing would differ from label alignment: use pandas.
            plan.new_pandas_required.add(id(op))
        in_libs = []
        for k, producer in enumerate(op.inputs):
            # State edges do not inherit the producer's data-output frame library.
            lib = None if op.reads_state(k) else plan.libs.get(id(producer))
            if lib is not None and not accepts_input(bound_cls, op, k, lib):
                lib = lib.other
                _enforce_input_lib(plan, producer, op, k, lib, pandas_required, index_memo)
            in_libs.append(lib)
        lib = output_lib(bound_cls, op, in_libs)
        if id(op) in pandas_required and lib is FrameLib.POLARS:
            # Pass-through op: pandas output requires pandas inputs.
            for k, producer in enumerate(op.inputs):
                if in_libs[k] is FrameLib.POLARS:
                    _enforce_input_lib(plan, producer, op, k, FrameLib.PANDAS,
                                       pandas_required, index_memo)
            lib = FrameLib.PANDAS
        plan.impls[id(op)] = impl
        plan.libs[id(op)] = lib
    return plan


def _requires_label_alignment(op: IRNode, index_memo: dict) -> bool:
    """Whether polars' positional pairing differs from pandas' label alignment."""
    if op.aligns_by_label is not None:
        return op.aligns_by_label
    frames = [i for k, i in enumerate(op.inputs)
              if i.output_type in FRAME_TYPES and not op.reads_state(k)]
    return len({index_origin(i, index_memo) for i in frames}) > 1


def _enforce_input_lib(plan: _SelectionPlan, producer: IRNode, consumer: IRNode,
                       k: int, target: FrameLib, pandas_required: set[int],
                       index_memo: dict) -> None:
    """Enforce ``target`` on edge ``producer -> consumer[k]``: convert, or require
    the producer to run on pandas."""
    if target is FrameLib.PANDAS and not has_default_index(producer, index_memo):
        if id(producer) in pandas_required:
            plan.conflicts.append(f"{consumer} reads input {k} ({producer}) as pandas")
        plan.new_pandas_required.add(id(producer))
        return
    plan.conversions.setdefault((producer, target), []).append((consumer, k))


def _insert_conversions(conversions: dict) -> None:
    """Insert one ConversionOp per (producer, target), shared by its consumers."""
    for (producer, target), uses in conversions.items():
        conversion = ConversionOp(target.other, target, inputs=[producer])
        conversion.output_type = producer.output_type
        conversion._frame_lib = target
        for consumer, k in uses:
            consumer.inputs[k] = conversion
            conversion.add_output(consumer)
            producer.outputs = [o for o in producer.outputs if o is not consumer]
        producer.add_output(conversion)


def _assert_no_abstract_ops(root: IRNode,
                            selector: ImplementationSelector) -> None:
    """Guard: no abstract physical op may reach the scheduler.

    A surviving abstract operator has no selected implementation. Reject the
    plan before its ``process`` method can raise during execution.
    """
    for op in topological_iterator(root):
        if isinstance(op, PhysicalOp) and getattr(op, "is_abstract", False):
            raise RuntimeError(
                f"Abstract physical op {op!r} survived implementation selection; "
                f"{selector!r} matched none of its registered implementations. "
                f"Register one with @physical_impl or fix its supports() checks."
            )
