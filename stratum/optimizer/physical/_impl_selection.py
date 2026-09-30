"""Implementation-selection pass.

# TODO: Rewrite all comments
Lowering fixes the *shape* of the physical plan; this pass fixes the *impl* of
each op: which concrete backend-specific implementation actually runs. For every
op in the DAG it asks the :class:`~stratum.optimizer.physical._registry.PhysicalRegistry`
for the candidate :class:`PhysicalImpl` entries registered under the op's type,
filters them through each candidate's ``supports(op, ctx)`` check, and lets an
:class:`ImplementationSelector` choose one. The choice is then *bound* into the
op at plan time: the op is swapped to the impl's concrete
:class:`~stratum.optimizer.physical._physical_ops.PhysicalOp` class in place
(identity preserved -- the buffer pool and all DAG edges key on identity), and
its ``on_impl_selected(ctx)`` folds any plan-time state into the op. A backend
that binds by mutation (e.g. the Rust kernels, which swap a transformer's
estimator for the Rust adapter) does so in ``on_impl_selected`` on its concrete
class.

Execution afterwards is plain ``op.process`` with **no selection control flow
left**.

Ops with no candidates (un-migrated logical families, ValueOp, ChoiceOp, ...)
pass through and keep executing their own ``process``. An op that *has*
candidates but ends up with none bound is a different case: if it has no
``process`` of its own, as the frame families do not, nothing can run it, so that
fails here rather than at execution.

"""
from __future__ import annotations

from stratum.optimizer.logical._base import IRNode
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.physical._predictor_execs import RandomForestOp
from stratum.optimizer.physical._registry import (PhysicalImpl, PhysicalRegistry,
                                                  get_default_physical_registry)
from stratum.optimizer._op_utils import topological_iterator
from stratum.utils._utils import start_time, log_time

import logging
logger = logging.getLogger(__name__)


class ImplementationSelector:
    """Strategy interface: pick one impl for ``op`` from ``candidates``.

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


class DefaultImplementationSelector(ImplementationSelector):
    """Choose implementations using the stable default backend preference.

    The selector is intentionally local to one operator.  Candidates have
    already been filtered through ``supports(op, ctx)`` by :func:`bind_op`, so
    this policy only ranks the supported implementations and falls back to
    registration order when none of the preferred backends is available.
    """

    _PREFERRED_BACKENDS = ("pandas", "sklearn-skrub", "numpy")
    _OP_PREFERENCES = {
        RandomForestOp: ("sklearn_rf",),
    }

    # FIXME: PandasInMemoryFrame may fail if the in-memory dataframe is Polars
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


class GreedyImplementationSelector(ImplementationSelector):
    """Choose implementations using an efficient-backend-first preference.

    Like DefaultImplementationSelector, this policy only ranks the
    already ``supports``-filtered candidates for one operator at a time. It
    does not call :meth:`PhysicalImpl.cost` (still a placeholder) and does not
    look at neighboring operators, formats, conversion costs, or plan-wide
    costs. This is a baseline.
    stratum backend contains hybrid implementations (python + rust).
    """

    _PREFERRED_BACKENDS = (
        "rust", "stratum", "polars", "numpy", "sklearn-skrub", "pandas"
    )
    _OP_PREFERENCES = {
        RandomForestOp: _GREEDY_RF_PREFERENCES,
    }

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


_IMPLEMENTATION_SELECTOR_FACTORIES: dict[str, type[ImplementationSelector]] = {
    "default": DefaultImplementationSelector,
    "greedy": GreedyImplementationSelector,
}


def get_implementation_selector(mode: str) -> ImplementationSelector:
    """Get the configured implementation-selection policy.

    Keeping construction in one place makes adding a future selector a small,
    explicit change while preserving selector injection for focused tests.
    """
    try:
        selector_type = _IMPLEMENTATION_SELECTOR_FACTORIES[mode]
    except KeyError as exc:
        raise ValueError(
            f"Unknown implementation selector {mode!r}; available selectors "
            f"are {sorted(_IMPLEMENTATION_SELECTOR_FACTORIES)}.") from exc
    return selector_type()


class FlagBasedSelector(ImplementationSelector):
    """Pin every frame op to one named dataframe backend.

    Preference order: a Rust kernel when ``ctx.prefer_rust`` (the old
    ``allow_patch and rust_backend`` gate, decided per op by ``supports``), then
    the impl matching :attr:`backend`, then a backend-agnostic impl
    (sklearn/skrub estimators, numpy sources).

    The backend is constructor state rather than a field on the plan context:
    it is a selection policy, so the selector that acts on it is the only thing
    that should carry it. Returning ``None`` when no candidate matches is what
    makes that pin strict -- an abstract op then survives selection and
    :func:`_assert_no_abstract_ops` fails the plan, instead of the op silently
    running on the other backend.

    Not registered in :data:`_IMPLEMENTATION_SELECTOR_FACTORIES`: a pinned
    backend is a testing tool (``OptConfig(selector=...)``), not a user-facing
    mode. Users reach polars through ``implementation_selector="greedy"``.
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

    Looks up the registry candidates for the op's type, filters by
    ``supports(op, ctx)``, lets the selector choose, and binds the choice by
    swapping ``op.__class__`` to the impl's concrete class (identity preserved;
    a logical op that is a pure backend refinement becomes its physical subclass)
    and running its ``on_impl_selected(ctx)``.

    Ops with no candidate are left untouched (un-migrated families / structural
    ops run their own ``process``). Raises when an op that needs an
    implementation gets none, either because every backend refused it or because
    the selector matched none of the ones that did. Returns ``op``.
    """
    if registry is None:
        registry = get_default_physical_registry()
    if selector is None:
        selector = get_implementation_selector(ctx.implementation_selector)

    registered = registry.candidates_for(type(op))
    candidates = [c for c in registered if c.supports(op, ctx)]
    impl = selector.choose(op, candidates, ctx)
    if impl is None:
        if registered and _needs_an_impl(op):
            raise NotImplementedError(_no_impl_message(op, selector, registered, candidates))
        return op
    logger.debug(f"Selected {impl.backend_name} implementation for {op}")
    if impl.impl_class is not None and impl.impl_class is not type(op):
        op.__class__ = impl.impl_class # late-binding
    # Downstream format-sensitive operators can inspect the already-bound
    # producer while this topological pass selects their own implementation.
    op._selected_backend = impl.backend_name
    if isinstance(op, PhysicalOp):
        op.on_impl_selected(ctx)
    return op


def _no_impl_message(op: IRNode, selector: ImplementationSelector,
                     registered, candidates) -> str:
    backends = sorted({c.backend_name for c in registered})
    if not candidates:
        reason = ("every registered backend refused it in supports(), which is how "
                  "an operator whose shape a backend cannot express opts out")
    else:
        reason = (f"the selector matched none of the supported backends "
                  f"{sorted({c.backend_name for c in candidates})} against "
                  f"{selector!r}")
    return (f"No implementation was bound for {op}: {reason}. Registered backends "
            f"for {type(op).__name__} are {backends}.")


def _needs_an_impl(op: IRNode) -> bool:
    """Whether ``op`` would be unrunnable if no implementation were bound.

    The frame families are pure config: they carry no ``process`` of their own, so
    an unbound one raises a bare NotImplementedError at execution time, far from
    the decision that caused it. The estimator families do define ``process`` and
    legitimately run unbound when an optional accelerator refuses them.
    """
    return type(op).process is IRNode.process


def select_implementations(root: IRNode, ctx: PlanContext,
                           registry: PhysicalRegistry | None = None,
                           selector: ImplementationSelector | None = None) -> IRNode:
    """Resolve every op with registered candidates to a concrete implementation.

    Ops without candidates (un-migrated families and structural ops) are left
    as-is; they keep executing via their own ``process``. Returns ``root``
    (selection binds in place, so the root object is unchanged).
    """
    start = start_time()
    if registry is None:
        registry = get_default_physical_registry()
    if selector is None:
        selector = get_implementation_selector(ctx.implementation_selector)

    for op in topological_iterator(root):
        bind_op(op, ctx, registry=registry, selector=selector)
    log_time("implementation selection took", start)
    _assert_no_abstract_ops(root, selector)
    return root


def _assert_no_abstract_ops(root: IRNode,
                            selector: ImplementationSelector) -> None:
    """Guard: no abstract physical op may reach the scheduler.

    A surviving abstract op means lowering produced it but the selector chose
    nothing for it -- its ``process`` would raise at run time. Fail loudly at
    plan time instead.
    """
    for op in topological_iterator(root):
        if isinstance(op, PhysicalOp) and getattr(op, "is_abstract", False):
            raise RuntimeError(
                f"Abstract physical op {op!r} survived implementation selection; "
                f"{selector!r} matched none of its registered implementations. "
                f"Register one with @physical_impl or fix its supports() checks."
            )
