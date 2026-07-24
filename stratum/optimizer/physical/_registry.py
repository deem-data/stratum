from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from stratum.optimizer.ir._base import IRNode
BackendName = str


"""Descriptor for one physical implementation of a plannable operator.

``op_type`` is the type the implementation is registered under: an *abstract
physical* op type for families already compiled to the physical layer (e.g.
``ReadCSV``), or a *logical* op type for families that still pass through
lowering unchanged (e.g. ``TransformerOp``). ``supports``/``cost``/``exec_mem``
form the fixed selector-facing API; ``impl_class`` names the concrete
``PhysicalOp`` the op is swapped to when this impl is chosen (identity preserved),
after which its ``on_impl_selected`` folds in any plan-time state.

This is the shared, backend-agnostic schema. A backend that carries extra
scheduling metadata *subclasses* this (e.g., :class:`RustPhysicalImpl`) instead of
widening the base, so the common schema stays small."""
@dataclass(frozen=True, slots=True)
class PhysicalImpl:
    op_type: type[IRNode]
    backend_name: BackendName
    input_format: str
    output_format: str
    supports: Callable[[IRNode, Any], bool]
    cost: Callable[[IRNode, Any], float]
    exec_mem: Callable[[IRNode, Any], int]
    execute: Callable[[IRNode, str, list[Any]], Any]
    # Concrete PhysicalOp class the op is swapped to when this impl is chosen.
    impl_class: type | None = None


@dataclass(frozen=True, slots=True)
class RustPhysicalImpl(PhysicalImpl):
    """Registry entry for a native Rust implementation.

    Extends the shared schema with the Rust scheduling capabilities. ``@rust_impl``
    reads these off the op class
    (:class:`~stratum.optimizer.physical._physical_ops.RustPhysicalOp`) rather than
    setting them independently, so the entry the planner reasons over and the
    operator that runs cannot disagree. No selector consumes them yet (the cost
    model is pending); they exist for the parallelization planner."""

    #: GIL released while the kernel runs; safe to run concurrently with others.
    releases_gil: bool = False
    #: Kernel parallelizes internally; the planner should not also fan it out.
    data_parallel: bool = False


def _unsupported_supports(op: IRNode, ctx: Any) -> bool:
    return False


def _unsupported_cost(op: IRNode, stats: Any) -> float:
    raise NotImplementedError("No physical cost model has been registered for this operator yet.")


def _unsupported_exec_mem(op: IRNode, stats: Any) -> int:
    raise NotImplementedError("No execution-memory model has been registered for this operator yet.")


def _unsupported_execute(op: IRNode, mode: str, inputs: list[Any]) -> Any:
    raise NotImplementedError("No physical implementation has been registered for this operator yet.")


def _current_process_execute(op: IRNode, mode: str, inputs: list[Any]) -> Any:
    return op.process(mode, inputs)


def _placeholder_cost(op: IRNode, stats: Any) -> float:
    return 1.0


def _placeholder_exec_mem(op: IRNode, stats: Any) -> int:
    return 0


"""Dictionary for physical implementations keyed by operator type.
Extensibile to support new backends. """
class PhysicalRegistry:
    def __init__(
        self,
        implementations: Iterable[PhysicalImpl] = (),
    ) -> None:
        self._implementations: dict[type[IRNode], list[PhysicalImpl]] = {}
        self._implementations_by_backend: dict[BackendName, list[PhysicalImpl]] = {}
        for impl in implementations:
            self.register(impl)

    def register(self, impl: PhysicalImpl) -> PhysicalImpl:
        self._implementations.setdefault(impl.op_type, []).append(impl)
        self._implementations_by_backend.setdefault(impl.backend_name, []).append(impl)
        return impl

    def op_types(self) -> tuple[type[IRNode], ...]:
        """Return operator types with at least one registered implementation."""
        return tuple(self._implementations)

    def candidates_for(
        self,
        op: type[IRNode] | IRNode,
        backend_name: BackendName | None = None,
    ) -> tuple[PhysicalImpl, ...]:
        op_type = op if isinstance(op, type) else type(op)
        candidates = self._implementations.get(op_type, ())
        if backend_name is not None:
            candidates = [impl for impl in candidates if impl.backend_name == backend_name]
        return tuple(candidates)

    def candidates_for_op(
        self,
        op: IRNode,
        backend_name: BackendName | None = None,
    ) -> tuple[PhysicalImpl, ...]:
        return self.candidates_for(op, backend_name=backend_name)

    def backends_for(self, op: type[IRNode] | IRNode) -> tuple[BackendName, ...]:
        return tuple(impl.backend_name for impl in self.candidates_for(op))

    def has_candidates(self, op: type[IRNode] | IRNode) -> bool:
        return len(self.candidates_for(op)) > 0

    def candidates_by_backend(self, backend_name: BackendName) -> tuple[PhysicalImpl, ...]:
        return tuple(self._implementations_by_backend.get(backend_name, ()))

    def empty(self) -> bool:
        return not self._implementations


# PhysicalImpl descriptors collected by the @physical_impl class decorator, in
# declaration order. build_default_physical_registry imports the exec modules
# (triggering decoration) and registers everything gathered here.
_DECORATED_IMPLS: list[PhysicalImpl] = []


def physical_impl(of: type[IRNode], backend: BackendName,
                  input_format: str = "frame", output_format: str = "frame"):
    """Class decorator registering a concrete PhysicalOp as a base PhysicalImpl.

    ``of`` is the (abstract) op type the class implements, e.g.::

        @physical_impl(of=ReadCSV, backend="pandas", input_format="value")
        class PandasReadCSV(ReadCSV): ...

    The selector-facing hooks (``supports``/``cost``/``exec_mem``) are taken from
    the class (PhysicalOp provides placeholder defaults); ``execute`` delegates to
    the op's own ``process``, which is concrete once the selection pass has swapped
    the op to this class.

    This builds a base :class:`PhysicalImpl`. Backends that carry extra scheduling
    metadata have their own decorator (:func:`rust_impl`) that builds the matching
    :class:`PhysicalImpl` subclass; the plain per-backend decorators below only fix
    ``backend`` until their backend grows its own schema.
    """
    def deco(cls):
        _DECORATED_IMPLS.append(PhysicalImpl(
            op_type=of,
            backend_name=backend,
            input_format=input_format,
            output_format=output_format,
            supports=cls.supports,
            cost=cls.cost,
            exec_mem=cls.exec_mem,
            execute=_current_process_execute,
            impl_class=cls,
        ))
        return cls
    return deco


def rust_impl(of: type[IRNode], input_format: str = "frame",
              output_format: str = "frame"):
    """Register a native Rust ``PhysicalOp`` as a :class:`RustPhysicalImpl`.

    Like :func:`physical_impl`, but builds the Rust-specific registry entry and
    reads the Rust capability hints (``releases_gil`` / ``data_parallel``) off the
    op class -- which should derive from
    :class:`~stratum.optimizer.physical._physical_ops.RustPhysicalOp` -- so the
    registry entry and the operator carry the same values.
    """
    def deco(cls):
        _DECORATED_IMPLS.append(RustPhysicalImpl(
            op_type=of,
            backend_name="rust",
            input_format=input_format,
            output_format=output_format,
            supports=cls.supports,
            cost=cls.cost,
            exec_mem=cls.exec_mem,
            execute=_current_process_execute,
            impl_class=cls,
            releases_gil=cls.releases_gil,
            data_parallel=cls.data_parallel,
        ))
        return cls
    return deco


# Plain per-backend decorators: fix ``backend`` and build a base PhysicalImpl. They
# exist so call sites read as one backend each; a backend grows its own decorator
# + PhysicalImpl subclass (as Rust has) once it needs backend-specific fields.
def _backend_impl(backend: BackendName):
    def decorator(of: type[IRNode], input_format: str = "frame",
                  output_format: str = "frame"):
        return physical_impl(of=of, backend=backend,
                             input_format=input_format, output_format=output_format)
    return decorator


polars_impl = _backend_impl("polars")
pandas_impl = _backend_impl("pandas")
numpy_impl = _backend_impl("numpy")
sklearn_skrub_impl = _backend_impl("sklearn-skrub")


def _register_current_estimator_impls(registry: PhysicalRegistry) -> None:
    # These are transitional registrations for estimator families that do not
    # have abstract physical operators yet. Once those lowerings land, their
    # implementations should be keyed by the corresponding physical type.
    from stratum.optimizer.ir._ops import PredictorOp, TransformerOp

    for op_type in (TransformerOp, PredictorOp):
        registry.register(
            PhysicalImpl(
                op_type=op_type,
                backend_name="sklearn-skrub",
                input_format="frame",
                output_format="frame",
                supports=lambda op, ctx: True,
                cost=_placeholder_cost,
                exec_mem=_placeholder_exec_mem,
                execute=_current_process_execute,
            )
        )


"""Create the default registry with every known implementation registered."""
def build_default_physical_registry() -> PhysicalRegistry:
    registry = PhysicalRegistry()

    # Imported lazily: the exec modules import back into this module (the
    # decorator / PhysicalImpl), so pulling them in at module level would cycle.
    # Importing each exec module triggers its lowering rules and
    # @physical_impl/@rust_impl registrations. Keep this list complete so a
    # standalone registry construction does not depend on optimizer imports.
    from stratum.optimizer.physical import _source_execs  # noqa: F401
    from stratum.optimizer.physical import _transform_execs  # noqa: F401
    from stratum.optimizer.physical import _concat_execs  # noqa: F401
    from stratum.optimizer.physical import _join_execs  # noqa: F401
    from stratum.optimizer.physical import _aggregation_execs  # noqa: F401
    from stratum.optimizer.physical import _projection_execs  # noqa: F401
    from stratum.optimizer.physical import _selection_execs  # noqa: F401
    from stratum.optimizer.physical import _map_execs  # noqa: F401
    from stratum.optimizer.physical import _getitem_execs  # noqa: F401

    for impl in _DECORATED_IMPLS:
        registry.register(impl)
    _register_current_estimator_impls(registry)
    return registry


_default_registry: PhysicalRegistry | None = None


"""Shared default registry, built once on first use."""
def get_default_physical_registry() -> PhysicalRegistry:
    """The implementation-selection pass consults this unless a registry is
    injected explicitly (tests, custom planners)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = build_default_physical_registry()
    return _default_registry
