from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from stratum.optimizer.ir._aggregation_ops import AggregateOp, GroupedDataframeOp
from stratum.optimizer.ir._dataframe_ops import (
    ApplyUDFOp,
    AssignOp,
    ConcatOp,
    DatetimeConversionOp,
    DropOp,
    GetAttrProjectionOp,
    MetadataOp,
    ProjectionOp,
    SplitOp,
    SplitOutput,
    StringMethodOp,
)
from stratum.optimizer.ir._join_ops import JoinOp
from stratum.optimizer.ir._numeric_ops import NumericOp
from stratum.optimizer.ir._ops import (
    BaseEstimatorOp,
    BinOp,
    CallOp,
    ChoiceOp,
    GetAttrOp,
    GetItemOp,
    ImplOp,
    MethodCallOp,
    Op,
    SearchEvalOp,
    ValueOp,
    VariableOp,
    EstimatorOp,
    TransformerOp,
)
BackendName = str


"""Descriptor for one physical implementation of a logical operator."""
@dataclass(frozen=True, slots=True)
class PhysicalImpl:
    logical_op_type: type[Op]
    backend_name: BackendName
    input_format: str
    output_format: str
    supports: Callable[[Op], bool]
    cost: Callable[[Op, Any], float]
    exec_mem: Callable[[Op, Any], int]
    execute: Callable[[Op, str, list[Any]], Any]


"""Logical operator family used to keep the registry extensible."""
@dataclass(frozen=True, slots=True)
class OperatorFamily:
    name: str
    logical_op_types: tuple[type[Op], ...]
    default_backends: tuple[BackendName, ...] = ()
    notes: str = ""


"""A physical execution backend understood by the registry."""
@dataclass(frozen=True, slots=True)
class BackendSpec:
    name: str
    notes: str = ""


# FIXME: Only list the stratum's logical operators after compilation from skrub IR
CURRENT_LOGICAL_OPERATOR_TYPES: tuple[type[Op], ...] = (
    AggregateOp,
    ApplyUDFOp,
    AssignOp,
    BaseEstimatorOp,
    BinOp,
    CallOp,
    ChoiceOp,
    ConcatOp,
    DatetimeConversionOp,
    DropOp,
    EstimatorOp,
    GetAttrOp,
    GetAttrProjectionOp,
    GetItemOp,
    GroupedDataframeOp,
    ImplOp,
    JoinOp,
    MetadataOp,
    MethodCallOp,
    NumericOp,
    ProjectionOp,
    SearchEvalOp,
    SplitOp,
    SplitOutput,
    StringMethodOp,
    TransformerOp,
    ValueOp,
    VariableOp,
)


CURRENT_BACKENDS: tuple[BackendSpec, ...] = (
    BackendSpec("pandas", "Pandas dataframe implementation."),
    BackendSpec("polars", "Polars dataframe implementation."),
    BackendSpec("numpy", "NumPy array implementation."),
    BackendSpec("sklearn-skrub", "Existing sklearn/skrub implementation."),
    BackendSpec("rust", "Native Rust implementation selected like any other backend."),
)


CURRENT_OPERATOR_FAMILIES: tuple[OperatorFamily, ...] = (
    OperatorFamily(
        name="logical",
        logical_op_types=CURRENT_LOGICAL_OPERATOR_TYPES,
        default_backends=tuple(backend.name for backend in CURRENT_BACKENDS),
        notes="Current logical IR surface; backends are attached later by the planner.",
    ),
)


def _unsupported_supports(op: Op) -> bool:
    return False


def _unsupported_cost(op: Op, stats: Any) -> float:
    raise NotImplementedError("No physical cost model has been registered for this operator yet.")


def _unsupported_exec_mem(op: Op, stats: Any) -> int:
    raise NotImplementedError("No execution-memory model has been registered for this operator yet.")


def _unsupported_execute(op: Op, mode: str, inputs: list[Any]) -> Any:
    raise NotImplementedError("No physical implementation has been registered for this operator yet.")


def _current_process_execute(op: Op, mode: str, inputs: list[Any]) -> Any:
    return op.process(mode, inputs)


def _placeholder_cost(op: Op, stats: Any) -> float:
    return 1.0


def _placeholder_exec_mem(op: Op, stats: Any) -> int:
    return 0


"""Container for physical implementations and their logical families."""
class PhysicalRegistry:
    def __init__(
        self,
        families: Iterable[OperatorFamily] = (),
        implementations: Iterable[PhysicalImpl] = (),
    ) -> None:
        self._families: list[OperatorFamily] = list(families)
        self._implementations: dict[type[Op], list[PhysicalImpl]] = {}
        self._implementations_by_backend: dict[BackendName, list[PhysicalImpl]] = {}
        for impl in implementations:
            self.register(impl)

    def register_family(self, family: OperatorFamily) -> None:
        self._families.append(family)

    def register(self, impl: PhysicalImpl) -> PhysicalImpl:
        self._implementations.setdefault(impl.logical_op_type, []).append(impl)
        self._implementations_by_backend.setdefault(impl.backend_name, []).append(impl)
        return impl

    def families(self) -> tuple[OperatorFamily, ...]:
        return tuple(self._families)

    def logical_op_types(self) -> tuple[type[Op], ...]:
        types: list[type[Op]] = []
        seen: set[type[Op]] = set()
        for family in self._families:
            for logical_type in family.logical_op_types:
                if logical_type not in seen:
                    seen.add(logical_type)
                    types.append(logical_type)
        for logical_type in self._implementations:
            if logical_type not in seen:
                seen.add(logical_type)
                types.append(logical_type)
        return tuple(types)

    def candidates_for(
        self,
        logical_op: type[Op] | Op,
        backend_name: BackendName | None = None,
    ) -> tuple[PhysicalImpl, ...]:
        logical_type = logical_op if isinstance(logical_op, type) else type(logical_op)
        candidates = self._implementations.get(logical_type, ())
        if backend_name is not None:
            candidates = [impl for impl in candidates if impl.backend_name == backend_name]
        return tuple(candidates)

    """Return the physical implementations available for a given logical operator."""
    def candidates_for_op(
        self,
        op: Op,
        backend_name: BackendName | None = None,
    ) -> tuple[PhysicalImpl, ...]:
        return self.candidates_for(op, backend_name=backend_name)

    def backends_for(self, logical_op: type[Op] | Op) -> tuple[BackendName, ...]:
        return tuple(impl.backend_name for impl in self.candidates_for(logical_op))

    def has_candidates(self, logical_op: type[Op] | Op) -> bool:
        return len(self.candidates_for(logical_op)) > 0

    def candidates_by_backend(self, backend_name: BackendName) -> tuple[PhysicalImpl, ...]:
        return tuple(self._implementations_by_backend.get(backend_name, ()))

    def empty(self) -> bool:
        return not self._implementations


def _register_current_estimator_impls(registry: PhysicalRegistry) -> None:
    for logical_op_type in (TransformerOp, EstimatorOp):
        registry.register(
            PhysicalImpl(
                logical_op_type=logical_op_type,
                backend_name="sklearn-skrub",
                input_format="frame",
                output_format="frame",
                supports=lambda op: True,
                cost=_placeholder_cost,
                exec_mem=_placeholder_exec_mem,
                execute=_current_process_execute,
            )
        )


"""Create the registry skeleton without registering any operators yet."""
def build_default_physical_registry() -> PhysicalRegistry:
    registry = PhysicalRegistry(families=CURRENT_OPERATOR_FAMILIES)

    from stratum.optimizer.physical._rust_registry import register_rust_physical_operators

    _register_current_estimator_impls(registry)
    register_rust_physical_operators(registry)
    return registry
