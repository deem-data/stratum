from stratum.optimizer.ir._dataframe_ops import ConcatOp
from stratum.optimizer.ir._numeric_ops import NumericOp
from stratum.optimizer.ir._ops import EstimatorOp, Op, TransformerOp
from stratum.optimizer.physical import (
    CURRENT_BACKENDS,
    CURRENT_LOGICAL_OPERATOR_TYPES,
    OperatorFamily,
    PhysicalImpl,
    PhysicalRegistry,
    build_default_physical_registry,
)
from stratum.optimizer.physical._rust_registry import RUST_KERNELS


def test_default_registry_has_logical_surface_and_adapter_candidates():
    registry = build_default_physical_registry()

    assert not registry.empty()
    assert registry.logical_op_types()
    assert ConcatOp in CURRENT_LOGICAL_OPERATOR_TYPES
    assert NumericOp in CURRENT_LOGICAL_OPERATOR_TYPES
    assert "rust" in {backend.name for backend in CURRENT_BACKENDS}
    rust_candidates = registry.candidates_for(TransformerOp, backend_name="rust")
    sklearn_candidates = registry.candidates_for(TransformerOp, backend_name="sklearn-skrub")
    assert len(rust_candidates) == 2
    assert all(candidate.backend_name == "rust" for candidate in rust_candidates)
    assert len(sklearn_candidates) == 1
    assert len(registry.candidates_for(EstimatorOp, backend_name="sklearn-skrub")) == 1


def test_rust_kernel_registration_list_is_the_source_of_rust_candidates():
    registry = build_default_physical_registry()

    rust_candidates = registry.candidates_for(TransformerOp, backend_name="rust")

    assert len(rust_candidates) == len(RUST_KERNELS)
    assert {kernel.name for kernel in RUST_KERNELS} == {"one_hot_encoder", "string_encoder"}


def test_registry_registers_and_queries_impls_by_logical_type():
    registry = PhysicalRegistry()

    class DummyOp(Op):
        pass

    pandas_impl = PhysicalImpl(
        logical_op_type=DummyOp,
        backend_name="pandas",
        input_format="frame",
        output_format="frame",
        supports=lambda op: isinstance(op, DummyOp),
        cost=lambda op, stats: 1.0,
        exec_mem=lambda op, stats: 1,
        execute=lambda op, mode, inputs: ("concat", mode, len(inputs)),
    )
    rust_impl = PhysicalImpl(
        logical_op_type=DummyOp,
        backend_name="rust",
        input_format="frame",
        output_format="frame",
        supports=lambda op: isinstance(op, DummyOp),
        cost=lambda op, stats: 0.5,
        exec_mem=lambda op, stats: 1,
        execute=lambda op, mode, inputs: ("rust-concat", mode, len(inputs)),
    )

    registry.register(pandas_impl)
    registry.register(rust_impl)

    assert registry.candidates_for(DummyOp) == (pandas_impl, rust_impl)
    assert registry.candidates_for_op(DummyOp()) == (pandas_impl, rust_impl)
    assert registry.candidates_for(DummyOp, backend_name="rust") == (rust_impl,)
    assert registry.backends_for(DummyOp) == ("pandas", "rust")
    assert registry.candidates_by_backend("pandas") == (pandas_impl,)
    assert registry.candidates_by_backend("rust") == (rust_impl,)


def test_register_family_tracks_known_logical_types():
    registry = PhysicalRegistry()

    family = OperatorFamily(
        name="custom",
        logical_op_types=(ConcatOp,),
        default_backends=("pandas",),
    )
    registry.register_family(family)

    assert registry.families() == (family,)
    assert ConcatOp in registry.logical_op_types()
