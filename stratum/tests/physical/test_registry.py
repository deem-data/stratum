from stratum.optimizer.ir._dataframe_ops import ConcatOp
from stratum.optimizer.ir._numeric_ops import NumericOp
from stratum.optimizer.ir._ops import PredictorOp, Op, TransformerOp
from stratum.optimizer.physical import (
    PhysicalImpl,
    PhysicalOp,
    PhysicalRegistry,
    RustPhysicalImpl,
    build_default_physical_registry,
)
from stratum.optimizer.physical._source_execs import (
    InMemoryFrame,
    NumpyLoad,
    ReadCSV,
    ReadParquet,
    PandasInMemoryFrame,
    PandasReadCSV,
    PandasReadParquet,
    PolarsInMemoryFrame,
    PolarsReadCSV,
    PolarsReadParquet,
)
from stratum.optimizer.physical._transform_execs import (RustOneHotEncoder,
                                                        RustStringEncoder,
                                                        SkrubStringEncoder,
                                                        StringEncoderOp)


def test_default_registry_discovers_registered_operator_types():
    registry = build_default_physical_registry()

    assert not registry.empty()
    assert registry.op_types()
    # The registry surface comes from registrations, not a catalog of every
    # logical IR type. NumericOp has no implementation and must not appear.
    assert ConcatOp in registry.op_types()
    assert NumericOp not in registry.op_types()

    # Standalone construction imports source and transformer modules, so the
    # already-migrated abstract physical operators are discoverable directly.
    for op_type in (ReadCSV, ReadParquet, InMemoryFrame, NumpyLoad,
                    StringEncoderOp):
        assert op_type in registry.op_types()

    # StringEncoder migrated to its own physical op, so only OneHotEncoder's
    # Rust kernel is still keyed on the logical TransformerOp.
    rust_candidates = registry.candidates_for(TransformerOp, backend_name="rust")
    sklearn_candidates = registry.candidates_for(TransformerOp, backend_name="sklearn-skrub")
    assert len(rust_candidates) == 1
    assert all(candidate.backend_name == "rust" for candidate in rust_candidates)
    assert len(sklearn_candidates) == 1
    assert len(registry.candidates_for(PredictorOp, backend_name="sklearn-skrub")) == 1
    # The migrated StringEncoder physical op carries both a skrub and a rust impl.
    assert len(registry.candidates_for(StringEncoderOp, backend_name="rust")) == 1
    assert len(registry.candidates_for(StringEncoderOp, backend_name="sklearn-skrub")) == 1

    source_candidates = {
        ReadCSV: {PandasReadCSV, PolarsReadCSV},
        ReadParquet: {PandasReadParquet, PolarsReadParquet},
        InMemoryFrame: {PandasInMemoryFrame, PolarsInMemoryFrame},
    }
    for op_type, impl_classes in source_candidates.items():
        assert {candidate.impl_class for candidate in registry.candidates_for(op_type)} == impl_classes


def test_rust_kernels_are_class_based_impls():
    # After unification every Rust kernel is a class-based @rust_impl: OneHotEncoder
    # is still keyed on the logical TransformerOp, StringEncoder on its own op.
    registry = build_default_physical_registry()

    ohe_rust = registry.candidates_for(TransformerOp, backend_name="rust")
    se_rust = registry.candidates_for(StringEncoderOp, backend_name="rust")

    assert len(ohe_rust) == 1 and ohe_rust[0].impl_class is RustOneHotEncoder
    assert len(se_rust) == 1 and se_rust[0].impl_class is RustStringEncoder


def test_rust_impl_is_its_own_dataclass_with_capability_hints():
    # Rust has a dedicated PhysicalImpl subclass carrying scheduling capabilities,
    # read off the op class (RustPhysicalOp). Other backends stay on the base
    # PhysicalImpl, which has no such fields -- the schema is not shared.
    registry = build_default_physical_registry()

    (rust,) = registry.candidates_for(StringEncoderOp, backend_name="rust")
    (skrub,) = registry.candidates_for(StringEncoderOp, backend_name="sklearn-skrub")

    assert isinstance(rust, RustPhysicalImpl)
    assert rust.impl_class is RustStringEncoder
    assert rust.releases_gil and rust.data_parallel
    # Hints are sourced from the op class, so the entry and the operator agree.
    assert RustStringEncoder.releases_gil and RustStringEncoder.data_parallel

    assert type(skrub) is PhysicalImpl
    assert skrub.impl_class is SkrubStringEncoder
    assert not hasattr(skrub, "releases_gil")


def test_registry_registers_and_queries_impls_by_registered_type():
    registry = PhysicalRegistry()

    class DummyPhysicalOp(PhysicalOp):
        pass

    pandas_impl = PhysicalImpl(
        op_type=DummyPhysicalOp,
        backend_name="pandas",
        input_format="frame",
        output_format="frame",
        supports=lambda op, ctx: isinstance(op, DummyPhysicalOp),
        cost=lambda op, stats: 1.0,
        exec_mem=lambda op, stats: 1,
        execute=lambda op, mode, inputs: ("concat", mode, len(inputs)),
    )
    rust_impl = PhysicalImpl(
        op_type=DummyPhysicalOp,
        backend_name="rust",
        input_format="frame",
        output_format="frame",
        supports=lambda op, ctx: isinstance(op, DummyPhysicalOp),
        cost=lambda op, stats: 0.5,
        exec_mem=lambda op, stats: 1,
        execute=lambda op, mode, inputs: ("rust-concat", mode, len(inputs)),
    )

    registry.register(pandas_impl)
    registry.register(rust_impl)

    assert registry.candidates_for(DummyPhysicalOp) == (pandas_impl, rust_impl)
    assert registry.candidates_for_op(DummyPhysicalOp()) == (pandas_impl, rust_impl)
    assert registry.candidates_for(DummyPhysicalOp, backend_name="rust") == (rust_impl,)
    assert registry.backends_for(DummyPhysicalOp) == ("pandas", "rust")
    assert registry.candidates_by_backend("pandas") == (pandas_impl,)
    assert registry.candidates_by_backend("rust") == (rust_impl,)

    class UnregisteredLogicalOp(Op):
        pass

    assert DummyPhysicalOp in registry.op_types()
    assert UnregisteredLogicalOp not in registry.op_types()
