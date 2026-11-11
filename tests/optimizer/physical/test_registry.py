import pytest

from stratum.optimizer.logical._dataframe_ops import ConcatOp
from stratum.optimizer.logical._numeric_ops import NumericOp
from stratum.optimizer.logical._ops import PredictorOp, Op, TransformerOp
from stratum.optimizer.physical import (
    PhysicalImpl,
    PhysicalOp,
    PhysicalRegistry,
    RustPhysicalImpl,
    _registry,
    build_default_physical_registry,
)
from stratum.optimizer.logical._source_ops import (
    InMemoryFrameOp, LoadNumpyOp, ReadCSVOp, ReadParquetOp)
from stratum.optimizer.physical._source_execs import (
    PandasInMemoryFrame,
    PandasReadCSV,
    PandasReadParquet,
    PolarsInMemoryFrame,
    PolarsReadCSV,
    PolarsReadParquet,
)
from stratum.optimizer.physical._predictor_execs import (
    CatBoostOp,
    DecisionTreeOp,
    ElasticNetOp,
    ExtraTreesOp,
    HistGradientBoostingOp,
    KNeighborsOp,
    LassoOp,
    LightGBMOp,
    LinearRegressionOp,
    LogisticRegressionOp,
    RandomForestOp,
    StratumExactRandomForestClassifier,
    StratumHistogramRandomForestClassifier,
    RidgeOp,
    SGDOp,
    XGBoostOp,
)
from stratum.optimizer.physical._transform_execs import (RustOneHotEncoder,
                                                        RustStringEncoder,
                                                        SkrubStringEncoder,
                                                        StringEncoderOp,
                                                        PassthroughTransformer)


def test_default_registry_discovers_registered_operator_types():
    registry = build_default_physical_registry()

    assert not registry.empty()
    assert registry.op_types()
    # The registry surface comes from registrations, not a catalog of every
    # logical IR type. NumericOp has no implementation and must not appear.
    assert ConcatOp in registry.op_types()
    assert NumericOp not in registry.op_types()

    # Standalone construction imports source and transformer modules, so their
    # operators are discoverable directly.
    for op_type in (ReadCSVOp, ReadParquetOp, InMemoryFrameOp, LoadNumpyOp,
                    StringEncoderOp):
        assert op_type in registry.op_types()

    # OneHotEncoder's Rust kernel is keyed on the generic physical transformer.
    rust_candidates = registry.candidates_for(PassthroughTransformer, backend_name="rust")
    assert len(rust_candidates) == 1
    assert all(candidate.backend_name == "rust" for candidate in rust_candidates)
    assert not registry.candidates_for(TransformerOp)
    assert len(registry.candidates_for(PredictorOp, backend_name="sklearn-skrub")) == 1
    # The migrated StringEncoder physical op carries both a skrub and a rust impl.
    assert len(registry.candidates_for(StringEncoderOp, backend_name="rust")) == 1
    assert len(registry.candidates_for(StringEncoderOp, backend_name="sklearn-skrub")) == 1

    # Random forests expose exact and histogram Rust variants alongside sklearn.
    forest_candidates = registry.candidates_for(RandomForestOp)
    assert {candidate.implementation_name for candidate in forest_candidates} == {
        "sklearn_rf", "rf_exact", "rf_hist"
    }
    assert {
        candidate.impl_class for candidate in forest_candidates
        if candidate.backend_name == "rust"
    } == {StratumExactRandomForestClassifier, StratumHistogramRandomForestClassifier}

    # Every other migrated predictor family carries its sklearn reference impl.
    for op_type in (ExtraTreesOp, DecisionTreeOp,
                    HistGradientBoostingOp, KNeighborsOp, LinearRegressionOp,
                    RidgeOp, LassoOp, ElasticNetOp, LogisticRegressionOp, SGDOp,
                    LightGBMOp, XGBoostOp, CatBoostOp):
        (candidate,) = registry.candidates_for(op_type)
        assert candidate.backend_name == "sklearn-skrub"

    source_candidates = {
        ReadCSVOp: {PandasReadCSV, PolarsReadCSV},
        ReadParquetOp: {PandasReadParquet, PolarsReadParquet},
        InMemoryFrameOp: {PandasInMemoryFrame, PolarsInMemoryFrame},
    }
    for op_type, impl_classes in source_candidates.items():
        assert {candidate.impl_class for candidate in registry.candidates_for(op_type)} == impl_classes


def test_rust_kernels_are_class_based_impls():
    # Every Rust kernel is a class-based @rust_impl.
    registry = build_default_physical_registry()

    ohe_rust = registry.candidates_for(PassthroughTransformer, backend_name="rust")
    se_rust = registry.candidates_for(StringEncoderOp, backend_name="rust")

    assert len(ohe_rust) == 1 and ohe_rust[0].impl_class is RustOneHotEncoder
    assert len(se_rust) == 1 and se_rust[0].impl_class is RustStringEncoder


def test_rust_impl_is_its_own_dataclass_with_capability_hints():
    # Rust scheduling capabilities belong to the registry entry.
    registry = build_default_physical_registry()

    (rust,) = registry.candidates_for(StringEncoderOp, backend_name="rust")
    (skrub,) = registry.candidates_for(StringEncoderOp, backend_name="sklearn-skrub")

    assert isinstance(rust, RustPhysicalImpl)
    assert rust.impl_class is RustStringEncoder
    assert rust.releases_gil and rust.data_parallel
    assert not hasattr(RustStringEncoder, "releases_gil")
    assert not hasattr(RustStringEncoder, "data_parallel")

    assert type(skrub) is PhysicalImpl
    assert skrub.impl_class is SkrubStringEncoder
    assert not hasattr(skrub, "releases_gil")


@pytest.mark.parametrize("releases_gil, data_parallel", [(False, True), (True, False)])
def test_rust_impl_accepts_capability_overrides(monkeypatch, releases_gil, data_parallel):
    monkeypatch.setattr(_registry, "_DECORATED_IMPLS", [])

    class CustomOp(PhysicalOp):
        pass

    @_registry.rust_impl(of=CustomOp, releases_gil=releases_gil, data_parallel=data_parallel)
    class CustomRustOp(CustomOp):
        pass

    registry = PhysicalRegistry(_registry._DECORATED_IMPLS)
    (impl,) = registry.candidates_for(CustomOp, backend_name="rust")

    assert impl.impl_class is CustomRustOp
    assert impl.releases_gil is releases_gil
    assert impl.data_parallel is data_parallel


def test_implementation_name_defaults_to_concrete_class_name():
    registry = build_default_physical_registry()
    (rust,) = registry.candidates_for(StringEncoderOp, backend_name="rust")

    assert rust.implementation_name == RustStringEncoder.__name__


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
    )
    rust_impl = PhysicalImpl(
        op_type=DummyPhysicalOp,
        backend_name="rust",
        input_format="frame",
        output_format="frame",
        supports=lambda op, ctx: isinstance(op, DummyPhysicalOp),
        cost=lambda op, stats: 0.5,
        exec_mem=lambda op, stats: 1,
    )

    registry.register(pandas_impl)
    registry.register(rust_impl)

    assert registry.candidates_for(DummyPhysicalOp) == (pandas_impl, rust_impl)
    assert registry.candidates_for(DummyPhysicalOp()) == (pandas_impl, rust_impl)
    assert registry.candidates_for(DummyPhysicalOp, backend_name="rust") == (rust_impl,)
    assert registry.backends_for(DummyPhysicalOp) == ("pandas", "rust")
    assert registry.candidates_by_backend("pandas") == (pandas_impl,)
    assert registry.candidates_by_backend("rust") == (rust_impl,)

    class UnregisteredLogicalOp(Op):
        pass

    assert DummyPhysicalOp in registry.op_types()
    assert UnregisteredLogicalOp not in registry.op_types()
