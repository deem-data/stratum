from sklearn.preprocessing import OneHotEncoder
from skrub import StringEncoder

import stratum as st
from stratum.optimizer.physical import build_default_physical_registry
from stratum.optimizer.physical._transform_execs import PassthroughTransformer, StringEncoderOp


def test_skrub_and_sklearn_estimators_are_not_monkey_patched():
    assert StringEncoder.__module__.startswith("skrub")
    assert OneHotEncoder.__module__.startswith("sklearn")


def test_rust_estimators_are_registered_as_physical_operators():
    registry = build_default_physical_registry()

    # OneHotEncoder is keyed on the generic physical transformer.
    transformer_rust = registry.candidates_for(PassthroughTransformer, backend_name="rust")
    string_encoder_rust = registry.candidates_for(StringEncoderOp, backend_name="rust")

    assert len(transformer_rust) == 1
    assert len(string_encoder_rust) == 1
    assert {c.backend_name for c in (*transformer_rust, *string_encoder_rust)} == {"rust"}


def test_stratum_still_exposes_adapter_classes_for_direct_legacy_use():
    assert st.StringEncoder.__name__ == "RustyStringEncoder"
    assert st.OneHotEncoder.__name__ == "RustyOneHotEncoder"
