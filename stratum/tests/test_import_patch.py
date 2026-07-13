from sklearn.preprocessing import OneHotEncoder
from skrub import StringEncoder

import stratum as st
from stratum.optimizer.ir._ops import TransformerOp
from stratum.optimizer.physical import build_default_physical_registry


def test_skrub_and_sklearn_estimators_are_not_monkey_patched():
    assert StringEncoder.__module__.startswith("skrub")
    assert OneHotEncoder.__module__.startswith("sklearn")


def test_rust_estimators_are_registered_as_physical_operators():
    registry = build_default_physical_registry()

    rust_candidates = registry.candidates_for(TransformerOp, backend_name="rust")

    assert len(rust_candidates) == 2
    assert {candidate.backend_name for candidate in rust_candidates} == {"rust"}


def test_stratum_still_exposes_adapter_classes_for_direct_legacy_use():
    assert st.StringEncoder.__name__ == "RustyStringEncoder"
    assert st.OneHotEncoder.__name__ == "RustyOneHotEncoder"
