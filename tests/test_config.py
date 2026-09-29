import os

import pytest

import stratum as st
from stratum._config import get_config


def test_config_restores_environment_after_nested_contexts(monkeypatch):
    monkeypatch.delenv("STRATUM_IMPLEMENTATION_SELECTOR", raising=False)
    monkeypatch.setenv("SKRUB_RUST", "original")
    original = get_config()

    with st.config(implementation_selector="greedy", rust_backend=True):
        assert os.environ["STRATUM_IMPLEMENTATION_SELECTOR"] == "greedy"
        assert os.environ["SKRUB_RUST"] == "1"
        with st.config(implementation_selector="default", rust_backend=False):
            assert os.environ["STRATUM_IMPLEMENTATION_SELECTOR"] == "default"
            assert os.environ["SKRUB_RUST"] == "0"
        assert os.environ["STRATUM_IMPLEMENTATION_SELECTOR"] == "greedy"
        assert os.environ["SKRUB_RUST"] == "1"

    assert get_config() == original
    assert "STRATUM_IMPLEMENTATION_SELECTOR" not in os.environ
    assert os.environ["SKRUB_RUST"] == "original"


def test_config_restores_environment_when_body_raises(monkeypatch):
    monkeypatch.delenv("STRATUM_IMPLEMENTATION_SELECTOR", raising=False)
    with pytest.raises(RuntimeError):
        with st.config(implementation_selector="greedy"):
            raise RuntimeError("test")
    assert "STRATUM_IMPLEMENTATION_SELECTOR" not in os.environ


def test_config_restores_partial_changes_when_setup_raises(monkeypatch):
    monkeypatch.delenv("SKRUB_RUST", raising=False)
    original = get_config()
    with pytest.raises(ValueError, match="num_threads"):
        with st.config(rust_backend=True, num_threads=-1):
            pass
    assert get_config() == original
    assert "SKRUB_RUST" not in os.environ
