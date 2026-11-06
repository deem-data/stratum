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


@pytest.mark.parametrize("bad", [0, -0.1, 1.1, float("nan"), float("inf"), "half", []])
def test_invalid_reuse_fraction_does_not_change_config(bad):
    original = get_config()
    with pytest.raises(ValueError, match="reuse_cache_memory_fraction"):
        st.set_config(reuse_cache=True, reuse_cache_memory_fraction=bad)
    assert get_config() == original


def test_config_changes_only_what_it_is_given():
    with st.config(implementation_selector="greedy", cse=False, explain=True,
                   graph_format="png"):
        outer = get_config()
        with st.config(scheduler=True):
            inner = get_config()
            assert inner["scheduler"] is True
            assert {k: v for k, v in inner.items() if k != "scheduler"} == {
                k: v for k, v in outer.items() if k != "scheduler"}
            assert os.environ["STRATUM_IMPLEMENTATION_SELECTOR"] == "greedy"
        with st.config(explain=False):
            assert get_config()["explain"] == ()


def test_reuse_settings_survive_other_config_calls_and_restore_after_failure():
    original = get_config()
    with st.config(reuse_cache=True, reuse_cache_memory_fraction=0.25):
        st.set_config(stats=True)
        assert get_config()["reuse_cache"] is True
        assert get_config()["reuse_cache_memory_fraction"] == 0.25
        with pytest.raises(RuntimeError):
            with st.config(reuse_cache=False, reuse_cache_memory_fraction=1):
                assert get_config()["reuse_cache"] is False
                assert get_config()["reuse_cache_memory_fraction"] == 1.0
                raise RuntimeError("test")
        assert get_config()["reuse_cache"] is True
        assert get_config()["reuse_cache_memory_fraction"] == 0.25
    assert get_config() == original
