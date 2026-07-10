"""Small upstream patches that are not physical-operator choices.

Rust estimator implementations are registered with the physical operator
registry. They are no longer installed by replacing upstream skrub or sklearn
classes at import time.
"""
from __future__ import annotations

import importlib
import threading
from types import ModuleType
from typing import Dict, Tuple, List

from stratum.patching._gridsearch import make_grid_search as StratumMakeGridSearch

# ------------------------
# Manual registry
# ------------------------
# Definition: (module, symbol) -> adapter
_DEFINITION_REPLACEMENTS: Dict[Tuple[str, str], object] = {
}

# Method-level replacements (for methods on classes)
# Format: (module, class_name, method_name) -> adapter
_METHOD_REPLACEMENTS: Dict[Tuple[str, str, str], object] = {
    ("skrub._data_ops._skrub_namespace", "SkrubNamespace", "make_grid_search"): StratumMakeGridSearch,
}

# Replace/override names in these upstream usage modules if present.
# Keep this list manually maintained. Add to it if skrub adds new direct imports.
_USAGE_MODULES: List[str] = [
]

# Symbol-level overrides for usage modules and top-level exposure on `skrub`
# (symbol name) -> adapter
_SYMBOL_OVERRIDES: Dict[str, object] = {
}

# Idempotence sentinel + lock
_PATCH_SENTINEL_NAME = "_STRATUM_PATCHED"
_LOCK = threading.RLock()


def _import_module(modname: str) -> ModuleType:
    return importlib.import_module(modname)


def _ensure_upstream() -> ModuleType:
    # Import the upstream package object
    return _import_module("skrub")


def _set_symbol(mod: ModuleType, name: str, value: object) -> None:
    try:
        setattr(mod, name, value)
    except Exception as exc:
        # Fail-soft: we keep going; this is safe because adapters fallback to parent behavior
        # if they get used elsewhere and unsupported settings occur.
        # In practice, setattr should not fail for valid modules.
        pass


def _patch_definitions() -> None:
    for (modname, symbol), adapter in _DEFINITION_REPLACEMENTS.items():
        # Ensure sklearn is imported if we are patching it
        if modname.startswith("sklearn"):
            # skrub already imports sklearn modules internally, but it's safer
            # to ensure it's loaded before trying to patch.
            _import_module("sklearn.preprocessing")

        mod = _import_module(modname)
        _set_symbol(mod, symbol, adapter)


def _patch_methods() -> None:
    """Patch methods on classes."""
    for (modname, class_name, method_name), adapter in _METHOD_REPLACEMENTS.items():
        try:
            mod = _import_module(modname)
            cls = getattr(mod, class_name, None)
            if cls is not None:
                _set_symbol(cls, method_name, adapter)
        except Exception:
            # If the module, class, or method doesn't exist, skip it.
            continue


def _patch_usage_modules() -> None:
    for modname in _USAGE_MODULES:
        try:
            mod = _import_module(modname)
        except Exception:
            # If a usage module doesn't exist in this skrub version, skip it.
            continue
        for symbol, adapter in _symbol_OVERRIDES_ITEMS():
            if hasattr(mod, symbol):
                _set_symbol(mod, symbol, adapter)


def _symbol_OVERRIDES_ITEMS():
    # Helper to avoid global lookup in hot loops
    return _SYMBOL_OVERRIDES.items()

def patch_skrub() -> None:
    """Patch upstream `skrub` in-place for non-operator-selector hooks.

    This function is safe to call multiple times (idempotent).
    """
    with _LOCK:
        upstream = _ensure_upstream()
        if getattr(upstream, _PATCH_SENTINEL_NAME, False):
            return  # already patched

        # 1) Patch definitions (so future internal imports resolve to adapters)
        _patch_definitions()

        # 2) Patch methods on classes
        _patch_methods()

        # 3) Patch usage modules (so already-imported names are overwritten)
        _patch_usage_modules()

        # 4) Patch top-level `skrub` for user-facing imports
        for symbol, adapter in _symbol_OVERRIDES_ITEMS():
            _set_symbol(upstream, symbol, adapter)

        # Mark as patched
        setattr(upstream, _PATCH_SENTINEL_NAME, True)
