"""
Automatic monkey-patching of selected classes inside upstream skrub
so that internal imports like `from ._string_encoder import StringEncoder` and
usage in `TableVectorizer` resolve to Stratum adapters.

Design notes:
- Runs once, idempotent, without config flags.
- Maintains a manual registry. Add to it when a new adapter is implemented in stratum.
- Patches:
  (1) the *defining modules* (e.g., `skrub._string_encoder.StringEncoder`),
  (2) relevant *usage modules* that may already have bound names, and
  (3) the top-level `skrub` package symbols for user-facing imports.

Special case: OneHotEncoder
- `skrub` uses `sklearn.preprocessing.OneHotEncoder` internally.
- We therefore *cannot* patch a `skrub`-defined class. Instead, we override the
  names `OneHotEncoder` in `skrub`'s usage modules and also expose our
  `OneHotEncoder` at `skrub.OneHotEncoder` for convenience.
"""
from __future__ import annotations

import importlib
import threading
from types import ModuleType
from typing import Dict, Tuple, List

# --- Import adapters (raises at import time if not available) ---
from stratum.adapters.string_encoder import RustyStringEncoder
from stratum.adapters.one_hot_encoder import RustyOneHotEncoder
from stratum.patching._gridsearch import make_grid_search as StratumMakeGridSearch

# ------------------------
# Manual registry
# ------------------------
# Definition: (module, symbol) -> adapter
_DEFINITION_REPLACEMENTS: Dict[Tuple[str, str], object] = {
    ("skrub._string_encoder", "StringEncoder"): RustyStringEncoder,
    ("sklearn.preprocessing", "OneHotEncoder"): RustyOneHotEncoder,
    # Note: There is no skrub-defined OneHotEncoder to replace at definition site.
}

# Method-level replacements (for methods on classes)
# Format: (module, class_name, method_name) -> adapter
_METHOD_REPLACEMENTS: Dict[Tuple[str, str, str], object] = {
    ("skrub._data_ops._skrub_namespace", "SkrubNamespace", "make_grid_search"): StratumMakeGridSearch,
}

# Replace/override names in these upstream usage modules if present.
# Keep this list manually maintained. Add to it if skrub adds new direct imports.
_USAGE_MODULES: List[str] = [
    "skrub._table_vectorizer",
    "skrub._tabular_pipeline",
    # Add other skrub modules here if they import our adapters (e.g., StringEncoder, OneHotEncoder) directly.
]

# Symbol-level overrides for usage modules and top-level exposure on `skrub`
# (symbol name) -> adapter
_SYMBOL_OVERRIDES: Dict[str, object] = {
    "StringEncoder": RustyStringEncoder,
    "OneHotEncoder": RustyOneHotEncoder,  # our adapter extends sklearn's
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
    """Patch upstream `skrub` in-place so its internals use Stratum adapters.

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
