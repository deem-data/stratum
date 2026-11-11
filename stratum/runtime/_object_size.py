# Estimate object sizes for the buffer pool and intermediate-result cache.

from sys import getsizeof
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType
from pandas import DataFrame, Series
from pandas.core.groupby.groupby import GroupBy
from polars import DataFrame as PolarsDataFrame, Series as PolarsSeries
import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator
from logging import getLogger
logger = getLogger(__name__)
size_cache = {}

def get_size(obj):
    if isinstance(obj, tuple) or isinstance(obj, list):
        return sum(get_size(item) for item in obj)
    elif isinstance(obj, dict):
        return sum(get_size(item) + get_size(key) for key, item in obj.items())
    else:
        return get_size_single_object(obj)

# TODO: replace the one-byte fallback with estimates for unsupported types.
# It can substantially undercount retained memory, delaying budget-triggered
# eviction. Such entries are still eviction candidates, but unsupported spill
# types remain resident until removed. The fallback allows plans to continue
# when an intermediate's size is unknown.
UNKNOWN_SIZE = 1


def _unknown_size(obj, kind: str = "") -> int:
    logger.debug(f"No memory estimate for {kind or 'type'} {type(obj)}; "
                 f"counting it as {UNKNOWN_SIZE} byte")
    return UNKNOWN_SIZE


def get_size_single_object(obj):
    if type(obj).__module__.startswith("pandas"):
        return get_size_pandas(obj)
    if type(obj).__module__.startswith("polars"):
        return get_size_polars(obj)
    if type(obj).__module__.startswith("numpy"):
        return get_size_numpy(obj)
    if isinstance(obj, (str, int, float, bool, bytes)) or obj is None:
        return getsizeof(obj)
    if isinstance(obj, BaseEstimator) or type(obj).__module__.startswith("sklearn."):
        return get_size_estimator(obj)
    return _unknown_size(obj)


def get_size_estimator(estimator):
    """Estimate retained estimator state, including private and nested objects.

    Count shared objects once and array views by their underlying numpy buffer.
    Return UNKNOWN_SIZE for the estimator if any object requires a custom
    __getstate__ implementation from an unsupported package.

    Native storage absent from serialization state is not included. Physical
    implementations with a known layout measure their state directly instead.
    """
    try:
        return _retained(estimator, {}) or _unknown_size(estimator, "estimator")
    except _UserState as exc:
        return _unknown_size(estimator, f"estimator holding a {exc.args[0].__qualname__}")


#: Classes, functions, modules and methods excluded from retained-state estimates.
_NOT_OWNED = (type, ModuleType, FunctionType, BuiltinFunctionType, MethodType)

#: Packages allowed to supply __getstate__ for memory estimation. Other
#: implementations may have side effects: skorch serializes its network and
#: imports referenced modules, which can re-execute a script loaded by path.
_KNOWN_STATE_PACKAGES = frozenset({"sklearn", "skrub", "stratum", "lightgbm", "xgboost",
                                   "catboost"})


class _UserState(Exception):
    """Unsupported __getstate__ implementation; args[0] is the defining class."""


def _state_builder(cls: type) -> type:
    """The class whose ``__getstate__`` builds the state of a ``cls`` instance."""
    return next(k for k in cls.__mro__ if "__getstate__" in vars(k))


def _retained(obj, seen: dict) -> int:
    """Estimate retained bytes for objects not already counted in ``seen``.

    ``seen`` maps object IDs to references. Retaining temporary serialization
    state prevents its ID from being reused during traversal.
    """
    if isinstance(obj, ndarray):
        while isinstance(obj.base, ndarray):
            obj = obj.base
    if id(obj) in seen or isinstance(obj, _NOT_OWNED):
        return 0
    seen[id(obj)] = obj
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sum(_retained(item, seen) for item in obj)
    if isinstance(obj, dict):
        return sum(_retained(key, seen) + _retained(item, seen) for key, item in obj.items())
    if isinstance(obj, (str, int, float, bool, bytes)) or obj is None:
        return getsizeof(obj)
    if type(obj).__module__.split(".")[0] in ("pandas", "polars", "numpy"):
        return get_size_single_object(obj)
    # Estimate estimator and native state from serialization state, for example
    # sklearn tree arrays or a LightGBM model string. Only call Python's default
    # __getstate__ or implementations from supported libraries.
    builder = _state_builder(type(obj))
    if builder is not object and builder.__module__.partition(".")[0] not in _KNOWN_STATE_PACKAGES:
        raise _UserState(builder)
    try:
        state = obj.__getstate__()
    except Exception:
        return _unknown_size(obj)
    return _retained(state, seen) if state is not None else _unknown_size(obj)

def get_size_pandas(obj):
    if isinstance(obj, DataFrame):
        return obj.memory_usage(deep=True).sum()
    elif isinstance(obj, Series):
        return obj.memory_usage(deep=True)
    elif isinstance(obj, GroupBy):
        # Estimate a GroupBy by its retained source frame, excluding grouper state.
        # This path is used when a groupby(...) op is
        # left unfused (no `by` argument, a column selection in between, or an
        # aggregation spec the logical IR cannot represent) and its intermediate
        # lands in the pool.
        return get_size(obj.obj)
    else:
        return _unknown_size(obj, "pandas type")

def get_size_polars(obj):
    if isinstance(obj, PolarsDataFrame):
        return obj.estimated_size(unit="b")
    elif isinstance(obj, PolarsSeries):
        return obj.estimated_size(unit="b")
    else:
        return _unknown_size(obj, "polars type")

def get_size_numpy(obj):
    if isinstance(obj, ndarray):
        return obj.nbytes
    elif isinstance(obj, np.generic):
        return obj.itemsize
    else:
        return _unknown_size(obj, "numpy type")

def prettify_bytes(num_bytes: float) -> str:
    """Format a byte count as a human-readable string (e.g. ``"1.50 MB"``)."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(num_bytes)
    for unit in units:
        # Stop at the last unit even if `size` is still >= 1024, so the number
        # and unit always stay paired (values > 1024 PB report as "N.NN PB").
        if abs(size) < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} {units[-1]}"
