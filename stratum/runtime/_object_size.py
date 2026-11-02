# This util function helps to determine the memory usage of a certain object
# this is relevant for the buffer pool and intermediate cache

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

# TODO: an unsized object is accounted for as 1 byte, so the pool never counts
# it towards its memory budget and never picks it for eviction: whatever it holds
# (possibly a whole frame behind a lazy view) leaks until the plan tears the pool
# down. Add a real estimate for every type that reaches the pool; the fallback
# only exists so an unknown intermediate cannot crash a plan mid-execution.
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
    """A fitted estimator, or a compiled sklearn structure such as a tree: all it
    retains, private and native state and nested estimators included.

    What is shared counts once, an array by the buffer it views, which a view
    keeps alive whole. An estimator holding anything whose state only user code
    can build is not sized at all, rather than counted in part.
    """
    try:
        return _retained(estimator, {}) or _unknown_size(estimator, "estimator")
    except _UserState as exc:
        return _unknown_size(estimator, f"estimator holding a {exc.args[0].__qualname__}")


#: Held by reference, not owned: a class, a function or a module.
_NOT_OWNED = (type, ModuleType, FunctionType, BuiltinFunctionType, MethodType)

#: Packages whose ``__getstate__`` is called to size what an object retains. Any
#: other is user code and may do anything: skorch's pickles the net, importing
#: the modules of the classes it names, which re-runs a script loaded by path.
_KNOWN_STATE_PACKAGES = frozenset({"sklearn", "skrub", "stratum", "lightgbm", "xgboost",
                                   "catboost"})


class _UserState(Exception):
    """An object's state would be built by user code (the class, as ``args[0]``)."""


def _state_builder(cls: type) -> type:
    """The class whose ``__getstate__`` builds the state of a ``cls`` instance."""
    return next(k for k in cls.__mro__ if "__getstate__" in vars(k))


def _retained(obj, seen: dict) -> int:
    """The bytes ``obj`` retains that ``seen`` does not hold already.

    ``seen`` maps ids to the objects counted, keeping each alive, as a pickled
    state is built anew and its id would otherwise be reused by the next one.
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
    # An estimator holds its attributes in the state it pickles, and so do native
    # structures (sklearn's trees, a LightGBM booster as its model string). Only
    # Python's default (the instance dict) or a known library builds that state.
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
        # A groupby is a lazy view: it holds the frame it was built from plus the
        # grouper, so the frame dominates. Reached whenever a groupby(...) op is
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
