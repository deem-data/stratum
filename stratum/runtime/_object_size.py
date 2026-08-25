# This util function helps to determine the memory usage of a certain object
# this is relevant for the buffer pool and intermediate cache

from sys import getsizeof
from pandas import DataFrame, Series
from pandas.core.groupby.groupby import GroupBy
from polars import DataFrame as PolarsDataFrame, Series as PolarsSeries
import numpy as np
from numpy import ndarray
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
    return _unknown_size(obj)

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
