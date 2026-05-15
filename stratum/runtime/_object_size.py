# This util function helps to determine the memory usage of a certain object
# this is relevant for the buffer pool and intermediate cache

from sys import getsizeof
from pandas import DataFrame, Series
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

def get_size_single_object(obj):
    if type(obj).__module__.startswith("pandas"):
        return get_size_pandas(obj)
    if type(obj).__module__.startswith("polars"):
        return get_size_polars(obj)
    if type(obj).__module__.startswith("numpy"):
        return get_size_numpy(obj)
    if isinstance(obj, (str, int, float, bool, bytes)) or obj is None:
        return getsizeof(obj)
    raise ValueError(f"Unsupported type for memory estimation: {type(obj)}")

def get_size_pandas(obj):
    if isinstance(obj, DataFrame):
        return obj.memory_usage(deep=True).sum()
    elif isinstance(obj, Series):
        return obj.memory_usage(deep=True)
    else:
        raise ValueError(f"Unsupported pandas type for memory estimation: {type(obj)}")

def get_size_polars(obj):
    if isinstance(obj, PolarsDataFrame):
        return obj.estimated_size(unit="b")
    elif isinstance(obj, PolarsSeries):
        return obj.estimated_size(unit="b")
    else:
        raise ValueError(f"Unsupported polars type for memory estimation: {type(obj)}")

def get_size_numpy(obj):
    if isinstance(obj, ndarray):
        return obj.nbytes
    elif isinstance(obj, np.generic):
        return obj.itemsize
    else:
        raise ValueError(f"Unsupported numpy type for memory estimation: {type(obj)}")
