"""pandas ``.str`` methods under the greedy selector: exact or refused, never different.

``str.replace`` has its own suite (``test_string_replace``).
"""
import pandas as pd
import polars as pl
import pytest
import skrub

import stratum
from stratum._api import evaluate
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import optimize
from stratum.optimizer.physical._conversion import ConversionOp
from stratum.optimizer.physical._map_execs import PolarsAssignMapOp, PolarsStringMethodOp

FRAME = pd.DataFrame({"s": ["abcdef", "ab", "", "-42", "a,b", "Ａ"]})

TRANSLATED = [
    ("lower", (), {}),
    ("upper", (), {}),
    ("len", (), {}),
    ("strip", ("a-",), {}),
    ("lstrip", (), {}),
    ("rstrip", (), {"to_strip": "f"}),
    ("startswith", ("ab",), {}),
    ("endswith", (), {"pat": "f"}),
    ("contains", ("b",), {}),
    ("contains", ("b.",), {"regex": True}),
    ("contains", (".",), {"regex": False}),
    ("count", ("[ab]",), {}),
    # pandas slices [start:stop], polars by offset and length.
    ("slice", (1, 3), {}),
    ("slice", (), {"start": 2}),
    ("slice", (-3,), {}),
    ("slice", (3, 1), {}),
    ("zfill", (5,), {}),
    ("normalize", ("NFKC",), {}),
    ("split", (",",), {}),
]

REFUSED = [
    ("find", ("c",), {}),  # misses to -1 in pandas, null in polars
    ("contains", ("B",), {"case": False}),
    ("startswith", (("a", "b"),), {}),
    ("count", ("(?i)a",), {}),
    ("slice", (0, -1), {}),
    ("slice", (0, 4, 2), {}),
    ("split", (",",), {"n": 1}),
    ("split", (",b",), {}),  # a regex in pandas
    ("title", (), {}),
]


def _plan(method, args, kwargs, path):
    with skrub.config_context(eager_data_ops=False):
        source = skrub.var("frame", FRAME)
        called = getattr(source["s"].str, method)(*args, **kwargs)
        return called if path == "standalone" else source.assign(called=called)


def _expected(method, args, kwargs, path):
    called = getattr(FRAME["s"].str, method)(*args, **kwargs)
    return called if path == "standalone" else FRAME.assign(called=called)


def _evaluate(plan):
    with stratum.config(scheduler=True, implementation_selector="greedy"):
        ops, *_ = optimize(plan, env=get_data(plan))
        return ops, evaluate(plan)


def _values(series) -> list:
    """Series values with lists as lists and every missing value as ``None``."""
    return [list(v) if hasattr(v, "__len__") and not isinstance(v, str)
            else None if pd.isna(v) else v for v in series.tolist()]


def _assert_matches(result, expected):
    result = result.to_pandas() if hasattr(result, "to_pandas") else result
    if isinstance(expected, pd.Series):
        assert _values(expected) == _values(result)
    else:
        assert list(expected.columns) == list(result.columns)
        for name in expected.columns:
            assert _values(expected[name]) == _values(result[name])


@pytest.mark.parametrize("path", ["standalone", "assign"])
@pytest.mark.parametrize("method,args,kwargs", TRANSLATED)
def test_translated_method_matches_pandas(method, args, kwargs, path):
    ops, result = _evaluate(_plan(method, args, kwargs, path))
    native = PolarsStringMethodOp if path == "standalone" else PolarsAssignMapOp
    assert any(isinstance(op, native) for op in ops)
    assert isinstance(result, (pl.Series, pl.DataFrame))
    _assert_matches(result, _expected(method, args, kwargs, path))


@pytest.mark.parametrize("path", ["standalone", "assign"])
@pytest.mark.parametrize("method,args,kwargs", REFUSED)
def test_untranslatable_method_runs_on_pandas(method, args, kwargs, path):
    ops, result = _evaluate(_plan(method, args, kwargs, path))
    assert not any(isinstance(op, PolarsStringMethodOp) for op in ops)
    assert any(isinstance(op, ConversionOp) for op in ops)
    _assert_matches(result, _expected(method, args, kwargs, path))
