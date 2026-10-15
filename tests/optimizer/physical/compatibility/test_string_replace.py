"""pandas ``str.replace`` under the greedy selector: exact or refused, never different."""
import re

import pandas as pd
import polars as pl
import pytest
import skrub

import stratum
from stratum._api import evaluate
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import optimize
from stratum.optimizer.logical._map_ops import polars_string_method
from stratum.optimizer.physical._conversion import ConversionOp
from stratum.optimizer.physical._map_execs import PolarsAssignMapOp, PolarsStringMethodOp
from stratum.optimizer.physical._selection_execs import PolarsSelectionOp

FRAME = pd.DataFrame({"s": ["1000010001.0", "aaa", "a.a", "AaA", "", None]})

TRANSLATED = [
    ((r"\.0$", ""), {"regex": True}),
    (("a", "x"), {}),
    ((".", "x"), {}),
    ((".", "x"), {"regex": False}),
    (("a", "x"), {"regex": True}),
    (("a", "x", 2), {}),
    (("a", "x"), {"n": 1, "regex": True}),
    (("a", "x"), {"n": 0}),
    (("a", "x"), {"n": 0, "regex": True}),
    (("a", "x"), {"n": -2, "regex": True}),
    ((), {"pat": "a", "repl": "x", "regex": True}),
    (("a", "$1"), {"regex": False}),
    (("a", "x"), {"case": True}),
]

REFUSED = [
    (("a", "$1"), {"regex": True}),
    (("(a)", r"\1\1"), {"regex": True}),
    (("", "x"), {"n": 2, "regex": True}),
    (("a", "x"), {"case": False}),
    (("a", "x"), {"regex": True, "flags": re.IGNORECASE}),
    ((r"a(?=a)", "x"), {"regex": True}),
    ((re.compile("a", re.IGNORECASE), "x"), {"regex": True}),
    (("a", lambda match: match.group(0).upper()), {"regex": True}),
]


def _plan(path, args, kwargs):
    with skrub.config_context(eager_data_ops=False):
        source = skrub.var("frame", FRAME)
        replaced = source["s"].str.replace(*args, **kwargs)
        if path == "standalone":
            return replaced, PolarsStringMethodOp
        if path == "assign":
            return source.assign(replaced=replaced), PolarsAssignMapOp
        return source[replaced == "xxx"], PolarsSelectionOp


@pytest.mark.parametrize("path", ["standalone", "assign", "selection"])
@pytest.mark.parametrize("args,kwargs", TRANSLATED)
def test_translated_replace_matches_pandas(path, args, kwargs):
    plan, expected_op = _plan(path, args, kwargs)
    replaced = FRAME["s"].str.replace(*args, **kwargs)
    expected = {"standalone": replaced,
                "assign": FRAME.assign(replaced=replaced),
                "selection": FRAME[replaced == "xxx"]}[path].reset_index(drop=True)

    with stratum.config(scheduler=True, implementation_selector="greedy"):
        ops, *_ = optimize(plan, env=get_data(plan))
        assert any(isinstance(op, expected_op) for op in ops)
        result = evaluate(plan)

    assert isinstance(result, (pl.Series, pl.DataFrame))
    result = result.to_pandas().reset_index(drop=True)
    if path == "standalone":
        pd.testing.assert_series_equal(result, expected, check_dtype=False)
    else:
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize("path", ["standalone", "assign", "selection"])
@pytest.mark.parametrize("args,kwargs", REFUSED)
def test_untranslatable_replace_falls_back_to_pandas(path, args, kwargs):
    # Polars refuses the call, so it runs as the recorded pandas call behind an
    # explicit conversion, and the result is pandas'.
    plan, _ = _plan(path, args, kwargs)
    replaced = FRAME["s"].str.replace(*args, **kwargs)
    expected = {"standalone": replaced,
                "assign": FRAME.assign(replaced=replaced),
                "selection": FRAME[replaced == "xxx"]}[path].reset_index(drop=True)
    with stratum.config(scheduler=True, implementation_selector="greedy"):
        ops, *_ = optimize(plan, env=get_data(plan))
        assert any(isinstance(op, ConversionOp) for op in ops)
        result = evaluate(plan)
    result = result.to_pandas() if hasattr(result, "to_pandas") else result
    result = result.reset_index(drop=True)
    if path == "standalone":
        pd.testing.assert_series_equal(result, expected, check_dtype=False)
    else:
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize("args,kwargs", TRANSLATED + REFUSED)
def test_default_selector_is_unchanged(args, kwargs):
    plan, _ = _plan("standalone", args, kwargs)
    with stratum.config(scheduler=True):
        result = evaluate(plan)
    pd.testing.assert_series_equal(result, FRAME["s"].str.replace(*args, **kwargs))


@pytest.mark.parametrize("expression", [False, True])
def test_common_regex_replacement_stays_native(monkeypatch, expression):
    def reject_pandas_conversion(*args, **kwargs):
        pytest.fail("the common regex replacement should use a native kernel")

    monkeypatch.setattr(pl.Series, "to_pandas", reject_pandas_conversion)
    series = pl.Series("bbl", ["1000010001.0", "2000020002", None])
    obj = pl.col("bbl") if expression else series
    result = polars_string_method(obj, "replace", (r"\.0$", ""), {"regex": True})
    if expression:
        result = series.to_frame().select(result).to_series()
    assert result.to_list() == ["1000010001", "2000020002", None]
