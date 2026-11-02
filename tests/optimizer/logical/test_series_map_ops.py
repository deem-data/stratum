"""SeriesMapOp: a standalone chain of row-local series ops, folded into one op."""
import operator

import numpy as np
import pandas as pd
import polars as pl
import pytest
import skrub

from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.logical._base import IndexEffect
from stratum.optimizer.logical._column_expr import (
    UFUNCS, BinOpExpr, Col, Const, FillnaExpr, UfuncExpr)
from stratum.optimizer.logical._map_ops import UNKNOWN_NAME, SeriesMapOp
from stratum.optimizer.logical._numeric_ops import NumericOp
from stratum.optimizer.physical import FlagBasedSelector
from stratum.optimizer.physical._impl_selection import get_implementation_selector
from stratum.optimizer.physical._map_execs import PolarsSeriesMapOp
from stratum.runtime._scheduler import SequentialScheduler

_FRAME = pd.DataFrame({"n": [1, 2, 3, 4], "f": [0.5, None, 2.5, 4.0],
                       "s": ["a", "b", None, "d"]})
_OTHER = pd.DataFrame({"n": [10, 20, 30, 40]}, index=[2, 3, 4, 5])

_SELECTORS = {
    "pandas": lambda: FlagBasedSelector(backend="pandas"),
    "polars": lambda: FlagBasedSelector(backend="polars"),
    "greedy": lambda: get_implementation_selector("greedy"),
}


def _plan(build, selector="pandas"):
    with skrub.config_context(eager_data_ops=False):
        dag = build(skrub.var("x", _FRAME), skrub.var("y", _OTHER))
    plan, split = optimize(dag, OptConfig(selector=_SELECTORS[selector]()),
                           env={"x": _FRAME, "y": _OTHER})
    return plan, split


def _maps(plan):
    return [op for op in plan if isinstance(op, SeriesMapOp)]


def _evaluate(build, selector):
    plan, split = _plan(build, selector)
    return plan, SequentialScheduler(plan, split).evaluate()


# --- Folding ----------------------------------------------------------------------

def test_a_chain_folds_into_one_series_map():
    plan, _ = _plan(lambda x, y: (x["f"] * 2).fillna(0))
    [series_map] = _maps(plan)
    assert series_map.expr == FillnaExpr(BinOpExpr(operator.mul, Col("f"), Const(2)), 0)
    assert len(series_map.inputs) == 1  # the frame alone
    assert not any(isinstance(op, NumericOp) for op in plan)


@pytest.mark.parametrize("name, build", {
    "a column read": lambda x, y: x["n"],
    "a chain an assign takes in": lambda x, y: x.assign(m=x["n"] * 2),
    "a chain a mask takes in": lambda x, y: x[x["n"] * 2 > 3],
    "a chain over no column": lambda x, y: x["n"].skb.apply_func(lambda s: s) * 2,
}.items())
def test_what_is_not_a_series_map(name, build):
    plan, _ = _plan(build)
    assert not _maps(plan)


def test_an_op_two_chains_share_is_a_map_of_its_own():
    def build(x, y):
        missing = x["f"].isna()
        return skrub.deferred(lambda p, q: p.sum() + q.sum())(missing.astype(int) * 2, ~missing)

    plan, _ = _plan(build, "polars")
    maps = _maps(plan)
    [shared] = [m for m in maps if not any(isinstance(i, SeriesMapOp) for i in m.inputs)]
    consumers = [m for m in maps if m is not shared]
    assert len(consumers) == 2 and all(m.inputs[1] is shared for m in consumers)
    # It reads the source's rows, so its consumers keep them and run on polars.
    assert all(m.index_effect is IndexEffect.KEEP for m in consumers)
    assert all(isinstance(m, PolarsSeriesMapOp) for m in maps)


def test_a_column_read_elsewhere_too_stays_a_column():
    def build(x, y):
        column = x["n"]
        return skrub.deferred(lambda a, b: a + b)(column, column * 2)

    plan, _ = _plan(build)
    [series_map] = _maps(plan)
    # Another op reads the column too, so the folder keeps it an operand; the
    # series map reads it again by name.
    assert series_map.expr == BinOpExpr(operator.mul, Col("n"), Const(2))
    assert len(series_map.inputs) == 1


def test_columns_of_another_frame_are_aligned_by_label():
    build = lambda x, y: x["n"] + y["n"]
    plan, result = _evaluate(build, "greedy")
    [series_map] = _maps(plan)
    assert series_map.index_effect is IndexEffect.COMMON
    assert not isinstance(series_map, PolarsSeriesMapOp)
    pd.testing.assert_series_equal(_FRAME["n"] + _OTHER["n"], result)


# --- Names --------------------------------------------------------------------------

#: Spellings, and whether polars can run them: it names every series by a string.
_SPELLINGS = {
    "scalar arithmetic": (lambda x, y: x["n"] * 2 + 1, True),
    "the same column twice": (lambda x, y: x["n"] - x["n"] ** 2, True),
    "a method chain": (lambda x, y: (x["f"] * 2).fillna(0), True),
    "a string method": (lambda x, y: x["s"].str.upper() + "_", True),
    "negation": (lambda x, y: -x["n"], True),
    "where": (lambda x, y: x["f"].where(x["n"] > 1, 0.0), True),
    "minus a reduction": (lambda x, y: x["n"] - x["n"].mean(), True),
    "two columns": (lambda x, y: x["n"] + x["f"], False),
    "two masks": (lambda x, y: (x["n"] > 1) & x["f"].isna(), False),
}


@pytest.mark.parametrize("name", _SPELLINGS)
def test_the_name_is_pandas(name):
    build, _ = _SPELLINGS[name]
    plan, _ = _plan(build)
    expected = build(_FRAME, _OTHER).name
    assert _maps(plan)[-1].result_name() == expected


def test_a_series_of_other_rows_names_the_result_by_its_values():
    plan, _ = _plan(lambda x, y: x["n"] + y["n"])
    assert _maps(plan)[-1].result_name() is UNKNOWN_NAME


@pytest.mark.parametrize("selector", _SELECTORS)
@pytest.mark.parametrize("name", _SPELLINGS)
def test_a_series_map_matches_pandas(name, selector):
    build, on_polars = _SPELLINGS[name]
    if selector == "polars" and not on_polars:
        pytest.skip("pandas names the result None, which polars cannot")
    expected = build(_FRAME, _OTHER)
    plan, result = _evaluate(build, selector)
    assert _maps(plan)
    if selector != "pandas":
        assert isinstance(result, pl.Series) is on_polars
    if isinstance(result, pl.Series):
        result = result.to_pandas()
    pd.testing.assert_series_equal(expected, result, check_dtype=False)


def test_an_unnamed_result_runs_on_polars_where_no_one_reads_the_name():
    # pandas names `n + f` None; a mask only reads which rows it keeps.
    plan, _ = _plan(lambda x, y: (lambda m: x[m].assign(k=m))(x["n"] + x["f"] > 2), "greedy")
    [series_map] = _maps(plan)
    assert series_map.result_name() is None
    assert isinstance(series_map, PolarsSeriesMapOp)


def test_a_mask_over_an_unnamed_series_keeps_its_name():
    # Planned only: a polars mask over a series fails at run time either way.
    plan, _ = _plan(lambda x, y: (x["n"] + x["f"])[x["n"] > 1], "greedy")
    [series_map] = [m for m in _maps(plan) if m.result_name() is None]
    assert not isinstance(series_map, PolarsSeriesMapOp)


# --- numpy functions ----------------------------------------------------------------

#: Values where polars' own log/exp round differently from numpy's in the last bit.
_FLOATS = pd.DataFrame({"v": np.random.default_rng(0).lognormal(0, 1, 2000)})


def test_a_numpy_function_folds_into_the_chain():
    plan, _ = _plan(lambda x, y: x["f"].skb.apply_func(np.log) + 1)
    [series_map] = _maps(plan)
    assert series_map.expr == BinOpExpr(operator.add, UfuncExpr(Col("f"), "log"), Const(1))
    assert not any(isinstance(op, NumericOp) for op in plan)


@pytest.mark.parametrize("name, build", {
    "another numpy function": lambda x, y: x["n"].skb.apply_func(np.cumsum),
    "one given an option": lambda x, y: x["f"].skb.apply_func(np.log, where=True),
    "one over a frame": lambda x, y: x[["n", "f"]].skb.apply_func(np.log),
}.items())
def test_what_stays_a_numeric_op(name, build):
    plan, _ = _plan(build)
    assert not _maps(plan)
    assert any(isinstance(op, NumericOp) for op in plan)


@pytest.mark.parametrize("selector", _SELECTORS)
@pytest.mark.parametrize("path", ["standalone", "assign"])
@pytest.mark.parametrize("func", UFUNCS)
def test_a_numpy_function_is_numpys_exactly(func, path, selector):
    ufunc = UFUNCS[func]
    with skrub.config_context(eager_data_ops=False):
        x = skrub.var("x", _FLOATS)
        applied = x["v"].skb.apply_func(ufunc)
        dag = applied if path == "standalone" else x.assign(w=applied)
    plan, split = optimize(dag, OptConfig(selector=_SELECTORS[selector]()), env={"x": _FLOATS})
    result = SequentialScheduler(plan, split).evaluate()
    if selector != "pandas":
        assert isinstance(result, (pl.Series, pl.DataFrame))
    result = result.to_pandas() if hasattr(result, "to_pandas") else result
    expected = ufunc(_FLOATS["v"])
    if path == "assign":
        result, expected = result["w"], expected.rename("w")
    pd.testing.assert_series_equal(expected, result, check_exact=True)


# --- Schema -----------------------------------------------------------------------

def test_the_schema_is_one_column_under_the_result_name():
    plan, _ = _plan(lambda x, y: x["n"] * 2)
    [series_map] = _maps(plan)
    series_map.propagate_output_schema()
    assert series_map.output_schema == pl.Schema({"n": pl.Unknown})
    plan, _ = _plan(lambda x, y: x["n"] + x["f"])
    [series_map] = _maps(plan)
    series_map.propagate_output_schema()
    assert series_map.output_schema is None
