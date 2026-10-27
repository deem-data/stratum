"""``DataFrame.fillna`` with a dict: exact under greedy, or refused."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
import skrub

import stratum
from stratum._api import evaluate
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import optimize
from stratum.optimizer.logical._column_methods import ColumnMethodOp

FRAME = pd.DataFrame({
    "f": [1.5, np.nan, None],
    "i": [1, 2, 3],
    "s": pd.Series(["a", None, "c"], dtype="str"),
    "b": [True, None, False],
})


@pytest.mark.parametrize("value", [
    {"f": 0},
    {"f": 2.5, "s": "missing"},
    {"s": "missing", "unknown": 1},
    {"i": 0},
    {},
])
@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize("then", ["nothing", "assign", "selection"])
def test_frame_fillna_matches_pandas(value, keyword, then):
    call = {"value": value} if keyword else {}
    args = () if keyword else (value,)
    filled_pd = FRAME.fillna(*args, **call)
    expected = {"nothing": filled_pd,
                "assign": filled_pd.assign(g=filled_pd["f"] * 2),
                "selection": filled_pd[filled_pd["f"] > 1]}[then]
    with skrub.config_context(eager_data_ops=False):
        filled = skrub.var("frame", FRAME).fillna(*args, **call)
        plan = {"nothing": filled, "assign": filled.assign(g=filled["f"] * 2),
                "selection": filled[filled["f"] > 1]}[then]
    with stratum.config(scheduler=True, implementation_selector="greedy"):
        ops, *_ = optimize(plan, env=get_data(plan))
        assert any(isinstance(op, ColumnMethodOp) and op.method == "fillna" for op in ops)
        result = evaluate(plan)
    assert isinstance(result, pl.DataFrame)
    pd.testing.assert_frame_equal(result.to_pandas().reset_index(drop=True),
                                  expected.reset_index(drop=True), check_dtype=False)


@pytest.mark.parametrize("value", [{"i": "x"}, {"s": 0}, {"f": True}])
def test_a_fill_value_of_another_kind_runs_on_pandas(value):
    # The schema knows each column's dtype at plan time, so polars refuses the
    # fill and pandas runs it, upcasting the column to object.
    with skrub.config_context(eager_data_ops=False):
        plan = skrub.var("frame", FRAME).fillna(value)
    with stratum.config(scheduler=True, implementation_selector="greedy"):
        ops, *_ = optimize(plan, env=get_data(plan))
        [fillna] = [op for op in ops if isinstance(op, ColumnMethodOp)]
        assert "mix types" in dict(fillna._refusals)["polars"]
        result = evaluate(plan)
    pd.testing.assert_frame_equal(result, FRAME.fillna(value))


def test_an_unknown_dtype_is_checked_when_the_fill_runs():
    # `b` holds bools and None, an object column whose dtype the schema leaves
    # unknown, so only the run can tell the fill value does not fit.
    with skrub.config_context(eager_data_ops=False):
        plan = skrub.var("frame", FRAME).fillna({"b": 1})
    with stratum.config(scheduler=True, implementation_selector="greedy"):
        with pytest.raises((NotImplementedError, RuntimeError),
                           match="no exact polars translation"):
            evaluate(plan)


@pytest.mark.parametrize("args,kwargs", [
    ((0,), {}),                      # scalar on a frame fills every column
    (({"f": None},), {}),            # missing fill value
    ((), {"value": {"f": 0}, "limit": 1}),
])
def test_other_frame_fillna_calls_are_not_folded(args, kwargs):
    with skrub.config_context(eager_data_ops=False):
        plan = skrub.var("frame", FRAME).fillna(*args, **kwargs)
    with stratum.config(scheduler=True):
        ops, *_ = optimize(plan, env=get_data(plan))
    assert not any(isinstance(op, ColumnMethodOp) for op in ops)


def test_series_fillna_with_a_dict_is_not_folded():
    # On a Series, a dict fills by index label, not by column.
    with skrub.config_context(eager_data_ops=False):
        plan = skrub.var("frame", FRAME)["f"].fillna({1: 9.0})
    with stratum.config(scheduler=True):
        ops, *_ = optimize(plan, env=get_data(plan))
        result = evaluate(plan)
    assert not any(isinstance(op, ColumnMethodOp) for op in ops)
    pd.testing.assert_series_equal(result, FRAME["f"].fillna({1: 9.0}))


@pytest.mark.parametrize("value", [{"f": 0}, {"s": "missing", "unknown": 1}])
def test_default_selector_is_unchanged(value):
    with skrub.config_context(eager_data_ops=False):
        plan = skrub.var("frame", FRAME).fillna(value)
    with stratum.config(scheduler=True):
        result = evaluate(plan)
    pd.testing.assert_frame_equal(result, FRAME.fillna(value))
