"""``sort_values`` is extracted into a SortOp that orders rows like pandas."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
import skrub

import stratum
from stratum._api import evaluate
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import optimize
from stratum.optimizer.logical._sort_ops import SortOp

FRAME = pd.DataFrame({
    "f": [2.0, np.nan, 1.0, None, 2.0, 3.0],
    "s": pd.Series(["b", None, "a", "c", "a", "b"], dtype="str"),
    "i": [3, 1, 2, 1, 3, 0],
    "row": range(6),
})


def _plan(call):
    with skrub.config_context(eager_data_ops=False):
        return call(skrub.var("frame", FRAME))


SORTS = [
    ("f", {}),
    ("f", {"ascending": False}),
    ("f", {"ascending": False, "na_position": "first"}),
    ("s", {}),
    (["i", "f"], {"ascending": [True, False]}),
    (["s", "i"], {"na_position": "first", "kind": "mergesort"}),
    ("i", {"ignore_index": True}),
]


def _stable_pandas(by, kwargs):
    # Ties keep their input order: a stable sort is a valid pandas outcome for
    # every kind, and the one both impls produce.
    return FRAME.sort_values(by, **{**kwargs, "kind": "stable"})


@pytest.mark.parametrize("by,kwargs", SORTS)
def test_frame_sort_matches_pandas_under_greedy(by, kwargs):
    plan = _plan(lambda df: df.sort_values(by, **kwargs))
    with stratum.config(scheduler=True, implementation_selector="greedy"):
        ops, *_ = optimize(plan, env=get_data(plan))
        assert any(isinstance(op, SortOp) for op in ops)
        result = evaluate(plan)
    assert isinstance(result, pl.DataFrame)
    pd.testing.assert_frame_equal(result.to_pandas(),
                                  _stable_pandas(by, kwargs).reset_index(drop=True),
                                  check_dtype=False)


@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("na_position", ["last", "first"])
@pytest.mark.parametrize("column", ["f", "s", "i"])
def test_series_sort_matches_pandas_under_greedy(ascending, na_position, column):
    plan = _plan(lambda df: df[column].sort_values(ascending=ascending,
                                                   na_position=na_position))
    with stratum.config(scheduler=True, implementation_selector="greedy"):
        result = evaluate(plan)
    expected = FRAME[column].sort_values(ascending=ascending, na_position=na_position)
    assert result.to_pandas().isna().tolist() == expected.isna().tolist()
    pd.testing.assert_series_equal(result.to_pandas().dropna().reset_index(drop=True),
                                   expected.dropna().reset_index(drop=True),
                                   check_dtype=False)


@pytest.mark.parametrize("by,kwargs", SORTS)
def test_default_selector_keeps_pandas_index(by, kwargs):
    plan = _plan(lambda df: df.sort_values(by, **kwargs))
    with stratum.config(scheduler=True):
        ops, *_ = optimize(plan, env=get_data(plan))
        assert any(isinstance(op, SortOp) for op in ops)
        result = evaluate(plan)
    pd.testing.assert_frame_equal(result, _stable_pandas(by, kwargs))


@pytest.mark.parametrize("call", [
    lambda df: df.sort_values("f", key=lambda s: -s),
    lambda df: df.sort_values(0, axis=1),
    lambda df: df.sort_values("f", ascending=[True, False]),
    lambda df: df["f"].sort_values(0),
])
def test_other_sorts_stay_method_calls(call):
    plan = _plan(call)
    with stratum.config(scheduler=True):
        ops, *_ = optimize(plan, env=get_data(plan))
    assert not any(isinstance(op, SortOp) for op in ops)
