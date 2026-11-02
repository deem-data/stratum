"""A polars value holds its pandas index beside it, where an op uses it."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
import skrub
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import stratum as st
from stratum._api import evaluate
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import optimize
from stratum.optimizer.logical._selection_ops import SelectionOp
from stratum.optimizer.logical._sort_ops import SortOp
from stratum.optimizer.physical._aggregation_execs import PolarsAggregateOp
from stratum.optimizer.physical._conversion import ConversionOp
from stratum.optimizer.physical._join_execs import PandasJoinOp, PolarsJoinOp

_RNG = np.random.default_rng(0)
X = pd.DataFrame({"k": _RNG.integers(0, 4, 30), "j": _RNG.choice(["x", "y"], 30),
                  "a": _RNG.normal(size=30)}, index=_RNG.permutation(100)[:30])
EVENTS = pd.DataFrame({"k": _RNG.integers(0, 3, 50), "j": _RNG.choice(["x", "y", None], 50),
                       "v": _RNG.normal(size=50)})


@pytest.fixture
def frames():
    with skrub.config_context(eager_data_ops=False):
        events = skrub.var("events", EVENTS)
        feats = events.groupby(["k", "j"]).agg(s=("v", "sum"), m=("v", "min"))
        yield skrub.var("X", X), events, feats.assign(m=feats["m"] // 1)


def _greedy_ops(plan):
    with st.config(scheduler=True, implementation_selector="greedy"):
        return optimize(plan, env=get_data(plan))[0]


def _both(plan):
    """The plan's result under the default selector and under greedy."""
    results = []
    for selector in ("default", "greedy"):
        with st.config(scheduler=True, implementation_selector=selector):
            result = evaluate(plan)
        results.append(result.to_pandas() if isinstance(result, pl.DataFrame) else result)
    return results


def _of(ops, cls):
    return [op for op in ops if isinstance(op, cls)]


def test_a_polars_aggregation_hides_its_keys_from_column_reads(frames):
    # pandas holds the keys in the index, not among the columns.
    _, events, _ = frames
    with skrub.config_context(eager_data_ops=False):
        agg = events.groupby("k").agg(s=("v", "sum"))
        plans = [agg.columns, agg.reset_index(drop=True),
                 agg.skb.apply(StandardScaler()).reset_index(drop=True)]
    for plan in plans:
        ops = _greedy_ops(plan)
        assert _of(ops, PolarsAggregateOp)
        expected, result = _both(plan)
        if isinstance(expected, pd.Index):
            pd.testing.assert_index_equal(pd.Index(result), expected)
        else:
            pd.testing.assert_frame_equal(result, expected)


def test_an_index_join_runs_on_polars(frames):
    x, _, feats = frames
    for how in ("left", "inner"):
        with skrub.config_context(eager_data_ops=False):
                joined = x.join(feats, on=["k", "j"], how=how).fillna({"s": 0})
        # The left index is used, or not.
        for plan, converts in ((joined.reset_index(drop=True), False), (joined.index, True)):
            ops = _greedy_ops(plan)
            assert _of(ops, PolarsJoinOp) and not _of(ops, PandasJoinOp)
            assert bool(_of(ops, ConversionOp)) is converts
            expected, result = _both(plan)
            if converts:
                pd.testing.assert_index_equal(result, expected)
            else:
                pd.testing.assert_frame_equal(result, expected)


def test_an_index_join_repeats_the_left_index_with_its_rows():
    # A right index of repeated labels matches a left row more than once.
    events = EVENTS[["k", "v"]]
    for how in ("left", "inner"):
        with skrub.config_context(eager_data_ops=False):
            right = skrub.var("events", events).set_index("k")
            plan = skrub.var("X", X[["k", "a"]]).join(right, on="k", how=how).index
        assert _of(_greedy_ops(plan), PolarsJoinOp)
        expected, result = _both(plan)
        pd.testing.assert_index_equal(result, expected)


def test_converting_to_pandas_restores_the_index(frames):
    # The aggregation's keys, and a filter's rows of the frame's own index.
    x, _, feats = frames
    with skrub.config_context(eager_data_ops=False):
        plans = [feats.index, x[x["a"] > 0].index, x[x["a"] > 0].head(-3).index]
    for plan in plans:
        ops = _greedy_ops(plan)
        assert [op._indexed for op in _of(ops, ConversionOp)] == [True]
        expected, result = _both(plan)
        pd.testing.assert_index_equal(result, expected)


def test_an_unused_index_is_not_held(frames):
    _, events, _ = frames
    with skrub.config_context(eager_data_ops=False):
        agg = events.groupby("k").agg(s=("v", "sum"))
    for plan, held in ((agg.reset_index(drop=True), False), (agg.index, True)):
        assert [op._tracks_index for op in _of(_greedy_ops(plan), PolarsAggregateOp)] == [held]


def test_set_index_holds_the_keys_as_the_index():
    with skrub.config_context(eager_data_ops=False):
        plan = skrub.var("events", EVENTS).set_index(["k", "j"]).index
    expected, result = _both(plan)
    pd.testing.assert_index_equal(result, expected)


def test_a_filter_carries_them_to_a_read_of_the_group_keys():
    frame = pd.DataFrame({"t": list("aaabbbbcc"), "x": range(9)})
    with skrub.config_context(eager_data_ops=False):
        source = skrub.var("frame", frame)
        counts = source["t"].value_counts()
        plan = source[source["t"].isin(counts[counts >= 3].index)].reset_index(drop=True)
    ops = _greedy_ops(plan)
    assert not _of(ops, ConversionOp)
    expected, result = _both(plan)
    pd.testing.assert_frame_equal(result, expected)


def test_an_aggregation_by_a_computed_key_runs_on_pandas(frames):
    # Its keys would have no column name of their own on polars.
    _, events, _ = frames
    with skrub.config_context(eager_data_ops=False):
        plan = events.groupby(events["k"] * 2).agg(s=("v", "sum"))
    assert not _of(_greedy_ops(plan), PolarsAggregateOp)
    expected, result = _both(plan)
    pd.testing.assert_frame_equal(result, expected)


def test_index_columns_reach_the_plan_after_the_split(frames, monkeypatch):
    # The aggregation runs once, before the split; each fold's join reads it.
    _, events, feats = frames
    joined = []
    index_join = PolarsJoinOp._index_join
    monkeypatch.setattr(PolarsJoinOp, "_index_join",
                        lambda self, *a: joined.append(self) or index_join(self, *a))
    # The split reads rows by position.
    frame = X.reset_index(drop=True)
    y = pd.Series(_RNG.normal(size=len(frame)), name="y")
    with skrub.config_context(eager_data_ops=False):
        x = skrub.var("X", frame).skb.mark_as_X(cv=KFold(2, shuffle=True, random_state=0))
        yv = skrub.var("y", y).skb.mark_as_y()
        features = x.join(feats, on=["k", "j"]).fillna({"s": 0, "m": 0})
        plan = features.drop(columns=["j"]).skb.apply(Ridge(), y=yv)
    scores = {}
    for selector in ("default", "greedy"):
        with st.config(scheduler=True, implementation_selector=selector):
            search = plan.skb.make_grid_search(fitted=True, refit=False, n_jobs=1, scoring="r2")
        scores[selector] = list(search.results_["scores"])
    assert joined
    assert scores["greedy"] == scores["default"]


ROWS = pd.DataFrame({"k": [2, 1, 2, None, 1, 3], "f": [0.5, np.nan, 0.5, 2.0, np.nan, 1.0],
                     "s": ["b", "a", "b", "c", None, "a"]}, index=[50, 40, 30, 20, 10, 0])
MOVES = {
    "sort": lambda x: x.sort_values(["k", "f"], ascending=[True, False]),
    "sort, nan first": lambda x: x.sort_values("f", na_position="first"),
    "series sort, ties": lambda x: x["s"].sort_values(),
    "float series sort": lambda x: x["f"].sort_values(ascending=False),
    "sort, head": lambda x: x.sort_values("f").head(3),
    "drop_duplicates": lambda x: x.drop_duplicates(),
    "drop_duplicates subset, last": lambda x: x.drop_duplicates(subset=["k"], keep="last"),
    "drop_duplicates, none": lambda x: x.drop_duplicates(subset=["s"], keep=False),
    "series drop_duplicates": lambda x: x["k"].drop_duplicates(),
    "dropna": lambda x: x.dropna(),
    "dropna thresh": lambda x: x.dropna(thresh=3),
    "series dropna": lambda x: x["f"].dropna(),
}


@pytest.mark.parametrize("move", MOVES)
def test_a_row_moving_op_takes_the_index_with_its_rows(move):
    with skrub.config_context(eager_data_ops=False):
        plan = MOVES[move](skrub.var("rows", ROWS)).index
    moving = [op for op in _greedy_ops(plan) if isinstance(op, (SortOp, SelectionOp))]
    assert moving and all(type(op).__name__.startswith("Polars") and op._tracks_index
                          for op in moving)
    expected, result = _both(plan)
    pd.testing.assert_index_equal(result, expected)
