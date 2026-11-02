"""Index use: a polars frame converts to pandas wherever its index is not used."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
import skrub
from sklearn.preprocessing import KBinsDiscretizer

import stratum as st
from stratum._api import evaluate
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import optimize
from stratum.optimizer.logical._aggregation_ops import AggregateOp
from stratum.optimizer.logical._candidate_ops import ScoreCandidatesOp
from stratum.optimizer.logical._ops import OperandRef, ValueOp
from stratum.optimizer.logical._scoring import Metric
from stratum.optimizer.logical._selection_ops import SelectionOp
from stratum.optimizer.physical._conversion import ConversionOp
from stratum.optimizer.physical._index_use import IndexUse
from stratum.optimizer.physical._representation import FrameLib, frame_lib

_RNG = np.random.default_rng(0)
FRAME = pd.DataFrame({"a": _RNG.normal(size=60), "b": _RNG.normal(size=60),
                      "k": _RNG.integers(0, 3, 60)})


@pytest.fixture
def data():
    with skrub.config_context(eager_data_ops=False):
        yield skrub.var("frame", FRAME)


def _plan_ops(plan, selector="default"):
    with st.config(scheduler=True, implementation_selector=selector):
        return optimize(plan, env=get_data(plan))[0]


def _filter_index_used(plan) -> bool:
    """Whether the plan uses the index of its row filter's output."""
    ops = _plan_ops(plan)
    kept = next(op for op in ops if isinstance(op, SelectionOp))
    return IndexUse(ops).output_used(kept)


# --- Which edges use the index ------------------------------------------------------

def test_the_result_uses_the_index(data):
    assert _filter_index_used(data[data["a"] > 0])


def test_a_grouped_aggregation_drops_the_index(data):
    with skrub.config_context(eager_data_ops=False):
        plan = data[data["a"] > 0].groupby("k").agg({"b": "sum"})
    assert not _filter_index_used(plan)


def test_renumbering_drops_the_index(data):
    with skrub.config_context(eager_data_ops=False):
        plan = data[data["a"] > 0].reset_index(drop=True)
    assert not _filter_index_used(plan)


def test_a_map_passes_the_index_on(data):
    with skrub.config_context(eager_data_ops=False):
        kept = data[data["a"] > 0]
        used = kept.assign(c=kept["a"] * 2)
        dropped = used.groupby("k").agg({"c": "sum"})
    assert _filter_index_used(used)
    assert not _filter_index_used(dropped)


def test_recorded_code_uses_the_index(data):
    with skrub.config_context(eager_data_ops=False):
        plan = (data[data["a"] > 0].skb.apply_func(lambda df: df)
                .groupby("k").agg({"b": "sum"}))
    assert _filter_index_used(plan)


def test_label_alignment_uses_the_index(data):
    # pandas pairs the filtered column with the frame's rows by label.
    with skrub.config_context(eager_data_ops=False):
        plan = data.assign(c=data[data["a"] > 0]["b"]).groupby("k").agg({"c": "sum"})
    assert _filter_index_used(plan)


def test_alignment_with_a_value_of_unknown_kind_uses_the_index(data):
    # Recorded code may return a series, which pandas aligns with by label.
    with skrub.config_context(eager_data_ops=False):
        other = data.skb.apply_func(lambda df: df["b"])
        plan = (data[data["a"] > 0]["a"] + other).reset_index(drop=True)
    assert _filter_index_used(plan)


def test_an_index_join_matches_one_side_and_passes_the_other(data):
    with skrub.config_context(eager_data_ops=False):
        totals = data.groupby("k").agg({"b": "sum"}).rename(columns={"b": "total"})
        joined = data[data["a"] > 0].join(totals, on="k")
        dropped = joined.groupby("k").agg({"total": "sum"})
    assert _filter_index_used(joined)
    assert not _filter_index_used(dropped)
    ops = _plan_ops(dropped)
    grouped = next(op for op in ops if isinstance(op, AggregateOp))
    assert IndexUse(ops).output_used(grouped)


def test_a_rename_pairs_no_rows(data):
    # The new name comes from the graph: an input, but no rows to align.
    with skrub.config_context(eager_data_ops=False):
        name = data.skb.apply_func(lambda df: "c")
        plan = data[data["a"] > 0].rename(columns={"b": name}).reset_index(drop=True)
    assert not _filter_index_used(plan)


@pytest.mark.parametrize("source, aligns", [("sklearn", False), ("stratum", False),
                                            ("callable", True)])
def test_scoring_pairs_by_position_unless_the_metric_is_a_callable(source, aligns):
    predictions, y = ValueOp(pd.Series([1.0])), ValueOp(pd.Series([1.0]))
    metric = Metric(name="m", fn=lambda t, p: 0.0, source=source)
    score = ScoreCandidatesOp(["c"], metric, inputs=[predictions, y], y_ref=OperandRef(1))
    for producer in (predictions, y):
        producer.add_output(score)
    index_use = IndexUse([predictions, y, score])
    assert index_use.uses_input(score, 0) is aligns
    assert index_use.uses_input(score, 1) is aligns


# --- Selection converts where the index is not used ---------------------------------

def _filter_then_transform(data):
    with skrub.config_context(eager_data_ops=False):
        kept = data[data["a"] > 0][["a", "b"]]
        # KBinsDiscretizer reads pandas only: it is not known to answer the same on polars.
        return kept.skb.apply(KBinsDiscretizer(n_bins=3, encode="ordinal"))


def _pandas_result(plan):
    with st.config(scheduler=True):
        return evaluate(plan)


def test_a_filter_whose_index_is_not_used_runs_on_polars(data):
    with skrub.config_context(eager_data_ops=False):
        plan = _filter_then_transform(data).groupby("a").agg({"b": "sum"})
    ops = _plan_ops(plan, "greedy")
    kept = next(op for op in ops if isinstance(op, SelectionOp))
    assert frame_lib(kept) is FrameLib.POLARS
    assert any(isinstance(op, ConversionOp) and op.target is FrameLib.PANDAS for op in ops)
    with st.config(scheduler=True, implementation_selector="greedy"):
        result = evaluate(plan)
    expected = _pandas_result(plan)
    result = result.to_pandas() if isinstance(result, pl.DataFrame) else result
    pd.testing.assert_frame_equal(
        result.sort_values("a", ignore_index=True), expected.reset_index(), check_dtype=False)


def test_a_filter_whose_index_reaches_the_result_runs_on_pandas(data):
    plan = _filter_then_transform(data)
    ops = _plan_ops(plan, "greedy")
    kept = next(op for op in ops if isinstance(op, SelectionOp))
    assert frame_lib(kept) is FrameLib.PANDAS
    with st.config(scheduler=True, implementation_selector="greedy"):
        result = evaluate(plan)
    pd.testing.assert_frame_equal(result, _pandas_result(plan))
