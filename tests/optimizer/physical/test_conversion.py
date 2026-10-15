"""Explicit conversions between frame libraries and correctness-first selection (#227)."""
import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest
import skrub

import stratum as st
from stratum._api import evaluate
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import optimize
from stratum.optimizer.physical._conversion import ConversionOp, to_pandas, to_polars
from stratum.optimizer.physical._representation import FrameLib, frame_lib

FRAME = pd.DataFrame({
    "k": [3, 1, 2, 1],
    "x": [1.5, np.nan, -3.0, 4.0],
    "s": pd.Series(["a", None, "c", "d"], dtype="str"),
    "d": [datetime.date(2020, 1, d) for d in (1, 2, 3, 4)],
})


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "f.parquet"
    FRAME.to_parquet(path)
    with skrub.config_context(eager_data_ops=False):
        yield st.as_data_op(str(path)).skb.apply_func(pd.read_parquet)


def _greedy(plan):
    with st.config(scheduler=True, implementation_selector="greedy"):
        ops = optimize(plan, env=get_data(plan))[0]
        return ops, evaluate(plan)


def _pandas(plan):
    with st.config(scheduler=True):
        return evaluate(plan)


def _as_pandas(value):
    return to_pandas(value) if isinstance(value, (pl.DataFrame, pl.Series)) else value


# --- the conversion itself ------------------------------------------------------

def test_round_trip_keeps_pandas_reader_dtypes():
    back = to_pandas(to_polars(pd.read_parquet(_parquet(FRAME))))
    expected = pd.read_parquet(_parquet(FRAME))
    pd.testing.assert_frame_equal(back, expected, check_dtype=False)
    assert back["s"].dtype == expected["s"].dtype           # pandas' str, not object
    assert isinstance(back["d"].iloc[0], datetime.date)     # not datetime64
    assert back["x"].isna().tolist() == [False, True, False, False]


def _parquet(frame):
    import io
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    buffer.seek(0)
    return buffer


def test_a_series_converts_with_its_name():
    series = to_pandas(pl.Series("s", ["a", None]))
    assert series.name == "s" and series.tolist()[0] == "a" and series.isna().tolist()[1]


# --- conversions in plans -------------------------------------------------------

def test_a_pandas_call_on_a_default_index_reads_a_conversion(source):
    plan = source.assign(c=source["x"].where(source["x"] > 0))
    ops, result = _greedy(plan)
    where = next(op for op in ops if type(op).__name__ == "MethodCallExec")
    assert all(isinstance(i, ConversionOp) and i.target is FrameLib.PANDAS
               for i in where.inputs)
    # and back, explicitly, into the polars assign
    assert [o.target for o in where.outputs] == [FrameLib.POLARS]
    assert isinstance(result, pl.DataFrame)
    pd.testing.assert_frame_equal(_as_pandas(result), _pandas(plan), check_dtype=False)


def test_one_conversion_serves_every_consumer(source):
    x = source["x"]
    plan = source.assign(a=x.where(x > 0), b=x.mask(x > 0))
    ops, result = _greedy(plan)
    from_x = [op for op in ops if isinstance(op, ConversionOp)
              and type(op.inputs[0]).__name__ == "PolarsColumnProjectionOp"]
    assert len(from_x) == 1 and len(from_x[0].outputs) == 2
    pd.testing.assert_frame_equal(_as_pandas(result), _pandas(plan), check_dtype=False)


def test_the_default_selector_inserts_no_conversion(source):
    plan = source.assign(c=source["x"].where(source["x"] > 0))
    with st.config(scheduler=True):
        ops = optimize(plan, env=get_data(plan))[0]
    assert not any(isinstance(op, ConversionOp) for op in ops)
    assert {frame_lib(op) for op in ops} <= {FrameLib.PANDAS, None}


# --- correctness-first fallback -------------------------------------------------

def test_index_labels_a_pandas_consumer_needs_keep_the_filter_on_pandas(source):
    # `.index` of a filtered frame: a conversion from polars would renumber the
    # rows, so the filter itself has to run on pandas.
    filtered = source[source["x"] > 0]
    plan = filtered.assign(pos=filtered.index)
    ops, result = _greedy(plan)
    selection = next(op for op in ops if type(op).__name__.endswith("SelectionOp"))
    assert frame_lib(selection) is FrameLib.PANDAS
    assert "index labels" in dict(selection._refusals)["polars"]
    pd.testing.assert_frame_equal(_as_pandas(result), _pandas(plan), check_dtype=False)
    assert _as_pandas(result)["pos"].tolist() == [0, 3]


def test_a_filtered_series_is_aligned_by_label_not_by_position(source):
    # pandas aligns the filtered column on the frame's labels (NaN elsewhere);
    # polars would pair rows by position. Before #227 greedy computed the latter.
    positive = source["x"][source["x"] > 0]
    plan = source.assign(p=positive)
    ops, result = _greedy(plan)
    expected = _pandas(plan)
    assert expected["p"].isna().tolist() == [False, True, True, False]
    pd.testing.assert_frame_equal(_as_pandas(result), expected, check_dtype=False)


def test_columns_of_the_same_frame_stay_on_polars(source):
    # Same rows, same order: positional pairing is label alignment here.
    filtered = source[source["k"] > 1]
    plan = filtered.assign(y=filtered["x"] * 2, z=filtered["x"].where(filtered["x"] > 0))
    ops, result = _greedy(plan)
    assign = next(op for op in ops if type(op).__name__.endswith("AssignMapOp"))
    assert frame_lib(assign) is FrameLib.POLARS
    pd.testing.assert_frame_equal(_as_pandas(result).reset_index(drop=True),
                                  _pandas(plan).reset_index(drop=True), check_dtype=False)


def test_a_row_drop_starts_a_new_index_origin():
    # `drop` of rows by label leaves gaps in a RangeIndex, so the dropped frame's
    # labels no longer match its source's positions.
    from stratum.optimizer.physical._conversion import has_default_index
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    with skrub.config_context(eager_data_ops=False):
        plan = skrub.var("frame", frame).drop([1])
    with st.config(scheduler=True, implementation_selector="greedy"):
        ops = optimize(plan, env=get_data(plan))[0]
    drop = next(op for op in ops if type(op).__name__.endswith(("DropOp", "SelectionOp")))
    assert not has_default_index(drop)


def test_a_row_dropped_column_is_aligned_by_label():
    # pandas aligns the shorter column on the frame's labels (NaN for the dropped
    # row); pairing by position fails on the length mismatch.
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    with skrub.config_context(eager_data_ops=False):
        var = skrub.var("frame", frame)
        plan = var.assign(p=var.drop([1])["x"])
    with st.config(scheduler=True, implementation_selector="greedy"):
        result = evaluate(plan)
    expected = frame.assign(p=frame.drop([1])["x"])
    pd.testing.assert_frame_equal(_as_pandas(result), expected, check_dtype=False)
