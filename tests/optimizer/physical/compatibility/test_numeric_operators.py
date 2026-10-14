"""Arithmetic lowered to NumericOp keeps the recorded operator's meaning on polars."""
import datetime

import pandas as pd
import pytest
import skrub

import stratum as st
from stratum._api import evaluate
from stratum.optimizer.physical._conversion import to_pandas

FRAME = pd.DataFrame({
    "boro": pd.Series(["1", "2"], dtype="str"),
    "block": pd.Series(["7", "88"], dtype="str"),
    "x": [1.5, 2.0],
    "d": pd.to_datetime(["2020-02-29", "2021-03-01"]),
})


@pytest.mark.parametrize("build", [
    lambda s: s.assign(bbl=s["boro"] + s["block"].str.zfill(5)),   # string concatenation
    lambda s: s.assign(y=s["x"] * 2 + 1),                           # numbers
    lambda s: s.assign(e=s["d"] + pd.DateOffset(years=1)),          # pandas-only constant
    lambda s: s.assign(e=s["d"] + datetime.timedelta(days=1)),      # duration
    # standalone, so they are lowered to NumericOp rather than folded
    lambda s: (s["boro"] + s["block"].str.zfill(5)).to_frame(),
    lambda s: (s["x"] * 2 + 1).to_frame(),
    lambda s: (s["d"] + pd.DateOffset(years=1)).to_frame(),
])
def test_numeric_operators_match_pandas_under_greedy(build):
    with skrub.config_context(eager_data_ops=False):
        plan = build(skrub.var("frame", FRAME))
    with st.config(scheduler=True):
        expected = evaluate(plan)
    with st.config(scheduler=True, implementation_selector="greedy"):
        result = evaluate(plan)
    result = to_pandas(result) if not isinstance(result, pd.DataFrame) else result
    # Values only: pandas names `a + b` None when the names differ, polars keeps
    # the left name, which shows up as the column label after to_frame().
    pd.testing.assert_frame_equal(result.reset_index(drop=True).set_axis(range(result.shape[1]), axis=1),
                                  expected.reset_index(drop=True).set_axis(range(expected.shape[1]), axis=1),
                                  check_dtype=False)
