"""``drop`` keeps pandas' meaning under greedy."""
import pandas as pd
import polars as pl
import pytest
import skrub

import stratum as st
from stratum._api import evaluate
from stratum.optimizer.physical._conversion import to_pandas

FRAME = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [0.5, 1.5, 2.5]})


@pytest.mark.parametrize("call", [
    lambda df: df.drop(columns=["b"]),
    lambda df: df.drop(columns="b"),
    lambda df: df.drop(["b", "c"], axis=1),
    lambda df: df.drop(columns=["b", "missing"], errors="ignore"),
    lambda df: df.drop(index=[1]),                      # rows: stays on pandas
    lambda df: df.drop([0, 2]),                         # rows by default axis
])
def test_drop_matches_pandas_under_greedy(call):
    with skrub.config_context(eager_data_ops=False):
        plan = call(skrub.var("frame", FRAME))
    with st.config(scheduler=True, implementation_selector="greedy"):
        result = evaluate(plan)
    result = to_pandas(result) if isinstance(result, pl.DataFrame) else result
    expected = call(FRAME)
    pd.testing.assert_frame_equal(result.reset_index(drop=True),
                                  expected.reset_index(drop=True), check_dtype=False)


def test_a_missing_column_still_raises_without_errors_ignore():
    with skrub.config_context(eager_data_ops=False):
        plan = skrub.var("frame", FRAME).drop(columns=["missing"])
    with st.config(scheduler=True, implementation_selector="greedy"):
        with pytest.raises(Exception, match="missing"):
            evaluate(plan)
