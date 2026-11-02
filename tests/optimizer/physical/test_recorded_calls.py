"""A recorded user function or ``assign`` value reads the pandas it was written for."""
import numpy as np
import pandas as pd
import polars as pl
import pytest
import skrub

import stratum as st
from stratum._api import evaluate
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import optimize
from stratum.optimizer.physical._conversion import ConversionOp
from stratum.optimizer.physical._map_execs import PandasApplyUDFOp, PandasAssignOp

FRAME = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [4, 5, 6, 7]})

CASES = {
    # pandas hands a frame's function one column at a time.
    "frame apply": (lambda x: x.apply(lambda column: column.sum()), PandasApplyUDFOp),
    # pandas passes a missing value to the function.
    "series apply": (lambda x: x["a"].apply(lambda v: 0.0 if v != v else v), PandasApplyUDFOp),
    "series apply ufunc": (lambda x: x["a"].apply(np.sin), PandasApplyUDFOp),
    # A recorded Series aligns by index.
    "assign series": (lambda x: x.assign(c=pd.Series([1, 2, 3], index=[2, 1, 0])),
                      PandasAssignOp),
    # pandas infers int64 for the scalar.
    "assign scalar and list": (lambda x: x.assign(c=1, d=[1, 2, 3]), PandasAssignOp),
}


@pytest.mark.parametrize("case", CASES)
def test_greedy_runs_it_on_pandas(case):
    make, impl = CASES[case]
    results = []
    for selector in ("default", "greedy"):
        with skrub.config_context(eager_data_ops=False):
            source = skrub.var("x", FRAME)
            # Under greedy the rows before the call are on polars. The plan ends
            # at the call: greedy would convert its pandas result to polars for a
            # trailing `reset_index` and return polars, where an unnamed Series
            # comes back named "". Greedy is not optimal there; a cost-based
            # selector would leave it on pandas.
            plan = make(source[source["b"] > 4].reset_index(drop=True))
        with st.config(scheduler=True, implementation_selector=selector):
            ops = optimize(plan, env=get_data(plan))[0]
            result = evaluate(plan)
        assert any(isinstance(op, impl) for op in ops)
        if selector == "greedy":
            assert any(isinstance(op, ConversionOp) for op in ops)
        results.append(result.to_pandas() if isinstance(result, (pl.DataFrame, pl.Series))
                       else result)
    expected, result = results
    if isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(result, expected)
    else:
        pd.testing.assert_frame_equal(result, expected)
