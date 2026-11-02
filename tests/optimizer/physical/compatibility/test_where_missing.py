"""``where`` with a missing ``other``, ``where(cond)`` included, matches pandas.

pandas fills the operand dtype's missing value and changes the dtype with it (an
int column becomes float, a bool one object), which polars cannot do without
knowing that dtype, so these run on pandas under greedy too.
"""
import numpy as np
import pandas as pd
import pytest
import skrub

import stratum
from stratum._api import evaluate
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import optimize
from stratum.optimizer.logical._column_expr import WhereExpr
from stratum.optimizer.logical._map_ops import SeriesMapOp
from stratum.optimizer.physical._conversion import to_pandas

FRAME = pd.DataFrame({
    "str": pd.Series(["1", "x", "3"], dtype="str"),
    "int": [1, 2, 3],
    "bool": [True, False, True],
    "float": [1.0, 2.0, 3.0],
    "date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
    "Int64": pd.Series([1, None, 3], dtype="Int64"),
    "k": [1, 0, 1],
})
OTHERS = {"none given": (), "nan": (np.nan,), "None": (None,), "NA": (pd.NA,)}


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("other", OTHERS)
@pytest.mark.parametrize("path", ["standalone", "assign"])
@pytest.mark.parametrize("column", [c for c in FRAME if c != "k"])
def test_where_missing_matches_pandas(column, path, other, selector):
    args = OTHERS[other]
    try:
        expected = FRAME[column].where(FRAME["k"] == 1, *args)
    except (TypeError, ValueError):
        pytest.skip("pandas rejects this fill for this dtype")
    with skrub.config_context(eager_data_ops=False):
        source = skrub.var("frame", FRAME)
        kept = source[column].where(source["k"] == 1, *args)
        plan = kept if path == "standalone" else source.assign(w=kept)
    with stratum.config(scheduler=True, implementation_selector=selector):
        ops, *_ = optimize(plan, env=get_data(plan))
        result = evaluate(plan)
    result = to_pandas(result)
    if path == "assign":
        result = result["w"].rename(column)
    assert not any(type(op).__name__ in ("PolarsSeriesMapOp", "PolarsAssignMapOp")
                   for op in ops)
    if other == "none given":
        assert any(isinstance(op, SeriesMapOp) and isinstance(op.expr, WhereExpr)
                   for op in ops) or path == "assign"
    assert result.isna().tolist() == expected.isna().tolist()
    # Under greedy the source is read through polars, which gives strings back as
    # `str` and a nullable Int64 as float64 before `where` runs; pandas' own
    # dtype changes (int to float, bool to object) are checked everywhere.
    converted = selector == "greedy" and column in ("str", "Int64")
    pd.testing.assert_series_equal(result.reset_index(drop=True),
                                   expected.reset_index(drop=True),
                                   check_dtype=not converted)


def test_a_present_other_still_runs_on_polars():
    with skrub.config_context(eager_data_ops=False):
        source = skrub.var("frame", FRAME)
        plan = source["float"].where(source["k"] == 1, 0.0)
    with stratum.config(scheduler=True, implementation_selector="greedy"):
        ops, *_ = optimize(plan, env=get_data(plan))
    assert any(type(op).__name__ == "PolarsSeriesMapOp" for op in ops)
