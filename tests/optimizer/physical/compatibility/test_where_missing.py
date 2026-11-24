"""Use pandas for ``where`` with an omitted or missing scalar replacement.

Missing values can change the dtype: int to float, bool to object. The greedy
selector must also choose pandas for these expressions. Strict expected failures
document a later conversion issue: object-column NaN/NA values become Polars nulls.
"""
import numpy as np
import pandas as pd
import pytest
import skrub

import stratum
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.logical._column_expr import WhereExpr
from stratum.optimizer.logical._map_ops import ExpressionMapOp
from stratum.optimizer.physical._map_execs import PandasExpressionMapOp
from stratum.optimizer.physical._fused_map_execs import PolarsFusedMapOp
from stratum.runtime._scheduler import SequentialScheduler
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


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("other", OTHERS)
@pytest.mark.parametrize("path", ["standalone", "assign"])
@pytest.mark.parametrize("column", [c for c in FRAME if c != "k"])
def test_where_missing_matches_pandas(column, path, other, selector, fused, request):
    if (selector == "greedy" and path == "assign" and column == "bool"
            and other in {"none given", "nan", "NA"}):
        request.node.add_marker(pytest.mark.xfail(
            strict=True, raises=AssertionError,
            reason="pandas object missing sentinels become Polars nulls after where"))
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
        ops, split = optimize(plan, OptConfig(map_fusion=fused), env=get_data(plan))
        result = SequentialScheduler(ops, split).evaluate()
    result = to_pandas(result)
    if path == "assign":
        result = result["w"].rename(column)
    assert any(isinstance(op, PandasExpressionMapOp) and isinstance(op.expr, WhereExpr)
               for op in ops)
    assert result.isna().tolist() == expected.isna().tolist()
    # With greedy selection, conversion through Polars changes strings to `str`
    # and nullable Int64 to float64 before `where`. Skip dtype checks for those
    # inputs; retain checks for int -> float and bool -> object promotion.
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
    assert any(isinstance(op, (ExpressionMapOp, PolarsFusedMapOp))
               and isinstance(op.expr, WhereExpr) for op in ops)
    assert not any(isinstance(op, PandasExpressionMapOp) for op in ops)
