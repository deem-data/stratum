"""``pd.to_datetime`` under the greedy selector: exact or refused, never different."""
import pandas as pd
import polars as pl
import pytest
import skrub

import stratum
from stratum._api import evaluate
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import optimize
from stratum.optimizer.physical._map_execs import PolarsAssignMapOp, PolarsSeriesMapOp

FRAME = pd.DataFrame({"d": pd.to_datetime(["2020-01-31 10:00", None, "2021-06-01 00:00"])})


@pytest.fixture
def source(tmp_path):
    # Read through parquet, so the column reaches polars as a datetime already.
    path = tmp_path / "d.parquet"
    FRAME.to_parquet(path)
    with skrub.config_context(eager_data_ops=False):
        yield stratum.as_data_op(str(path)).skb.apply_func(pd.read_parquet)


@pytest.mark.parametrize("kwargs", [{}, {"format": "%d/%m/%Y"}, {"errors": "coerce"}])
@pytest.mark.parametrize("path", ["standalone", "assign"])
def test_a_datetime_column_is_returned_as_it_is(source, path, kwargs):
    # pandas returns datetimes unchanged, whatever the options.
    with skrub.config_context(eager_data_ops=False):
        parsed = source["d"].skb.apply_func(pd.to_datetime, **kwargs)
        plan = parsed if path == "standalone" else source.assign(p=parsed)
    expected = pd.to_datetime(FRAME["d"], **kwargs)
    with stratum.config(scheduler=True, implementation_selector="greedy"):
        ops, *_ = optimize(plan, env=get_data(plan))
        result = evaluate(plan)
    native = PolarsSeriesMapOp if path == "standalone" else PolarsAssignMapOp
    assert any(isinstance(op, native) for op in ops)
    assert isinstance(result, (pl.Series, pl.DataFrame))
    result = result.to_pandas()
    if path == "assign":
        result = result["p"].rename("d")
    # polars keeps microseconds where pandas reads nanoseconds: the same instants.
    pd.testing.assert_series_equal(expected, result, check_dtype=False)
