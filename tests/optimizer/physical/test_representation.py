"""Frame representations on plan edges (#226): checked at planning, not at run time."""
import pandas as pd
import polars as pl
import pytest
import skrub

import stratum as st
from stratum._api import evaluate
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import optimize
from stratum.optimizer.logical._base import FRAME_TYPES
from stratum.optimizer.physical._conversion import ConversionOp
from stratum.optimizer.physical._registry import get_default_physical_registry
from stratum.optimizer.physical._representation import (
    FrameLib, frame_lib)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "x.parquet"
    pd.DataFrame({"x": [1, -2, 3], "s": ["1.0", "2", "3.0"]}).to_parquet(path)
    with skrub.config_context(eager_data_ops=False):
        yield st.as_data_op(str(path)).skb.apply_func(pd.read_parquet)


def _optimize(plan, selector):
    with st.config(scheduler=True, implementation_selector=selector):
        return optimize(plan, env=get_data(plan))[0]


def test_an_opaque_call_fed_polars_gets_converted_input(source):
    # #226, first example: `where` without `other` stays the recorded pandas
    # call; the greedy plan converts its polars inputs explicitly (#227).
    plan = source.assign(c=source["x"].where(source["x"] > 0))
    ops = _optimize(plan, "greedy")
    where = next(op for op in ops if type(op).__name__ == "MethodCallExec")
    assert all(isinstance(i, ConversionOp) and i.target is FrameLib.PANDAS
               for i in where.inputs)
    with st.config(scheduler=True, implementation_selector="greedy"):
        result = evaluate(plan)
    with st.config(scheduler=True):
        expected = evaluate(plan)
    assert isinstance(result, pl.DataFrame), "where keeps the rows: the assign stays on polars"
    pd.testing.assert_frame_equal(result.to_pandas(), expected, check_dtype=False)


def test_a_translated_expression_runs_on_polars(source):
    # #226, second example: the regex replace now compiles exactly.
    plan = source.assign(c=source["s"].str.replace(r"\.0$", "", regex=True))
    ops = _optimize(plan, "greedy")
    assert not any(isinstance(op, ConversionOp) for op in ops)
    with st.config(scheduler=True, implementation_selector="greedy"):
        result = evaluate(plan)
    assert isinstance(result, pl.DataFrame)
    assert result["c"].to_list() == ["1", "2", "3"]


def test_an_untranslatable_expression_runs_on_pandas(source):
    plan = source.assign(c=source["s"].str.replace(r"(\d)", r"\1\1", regex=True))
    ops = _optimize(plan, "greedy")
    assign = next(op for op in ops if type(op).__name__.endswith("AssignMapOp"))
    assert type(assign).__name__ == "PandasAssignMapOp"
    assert "not portable" in dict(assign._refusals)["polars"]
    with st.config(scheduler=True, implementation_selector="greedy"):
        result = evaluate(plan)
    with st.config(scheduler=True):
        expected = evaluate(plan)
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("selector", ["default", "greedy"])
def test_every_frame_edge_has_a_library(source, selector):
    plan = source.assign(c=source["x"] * 2)
    ops = _optimize(plan, selector)
    expected = FrameLib.PANDAS if selector == "default" else FrameLib.POLARS
    libs = {frame_lib(op) for op in ops if op.output_type in FRAME_TYPES}
    assert libs == {expected}


def test_registered_impls_declare_their_backends_library():
    registry = get_default_physical_registry()
    for op_type in registry.op_types():
        for impl in registry.candidates_for(op_type):
            if impl.impl_class is None:
                continue
            expected = {"pandas": FrameLib.PANDAS, "polars": FrameLib.POLARS}.get(
                impl.backend_name)
            assert impl.impl_class.frame_lib == expected, impl
