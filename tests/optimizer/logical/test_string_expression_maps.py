"""String Maps preserve argument resolution, result shapes, and fusion boundaries."""
import pandas as pd
import polars as pl
import pytest
import skrub

from stratum.optimizer._op_utils import validate_dag
from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.logical._base import Uncacheable
from stratum.optimizer.logical._column_expr import OperandLeaf, StrExpr
from stratum.optimizer.logical._map_ops import ExpressionMapOp
from stratum.optimizer.physical._fused_map_execs import PolarsFusedAssignOp, PolarsFusedMapOp
from stratum.optimizer.physical._impl_selection import get_implementation_selector
from stratum.optimizer.physical._representation import FrameLib, frame_lib
from stratum.runtime._scheduler import SequentialScheduler


FRAME = pd.DataFrame({"s": ["a1", "bb", "c1"], "id": [1, 2, 3]})


def execute(build, selector):
    with skrub.config_context(eager_data_ops=False):
        query = build(skrub.var("frame", FRAME))
    ops, split = optimize(query, OptConfig(selector=get_implementation_selector(selector)),
                          env={"frame": FRAME})
    validate_dag(ops[-1])
    result = SequentialScheduler(ops, split).evaluate()
    if isinstance(result, (pl.Series, pl.DataFrame)):
        result = result.to_pandas()
    return ops, result


def string_map(ops):
    [op] = [op for op in ops if isinstance(op, ExpressionMapOp) and isinstance(op.expr, StrExpr)]
    return op


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("method,path", [
    ("contains", "standalone"), ("contains", "selection"),
    ("replace", "standalone"), ("replace", "assign"),
])
def test_graph_fed_string_parameters_are_resolved_at_execution(selector, path, method):
    def build(df):
        pattern = skrub.deferred(lambda: "1")()
        if method == "contains":
            called = df["s"].str.contains(pattern, regex=False)
        else:
            replacement = skrub.deferred(lambda: "x")()
            called = df["s"].str.replace(pattern, replacement, regex=False)
        if path == "assign":
            return df.assign(z=called)
        if path == "selection":
            return df[called if method == "contains" else called == "ax"]
        return called

    ops, result = execute(build, selector)
    mapped = string_map(ops)
    assert isinstance(mapped.expr.params["pat"], OperandLeaf)
    assert mapped.expr.has_parameter_exprs
    assert frame_lib(mapped) is (FrameLib.PANDAS if selector == "default" else FrameLib.POLARS)
    assert not any(isinstance(op, (PolarsFusedMapOp, PolarsFusedAssignOp)) for op in ops)
    called = (FRAME["s"].str.contains("1", regex=False) if method == "contains"
              else FRAME["s"].str.replace("1", "x", regex=False))
    expected = called
    if path == "assign":
        expected = FRAME.assign(z=called)
    elif path == "selection":
        expected = FRAME[called if method == "contains" else called == "ax"]
    check = pd.testing.assert_series_equal if path == "standalone" else pd.testing.assert_frame_equal
    check(result.reset_index(drop=True), expected.reset_index(drop=True))


@pytest.mark.parametrize("method,args,kwargs", [
    ("split", ("1",), {"expand": True}),
    ("extract", (r"(?P<letter>[a-z])",), {}),
    ("extractall", (r"(?P<letter>[a-z])",), {}),
    ("cat", (), {"sep": "|"}),
])
def test_shape_changing_string_calls_keep_their_pandas_results(method, args, kwargs):
    ops, result = execute(lambda df: getattr(df["s"].str, method)(*args, **kwargs), "greedy")
    mapped = string_map(ops)
    assert mapped.output_schema is None
    assert not mapped.expr.preserves_columns
    assert frame_lib(mapped) is FrameLib.PANDAS
    with pytest.raises(Uncacheable):
        mapped.expr.lineage()
    expected = getattr(FRAME["s"].str, method)(*args, **kwargs)
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize("method,args", [("normalize", ("NFKC",)), ("split", ("1",))])
def test_native_methods_outside_the_fusion_allowlist_stay_materialized(method, args):
    ops, _ = execute(lambda df: getattr(df["s"].str, method)(*args), "greedy")
    assert frame_lib(string_map(ops)) is FrameLib.POLARS
    assert not any(isinstance(op, PolarsFusedMapOp) for op in ops)


def test_string_mapping_arguments_preserve_dictionary_subclasses():
    class TranslationTable(dict):
        def __missing__(self, key):
            return ord("x")

    table = TranslationTable({ord("a"): ord("A")})
    _, result = execute(lambda df: df["s"].str.translate(table), "greedy")
    pd.testing.assert_series_equal(result, FRAME["s"].str.translate(table))
    assert table == {ord("a"): ord("A")}
