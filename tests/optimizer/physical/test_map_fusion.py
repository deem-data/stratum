"""Physical Polars fusion preserves the explicit logical IR and eager results."""
import gc
import operator
import weakref

import numpy as np
import pandas as pd
import polars as pl
import pytest
import skrub

from stratum.optimizer._op_utils import topological_iterator, validate_dag
from stratum.optimizer._optimize import OptConfig, logical_optimize, optimize
from stratum.optimizer._lineage import lineage_key
from stratum.optimizer.logical._column_expr import BinOpExpr, Col, Const, UFUNCS
from stratum.optimizer.logical._combine_ops import AssignOp
from stratum.optimizer.logical._map_ops import ExpressionMapOp
from stratum.optimizer.logical._ops import MethodCallOp
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp
from stratum.optimizer.physical._conversion import ConversionOp
from stratum.optimizer.physical._fused_map_execs import PolarsFusedAssignOp, PolarsFusedMapOp
from stratum.optimizer.physical._impl_selection import get_implementation_selector
from stratum.optimizer.physical._map_execs import PolarsExpressionMapOp
from stratum.optimizer.physical._plan_context import PlanContext
from stratum.optimizer.physical._representation import check_frame_representations
from stratum.runtime._scheduler import SequentialScheduler


FRAME = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0],
                      "s": ["a1", "b", "c"], "d": ["2020-01-01", "2020-02-02", "2020-03-03"]})


def plan(build, *, fused=True, selector="greedy", frame=FRAME, literal_inlining=True):
    with skrub.config_context(eager_data_ops=False):
        dag = build(skrub.var("frame", frame))
    return optimize(dag, OptConfig(map_fusion=fused,
                                  literal_inlining=literal_inlining,
                                  selector=get_implementation_selector(selector)),
                    env={"frame": frame})


def execute(build, **options):
    ops, split = plan(build, **options)
    validate_dag(ops[-1])
    check_frame_representations(ops)
    result = SequentialScheduler(ops, split).evaluate()
    if isinstance(result, (pl.Series, pl.DataFrame)):
        result = result.to_pandas()
    return ops, result


def fused_ops(ops):
    return [op for op in ops if isinstance(op, (PolarsFusedAssignOp, PolarsFusedMapOp))]


def test_config_is_snapshotted_and_logical_ir_stays_explicit():
    with skrub.config_context(eager_data_ops=False):
        frame = skrub.var("frame", FRAME)
        dag = frame.assign(c=frame["a"] + 2)
    config = OptConfig(map_fusion=True)
    root = logical_optimize(dag, config, env={"frame": FRAME})
    ops = list(topological_iterator(root))
    assert any(isinstance(op, ExpressionMapOp) for op in ops)
    assert any(isinstance(op, AssignOp) for op in ops)
    assert not fused_ops(ops)
    ctx = PlanContext.from_flags(config)
    config.map_fusion = False
    assert ctx.map_fusion and not PlanContext.from_flags(config).map_fusion


def test_assign_compiles_to_one_with_columns(monkeypatch):
    build = lambda df: df.assign(c=df["a"] + 2, label="constant", sequence=[1, 2, 3])
    ops, split = plan(build)
    assert len(ops) == 2
    assert isinstance(ops[-1], PolarsFusedAssignOp)
    assert ops[-1].entries["c"] == BinOpExpr(operator.add, Col("a"), Const(2))
    calls = []
    original = pl.DataFrame.with_columns

    def record(frame, *args, **kwargs):
        calls.append(1)
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "with_columns", record)
    monkeypatch.setattr(PolarsExpressionMapOp, "process",
                        lambda *args, **kwargs: pytest.fail("absorbed map executed"))
    result = SequentialScheduler(ops, split).evaluate().to_pandas()
    assert len(calls) == 1
    pd.testing.assert_frame_equal(result, FRAME.assign(c=FRAME["a"] + 2,
                                                      label="constant", sequence=[1, 2, 3]))


@pytest.mark.parametrize("value", [pd.NA, pd.NaT])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("literal_inlining", [False, True])
def test_unsupported_assignment_literals_use_pandas(value, fused, literal_inlining):
    from stratum.optimizer.physical._combine_execs import PandasAssignOp

    ops, result = execute(lambda df: df.assign(c=df["a"] + 1, missing=value), fused=fused,
                          literal_inlining=literal_inlining)
    assert isinstance(ops[-1], PandasAssignOp)
    assert "no polars literal" in dict(ops[-1]._refusals)["polars"]
    pd.testing.assert_frame_equal(result, FRAME.assign(c=FRAME["a"] + 1, missing=value))


def test_standalone_chain_compiles_to_one_select():
    ops, result = execute(lambda df: (df["a"] + 2) * 3)
    assert len(ops) == 2 and isinstance(ops[-1], PolarsFusedMapOp)
    assert ops[-1].result_name == "a"
    pd.testing.assert_series_equal(result, (FRAME["a"] + 2) * 3)


@pytest.mark.parametrize("name", ["*", "^a.*$", ""])
@pytest.mark.parametrize("path,fused", [
    ("standalone", False), ("standalone", True),
    ("assign", False), ("assign", True), ("projected", True),
    ("selector", False), ("selection", False), ("aggregate", False),
])
def test_column_names_are_literal(name, path, fused):
    frame = pd.DataFrame({name: [1, 2, 3], "a": [10, 20, 30], "g": [0, 0, 1]})

    def build(df):
        if path == "standalone":
            return df[name] + 1
        if path == "assign":
            return df.assign(z=df[name] + 1)
        if path == "projected":
            return df[[name]].assign(z=df["a"] + 1)
        if path == "selector":
            return df.skb.select([name]) if hasattr(df, "skb") else df[[name]]
        if path == "selection":
            return df[df[name] > 1]
        return df.groupby("g", as_index=False).agg(total=(name, "sum"))

    _, result = execute(build, frame=frame, fused=fused)
    expected = build(frame)
    check = pd.testing.assert_series_equal if isinstance(expected, pd.Series) else pd.testing.assert_frame_equal
    check(result.reset_index(drop=True), expected.reset_index(drop=True))


@pytest.mark.parametrize("fused", [False, True])
def test_missing_pattern_named_column_is_not_a_selector(fused):
    ops, split = plan(lambda df: df.assign(z=df["^a$"] + 1), fused=fused)
    with pytest.raises(RuntimeError, match=r"\^a\$"):
        SequentialScheduler(ops, split).evaluate()


@pytest.mark.parametrize("build", [
    lambda df: df.assign(a=df["a"] * 10, original=df["a"] * 2),
    lambda df: df.assign(c=(df["a"] + df["b"]) * 2),
    lambda df: df.assign(c=df["a"].fillna(0).astype("float32")),
    lambda df: df.assign(c=df["a"].where(df["b"] > 4, 0.0)),
    lambda df: df.assign(c=df["s"].str.upper().str.len()),
    lambda df: df.assign(c=df["s"].map({"a1": 1, "b": 2, "c": 3})),
    lambda df: df.assign(first=df["a"].isna(), second=df["a"].notna()),
    lambda df: df[["b", "a"]].assign(a=df["b"] * 10, c=df["a"] + 2),
    lambda df: df.drop(columns=["a"]).assign(a=df["a"] + 2),
    lambda df: df.assign(flag="yes", items=[1, 2, 3]),
])
def test_fused_and_unfused_results_match(build):
    original = FRAME.copy(deep=True)
    _, expected = execute(build, fused=False)
    _, result = execute(build)
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(result, build(FRAME), check_dtype=False)
    pd.testing.assert_frame_equal(FRAME, original)


def test_datetime_chain_fuses():
    build = lambda df: df.assign(year=df["d"].skb.apply_func(pd.to_datetime).dt.year)
    ops, result = execute(build)
    assert len(ops) == 2 and isinstance(ops[-1], PolarsFusedAssignOp)
    pd.testing.assert_frame_equal(result, FRAME.assign(year=pd.to_datetime(FRAME["d"]).dt.year),
                                  check_dtype=False)


def test_shared_computation_stays_materialized():
    def build(df):
        shared = df["a"] + 1
        assigned = df.assign(c=shared * 2)
        return skrub.deferred(lambda frame, column: frame)(assigned, shared)
    ops, result = execute(build)
    maps = [op for op in ops if isinstance(op, PolarsFusedMapOp)]
    assert len(maps) == 1
    [assign] = [op for op in ops if isinstance(op, PolarsFusedAssignOp)]
    assert maps[0] in assign.inputs
    assert len(maps[0].outputs) == 2
    pd.testing.assert_frame_equal(result, FRAME.assign(c=(FRAME["a"] + 1) * 2))


def test_shared_column_is_not_absorbed():
    def build(df):
        column = df["a"]
        assigned = df.assign(c=column + df["b"])
        return skrub.deferred(lambda frame, column: frame)(assigned, column)
    ops, result = execute(build)
    assert any(isinstance(op, ColumnProjectionOp) and op.key == "a" for op in ops)
    assert any(isinstance(op, PolarsFusedAssignOp) for op in ops)
    pd.testing.assert_frame_equal(result, FRAME.assign(c=FRAME["a"] + FRAME["b"]))


def test_shared_base_projection_keeps_its_other_consumer():
    def build(df):
        kept = df[["b"]]
        assigned = kept.assign(c=df["a"] + 2)
        return skrub.deferred(lambda frame, original: frame)(assigned, kept)
    ops, result = execute(build)
    assert any(isinstance(op, ColumnProjectionOp) and op.key == ["b"] for op in ops)
    [assign] = [op for op in ops if isinstance(op, PolarsFusedAssignOp)]
    assert assign.columns == ["b"]
    pd.testing.assert_frame_equal(result, FRAME[["b"]].assign(c=FRAME["a"] + 2))


def test_projected_assignment_keeps_column_order():
    ops, result = execute(lambda df: df[["s", "b"]].assign(b=df["a"] + 1, a=df["b"] * 2))
    assert len(ops) == 2 and isinstance(ops[-1], PolarsFusedAssignOp)
    assert list(result) == ["s", "b", "a"]
    assert list(ops[-1].output_schema) == list(result)


@pytest.mark.parametrize("fused", [False, True])
def test_overwriting_a_missing_projected_column_still_fails(fused):
    ops, split = plan(lambda df: df[["missing"]].assign(missing=df["a"] + 1), fused=fused)
    with pytest.raises(RuntimeError, match="missing"):
        SequentialScheduler(ops, split).evaluate()


def test_chained_assignments_read_the_correct_versions_of_columns():
    def build(df):
        first = df.assign(a=df["a"] + 1)
        return first.assign(c=first["a"] * 2, original=df["a"])
    _, expected = execute(build, fused=False)
    ops, result = execute(build)
    assert len([op for op in ops if isinstance(op, PolarsFusedAssignOp)]) == 2
    pd.testing.assert_frame_equal(result, expected)


def test_fused_lineage_includes_assignment_and_projection_order():
    def key(build):
        ops, _ = plan(build)
        op = ops[-1]
        assert isinstance(op, PolarsFusedAssignOp)
        return lineage_key(op, ["source"])
    first = lambda df: df[["a", "b"]].assign(c=df["a"] + 1, d=df["a"] + 2)
    assert key(first) == key(first)
    assert key(first) != key(lambda df: df[["b", "a"]].assign(c=df["a"] + 1, d=df["a"] + 2))
    assert key(first) != key(lambda df: df[["a", "b"]].assign(d=df["a"] + 2, c=df["a"] + 1))


def test_graph_fed_cast_and_fillna_remain_opaque():
    def build(df):
        cast = df["a"].astype(skrub.as_data_op("float32"))
        return df.assign(c=cast.fillna(0))
    ops, result = execute(build, literal_inlining=False)
    assert not fused_ops(ops)
    assert any(isinstance(op, MethodCallOp) and op.method_name == "fillna" for op in ops)
    pd.testing.assert_frame_equal(result, FRAME.assign(c=FRAME["a"].astype("float32").fillna(0)))


def test_conversions_remain_boundaries():
    build = lambda df: df.assign(c=df["s"].str.replace(r"(\d)", r"\1\1", regex=True),
                                 value=df["a"] + 2)
    ops, result = execute(build)
    assert any(isinstance(op, ConversionOp) for op in ops)
    assert any(isinstance(op, PolarsFusedAssignOp) for op in ops)
    pd.testing.assert_frame_equal(result, FRAME.assign(
        c=FRAME["s"].str.replace(r"(\d)", r"\1\1", regex=True), value=FRAME["a"] + 2))


@pytest.mark.parametrize("method,receiver", [
    ("fillna", lambda df: df["a"] / df["b"]),
    ("isna", lambda df: (df["a"] / df["b"]).clip(0, 1)),
    ("notna", lambda df: (df["a"] / df["b"]).fillna(0)),
])
@pytest.mark.parametrize("path", ["assign", "selection"])
def test_derived_missing_value_methods_match_pandas(method, receiver, path):
    frame = pd.DataFrame({"a": [0., 1., np.nan, 4.], "b": [0., 2., 1., np.nan]})

    def build(df):
        value = receiver(df)
        result = value.fillna(0) if method == "fillna" else getattr(value, method)()
        if path == "assign":
            return df.assign(c=result)
        return df[result > 0] if method == "fillna" else df[result]

    ops, result = execute(build, frame=frame)
    if path == "assign":
        assert len(ops) == 2 and isinstance(ops[-1], PolarsFusedAssignOp)
        _, unfused = execute(build, frame=frame, fused=False)
        pd.testing.assert_frame_equal(result, unfused)
    pd.testing.assert_frame_equal(result.reset_index(drop=True),
                                  build(frame).reset_index(drop=True))


def test_standalone_numpy_chain_with_fillna_fuses():
    frame = pd.DataFrame({"a": [-1., 0., 1., np.nan]})
    with np.errstate(invalid="ignore", divide="ignore"):
        ops, result = execute(lambda df: df["a"].skb.apply_func(np.log).fillna(0), frame=frame)
        expected = np.log(frame["a"]).fillna(0)
    assert len(ops) == 2 and isinstance(ops[-1], PolarsFusedMapOp)
    pd.testing.assert_series_equal(result, expected, check_exact=True)


def test_derived_datetime_conversion_fuses_without_reparsing():
    frame = pd.DataFrame({"d": [" 2020-01-01 ", "bad", None]})

    def build(df):
        date = df["d"].str.strip().skb.apply_func(pd.to_datetime, format="%Y-%m-%d", errors="coerce")
        shifted = date + pd.Timedelta(days=1)
        return df.assign(parsed=shifted.skb.apply_func(pd.to_datetime, format="%d/%m/%Y"))

    ops, result = execute(build, frame=frame)
    _, unfused = execute(build, frame=frame, fused=False)
    assert len(ops) == 2 and isinstance(ops[-1], PolarsFusedAssignOp)
    pd.testing.assert_frame_equal(result, unfused)
    expected = pd.to_datetime(frame["d"].str.strip(), format="%Y-%m-%d", errors="coerce")
    pd.testing.assert_frame_equal(result, frame.assign(parsed=expected + pd.Timedelta(days=1)),
                                  check_dtype=False)


def test_derived_dtype_metadata_is_local_to_each_execution():
    ops, _ = plan(lambda df: df.assign(m=(df["a"] + df["b"]).isna()))
    kernel = ops[-1]
    assert isinstance(kernel, PolarsFusedAssignOp)
    for values in ([float("nan"), 1., 2.], [None, "x", "y"]):
        borrowed = pl.from_pandas(FRAME).with_columns(
            pl.Series("a", values), pl.Series("b", values))
        original = borrowed.to_pandas()
        expected = original.assign(m=(original["a"] + original["b"]).isna())
        reference = weakref.ref(borrowed)
        result = kernel.process([borrowed])
        pd.testing.assert_frame_equal(result.to_pandas(), expected)
        del borrowed
        gc.collect()
        assert reference() is None, "kernel retained its borrowed input"


def test_pandas_stays_unfused():
    ops, result = execute(lambda df: df.assign(c=df["a"] + 2), selector="default")
    assert not fused_ops(ops)
    assert any(isinstance(op, ExpressionMapOp) for op in ops)
    pd.testing.assert_frame_equal(result, FRAME.assign(c=FRAME["a"] + 2))


@pytest.mark.parametrize("name", UFUNCS)
def test_fusion_keeps_numpy_numerical_results(name):
    frame = pd.DataFrame({"a": np.random.default_rng(0).lognormal(0, 1, 2000)})
    func = UFUNCS[name]

    ops, result = execute(lambda df: df.assign(c=df["a"].skb.apply_func(func)), frame=frame)
    assert len(ops) == 2 and fused_ops(ops)
    pd.testing.assert_series_equal(result["c"].rename("a"), func(frame["a"]), check_exact=True)
