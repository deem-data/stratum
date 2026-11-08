"""Lineage of deterministic dataframe operations and their folded expressions."""
import json
import operator
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

import stratum as st
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._lineage import (
    LineageEncoder, Uncacheable, compute_lineage_keys, lineage_key,
)
from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.logical._aggregation_ops import AggregateOp
from stratum.optimizer.logical._base import OperandRef, OutputType
from stratum.optimizer.logical._column_expr import (
    AggExpr, AstypeExpr, BinOpExpr, Col, Const, DatetimeExpr, DtExpr,
    FillnaExpr, OperandLeaf, StrExpr, UnaryOpExpr,
)
from stratum.optimizer.logical._column_methods import (
    AstypeOp, ClipOp, FillnaOp, IsinOp, IsnaOp, MapValuesOp, NotnaOp, WhereOp,
)
from stratum.optimizer.logical._dataframe_ops import ConcatOp
from stratum.optimizer.logical._index_ops import GroupKeysOp, IndexAccessOp
from stratum.optimizer.logical._join_ops import FilteringJoinOp, JoinOp
from stratum.optimizer.logical._map_ops import (
    ApplyUDFOp, AssignMapOp, AssignOp, DatetimeConversionOp, DatetimeFieldOp,
    StringMethodOp,
)
from stratum.optimizer.logical._metadata_ops import (
    ColumnNamesOp, RenameOp, ResetIndexOp, ShapeOp,
)
from stratum.optimizer.logical._ops import CallOp, GetItemOp
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp, ColumnSelectorOp, DropOp
from stratum.optimizer.logical._selection_ops import (
    DropDuplicatesSelectionOp, DropLabelsSelectionOp, DropMissingSelectionOp,
    MaskSelectionOp, SampleSelectionOp, SliceSelectionOp,
)
from stratum.optimizer.logical._sort_ops import SortOp
from stratum.optimizer.physical._conversion import ConversionOp
from stratum.optimizer.physical._impl_selection import FlagBasedSelector
from stratum.optimizer.physical._representation import FrameLib

REPO = Path(__file__).resolve().parents[2]


def digest(value):
    encoder = LineageEncoder()
    encoder.encode(value)
    return encoder.hexdigest()


@pytest.mark.parametrize("make_op", [
    lambda x: ColumnProjectionOp([x]),
    lambda x: ColumnSelectorOp([x]),
    lambda x: GetItemOp(key=x),
    lambda x: DropOp(columns=[x]),
    lambda x: MaskSelectionOp(BinOpExpr(operator.gt, Col(x), Const(0))),
    lambda x: DropMissingSelectionOp(subset=[x]),
    lambda x: DropDuplicatesSelectionOp(subset=[x]),
    lambda x: DropLabelsSelectionOp(labels=[x]),
    lambda x: SortOp(by=[x]),
    lambda x: JoinOp(left_on=x, right_on=x),
    lambda x: FilteringJoinOp(left_on=x, right_on=x),
    lambda x: AggregateOp(grouped=True, grouping=(Col(x),),
                          aggregations=(("total", AggExpr("sum", Col("v"))),)),
    lambda x: AssignMapOp({x: BinOpExpr(operator.add, Col("v"), Const(1))}),
    lambda x: AssignOp(kwargs={x: 1}),
    lambda x: FillnaOp(value=x),
    lambda x: MapValuesOp(mapping={"old": x}),
    lambda x: IsinOp(values=[x]),
    lambda x: RenameOp(columns={"old": x}),
    lambda x: GroupKeysOp(x),
])
def test_operator_configuration_changes_the_key(make_op):
    key = lineage_key(make_op("a"), ["source"])
    assert key == lineage_key(make_op("a"), ["source"])
    assert key != lineage_key(make_op("b"), ["source"])


@pytest.mark.parametrize("left,right", [
    (SliceSelectionOp(n=2), SliceSelectionOp(n=3)),
    (SampleSelectionOp(n=2, seed=0), SampleSelectionOp(n=2, seed=1)),
    (ConcatOp([OperandRef(0), OperandRef(1)], axis=0),
     ConcatOp([OperandRef(0), OperandRef(1)], axis=1)),
    (AstypeOp("int64"), AstypeOp("float64")),
    (ClipOp(lower=0), ClipOp(lower=1)),
    (WhereOp(cond=OperandRef(1), other=0), WhereOp(cond=OperandRef(1), other=1)),
    (StringMethodOp("lower"), StringMethodOp("upper")),
    (DatetimeFieldOp("year"), DatetimeFieldOp("month")),
    (DatetimeConversionOp(format="%Y-%m-%d"), DatetimeConversionOp(format="%d/%m/%Y")),
    (AggregateOp(output_type=OutputType.FRAME), AggregateOp(output_type=OutputType.SERIES)),
    (ConversionOp(FrameLib.PANDAS, FrameLib.POLARS),
     ConversionOp(FrameLib.POLARS, FrameLib.PANDAS)),
])
def test_result_affecting_options_are_part_of_lineage(left, right):
    assert lineage_key(left, ["source", "other"]) != lineage_key(right, ["source", "other"])


def test_parameterless_operations_remain_distinct():
    ops = [IndexAccessOp(), ResetIndexOp(), ShapeOp(), ColumnNamesOp(), IsnaOp(), NotnaOp()]
    keys = [lineage_key(op, ["source"]) for op in ops]
    assert len(set(keys)) == len(ops)
    assert all(lineage_key(op, ["changed"]) != key for op, key in zip(ops, keys))


def test_assignment_order_and_operand_order_are_semantic():
    entries = {"a": Col("x"), "b": Col("y")}
    first = lineage_key(AssignMapOp(entries), ["source"])
    assert first != lineage_key(AssignMapOp(dict(reversed(list(entries.items())))), ["source"])
    concat = ConcatOp([OperandRef(0), OperandRef(1)])
    assert lineage_key(concat, ["left", "right"]) != lineage_key(concat, ["right", "left"])


def test_expressions_encode_types_operands_and_options():
    expressions = [
        Col("x"), Col("y"), Const(1), Const(1.0),
        OperandLeaf(OperandRef(0)), OperandLeaf(OperandRef(1)),
        BinOpExpr(operator.add, Col("x"), Const(1)),
        BinOpExpr(operator.sub, Col("x"), Const(1)),
        BinOpExpr(operator.sub, Const(1), Col("x")),
        UnaryOpExpr(operator.neg, Col("x")), UnaryOpExpr(operator.pos, Col("x")),
        StrExpr(Col("s"), "contains", {"pat": "a"}),
        StrExpr(Col("s"), "contains", {"pat": "b"}),
        DtExpr(Col("d"), "year"), DtExpr(Col("d"), "month"),
        DatetimeExpr(Col("s"), format="%Y"), DatetimeExpr(Col("s"), format="%Y-%m"),
        AstypeExpr(Col("x"), "int64"), AstypeExpr(Col("x"), "float64"),
        FillnaExpr(Col("x"), Const(0)), FillnaExpr(Col("x"), Const(1)),
        AggExpr("sum", Col("x")), AggExpr("sum", Col("x"), {"skipna": False}),
        AggExpr("mean", Col("x")),
    ]
    keys = [digest(expression) for expression in expressions]
    assert len(set(keys)) == len(expressions)
    assert keys == [digest(expression) for expression in expressions]


@pytest.mark.parametrize("expression", [
    BinOpExpr(np.add, Col("x"), Const(1)),
    UnaryOpExpr(lambda x: -x, Col("x")),
    Const(object()),
    StrExpr(Col("s"), "split", {"pat": ","}),
    StrExpr(Col("s"), "replace", {"pat": "a", "repl": lambda match: "b"}),
])
def test_unknown_expression_code_has_no_lineage(expression):
    with pytest.raises(Uncacheable):
        digest(expression)


@pytest.mark.parametrize("op", [
    SampleSelectionOp(n=2),
    SampleSelectionOp(n=2, seed=np.random.RandomState(0)),
    SampleSelectionOp(n=2, seed=np.random.default_rng(0)),
    StringMethodOp("split", {"pat": ","}),
    RenameOp(columns=lambda name: name.upper()),
    AssignOp(kwargs={"x": lambda frame: frame["y"]}),
    ApplyUDFOp(args=(lambda frame: frame,)),
    CallOp(func=lambda frame: frame),
])
def test_nondeterministic_and_unsupported_operations_are_uncacheable(op):
    with pytest.raises(Uncacheable):
        lineage_key(op, ["source"])


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "group": ["a", "a", "b"],
                  "text": ["Alpha", "Beta", "Gamma"],
                  "date": ["2020-01-01", "2020-02-02", "2021-03-03"]}).to_csv(path, index=False)
    return path


def read_csv(path):
    return st.deferred(pd.read_csv)(str(path))


def plan_keys(dag, backend="pandas", reasons=None):
    plan, *_ = optimize(dag, OptConfig(selector=FlagBasedSelector(backend=backend)),
                        env=get_data(dag))
    keys = compute_lineage_keys(plan, reasons=reasons)
    return [(str(op), keys[op]) for op in plan]


def csv_pipeline(path, offset=1):
    data = read_csv(path)
    return data.assign(z=data["x"] + offset).sort_values("z")


@pytest.mark.parametrize("backend", ["pandas", "polars"])
@pytest.mark.parametrize("build", [
    lambda d: d.assign(z=d["x"] + 1).sort_values("z"),
    lambda d: d[d["x"] > 1][["y"]],
    lambda d: d.drop(columns=["text"]).drop_duplicates(subset=["x"]).head(2),
    lambda d: d[["x", "y"]].groupby("x").sum(),
    lambda d: d.assign(upper=d["text"].str.upper()),
    lambda d: d["date"].skb.apply_func(pd.to_datetime, format="%Y-%m-%d").dt.year,
    lambda d: d["x"].fillna(0).clip(lower=1, upper=2).astype("float64"),
    lambda d: d.rename(columns={"x": "renamed"}).reset_index(drop=True),
    lambda d: d.sample(n=2, random_state=0),
    lambda d: d[["x"]].merge(d[["x", "y"]], on="x"),
    lambda d: d[["x"]].skb.concat([d[["y"]]], axis=1),
])
def test_file_backed_dataframe_plans_are_keyed(csv_path, backend, build):
    reasons = {}
    keys = plan_keys(build(read_csv(csv_path)), backend, reasons)
    assert keys and all(key is not None for _, key in keys), reasons
    assert plan_keys(build(read_csv(csv_path)), backend) == keys


def test_edits_rekey_only_the_affected_part_of_a_plan(csv_path):
    before = plan_keys(csv_pipeline(csv_path))
    changed = plan_keys(csv_pipeline(csv_path, offset=2))
    assert len(before) == len(changed)
    different = [a != b for (_, a), (_, b) in zip(before, changed)]
    first = different.index(True)
    assert first > 0
    assert all(different[first:])
    metadata = csv_path.stat()
    os.utime(csv_path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000_000))
    assert all(a != b for (_, a), (_, b) in zip(before, plan_keys(csv_pipeline(csv_path))))


def test_an_unseeded_sample_blocks_downstream_keys(csv_path):
    reasons = {}
    keys = plan_keys(read_csv(csv_path).sample(n=2)[["x"]], reasons=reasons)
    assert keys[0][1] is not None
    assert all(key is None for _, key in keys[1:])
    assert any("random_state" in reason for reason in reasons.values())


def test_a_udf_blocks_downstream_keys(csv_path):
    reasons = {}
    dag = read_csv(csv_path).skb.apply_func(lambda frame: frame[["x", "y"]] * 2)[["x"]]
    keys = plan_keys(dag, reasons=reasons)
    assert keys[0][1] is not None
    assert all(key is None for _, key in keys[1:])
    assert "an input has no key" in reasons.values()


def test_folding_does_not_make_an_unsupported_string_method_cacheable(csv_path):
    data = read_csv(csv_path)
    reasons = {}
    keys = plan_keys(data.assign(parts=data["text"].str.split("a")), reasons=reasons)
    assert keys[0][1] is not None and keys[-1][1] is None
    assert any("not a known string method" in reason for reason in reasons.values())


def test_dataframe_keys_are_stable_across_processes(csv_path):
    script = textwrap.dedent("""
        import json
        import sys
        from tests.optimizer.test_operator_lineage import csv_pipeline, plan_keys
        print(json.dumps([key for _, key in plan_keys(csv_pipeline(sys.argv[1]))]))
    """)
    expected = [key for _, key in plan_keys(csv_pipeline(csv_path))]
    for seed in ("1", "2"):
        result = subprocess.run([sys.executable, "-c", script, str(csv_path)],
                                env={**os.environ, "PYTHONHASHSEED": seed}, cwd=REPO,
                                capture_output=True, text=True, check=True, timeout=60)
        assert json.loads(result.stdout.strip().splitlines()[-1]) == expected


def test_a_window_by_calendar_offset_is_keyed(tmp_path):
    path = tmp_path / "events.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2020-03-01", "2021-06-01"]),
                  "cutoff": pd.to_datetime(["2020-01-01", "2020-01-01"])}).to_parquet(path)
    events = st.as_data_op(str(path)).skb.apply_func(pd.read_parquet)
    in_window = ((events["date"] >= events["cutoff"])
                 & (events["date"] < events["cutoff"] + pd.DateOffset(years=1)))
    reasons = {}
    keys = plan_keys(events[in_window], reasons=reasons)
    assert all(key is not None for _, key in keys), reasons


def test_standalone_arithmetic_is_keyed(tmp_path):
    # Arithmetic no assign or mask consumes stays a NumericOp: a number, a string
    # concatenation, a calendar offset, a numpy ufunc.
    path = tmp_path / "lots.parquet"
    pd.DataFrame({"boro": ["1", "2"], "block": ["7", "88"], "x": [1.5, 2.0],
                  "d": pd.to_datetime(["2020-02-29", "2021-03-01"])}).to_parquet(path)
    lots = st.as_data_op(str(path)).skb.apply_func(pd.read_parquet)
    for dag in (lots["x"] * 2 + 1, lots["boro"] + lots["block"],
                lots["d"] + pd.DateOffset(years=1), lots["x"].skb.apply_func(np.log)):
        reasons = {}
        keys = plan_keys(dag, reasons=reasons)
        assert all(key is not None for _, key in keys), reasons


def test_numeric_ops_are_keyed_by_operation_and_operands(csv_path):
    data = read_csv(csv_path)
    x, y = data["x"], data["y"]
    variants = [x + 1, x - 1, 1 - x, x * 2, x / 2, x ** 3, x + y, y + x,
                x.skb.apply_func(np.exp), x.skb.apply_func(np.sqrt)]
    final_keys = [plan_keys(v)[-1][1] for v in variants]
    assert None not in final_keys
    assert len(set(final_keys)) == len(variants)


def test_an_unknown_numpy_function_is_uncacheable(csv_path):
    reasons = {}
    keys = plan_keys(read_csv(csv_path)["x"].skb.apply_func(np.cumsum), reasons=reasons)
    assert keys[-1][1] is None
    assert any("cumsum" in str(why) for why in reasons.values())
