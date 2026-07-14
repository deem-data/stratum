import operator
import unittest

import pytest
import numpy as np
import pandas as pd
import polars as pl

import stratum as st
from stratum.optimizer._optimize import OptConfig
from stratum.optimizer.ir._map_ops import AssignMapOp
from stratum.optimizer.ir._projection_ops import (
    AssignOp, DatetimeConversionOp, GetAttrProjectionOp)
from stratum.optimizer.ir._column_expr import (
    BinOpExpr, Col, Const, DatetimeExpr, DtExpr, OperandLeaf, StrExpr, _Folder)
from stratum.optimizer.ir._ops import (
    BinOp, GetItemOp, Op, OperandRef, UnaryOp)
from stratum.tests.logical_optimizer.test_dataframe_ops import (
    optimize, run_op, force_polars, make_map_op)


def _one(test, ops, cls):
    found = [o for o in ops if isinstance(o, cls)]
    test.assertEqual(1, len(found), f"expected exactly one {cls.__name__}")
    return found[0]


class TestAssignMapFolding(unittest.TestCase):
    """df.assign(...) folds into an AssignMapOp with one ColumnExpr per column."""

    def setUp(self):
        self.df = pd.DataFrame({"c1": [1, 2],
                                "c2": ["2020-01-01", "2020-02-03"],
                                "s": ["a1", "bb"]})

    def test_feature_engineering_pattern_folds(self):
        # The motivating pattern: arithmetic, a datetime conversion via apply_func
        # and two .dt accessors all fold; the shared date column folds once.
        src = st.as_data_op(self.df)
        date_col = src["c2"].skb.apply_func(pd.to_datetime)
        out = src.assign(c3=src["c1"] + 123, c4=date_col.dt.day, c5=date_col.dt.day)
        ops = optimize(out, OptConfig(dataframe_ops=True))
        map_op = _one(self, ops, AssignMapOp)
        day = DtExpr(DatetimeExpr(Col("c2")), "day")
        self.assertEqual({"c3": BinOpExpr(operator.add, Col("c1"), Const(123)),
                          "c4": day, "c5": day}, map_op.entries)
        # everything private to the assign is absorbed: only source + map remain
        self.assertEqual(1, len(map_op.inputs))
        for cls in (GetItemOp, BinOp, DatetimeConversionOp, GetAttrProjectionOp):
            self.assertEqual([], [o for o in ops if isinstance(o, cls)], cls.__name__)

    def test_shared_subexpression_folds_to_same_object(self):
        # c4 and c5 fold through the shared folder memo: one expression instance.
        src = st.as_data_op(self.df)
        date_col = src["c2"].skb.apply_func(pd.to_datetime)
        out = src.assign(c4=date_col.dt.day, c5=date_col.dt.day)
        map_op = _one(self, optimize(out, OptConfig(dataframe_ops=True)), AssignMapOp)
        self.assertIs(map_op.entries["c4"].operand, map_op.entries["c5"].operand)

    def test_same_root_reused_across_entries(self):
        src = st.as_data_op(self.df)
        derived = src["c1"] + 1
        out = src.assign(first=derived, second=derived)
        map_op = _one(self, optimize(out, OptConfig(dataframe_ops=True)), AssignMapOp)
        self.assertIs(map_op.entries["first"], map_op.entries["second"])

    def test_same_producer_used_for_both_operands(self):
        src = st.as_data_op(self.df)
        derived = src["c1"] + 1
        out = src.assign(total=derived + derived)
        map_op = _one(self, optimize(out, OptConfig(dataframe_ops=True)), AssignMapOp)
        total = map_op.entries["total"]
        self.assertIsInstance(total, BinOpExpr)
        self.assertIs(total.left, total.right)

    def test_str_method_folds_to_str_expr(self):
        src = st.as_data_op(self.df)
        out = src.assign(c=src["s"].str.count("1"))
        map_op = _one(self, optimize(out, OptConfig(dataframe_ops=True)), AssignMapOp)
        self.assertEqual({"c": StrExpr(Col("s"), "count", ("1",))}, map_op.entries)

    def test_scalar_constant_folds_to_const(self):
        src = st.as_data_op(self.df)
        out = src.assign(c3=src["c1"] + 1, flag=7)
        map_op = _one(self, optimize(out, OptConfig(dataframe_ops=True)), AssignMapOp)
        self.assertEqual(Const(7), map_op.entries["flag"])

    def test_sequence_constant_falls_back_to_assign_op(self):
        # A list-valued kwarg means "assign these values"; backends spell that
        # differently, so the opaque AssignOp handles it.
        src = st.as_data_op(self.df)
        ops = optimize(src.assign(vals=[10, 20]), OptConfig(dataframe_ops=True))
        _one(self, ops, AssignOp)
        self.assertEqual([], [o for o in ops if isinstance(o, AssignMapOp)])

    def test_flag_off_keeps_assign_op(self):
        src = st.as_data_op(self.df)
        with make_map_op(False):
            ops = optimize(src.assign(c3=src["c1"] + 1), OptConfig(dataframe_ops=True))
        _one(self, ops, AssignOp)

    def test_externally_consumed_producer_stays_as_leaf(self):
        # The comparison feeds the assign AND a selection, so it cannot be
        # absorbed: the entry references it via an OperandLeaf.
        src = st.as_data_op(self.df)
        mask = src["c1"] > 1
        out = src[mask].assign(keep=mask)
        ops = optimize(out, OptConfig(dataframe_ops=True))
        map_op = _one(self, ops, AssignMapOp)
        self.assertIsInstance(map_op.entries["keep"], OperandLeaf)
        self.assertEqual(2, len(map_op.inputs))  # [selection frame, shared mask]
        self.assertTrue(any(isinstance(o, BinOp) for o in ops))

    def test_transformed_column_stays_as_leaf(self):
        # A fitted transformer is outside the native-lazy grammar: the apply
        # stays a graph op and the entry references its output via a leaf.
        from sklearn.preprocessing import StandardScaler
        from stratum.optimizer.ir._ops import TransformerOp
        src = st.as_data_op(self.df)
        scaled = src.skb.apply(StandardScaler(), cols=["c1"])
        out = src.assign(keep=scaled["c1"])
        ops = optimize(out, OptConfig(dataframe_ops=True))
        map_op = _one(self, ops, AssignMapOp)
        self.assertIsInstance(map_op.entries["keep"], OperandLeaf)
        self.assertEqual(2, len(map_op.inputs))  # [src, scaled["c1"]]
        self.assertTrue(any(isinstance(o, TransformerOp) for o in ops))

    def test_unsupported_datetime_kwargs_stay_as_leaf(self):
        src = st.as_data_op(self.df)
        parsed = src["c2"].skb.apply_func(pd.to_datetime, dayfirst=True)
        ops = optimize(src.assign(parsed=parsed), OptConfig(dataframe_ops=True))
        map_op = _one(self, ops, AssignMapOp)
        self.assertIsInstance(map_op.entries["parsed"], OperandLeaf)
        _one(self, ops, DatetimeConversionOp)


class TestAssignMapProcess(unittest.TestCase):
    """AssignMapOp evaluates its entries and assigns them per backend."""

    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 3],
                                "d": ["2020-01-01", "2020-02-03", "2020-03-05"]})

    def test_expr_entries_pandas(self):
        op = AssignMapOp(entries={
            "y": BinOpExpr(operator.add, Col("x"), Const(10)),
            "day": DtExpr(DatetimeExpr(Col("d")), "day"),
            "flag": Const(1)})
        result = run_op(op, self.df)
        self.assertEqual([11, 12, 13], result["y"].tolist())
        self.assertEqual([1, 3, 5], result["day"].tolist())
        self.assertEqual([1, 1, 1], result["flag"].tolist())
        self.assertEqual(["x", "d", "y", "day", "flag"], list(result.columns))

    def test_datetime_expr_forwards_pandas_kwargs(self):
        frame = pd.DataFrame({"d": ["01/02/2020"]})
        op = AssignMapOp(entries={
            "parsed": DatetimeExpr(Col("d"), kwargs={"dayfirst": True})})
        result = run_op(op, frame)
        self.assertEqual(pd.Timestamp("2020-02-01"), result["parsed"].iloc[0])

    def test_expr_entries_polars(self):
        op = AssignMapOp(entries={
            "y": BinOpExpr(operator.add, Col("x"), Const(10)),
            "day": DtExpr(DatetimeExpr(Col("d")), "day")})
        with force_polars():
            result = run_op(op, pl.from_pandas(self.df))
        self.assertEqual([11, 12, 13], result["y"].to_list())
        self.assertEqual([1, 3, 5], result["day"].to_list())

    def test_is_month_end_entry_polars(self):
        frame = pl.DataFrame({
            "d": pl.Series("d", pd.to_datetime(["2025-01-31", "2025-01-15"]))})
        op = AssignMapOp(entries={"end": DtExpr(Col("d"), "is_month_end")})
        with force_polars():
            result = run_op(op, frame)
        self.assertEqual([True, False], result["end"].to_list())

    def test_operand_leaf_entry(self):
        op = AssignMapOp(entries={"ext": OperandLeaf(OperandRef(1))})
        result = run_op(op, self.df, pd.Series([7, 8, 9]))
        self.assertEqual([7, 8, 9], result["ext"].tolist())

    def test_pandas_leaf_converts_under_polars(self):
        # A leaf feeding pandas data into a polars plan is converted, mirroring
        # the AssignOp behaviour.
        op = AssignMapOp(entries={"ext": OperandLeaf(OperandRef(1))})
        with force_polars():
            result = run_op(op, pl.from_pandas(self.df), pd.Series([7, 8, 9]))
        self.assertEqual([7, 8, 9], result["ext"].to_list())

    def test_scalar_leaf_broadcasts_under_polars(self):
        op = AssignMapOp(entries={"ext": OperandLeaf(OperandRef(1))})
        with force_polars():
            result = run_op(op, pl.from_pandas(self.df), 7)
        self.assertEqual([7, 7, 7], result["ext"].to_list())

    def test_list_leaf_becomes_column_under_polars(self):
        op = AssignMapOp(entries={"ext": OperandLeaf(OperandRef(1))})
        with force_polars():
            result = run_op(op, pl.from_pandas(self.df), [7, 8, 9])
        self.assertEqual([7, 8, 9], result["ext"].to_list())

    def test_clone_shares_immutable_entries(self):
        op = AssignMapOp(entries={"y": BinOpExpr(operator.add, Col("x"), Const(1))})
        cloned = op.clone()
        self.assertIsNot(op.entries, cloned.entries)
        self.assertIs(op.entries["y"], cloned.entries["y"])


class TestMapStructureKey(unittest.TestCase):
    """MapOps expose structural keys so CSE can dedup identical maps."""

    def test_assign_map_equal_keys(self):
        src = Op()
        entries = {"y": BinOpExpr(operator.add, Col("x"), Const(1))}
        a = AssignMapOp(entries=dict(entries), inputs=[src])
        b = AssignMapOp(entries=dict(entries), inputs=[src])
        self.assertEqual(a.structure_key(), b.structure_key())

    def test_assign_map_differs_on_entries(self):
        src = Op()
        a = AssignMapOp(entries={"y": Const(1)}, inputs=[src])
        b = AssignMapOp(entries={"y": Const(2)}, inputs=[src])
        self.assertNotEqual(a.structure_key(), b.structure_key())

    def test_duplicate_assign_maps_dedup_after_cse(self):
        df = pd.DataFrame({"x": [1, 2]})
        data = st.as_data_op(df)
        root = data.assign(y=data["x"] + 1).skb.concat(
            [data.assign(y=data["x"] + 1)], axis=0)
        ops = optimize(root, OptConfig(dataframe_ops=True))
        self.assertEqual(1, len([o for o in ops if isinstance(o, AssignMapOp)]))


class TestMapExprRefContract(unittest.TestCase):
    """New expr nodes honour the ref-traversal contract used by CSE/validation."""

    def test_iter_refs_through_dt_chain(self):
        expr = DtExpr(DatetimeExpr(OperandLeaf(OperandRef(2))), "day")
        self.assertEqual([2], [r.k for r in expr.iter_operand_refs()])

    def test_remap_rebuilds_dt_chain(self):
        expr = DtExpr(DatetimeExpr(OperandLeaf(OperandRef(2)),
                                   kwargs={"format": "%Y"}), "day")
        self.assertEqual(
            DtExpr(DatetimeExpr(OperandLeaf(OperandRef(1)),
                                kwargs={"format": "%Y"}), "day"),
            expr.remap_operand_refs({2: 1}))

    def test_pure_dt_chain_has_no_refs(self):
        expr = DtExpr(DatetimeExpr(Col("d")), "month")
        self.assertEqual([], list(expr.iter_operand_refs()))
        self.assertEqual(expr, expr.remap_operand_refs({}))


class TestFolderEdgeCases(unittest.TestCase):
    def test_constant_unary_has_no_producer(self):
        folder = _Folder(Op())
        unary = UnaryOp(operator.neg, 1)
        self.assertEqual([], folder._producer_ops(unary))

    def test_source_root_becomes_input_zero_leaf(self):
        source = Op()
        consumer = Op(inputs=[source])
        source.outputs = [consumer]
        folder = _Folder(source)

        expr = folder.fold(source, root_consumer=consumer)

        self.assertEqual(OperandLeaf(OperandRef(0)), expr)
        self.assertEqual([], folder.leaf_ops)

    def test_absorb_is_idempotent(self):
        folder = _Folder(Op())
        node = Op()
        folder._absorb(node)
        folder._absorb(node)
        self.assertEqual([node], folder.absorbed)

    def test_shared_child_of_dropped_roots_is_dropped_once(self):
        source = Op()
        shared = UnaryOp(operator.neg, OperandRef(0))
        shared.inputs = [source]
        first = UnaryOp(operator.neg, OperandRef(0))
        first.inputs = [shared]
        second = UnaryOp(operator.pos, OperandRef(0))
        second.inputs = [shared]
        consumer = Op(inputs=[source, first, second])
        first_external = Op(inputs=[first])
        second_external = Op(inputs=[second])
        source.outputs = [shared, consumer]
        shared.outputs = [first, second]
        first.outputs = [consumer, first_external]
        second.outputs = [consumer, second_external]
        folder = _Folder(source)
        subgraph, children = folder._discover([first, second])

        absorbable = folder._absorbable(
            [first, second], consumer, subgraph, children)

        self.assertEqual(set(), absorbable)


"""End-to-end: folded maps run through the full optimize + execute path."""

@pytest.fixture(params=[False, True], ids=["pandas", "polars"])
def polars(request):
    with force_polars(request.param):
        yield request.param


def test_assign_pipeline_evaluates(polars):
    df = pd.DataFrame({"c1": [1, 2], "c2": ["2020-01-01", "2020-02-03"]})
    src = st.as_data_op(df)
    date_col = src["c2"].skb.apply_func(pd.to_datetime)
    out = src.assign(c3=src["c1"] + 123, c4=date_col.dt.day, c5=date_col.dt.day)
    result = st._api.evaluate(out)
    assert [124, 125] == list(result["c3"])
    assert [1, 3] == list(result["c4"])
    assert [1, 3] == list(result["c5"])


def test_chained_assign_maps_evaluate(polars):
    # Three chained assigns, each reading a column produced by the previous one:
    # every assign folds into its own map, and Col refs resolve against the
    # previous map's output frame.
    df = pd.DataFrame({"x": [1, 2, 3]})
    src = st.as_data_op(df)
    d1 = src.assign(x2=src["x"] * 2)
    d2 = d1.assign(x4=d1["x2"] * 2)
    d3 = d2.assign(x8=d2["x4"] * 2)
    ops = optimize(d3, OptConfig(dataframe_ops=True))
    assert 3 == len([o for o in ops if isinstance(o, AssignMapOp)])
    assert 4 == len(ops)  # source + three maps
    result = st._api.evaluate(d3)
    assert [2, 4, 6] == list(result["x2"])
    assert [4, 8, 12] == list(result["x4"])
    assert [8, 16, 24] == list(result["x8"])


def test_assign_overwrite_and_read_original_column(polars):
    # One entry overwrites "a" while another reads it: both must see the
    # ORIGINAL column (assign/with_columns evaluate against the input frame).
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    src = st.as_data_op(df)
    out = src.assign(a=src["a"] * 10, doubled=src["a"] * 2)
    result = st._api.evaluate(out)
    assert [10.0, 20.0, 30.0] == list(result["a"])
    assert [2.0, 4.0, 6.0] == list(result["doubled"])


def test_deep_arithmetic_expression_evaluates(polars):
    # A deep native tree: nested binops, unary negation, pow / floordiv / mod.
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    src = st.as_data_op(df)
    a, b = src["a"], src["b"]
    out = src.assign(y=-((a + b) ** 2) / 3 + (a * b) % 5 + b // a)
    ops = optimize(out, OptConfig(dataframe_ops=True))
    assert 2 == len(ops)  # everything private folds: source + map
    result = st._api.evaluate(out)
    expected = -((df["a"] + df["b"]) ** 2) / 3 + (df["a"] * df["b"]) % 5 + df["b"] // df["a"]
    np.testing.assert_allclose(expected.to_numpy(),
                               np.asarray(list(result["y"]), dtype=float), rtol=1e-9)


def test_boolean_logic_expression_evaluates(polars):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 4.0, 5.0]})
    src = st.as_data_op(df)
    out = src.assign(flag=~((src["a"] > 2) | (src["b"] < 4)))
    result = st._api.evaluate(out)
    assert [False, True, False] == list(result["flag"])


def test_chained_string_methods_evaluate(polars):
    # StrExpr over StrExpr: each .str call fuses, then the whole chain folds.
    # polars renames every method (strip_chars / to_uppercase / len_chars).
    df = pd.DataFrame({"s": ["  ab  ", " c ", "def"]})
    src = st.as_data_op(df)
    out = src.assign(up=src["s"].str.strip().str.upper(),
                     n=src["s"].str.strip().str.len() * 10)
    ops = optimize(out, OptConfig(dataframe_ops=True))
    assert 2 == len(ops)
    result = st._api.evaluate(out)
    assert ["AB", "C", "DEF"] == list(result["up"])
    assert [20, 10, 30] == list(result["n"])


def test_datetime_feature_engineering_evaluates(polars):
    # dt attributes combined with arithmetic below and above the conversion.
    df = pd.DataFrame({"ts": ["2021-03-05", "2023-11-20", "2020-01-31"]})
    src = st.as_data_op(df)
    date = src["ts"].skb.apply_func(pd.to_datetime)
    out = src.assign(months_since_2020=(date.dt.year - 2020) * 12 + date.dt.month,
                     day=date.dt.day)
    ops = optimize(out, OptConfig(dataframe_ops=True))
    assert 2 == len(ops)
    result = st._api.evaluate(out)
    assert [15, 47, 1] == list(result["months_since_2020"])
    assert [5, 20, 31] == list(result["day"])


@pytest.mark.parametrize(
    "values, kwargs, expected",
    [
        (["01/02/2020"], {"dayfirst": True},
         [pd.Timestamp("2020-02-01")]),
        ([1], {"unit": "D", "origin": "2020-01-01"},
         [pd.Timestamp("2020-01-02")]),
        (["2020-01-01"], {"utc": True},
         [pd.Timestamp("2020-01-01", tz="UTC")]),
    ],
)
def test_datetime_unsupported_kwargs_preserved(polars, values, kwargs, expected):
    df = pd.DataFrame({"ts": values})
    src = st.as_data_op(df)
    parsed = src["ts"].skb.apply_func(pd.to_datetime, **kwargs)
    result = st._api.evaluate(src.assign(parsed=parsed))
    assert expected == [pd.Timestamp(value) for value in result["parsed"]]


@pytest.mark.xfail(
    strict=True,
    reason="TODO: DatetimeExpr.to_polars only supports string operands")
def test_datetime_conversion_existing_datetime_polars():
    df = pd.DataFrame({"ts": pd.to_datetime(["2020-01-01", "2021-02-03"])})
    src = st.as_data_op(df)
    parsed = src["ts"].skb.apply_func(pd.to_datetime)
    with force_polars():
        st._api.evaluate(src.assign(parsed=parsed))


def test_weird_column_names_evaluate(polars):
    # Names with spaces and non-ASCII survive folding (Col carries the raw name).
    df = pd.DataFrame({"col with space": [1, 2], "größe": [3, 4]})
    src = st.as_data_op(df)
    out = src.assign(**{"new col": src["col with space"] * 2,
                        "größe²": src["größe"] ** 2})
    result = st._api.evaluate(out)
    assert [2, 4] == list(result["new col"])
    assert [9, 16] == list(result["größe²"])


def test_nan_propagation_evaluates(polars):
    # NaN flows through a folded arithmetic entry (polars may surface it as null).
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    src = st.as_data_op(df)
    result = st._api.evaluate(src.assign(y=src["a"] * 2 + 1))
    values = list(result["y"])
    assert 3.0 == values[0] and 7.0 == values[2]
    assert values[1] is None or np.isnan(values[1])


def test_external_data_op_operand_evaluates(polars):
    # A value fed from another data-op stays a graph input (OperandLeaf) but the
    # arithmetic around it still folds and runs on both backends.
    df = pd.DataFrame({"x": [1, 2, 3]})
    src = st.as_data_op(df)
    factor = st.as_data_op(3)
    out = src.assign(scaled=src["x"] * factor)
    map_op = _one(unittest.TestCase(), optimize(out, OptConfig(dataframe_ops=True)),
                  AssignMapOp)
    assert 2 == len(map_op.inputs)  # [src, factor]
    result = st._api.evaluate(out)
    assert [3, 6, 9] == list(result["scaled"])


def test_direct_external_scalar_operand_evaluates(polars):
    df = pd.DataFrame({"x": [1, 2, 3]})
    src = st.as_data_op(df)
    marker = st.as_data_op(7)
    result = st._api.evaluate(src.assign(marker=marker))
    assert [7, 7, 7] == list(result["marker"])


def test_filter_then_assign_pipeline_evaluates(polars):
    # Mask fold + assign fold chained: the pipeline collapses to three ops.
    df = pd.DataFrame({
        "qty": [1.0, 2.0, 3.0, 4.0, 5.0],
        "price": [10.0, 20.0, 30.0, 40.0, 50.0],
        "ts": ["2020-01-05", "2020-02-10", "2020-03-15",
               "2020-04-20", "2020-05-25"],
        "tag": ["a1", "bb", "c1", "d1", "ee"],
    })
    src = st.as_data_op(df)
    f = src[src["tag"].str.count("1") > 0]
    feat = f.assign(total=f["qty"] * f["price"],
                    month=f["ts"].skb.apply_func(pd.to_datetime).dt.month)

    ops = optimize(feat, OptConfig(dataframe_ops=True))
    assert 3 == len(ops)  # source -> selection -> assign map
    from stratum.optimizer.ir._selection_ops import SelectionOp
    for cls in (SelectionOp, AssignMapOp):
        assert 1 == len([o for o in ops if isinstance(o, cls)]), cls.__name__

    result = st._api.evaluate(feat)
    assert [10.0, 90.0, 160.0] == list(result["total"])
    assert [1, 3, 4] == list(result["month"])
    assert ["a1", "c1", "d1"] == list(result["tag"])


def test_wide_assign_evaluates(polars):
    # Many entries in one map (mix of shared source columns and constants).
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    src = st.as_data_op(df)
    kwargs = {f"c{i}": src["x"] * i + src["y"] for i in range(8)}
    kwargs["marker"] = -1
    out = src.assign(**kwargs)
    ops = optimize(out, OptConfig(dataframe_ops=True))
    assert 2 == len(ops)
    result = st._api.evaluate(out)
    for i in range(8):
        assert [1 * i + 3, 2 * i + 4] == list(result[f"c{i}"]), f"c{i}"
    assert [-1, -1] == list(result["marker"])


if __name__ == "__main__":
    unittest.main()
