import operator
import unittest

import pytest
import numpy as np
import pandas as pd
import polars as pl
import skrub

import stratum as st
from stratum.optimizer._optimize import OptConfig
from stratum.optimizer.logical._map_ops import (
    AccessorMapOp, ApplyUDFOp, AssignMapOp, AssignOp, CallMapOp,
    DatetimeConversionOp, StringMethodOp, make_datetime_conversion_op,
    make_frame_get_attr, make_string_method_op, polars_datetime_kwargs)
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp
from stratum.optimizer.logical._column_expr import (
    BinOpExpr, Col, ColumnMethodExpr, Const, DatetimeExpr, DtExpr, OperandLeaf,
    StrExpr, _Folder)
from stratum.optimizer.logical._column_methods import (
    ColumnMethodOp, is_supported_column_method)
from stratum.optimizer.logical._dataframe_ops import extract_dataframe_op
from stratum.optimizer.logical._selection_ops import SelectionOp
from stratum.optimizer.logical._ops import (
    BinOp, CallOp, GetAttrOp, GetItemOp, MethodCallOp, Op, OperandRef, OutputType,
    UnaryOp)
from .test_dataframe_ops import (
    PolarsTestCase, _inp, optimize, run_op, force_polars, make_map_op)


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
        for cls in (GetItemOp, BinOp, DatetimeConversionOp, AccessorMapOp):
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

    def test_fillna_astype_chain_folds(self):
        src = st.as_data_op(self.df)
        out = src.assign(c=src["c1"].fillna(0).astype("float32"))
        map_op = _one(self, optimize(out, OptConfig(dataframe_ops=True)), AssignMapOp)
        self.assertEqual(
            ColumnMethodExpr(
                ColumnMethodExpr(Col("c1"), "fillna", (0,)),
                "astype",
                ("float32",),
            ),
            map_op.entries["c"],
        )
        self.assertEqual(1, len(map_op.inputs))

    def test_unsupported_astype_options_stay_as_leaf(self):
        src = st.as_data_op(self.df)
        converted = src["c1"].astype("float32", errors="ignore")
        map_op = _one(
            self,
            optimize(src.assign(c=converted), OptConfig(dataframe_ops=True)),
            AssignMapOp,
        )
        self.assertIsInstance(map_op.entries["c"], OperandLeaf)

    def test_row_local_methods_fold_with_expression_arguments(self):
        src = st.as_data_op(self.df)
        c1 = src["c1"]
        out = src.assign(
            clipped=c1.clip(lower=1, upper=2),
            kept=c1.where(c1 > 1, 0),
            member=src["s"].isin(["a1", "missing"]),
            present=c1.notna(),
        )
        map_op = _one(self, optimize(out, OptConfig(dataframe_ops=True)), AssignMapOp)

        self.assertEqual(1, len(map_op.inputs))
        self.assertEqual(
            ColumnMethodExpr(Col("c1"), "clip", kwargs={"lower": 1, "upper": 2}),
            map_op.entries["clipped"],
        )
        self.assertEqual(
            ColumnMethodExpr(
                Col("c1"),
                "where",
                (BinOpExpr(operator.gt, Col("c1"), Const(1)), 0),
            ),
            map_op.entries["kept"],
        )
        self.assertEqual(
            ColumnMethodExpr(Col("s"), "isin", (["a1", "missing"],)),
            map_op.entries["member"],
        )
        self.assertEqual(
            ColumnMethodExpr(Col("c1"), "notna"),
            map_op.entries["present"],
        )

    def test_unsupported_fillna_options_stay_as_leaf(self):
        src = st.as_data_op(self.df)
        filled = src["c1"].fillna(0, limit=1)
        map_op = _one(
            self,
            optimize(src.assign(c=filled), OptConfig(dataframe_ops=True)),
            AssignMapOp,
        )
        self.assertIsInstance(map_op.entries["c"], OperandLeaf)

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
        from stratum.optimizer.logical._ops import TransformerOp
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

    def test_different_scalar_types_do_not_dedup_after_cse(self):
        df = pd.DataFrame({"x": [1, 2]})
        data = st.as_data_op(df)
        root = data.assign(y=data["x"] * st.as_data_op(1)).skb.concat(
            [data.assign(y=data["x"] * st.as_data_op(1.0))], axis=0)
        ops = optimize(root, OptConfig(dataframe_ops=True))
        self.assertEqual(2, len([o for o in ops if isinstance(o, AssignMapOp)]))
        result = st._api.evaluate(root)
        self.assertEqual([1.0, 2.0, 1.0, 2.0], list(result["y"]))


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

    def test_method_expression_remaps_refs_in_arguments(self):
        expr = ColumnMethodExpr(
            OperandLeaf(OperandRef(1)),
            "where",
            (
                OperandLeaf(OperandRef(2)),
                OperandLeaf(OperandRef(3)),
            ),
        )
        self.assertEqual([1, 2, 3], [ref.k for ref in expr.iter_operand_refs()])
        self.assertEqual(
            ColumnMethodExpr(
                OperandLeaf(OperandRef(4)),
                "where",
                (
                    OperandLeaf(OperandRef(5)),
                    OperandLeaf(OperandRef(6)),
                ),
            ),
            expr.remap_operand_refs({1: 4, 2: 5, 3: 6}),
        )


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


def test_scalar_comparison_methods_fold_and_evaluate(polars):
    df = pd.DataFrame({"age": [2, 5, 8], "cat": ["A", "B", "A"]})
    src = st.as_data_op(df)
    out = src.assign(recent=src["age"].le(5), category=src["cat"].eq("A"))
    maps = [op for op in optimize(out) if isinstance(op, AssignMapOp)]
    assert len(maps) == 1
    assert all(isinstance(expr, BinOpExpr) for expr in maps[0].entries.values())
    result = st._api.evaluate(out)
    assert list(result["recent"]) == [True, True, False]
    assert list(result["category"]) == [True, False, True]


def test_series_comparison_method_keeps_pandas_alignment():
    df = pd.DataFrame({"a": [1, 2], "b": [1, 3]})
    src = st.as_data_op(df)
    ops = optimize(src.assign(same=src["a"].eq(src["b"])))
    assert any(isinstance(op, MethodCallOp) and op.method_name == "eq"
               for op in ops)


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


# FIXME: Fix this test
#@pytest.mark.xfail(
#    strict=True,
#    reason="TODO: DatetimeExpr.to_polars only supports string operands")
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


def test_external_scalar_data_op_operand_folds_to_const(polars):
    # A scalar fed from another data-op arrives as a ValueOp input, and is inlined
    # into the expression as a Const -- the map takes only the source frame.
    df = pd.DataFrame({"x": [1, 2, 3]})
    src = st.as_data_op(df)
    factor = st.as_data_op(3)
    out = src.assign(scaled=src["x"] * factor)
    map_op = _one(unittest.TestCase(), optimize(out, OptConfig(dataframe_ops=True)),
                  AssignMapOp)
    assert 1 == len(map_op.inputs)  # [src]
    assert BinOpExpr(operator.mul, Col("x"), Const(3)) == map_op.entries["scaled"]
    result = st._api.evaluate(out)
    assert [3, 6, 9] == list(result["scaled"])


def test_external_container_data_op_operand_stays_leaf(polars):
    # A container value is not inlined (Const compiles to pl.lit(), which would make
    # a list one list-valued cell): it stays a graph input. The Polars leaf
    # converts it to a Series before evaluating the surrounding expression.
    df = pd.DataFrame({"x": [1, 2, 3]})
    src = st.as_data_op(df)
    factor = st.as_data_op([3, 3, 3])
    out = src.assign(scaled=src["x"] * factor)
    map_op = _one(unittest.TestCase(), optimize(out, OptConfig(dataframe_ops=True)),
                  AssignMapOp)
    assert 2 == len(map_op.inputs)  # [src, factor]
    with st.config(implementation_selector="greedy" if polars else "default"):
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
    from stratum.optimizer.logical._selection_ops import SelectionOp
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


def test_ttt_shaped_wide_fillna_astype_collapses_to_one_map():
    """The dominant TTT pattern is one map, not three ops per column."""
    columns = [f"direct_{i}" for i in range(32)]
    df = pd.DataFrame({name: [1.0, np.nan] for name in columns})
    src = st.as_data_op(df)
    out = src.assign(
        **{name: src[name].fillna(0.0).astype("float32") for name in columns}
    )

    ops = optimize(out, OptConfig(dataframe_ops=True))
    map_op = _one(unittest.TestCase(), ops, AssignMapOp)

    assert 2 == len(ops)  # source + one map (formerly 98 logical nodes)
    assert 32 == len(map_op.entries)
    assert 1 == len(map_op.inputs)  # no per-column OperandLeaf inputs


def test_fillna_astype_pipeline_evaluates(polars):
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    src = st.as_data_op(df)
    out = src.assign(x=src["x"].fillna(0.0).astype("float32"))

    ops = optimize(out, OptConfig(dataframe_ops=True))
    assert not any(isinstance(op, ColumnMethodOp) for op in ops)

    result = st._api.evaluate(out)
    np.testing.assert_array_equal(
        np.asarray(list(result["x"]), dtype=np.float32),
        np.asarray([1.0, 0.0, 3.0], dtype=np.float32),
    )
    if isinstance(result, pl.DataFrame):
        assert pl.Float32 == result.schema["x"]
    else:
        assert np.dtype("float32") == result["x"].dtype


def test_folded_fillna_astype_executes_on_polars():
    expr = ColumnMethodExpr(
        ColumnMethodExpr(Col("x"), "fillna", (0.0,)),
        "astype",
        ("float32",),
    )
    op = AssignMapOp(entries={"x": expr})
    with force_polars():
        result = run_op(op, pl.DataFrame({"x": [1.0, float("nan"), None]}))
    assert [1.0, 0.0, 0.0] == result["x"].to_list()
    assert pl.Float32 == result.schema["x"]


def test_standalone_fillna_handles_polars_nan_and_null():
    series = pl.Series("x", [1.0, float("nan"), None])
    op = ColumnMethodOp(method="fillna", args=(0.0,))
    with force_polars():
        result = run_op(op, series)
    assert [1.0, 0.0, 0.0] == result.to_list()


@pytest.mark.parametrize("use_polars", [False, True], ids=["pandas", "polars"])
def test_row_local_method_pipeline_evaluates(use_polars):
    frame = pd.DataFrame({
        "x": [0.0, 2.0, np.nan],
        "s": ["yes", "no", "yes"],
    })
    src = st.as_data_op(frame)
    x = src["x"]
    out = src.assign(
        clipped=x.clip(lower=1.0, upper=1.5),
        kept=x.where(x > 1.0, 0.0),
        member=src["s"].isin(["yes"]),
        present=x.notna(),
    )
    map_op = _one(
        unittest.TestCase(),
        optimize(out, OptConfig(dataframe_ops=True)),
        AssignMapOp,
    )

    input_frame = pl.from_pandas(frame) if use_polars else frame
    logical_map = AssignMapOp(entries=map_op.entries)
    with force_polars(use_polars):
        result = run_op(logical_map, input_frame)

    np.testing.assert_allclose(
        np.asarray(list(result["clipped"]), dtype=float),
        np.asarray([1.0, 1.5, np.nan]),
        equal_nan=True,
    )
    assert [0.0, 2.0, 0.0] == list(result["kept"])
    assert [True, False, True] == list(result["member"])
    assert [True, True, False] == list(result["present"])


def test_notna_is_foldable_in_selection_predicate():
    frame = pd.DataFrame({"x": [1.0, np.nan, 2.0]})
    src = st.as_data_op(frame)
    out = src[src["x"].notna()]
    ops = optimize(out, OptConfig(dataframe_ops=True))
    from stratum.optimizer.logical._selection_ops import SelectionOp

    selection = _one(unittest.TestCase(), ops, SelectionOp)
    assert ColumnMethodExpr(Col("x"), "notna") == selection.predicate
    assert 2 == len(ops)


def _series_input():
    src = Op()
    src.output_type = OutputType.SERIES
    return src


def _wire(op, *inputs):
    op.inputs = list(inputs)
    for in_op in inputs:
        in_op.outputs = [op]
    return op


class TestMissingValueMethods(unittest.TestCase):
    """isna/isnull/notna/notnull, as methods or as pd.isna(x), are column methods."""

    def setUp(self):
        self.df = pd.DataFrame({"a": [1.0, np.nan]})

    def test_method_extracts(self):
        src = st.as_data_op(self.df)
        for method in ("isna", "isnull", "notna", "notnull"):
            with self.subTest(method=method):
                ops = optimize(getattr(src["a"], method)(),
                               OptConfig(dataframe_ops=True))
                method_op = _one(self, ops, ColumnMethodOp)
                self.assertEqual(method, method_op.method)
                self.assertIs(OutputType.SERIES, method_op.output_type)
                self.assertFalse(any(isinstance(op, MethodCallOp) for op in ops))

    def test_function_extracts(self):
        # pd.isnull / pd.notnull are the very same objects as pd.isna / pd.notna.
        src = st.as_data_op(self.df)
        for func, method in ((pd.isna, "isna"), (pd.isnull, "isna"),
                             (pd.notna, "notna"), (pd.notnull, "notna")):
            with self.subTest(func=func.__name__):
                ops = optimize(src["a"].skb.apply_func(func),
                               OptConfig(dataframe_ops=True))
                method_op = _one(self, ops, ColumnMethodOp)
                self.assertEqual(method, method_op.method)
                self.assertIs(OutputType.SERIES, method_op.output_type)
                self.assertFalse(any(isinstance(op, CallOp) for op in ops))

    def test_function_accepts_obj_keyword(self):
        src = _series_input()
        call = _wire(CallOp(func=pd.notna, args=(), kwargs={"obj": OperandRef(0)}), src)

        root, changed = extract_dataframe_op(call, call)

        self.assertTrue(changed)
        self.assertIsInstance(root, ColumnMethodOp)
        self.assertEqual("notna", root.method)
        self.assertEqual((), root.args)
        self.assertEqual({}, root.kwargs)
        self.assertEqual([src], root.inputs)
        self.assertEqual([root], src.outputs)

    def test_method_rejects_args_and_kwargs(self):
        for args, kwargs in (((1,), {}), ((), {"foo": 1})):
            with self.subTest(args=args, kwargs=kwargs):
                call = _wire(MethodCallOp("isna", args=args, kwargs=kwargs),
                             _series_input())
                self.assertFalse(is_supported_column_method(call))
                self.assertFalse(extract_dataframe_op(call, call)[1])

    def test_function_rejects_unsupported_calls(self):
        a, b = _series_input(), _series_input()
        frame = _series_input()
        frame.output_type = OutputType.FRAME
        cases = {
            # pd.isna(1) is a plain bool, not a column method.
            "scalar": _wire(CallOp(func=pd.isna, args=(1,), kwargs={})),
            "two operands": _wire(
                CallOp(func=pd.isna, args=(OperandRef(0), OperandRef(1)), kwargs={}),
                a, b),
            "positional and obj": _wire(
                CallOp(func=pd.isna, args=(OperandRef(0),), kwargs={"obj": OperandRef(0)}),
                _series_input()),
            "no operand": _wire(CallOp(func=pd.isna, args=(), kwargs={}),
                                _series_input()),
            # Frame-level isna has no validate_frame yet.
            "frame": _wire(CallOp(func=pd.isna, args=(OperandRef(0),), kwargs={}),
                           frame),
        }
        for name, call in cases.items():
            with self.subTest(name):
                root, changed = extract_dataframe_op(call, call)
                self.assertFalse(changed)
                self.assertIs(call, root)

    def test_folds_into_selection(self):
        src = st.as_data_op(self.df)
        for out, expected in (
                (src[src["a"].isnull()], ColumnMethodExpr(Col("a"), "isnull")),
                (src[src["a"].skb.apply_func(pd.notna)],
                 ColumnMethodExpr(Col("a"), "notna"))):
            with self.subTest(expected=expected):
                ops = optimize(out, OptConfig(dataframe_ops=True))
                selection = _one(self, ops, SelectionOp)
                self.assertEqual(expected, selection.predicate)
                self.assertEqual([], [op for op in ops if isinstance(op, ColumnMethodOp)])

    def test_folds_into_assign(self):
        src = st.as_data_op(self.df)
        out = src.assign(missing=src["a"].isnull(),
                         present=src["a"].skb.apply_func(pd.notnull))
        ops = optimize(out, OptConfig(dataframe_ops=True))
        map_op = _one(self, ops, AssignMapOp)
        self.assertEqual({"missing": ColumnMethodExpr(Col("a"), "isnull"),
                          "present": ColumnMethodExpr(Col("a"), "notna")},
                         map_op.entries)
        self.assertEqual([], [op for op in ops if isinstance(op, ColumnMethodOp)])

    def test_standalone_polars_handles_nan_and_null(self):
        series = pl.Series("a", [1.0, float("nan"), None])
        for method, expected in (("isna", [False, True, True]),
                                 ("isnull", [False, True, True]),
                                 ("notna", [True, False, False]),
                                 ("notnull", [True, False, False])):
            with self.subTest(method=method), force_polars():
                self.assertEqual(expected, run_op(ColumnMethodOp(method), series).to_list())

    def test_standalone_polars_on_non_float(self):
        series = pl.Series("s", ["x", None])
        with force_polars():
            self.assertEqual([False, True], run_op(ColumnMethodOp("isna"), series).to_list())


_MISSING = pd.DataFrame({"a": [0.0, 1.0, np.nan, 4.0], "b": [0.0, 2.0, 1.0, None],
                         "s": ["x", None, "z", "w"]})


def _evaluate_on_polars(plan):
    """Run ``plan`` end to end, where greedy picks the native polars impls."""
    with st.config(scheduler=True, implementation_selector="greedy"):
        result = st._api.evaluate(plan)
    assert isinstance(result, (pl.Series, pl.DataFrame))
    return result.to_pandas()


_FOLDED_NAN_GAP = pytest.mark.xfail(
    strict=True, reason="FIXME(#216): a folded operand has no dtype, so polars "
                        "skips the NaN half")


def _cases():
    cases = {
        "isna": lambda f, call: f["a"].isna(),
        "isnull": lambda f, call: f["b"].isnull(),
        "notna": lambda f, call: f["s"].notna(),
        "notnull": lambda f, call: f["s"].notnull(),
        "pd.isna": lambda f, call: call(f["a"], pd.isna),
        "pd.notnull": lambda f, call: call(f["s"], pd.notnull),
        # 0 / 0 is a NaN, not a null, in polars too: is_null alone would miss it.
        "computed-isna": lambda f, call: (f["a"] / f["b"]).isna(),
        "computed-notnull": lambda f, call: (f["a"] / f["b"]).notnull(),
    }
    for path in ("standalone", "assign", "selection"):
        for name, build in cases.items():
            gap = name.startswith("computed") and path != "standalone"
            yield pytest.param(build, path, id=f"{path}-{name}",
                               marks=[_FOLDED_NAN_GAP] if gap else [])


@pytest.mark.parametrize("build,path", _cases())
def test_missing_value_methods_match_pandas(build, path):
    expected_mask = build(_MISSING, lambda x, func: func(x))
    expected = {"standalone": expected_mask,
                "assign": _MISSING.assign(m=expected_mask),
                "selection": _MISSING[expected_mask]}[path]
    with skrub.config_context(eager_data_ops=False):
        src = skrub.var("frame", _MISSING)
        mask = build(src, lambda x, func: x.skb.apply_func(func))
        plan = {"standalone": mask, "assign": src.assign(m=mask),
                "selection": src[mask]}[path]
    result = _evaluate_on_polars(plan)
    if path == "standalone":
        assert expected.tolist() == result.tolist()
    else:
        pd.testing.assert_frame_equal(expected.reset_index(drop=True),
                                      result.reset_index(drop=True),
                                      check_dtype=False)


if __name__ == "__main__":
    unittest.main()


class TestMapRewrites(unittest.TestCase):
    """`optimize` rewrites column-modifying calls produced by skrub DAGs into maps."""

    def setUp(self):
        self.df = pd.DataFrame({
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "datetime": ["2025-11-01 10:00:00",
                         "2025-11-02 15:30:00",
                         "2025-11-03 09:45:00"],
        })

    @unittest.skip("Skipping this test for now")
    def test_projection_fused_get_item(self):
        data = st.as_data_op(self.df)["x"].apply(lambda x: x + 1)
        ops = optimize(data)
        self.assertEqual(2, len(ops))
        self.assertIsInstance(ops[1], ApplyUDFOp)

    def test_projection_fused_get_item_with_choice(self):
        data = st.as_data_op(self.df)["x"]
        sub_dag1 = data.apply(lambda x, a: x + a, a=st.as_data_op(1))
        sub_dag2 = data
        root = st.choose_from([sub_dag1, sub_dag2]).as_data_op()
        ops = optimize(root, OptConfig(literal_inlining=False))
        self.assertEqual(5, len(ops))
        self.assertIsInstance(ops[1], ColumnProjectionOp)
        self.assertIsInstance(ops[3], ApplyUDFOp)

    def test_fused_get_attr(self):
        data = st.as_data_op(self.df)[["datetime"]].apply(
            pd.to_datetime, format='%Y-%m-%d %H:%M:%S')
        data = data.assign(year=data["datetime"].dt.year,
                           month=data["datetime"].dt.month)
        data = data.copy()
        ops = optimize(data)
        # The column getitem and both fused .dt accessors fold into the assign map.
        self.assertEqual(5, len(ops))
        map_op = next(o for o in ops if isinstance(o, AssignMapOp))
        self.assertEqual({"year": DtExpr(Col("datetime"), "year"),
                          "month": DtExpr(Col("datetime"), "month")},
                         map_op.entries)
        self.assertIsInstance(ops[-1], MethodCallOp)  # the trailing .copy()

    def test_supported_series_method_gets_typed_op(self):
        data = st.as_data_op(self.df)["x"].astype("float32")
        ops = optimize(data, OptConfig(dataframe_ops=True))
        method = next(op for op in ops if isinstance(op, ColumnMethodOp))
        self.assertEqual("astype", method.method)
        self.assertIs(OutputType.SERIES, method.output_type)
        self.assertFalse(any(isinstance(op, MethodCallOp) for op in ops))


class TestIndexAttribute(unittest.TestCase):
    """`.index` is a sequence of labels, so it types as a SERIES."""

    def _get_attr(self, attr, container_kind):
        container = _inp(None)
        container.output_type = container_kind
        op = GetAttrOp(attr_name=attr)
        op.inputs = [container]
        container.outputs = [op]
        return make_frame_get_attr(None, op)

    def test_index_is_a_series_whatever_it_is_read_off(self):
        from stratum.optimizer.logical._index_ops import IndexAccessOp
        for kind in (OutputType.SERIES, OutputType.FRAME):
            with self.subTest(container=kind.name):
                op = self._get_attr("index", kind)
                self.assertIsInstance(op, IndexAccessOp)
                self.assertIs(OutputType.SERIES, op.output_type)

    def test_other_accessors_still_keep_the_container_kind(self):
        self.assertIs(OutputType.FRAME,
                      self._get_attr("dt", OutputType.FRAME).output_type)

    def test_polars_refuses_index(self):
        # polars frames have no index, and what `.index` means depends on what
        # produced the value, so there is no per-op translation.
        from stratum.optimizer.physical._map_execs import (
            PolarsAccessorMapOp)
        self.assertFalse(PolarsAccessorMapOp.supports(
            AccessorMapOp(attr_name=["index"]), None))
        self.assertTrue(PolarsAccessorMapOp.supports(
            AccessorMapOp(attr_name=["dt", "year"]), None))

    def test_arbitrary_pandas_index_is_preserved(self):
        from stratum.optimizer.logical._index_ops import IndexAccessOp
        from stratum.optimizer.physical._index_execs import PandasIndexAccessOp
        frame = pd.DataFrame({"a": [1, 2]}, index=pd.Index(["x", "y"], name="row"))
        ops = optimize(st.as_data_op(frame).index, OptConfig(dataframe_ops=True))
        self.assertTrue(any(isinstance(op, IndexAccessOp) for op in ops))
        self.assertTrue(any(isinstance(op, PandasIndexAccessOp) for op in ops))
        from stratum.runtime._buffer_pool import BufferPool
        pool = BufferPool()
        for op in ops:
            pool.put(op, op.process("fit_transform", [pool.pin(inp) for inp in op.inputs]))
        pd.testing.assert_index_equal(pool.pin(ops[-1]), frame.index)


class TestCallMapOp(unittest.TestCase):
    def test_func_and_method_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            CallMapOp(func=lambda x: x, method="drop", args=(), kwargs={})

    def test_kwargs_can_be_omitted(self):
        op = CallMapOp(method="copy", args=(), kwargs=None)
        self.assertIsNone(op.kwargs)

    def test_no_func_no_method_raises(self):
        with self.assertRaises(TypeError):
            run_op(CallMapOp(args=(), kwargs={}), pd.DataFrame({"a": [1]}))

    def test_func_path(self):
        op = CallMapOp(func=lambda df, v: df * v,
                          args=(OperandRef(0), 2), kwargs={})
        result = run_op(op, pd.DataFrame({"a": [1, 2]}))
        self.assertEqual([2, 4], result["a"].tolist())

    def test_method_pandas_path(self):
        op = CallMapOp(method="drop", args=("y",), kwargs={"axis": 1})
        result = run_op(op, pd.DataFrame({"x": [1, 2], "y": [3, 4]}))
        self.assertNotIn("y", result.columns)

    def test_method_polars_raises(self):
        with force_polars():
            op = CallMapOp(method="drop", args=(), kwargs={})
            with self.assertRaises(ValueError):
                run_op(op, pl.DataFrame({"a": [1]}))


class TestApplyUDFOp(unittest.TestCase):
    def test_pandas_single_column_str(self):
        op = ApplyUDFOp(args=(lambda x: x * 10,), kwargs={}, columns="a")
        result = run_op(op, pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        self.assertEqual([10, 20], result.tolist())

    def test_pandas_multi_column(self):
        op = ApplyUDFOp(args=(lambda x: x * 2,), kwargs={}, columns=["a", "b"])
        result = run_op(op, pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        self.assertEqual([2, 4], result["a"].tolist())

    def test_polars_sin_rewrite(self):
        with force_polars():
            op = ApplyUDFOp(args=(np.sin,), kwargs={})
            result = run_op(op, pl.Series("a", [0.0, np.pi / 2]))
            self.assertAlmostEqual(1.0, result[1], places=5)

    def test_polars_cos_rewrite(self):
        with force_polars():
            op = ApplyUDFOp(args=(np.cos,), kwargs={})
            result = run_op(op, pl.Series("a", [0.0]))
            self.assertAlmostEqual(1.0, result[0], places=5)

    def test_polars_single_col_general_func(self):
        with force_polars():
            op = ApplyUDFOp(args=(lambda x: x + 1,), kwargs={})
            result = run_op(op, pl.Series("a", [1, 2, 3]))
            self.assertEqual([2, 3, 4], result.to_list())

    def test_polars_multi_col_map_rows(self):
        with force_polars():
            op = ApplyUDFOp(args=(lambda row: (row[0] + row[1],),),
                            kwargs={}, columns=["a", "b"])
            result = run_op(op, pl.DataFrame({"a": [1, 2], "b": [3, 4]}))
            self.assertIsNotNone(result)


class TestAssignOpPolars(PolarsTestCase):
    def test_polars_series(self):
        op = AssignOp(args=(), kwargs={"b": pl.Series([10, 20])})
        result = run_op(op, pl.DataFrame({"a": [1, 2]}))
        self.assertIn("b", result.columns)

    def test_pandas_series_converted_to_polars(self):
        op = AssignOp(args=(), kwargs={"b": pd.Series([10, 20])})
        result = run_op(op, pl.DataFrame({"a": [1, 2]}))
        self.assertIn("b", result.columns)

    def test_placeholder_raises(self):
        # An OperandRef surviving into a polars assign kwarg is unsupported.
        op = AssignOp(args=(), kwargs={"b": OperandRef(1)})
        with self.assertRaises(NotImplementedError):
            run_op(op, pl.DataFrame({"a": [1, 2]}), OperandRef(1))


class TestDatetimeConversionOp(unittest.TestCase):
    def test_polars_path(self):
        with force_polars():
            op = DatetimeConversionOp(args=(), kwargs={})
            result = run_op(op, pl.Series("dt", ["2025-01-01", "2025-06-15"]))
            self.assertEqual(pl.Datetime, result.dtype)

    def test_pandas_path_preserves_kwargs(self):
        op = DatetimeConversionOp(args=(), kwargs={"dayfirst": True})
        result = run_op(op, pd.Series(["01/02/2020"]))
        self.assertEqual(pd.Timestamp("2020-02-01"), result.iloc[0])

    def test_polars_unsupported_kwargs_use_pandas_semantics(self):
        with force_polars():
            op = DatetimeConversionOp(args=(), kwargs={"dayfirst": True})
            result = run_op(op, pl.Series("dt", ["01/02/2020"]))
        self.assertEqual([pd.Timestamp("2020-02-01")], result.to_list())

    def test_polars_positional_options_use_pandas_semantics(self):
        with force_polars():
            op = DatetimeConversionOp(args=("coerce",), kwargs={})
            result = run_op(op, pl.Series("dt", ["2025-01-01", "bad"]))
        self.assertEqual([pd.Timestamp("2025-01-01"), None], result.to_list())

    def test_polars_fallback_converts_datetime_index(self):
        with force_polars():
            op = DatetimeConversionOp(args=(), kwargs={"dayfirst": True})
            result = run_op(op, ["01/02/2020", "03/04/2021"])
        self.assertEqual(
            [pd.Timestamp("2020-02-01"), pd.Timestamp("2021-04-03")],
            result.to_list())

    def test_polars_fallback_keeps_scalar_result(self):
        with force_polars():
            op = DatetimeConversionOp(args=(), kwargs={"dayfirst": True})
            result = run_op(op, "01/02/2020")
        self.assertEqual(pd.Timestamp("2020-02-01"), result)

    def _convert_with_option(self, polars, values, args, kwargs, option):
        # A plan-fed option is an OperandRef into the op's inputs: here inputs[1].
        series = pl.Series("dt", values) if polars else pd.Series(values)
        with force_polars(polars):
            op = DatetimeConversionOp(args=args, kwargs=kwargs)
            return list(run_op(op, series, option))

    def test_resolves_graph_fed_kwarg(self):
        # On polars the kwargs translation must see the resolved format.
        for polars in (False, True):
            with self.subTest(polars=polars):
                result = self._convert_with_option(
                    polars, ["2025-01-06"], (), {"format": OperandRef(1)}, "%Y-%m-%d")
                self.assertEqual([pd.Timestamp("2025-01-06")], result)

    def test_resolves_graph_fed_positional(self):
        for polars in (False, True):
            with self.subTest(polars=polars):
                result = self._convert_with_option(
                    polars, ["2025-01-06", "bad"], (OperandRef(1),), {}, "coerce")
                self.assertEqual(pd.Timestamp("2025-01-06"), result[0])
                self.assertTrue(pd.isna(result[1]))

    def test_polars_datetime_kwargs_rejects_unsupported_errors(self):
        self.assertIsNone(
            polars_datetime_kwargs((), {"errors": "ignore"}))


class TestAccessorMapOp(unittest.TestCase):
    def test_init_with_none(self):
        self.assertEqual([], AccessorMapOp(attr_name=None).attr_name)

    def test_init_with_str(self):
        self.assertEqual(["dt"], AccessorMapOp(attr_name="dt").attr_name)

    def _run_polars(self, dt_values, attr_name):
        with force_polars():
            s = pl.Series("dt", pd.to_datetime(dt_values))
            op = AccessorMapOp(attr_name=attr_name)
            return run_op(op, s)

    def test_polars_year(self):
        result = self._run_polars(["2025-01-15", "2025-06-20"], ["dt", "year"])
        self.assertEqual([2025, 2025], result.to_list())

    def test_polars_matches_pandas_values(self):
        values = _DATES["d"]
        for attr in _DT_ATTRS_PANDAS_SEMANTICS:
            with self.subTest(attr=attr):
                result = self._run_polars(values, ["dt", attr])
                self.assertEqual(getattr(values.dt, attr).tolist(), result.to_list())

    def test_polars_is_month_end(self):
        result = self._run_polars(["2025-01-31", "2025-01-15"],
                                  ["dt", "is_month_end"])
        self.assertEqual([True, False], result.to_list())

    def test_polars_single_accessor_returns_namespace(self):
        result = self._run_polars(["2025-01-15"], ["dt"])
        self.assertEqual([2025], result.year().to_list())

    def test_shared_accessor_keeps_original_edge_when_fused(self):
        source = Op()
        source.output_type = OutputType.SERIES
        accessor = AccessorMapOp(
            attr_name=["dt"], inputs=[source], outputs=[])
        current = GetAttrOp(attr_name="year")
        current.inputs = [accessor]
        other_consumer = Op(inputs=[accessor])
        accessor.outputs = [current, other_consumer]
        source.outputs = [accessor]

        new_op = make_frame_get_attr(None, current)

        self.assertEqual([other_consumer], accessor.outputs)
        self.assertIn(accessor, source.outputs)
        self.assertIn(new_op, source.outputs)


_DATES = pd.DataFrame({
    "d": pd.to_datetime(["2025-01-06 00:00:00.000001001",  # a Monday
                         "2025-01-12 12:00:00.123456789"]),  # a Sunday
    "s": ["2025-01-06", "2025-01-12"]})

# The `.dt` attributes where polars disagrees with pandas: it counts weekdays
# from Monday=1 and nanoseconds past the second (pandas: Monday=0, past the
# microsecond), and it lacks some of pandas' alias spellings.
_DT_ATTRS_PANDAS_SEMANTICS = ("dayofweek", "day_of_week", "weekday", "nanosecond",
                              "day_of_year", "daysinmonth")


def _run_plan(dag, env, polars):
    """Plan ``dag`` against one backend, run it and return a pandas result."""
    from stratum.runtime._buffer_pool import BufferPool
    with force_polars(polars):
        ops = optimize(dag, env=env)
    pool = BufferPool()
    for op in ops:
        pool.put(op, op.process("fit_transform", [pool.pin(inp) for inp in op.inputs]))
    result = pool.pin(ops[-1])
    # Every op here has a native polars impl, so a pandas result would mean a
    # fallback hid the polars kernel.
    assert isinstance(result, (pl.Series, pl.DataFrame)) == polars
    return result.to_pandas() if polars else result


@pytest.mark.parametrize("polars", [False, True], ids=["pandas", "polars"])
@pytest.mark.parametrize("path", ["accessor", "assign", "selection"])
@pytest.mark.parametrize("attr", _DT_ATTRS_PANDAS_SEMANTICS)
def test_dt_attribute_matches_pandas(attr, path, polars):
    # The standalone accessor runs PolarsAccessorMapOp; assign and mask
    # selection fold the read into a DtExpr.
    values = getattr(_DATES["d"].dt, attr)
    expected = {"accessor": values, "assign": _DATES.assign(v=values),
                "selection": _DATES[values == values.iloc[-1]]}[path]
    with skrub.config_context(eager_data_ops=False):
        src = skrub.var("x", _DATES)
        read = getattr(src["d"].dt, attr)
        dag = {"accessor": read, "assign": src.assign(v=read),
               "selection": src[read == values.iloc[-1]]}[path]
    result = _run_plan(dag, {"x": _DATES}, polars)
    if path == "accessor":
        assert expected.tolist() == result.tolist()
    else:
        pd.testing.assert_frame_equal(expected.reset_index(drop=True),
                                      result.reset_index(drop=True),
                                      check_dtype=False)


@pytest.mark.parametrize("polars", [False, True], ids=["pandas", "polars"])
def test_to_datetime_resolves_graph_fed_format(polars):
    with skrub.config_context(eager_data_ops=False):
        src = skrub.var("x", _DATES)
        fmt = skrub.deferred(lambda: "%Y-%m-%d")()
        dag = skrub.deferred(pd.to_datetime)(src["s"], format=fmt)
    result = _run_plan(dag, {"x": _DATES}, polars)
    assert pd.to_datetime(_DATES["s"]).tolist() == result.tolist()


class TestStringMethodOp(unittest.TestCase):
    """`col.str.<method>(...)` fuses the .str accessor and the call into one op."""

    def setUp(self):
        self.df = pd.DataFrame({"s": ["a1", "bb", "c1"], "x": [1, 2, 3]})

    def _one(self, ops, cls):
        found = [o for o in ops if isinstance(o, cls)]
        self.assertEqual(1, len(found), f"expected exactly one {cls.__name__}")
        return found[0]

    def test_str_method_fuses_accessor_away(self):
        # A str call used as a column projection (here assigned) becomes a single
        # StringMethodOp; the AccessorMapOp(["str"]) accessor drops out.
        # Map folding is disabled so the intermediate op stays observable (with it
        # on, the assign absorbs the call into a StrExpr entry -- see test_map_ops).
        data = st.as_data_op(self.df)
        with make_map_op(False):
            ops = optimize(data.assign(c=data["s"].str.upper()),
                           OptConfig(dataframe_ops=True))
        sm = self._one(ops, StringMethodOp)
        self.assertEqual("upper", sm.method)
        self.assertIs(OutputType.SERIES, sm.output_type)
        self.assertIsInstance(sm.inputs[0], ColumnProjectionOp)  # the column, not the accessor
        self.assertEqual([], [o for o in ops if isinstance(o, AccessorMapOp)])

    def test_shared_accessor_detached_after_last_call(self):
        # When the same `.str` accessor feeds two calls, each folds to its own
        # StringMethodOp and the accessor is removed once its last consumer is fused.
        data = st.as_data_op(self.df)
        acc = data["s"].str
        with make_map_op(False):
            ops = optimize(data.assign(a=acc.count("1"), b=acc.upper()),
                           OptConfig(dataframe_ops=True))
        self.assertEqual(2, len([o for o in ops if isinstance(o, StringMethodOp)]))
        self.assertEqual([], [o for o in ops if isinstance(o, AccessorMapOp)])

    def test_process_pandas(self):
        op = StringMethodOp(method="count", args=("1",))
        result = run_op(op, pd.Series(["a1", "bb", "c1"]))
        self.assertEqual([1, 0, 1], result.tolist())

    def test_process_polars_renames_method(self):
        # polars renames .str.count -> .str.count_matches (STR_POLARS_METHODS).
        with force_polars():
            op = StringMethodOp(method="count", args=("1",))
            result = run_op(op, pl.Series(["a1", "bb", "c1"]))
        self.assertEqual([1, 0, 1], result.to_list())

    def test_graph_argument_is_rewired_to_fused_method(self):
        column = Op()
        column.output_type = OutputType.SERIES
        accessor = AccessorMapOp(
            attr_name=["str"], inputs=[column], outputs=[])
        argument = Op()
        call = MethodCallOp(
            method_name="contains", args=(OperandRef(1),), kwargs={})
        call.inputs = [accessor, argument]
        accessor.outputs = [call]
        argument.outputs = [call]
        column.outputs = [accessor]

        new_op = make_string_method_op(call)

        self.assertIn(new_op, argument.outputs)
        self.assertNotIn(call, argument.outputs)


class TestMakeDatetimeConversionOp(unittest.TestCase):
    def test_extra_positional_args(self):
        op = CallOp(func=pd.to_datetime,
                    args=(OperandRef(0), "ISO8601"), kwargs={})
        new_op = make_datetime_conversion_op(op)
        self.assertEqual(("ISO8601",), tuple(new_op.args))


if __name__ == "__main__":
    unittest.main()
