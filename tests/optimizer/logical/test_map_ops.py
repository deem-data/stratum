import operator
import unittest

import pytest
import numpy as np
import pandas as pd
import polars as pl
import skrub

import stratum as st
from stratum.frontend import _api
from stratum.optimizer._optimize import OptConfig
from stratum.optimizer.logical._map_ops import (
    ApplyUDFOp, AttributeReadOp, CallMapOp, DatetimeConversionOp, DatetimeFieldOp,
    make_datetime_conversion_op, make_frame_get_attr)
from stratum.optimizer.logical._combine_ops import AssignOp
from stratum.optimizer.logical._map_ops import ExpressionMapOp
from stratum.runtime._scheduler import SequentialScheduler
from stratum.optimizer.logical._projection_ops import ColumnProjectionOp
from stratum.optimizer.logical._column_expr import (
    AstypeExpr, BinOpExpr, ClipExpr, Col, Const, DatetimeExpr, DtExpr, FillnaExpr,
    IsinExpr, IsnaExpr, MapValuesExpr, NotnaExpr, OperandLeaf, StrExpr, WhereExpr)
from stratum.optimizer.logical._expr_folding import _Folder
from stratum.optimizer.logical._dataframe_ops import (
    extract_dataframe_op, make_column_method_op, make_string_method_op)
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


class TestMapExprRefContract(unittest.TestCase):
    """New expr nodes honour the ref-traversal contract used by CSE/validation."""

    def test_iter_refs_through_dt_chain(self):
        expr = DtExpr(DatetimeExpr(OperandLeaf(OperandRef(2))), "day")
        self.assertEqual([2], [r.k for r in expr.iter_operand_refs()])

    def test_remap_rebuilds_dt_chain(self):
        expr = DtExpr(DatetimeExpr(OperandLeaf(OperandRef(2)), format="%Y"), "day")
        self.assertEqual(
            DtExpr(DatetimeExpr(OperandLeaf(OperandRef(1)), format="%Y"), "day"),
            expr.remap_operand_refs({2: 1}))

    def test_pure_dt_chain_has_no_refs(self):
        expr = DtExpr(DatetimeExpr(Col("d")), "month")
        self.assertEqual([], list(expr.iter_operand_refs()))
        self.assertEqual(expr, expr.remap_operand_refs({}))

    def test_method_expression_remaps_refs_in_arguments(self):
        expr = WhereExpr(*(OperandLeaf(OperandRef(k)) for k in (1, 2, 3)))
        self.assertEqual([1, 2, 3], [ref.k for ref in expr.iter_operand_refs()])
        self.assertEqual(WhereExpr(*(OperandLeaf(OperandRef(k)) for k in (4, 5, 6))),
                         expr.remap_operand_refs({1: 4, 2: 5, 3: 6}))


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


"""End-to-end: explicit maps and assignments execute through the full plan."""

@pytest.fixture(params=[False, True], ids=["pandas", "polars"])
def polars(request):
    with force_polars(request.param):
        yield request.param


def test_assign_pipeline_evaluates(polars):
    df = pd.DataFrame({"c1": [1, 2], "c2": ["2020-01-01", "2020-02-03"]})
    src = st.as_data_op(df)
    date_col = src["c2"].skb.apply_func(pd.to_datetime)
    out = src.assign(c3=src["c1"] + 123, c4=date_col.dt.day, c5=date_col.dt.day)
    result = _api.evaluate(out)
    assert [124, 125] == list(result["c3"])
    assert [1, 3] == list(result["c4"])
    assert [1, 3] == list(result["c5"])


def test_scalar_comparison_methods_fold_and_evaluate(polars):
    df = pd.DataFrame({"age": [2, 5, 8], "cat": ["A", "B", "A"]})
    src = st.as_data_op(df)
    out = src.assign(recent=src["age"].le(5), category=src["cat"].eq("A"))
    maps = [op for op in optimize(out) if isinstance(op, ExpressionMapOp)]
    assert len(maps) == 2
    assert all(isinstance(op.expr, BinOpExpr) for op in maps)
    result = _api.evaluate(out)
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
    # every projection reads the frame produced by the previous assignment.
    df = pd.DataFrame({"x": [1, 2, 3]})
    src = st.as_data_op(df)
    d1 = src.assign(x2=src["x"] * 2)
    d2 = d1.assign(x4=d1["x2"] * 2)
    d3 = d2.assign(x8=d2["x4"] * 2)
    ops = optimize(d3, OptConfig(dataframe_ops=True))
    assert 3 == len([o for o in ops if isinstance(o, AssignOp)])
    assert 3 == len([o for o in ops if isinstance(o, ExpressionMapOp)])
    assert 3 == len([o for o in ops if isinstance(o, ColumnProjectionOp)])
    result = _api.evaluate(d3)
    assert [2, 4, 6] == list(result["x2"])
    assert [4, 8, 12] == list(result["x4"])
    assert [8, 16, 24] == list(result["x8"])


def test_deep_arithmetic_expression_evaluates(polars):
    # A deep native tree: nested binops, unary negation, pow / floordiv / mod.
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    src = st.as_data_op(df)
    a, b = src["a"], src["b"]
    out = src.assign(y=-((a + b) ** 2) / 3 + (a * b) % 5 + b // a)
    ops = optimize(out, OptConfig(dataframe_ops=True))
    assert 1 == len([op for op in ops if isinstance(op, AssignOp)])
    assert len([op for op in ops if isinstance(op, ExpressionMapOp)]) > 1
    result = _api.evaluate(out)
    expected = -((df["a"] + df["b"]) ** 2) / 3 + (df["a"] * df["b"]) % 5 + df["b"] // df["a"]
    np.testing.assert_allclose(expected.to_numpy(),
                               np.asarray(list(result["y"]), dtype=float), rtol=1e-9)


def test_boolean_logic_expression_evaluates(polars):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 4.0, 5.0]})
    src = st.as_data_op(df)
    out = src.assign(flag=~((src["a"] > 2) | (src["b"] < 4)))
    result = _api.evaluate(out)
    assert [False, True, False] == list(result["flag"])


def test_chained_string_methods_evaluate(polars):
    # StrExpr over StrExpr: each .str call fuses, then the whole chain folds.
    # polars renames every method (strip_chars / to_uppercase / len_chars).
    df = pd.DataFrame({"s": ["  ab  ", " c ", "def"]})
    src = st.as_data_op(df)
    out = src.assign(up=src["s"].str.strip().str.upper(),
                     n=src["s"].str.strip().str.len() * 10)
    ops = optimize(out, OptConfig(dataframe_ops=True))
    assert 1 == len([op for op in ops if isinstance(op, AssignOp)])
    result = _api.evaluate(out)
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
    assert 1 == len([op for op in ops if isinstance(op, AssignOp)])
    result = _api.evaluate(out)
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
    result = _api.evaluate(src.assign(parsed=parsed))
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
        _api.evaluate(src.assign(parsed=parsed))


def test_weird_column_names_evaluate(polars):
    # Names with spaces and non-ASCII survive folding (Col carries the raw name).
    df = pd.DataFrame({"col with space": [1, 2], "größe": [3, 4]})
    src = st.as_data_op(df)
    out = src.assign(**{"new col": src["col with space"] * 2,
                        "größe²": src["größe"] ** 2})
    result = _api.evaluate(out)
    assert [2, 4] == list(result["new col"])
    assert [9, 16] == list(result["größe²"])


def test_external_scalar_data_op_operand_folds_to_const(polars):
    # A scalar fed from another data-op arrives as a ValueOp input, and is inlined
    # into the expression as a Const; the map takes the projected column.
    df = pd.DataFrame({"x": [1, 2, 3]})
    src = st.as_data_op(df)
    factor = st.as_data_op(3)
    out = src.assign(scaled=src["x"] * factor)
    map_op = _one(unittest.TestCase(), optimize(out, OptConfig(dataframe_ops=True)),
                  ExpressionMapOp)
    assert 1 == len(map_op.inputs)
    assert isinstance(map_op.inputs[0], ColumnProjectionOp)
    assert BinOpExpr(operator.mul, OperandLeaf(OperandRef(0)), Const(3)) == map_op.expr
    result = _api.evaluate(out)
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
                  ExpressionMapOp)
    assert 2 == len(map_op.inputs)  # projected column and external operand
    with st.config(implementation_selector="greedy" if polars else "default"):
        result = _api.evaluate(out)
    assert [3, 6, 9] == list(result["scaled"])


def test_direct_external_scalar_operand_evaluates(polars):
    df = pd.DataFrame({"x": [1, 2, 3]})
    src = st.as_data_op(df)
    marker = st.as_data_op(7)
    result = _api.evaluate(src.assign(marker=marker))
    assert [7, 7, 7] == list(result["marker"])


def test_filter_then_assign_pipeline_evaluates(polars):
    # The mask remains folded; assigned computations remain explicit.
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
    assert any(isinstance(op, ExpressionMapOp) for op in ops)
    from stratum.optimizer.logical._selection_ops import SelectionOp
    for cls in (SelectionOp, AssignOp):
        assert 1 == len([o for o in ops if isinstance(o, cls)]), cls.__name__

    result = _api.evaluate(feat)
    assert [10.0, 90.0, 160.0] == list(result["total"])
    assert [1, 3, 4] == list(result["month"])
    assert ["a1", "c1", "d1"] == list(result["tag"])


def test_wide_assign_evaluates(polars):
    # Multiple maps share source projections; assignment also accepts literals.
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    src = st.as_data_op(df)
    kwargs = {f"c{i}": src["x"] * i + src["y"] for i in range(8)}
    kwargs["marker"] = -1
    out = src.assign(**kwargs)
    ops = optimize(out, OptConfig(dataframe_ops=True))
    assert 1 == len([op for op in ops if isinstance(op, AssignOp)])
    result = _api.evaluate(out)
    for i in range(8):
        assert [1 * i + 3, 2 * i + 4] == list(result[f"c{i}"]), f"c{i}"
    assert [-1, -1] == list(result["marker"])


def test_fillna_astype_pipeline_evaluates(polars):
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    src = st.as_data_op(df)
    out = src.assign(x=src["x"].fillna(0.0).astype("float32"))

    ops = optimize(out, OptConfig(dataframe_ops=True))
    assert 2 == len([op for op in ops if isinstance(op, ExpressionMapOp)])

    result = _api.evaluate(out)
    np.testing.assert_array_equal(
        np.asarray(list(result["x"]), dtype=np.float32),
        np.asarray([1.0, 0.0, 3.0], dtype=np.float32),
    )
    if isinstance(result, pl.DataFrame):
        assert pl.Float32 == result.schema["x"]
    else:
        assert np.dtype("float32") == result["x"].dtype


def test_expression_fillna_astype_executes_on_polars():
    from stratum.optimizer.physical._map_execs import PolarsExpressionMapOp
    expr = AstypeExpr(FillnaExpr(OperandLeaf(OperandRef(0)), 0.0), "float32")
    op = PolarsExpressionMapOp(expr)
    source = Op()
    source.output_schema = pl.Schema({"x": pl.Float64})
    op.inputs = [source]
    result = op.process([pl.Series("x", [1.0, float("nan"), None])])
    assert [1.0, 0.0, 0.0] == result.to_list()
    assert pl.Float32 == result.dtype


def test_standalone_fillna_handles_polars_nan_and_null():
    series = pl.Series("x", [1.0, float("nan"), None])
    op = ExpressionMapOp(FillnaExpr(OperandLeaf(OperandRef(0)), 0.0))
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
    with force_polars(use_polars):
        ops = optimize(out, OptConfig(dataframe_ops=True))
    result = SequentialScheduler(ops, None).evaluate()

    np.testing.assert_allclose(
        np.asarray(list(result["clipped"]), dtype=float),
        np.asarray([1.0, 1.5, np.nan]),
        equal_nan=True,
    )
    assert [0.0, 2.0, 0.0] == list(result["kept"])
    assert [True, False, True] == list(result["member"])
    assert [True, True, False] == list(result["present"])


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
        for method, cls in (("isna", IsnaExpr), ("isnull", IsnaExpr),
                            ("notna", NotnaExpr), ("notnull", NotnaExpr)):
            with self.subTest(method=method):
                ops = optimize(getattr(src["a"], method)(),
                               OptConfig(dataframe_ops=True))
                method_op = _one(self, ops, ExpressionMapOp)
                self.assertIsInstance(method_op.expr, cls)
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
                method_op = _one(self, ops, ExpressionMapOp)
                self.assertEqual(method, method_op.expr.method)
                self.assertIs(OutputType.SERIES, method_op.output_type)
                self.assertFalse(any(isinstance(op, CallOp) for op in ops))

    def test_function_accepts_obj_keyword(self):
        src = _series_input()
        call = _wire(CallOp(func=pd.notna, args=(), kwargs={"obj": OperandRef(0)}), src)

        root, changed = extract_dataframe_op(call, call)

        self.assertTrue(changed)
        self.assertIsInstance(root, ExpressionMapOp)
        self.assertIsInstance(root.expr, NotnaExpr)
        self.assertEqual([src], root.inputs)
        self.assertEqual([root], src.outputs)

    def test_method_rejects_args_and_kwargs(self):
        for args, kwargs in (((1,), {}), ((), {"foo": 1})):
            with self.subTest(args=args, kwargs=kwargs):
                call = _wire(MethodCallOp("isna", args=args, kwargs=kwargs),
                             _series_input())
                self.assertIsNone(make_column_method_op(call))
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
                (src[src["a"].isnull()], IsnaExpr(Col("a"))),
                (src[src["a"].skb.apply_func(pd.notna)], NotnaExpr(Col("a")))):
            with self.subTest(expected=expected):
                ops = optimize(out, OptConfig(dataframe_ops=True))
                selection = _one(self, ops, SelectionOp)
                self.assertEqual(expected, selection.predicate)
                self.assertEqual([], [op for op in ops if isinstance(op, ExpressionMapOp)])

    def test_assign_reads_missing_value_maps(self):
        src = st.as_data_op(self.df)
        out = src.assign(missing=src["a"].isnull(),
                         present=src["a"].skb.apply_func(pd.notnull))
        ops = optimize(out, OptConfig(dataframe_ops=True))
        map_op = _one(self, ops, AssignOp)
        self.assertIsInstance(map_op.inputs[map_op.kwargs["missing"].k].expr, IsnaExpr)
        self.assertIsInstance(map_op.inputs[map_op.kwargs["present"].k].expr, NotnaExpr)

    def test_standalone_polars_handles_nan_and_null(self):
        series = pl.Series("a", [1.0, float("nan"), None])
        for cls, expected in ((IsnaExpr, [False, True, True]),
                              (NotnaExpr, [True, False, False])):
            with self.subTest(cls=cls.__name__), force_polars():
                op = ExpressionMapOp(cls(OperandLeaf(OperandRef(0))))
                self.assertEqual(expected, run_op(op, series).to_list())

    def test_standalone_polars_on_non_float(self):
        series = pl.Series("s", ["x", None])
        with force_polars():
            op = ExpressionMapOp(IsnaExpr(OperandLeaf(OperandRef(0))))
            self.assertEqual([False, True], run_op(op, series).to_list())


class TestColumnMethodFields(unittest.TestCase):
    """Each admitted column method parses into its corresponding expression."""

    def _parse(self, method, *args, output_type=OutputType.SERIES, **kwargs):
        src = _series_input()
        src.output_type = output_type
        return make_column_method_op(_wire(MethodCallOp(method, args=args, kwargs=kwargs), src))

    def test_spellings_parse_to_the_same_fields(self):
        cases = [
            (("astype", "float32"), ("astype",), {"dtype": "float32"}, AstypeExpr, ("float32",)),
            (("fillna", 0), ("fillna",), {"value": 0}, FillnaExpr, (0,)),
            (("clip", 1, 2), ("clip",), {"lower": 1, "upper": 2}, ClipExpr, (1, 2)),
            (("where", OperandRef(0), 0), ("where", OperandRef(0)), {"other": 0},
             WhereExpr, (OperandLeaf(OperandRef(0)), 0)),
            (("isin", [1]), ("isin",), {"values": [1]}, IsinExpr, ([1],)),
            (("map", {1: 2}), ("map",), {"arg": {1: 2}}, MapValuesExpr, ({1: 2}, None)),
        ]
        for positional, keyword_args, keywords, cls, fields in cases:
            for op in (self._parse(*positional), self._parse(*keyword_args, **keywords)):
                with self.subTest(method=positional[0]):
                    self.assertIsInstance(op, ExpressionMapOp)
                    self.assertIsInstance(op.expr, cls)
                    self.assertEqual(OperandLeaf(OperandRef(0)), op.expr.operand)
                    self.assertEqual(fields, tuple(getattr(op.expr, f) for f in cls.__slots__[1:]))
                    self.assertIs(OutputType.SERIES, op.output_type)

    def test_a_frame_admits_astype_and_a_per_column_fillna(self):
        frame = OutputType.FRAME
        self.assertEqual("int32", self._parse("astype", "int32", output_type=frame).expr.dtype)
        self.assertEqual({"a": 0}, self._parse("fillna", {"a": 0}, output_type=frame).expr.value)
        for method, args in (("fillna", (0,)), ("clip", (1,)), ("isna", ()),
                             ("isin", ([1],)), ("map", ({1: 2},))):
            with self.subTest(method=method):
                self.assertIsNone(self._parse(method, *args, output_type=frame))

    def test_unmodelled_calls_stay_opaque(self):
        cases = [
            ("astype", ("float32",), {"errors": "ignore"}),
            ("astype", ("not a dtype",), {}),
            ("fillna", (0,), {"limit": 1}),
            ("fillna", ([0, 1],), {}),
            ("clip", (), {}),
            ("clip", (1,), {"axis": 0}),
            ("where", (OperandRef(0),), {}),  # pandas' implicit NaN
            ("isin", ("ab",), {}),
            ("map", (lambda v: v,), {}),
            ("map", ({1: 2},), {"na_action": "all"}),
            ("map", ({1: "a", 2: 3},), {}),  # mixed value types
            ("isna", (1,), {}),
        ]
        for method, args, kwargs in cases:
            with self.subTest(method=method, args=args, kwargs=kwargs):
                self.assertIsNone(self._parse(method, *args, **kwargs))


@pytest.mark.parametrize("expr, values, expected", [
    pytest.param(AstypeExpr(OperandLeaf(OperandRef(0)), "float32"), [1, 2], [1.0, 2.0], id="astype"),
    pytest.param(FillnaExpr(OperandLeaf(OperandRef(0)), 0.0), [1.0, None], [1.0, 0.0], id="fillna"),
    pytest.param(ClipExpr(OperandLeaf(OperandRef(0)), lower=1, upper=2), [0, 3], [1, 2], id="clip"),
    pytest.param(IsinExpr(OperandLeaf(OperandRef(0)), [1]), [1, 2], [True, False], id="isin"),
    pytest.param(IsnaExpr(OperandLeaf(OperandRef(0))), [1.0, None], [False, True], id="isna"),
    pytest.param(NotnaExpr(OperandLeaf(OperandRef(0))), [1.0, None], [True, False], id="notna"),
    pytest.param(MapValuesExpr(OperandLeaf(OperandRef(0)), {1: "a"}), [1, 2], ["a", None], id="map"),
])
def test_column_methods_run_on_both_backends(polars, expr, values, expected):
    series = pl.Series("x", values) if polars else pd.Series(values)
    result = run_op(ExpressionMapOp(expr), series)
    assert expected == [None if pd.isna(v) else v for v in result]


def test_where_reads_its_condition_from_the_graph(polars):
    values = [1, 2, 3]
    series = pl.Series("x", values) if polars else pd.Series(values)
    cond = pl.Series([True, False, True]) if polars else pd.Series([True, False, True])
    expr = WhereExpr(OperandLeaf(OperandRef(0)), OperandLeaf(OperandRef(1)), 0)
    assert [1, 0, 3] == list(run_op(ExpressionMapOp(expr), series, cond))


_MISSING = pd.DataFrame({"a": [0.0, 1.0, np.nan, 4.0], "b": [0.0, 2.0, 1.0, None],
                         "s": ["x", None, "z", "w"]})


def _evaluate_greedy(plan, *, unnamed=False):
    """Use native kernels where they preserve the result, including its name."""
    with st.config(scheduler=True, implementation_selector="greedy"):
        result = _api.evaluate(plan)
    if unnamed:
        assert isinstance(result, pd.Series) and result.name is None
        return result
    assert isinstance(result, (pl.Series, pl.DataFrame))
    return result.to_pandas()


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
            yield pytest.param(build, path, id=f"{path}-{name}")


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
    result = _evaluate_greedy(plan, unnamed=path == "standalone" and expected.name is None)
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
        assign = next(o for o in ops if isinstance(o, AssignOp))
        self.assertEqual({"year", "month"}, set(assign.kwargs))
        self.assertTrue(all(isinstance(assign.inputs[ref.k], DatetimeFieldOp)
                            for ref in assign.kwargs.values()))
        self.assertIsInstance(ops[-1], MethodCallOp)  # the trailing .copy()

    def test_supported_series_method_gets_typed_op(self):
        data = st.as_data_op(self.df)["x"].astype("float32")
        ops = optimize(data, OptConfig(dataframe_ops=True))
        method = _one(self, ops, ExpressionMapOp)
        self.assertEqual("astype", method.expr.method)
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
        return make_frame_get_attr(op)

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
            pool.put(op, op.process([pool.pin(inp) for inp in op.inputs]))
        pd.testing.assert_index_equal(pool.pin(ops[-1]), frame.index)


class TestCallMapOp(unittest.TestCase):
    def test_kwargs_can_be_omitted(self):
        op = CallMapOp(args=())
        self.assertIsNone(op.kwargs)


class TestApplyUDFOp(unittest.TestCase):
    def test_pandas_series(self):
        op = ApplyUDFOp(args=(lambda x: x * 10,), kwargs={})
        result = run_op(op, pd.Series([1, 2]))
        self.assertEqual([10, 20], result.tolist())

    def test_pandas_multi_column(self):
        op = ApplyUDFOp(args=(lambda x: x * 2,), kwargs={})
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
            op = ApplyUDFOp(args=(lambda row: (row[0] + row[1],),), kwargs={})
            result = run_op(op, pl.DataFrame({"a": [1, 2], "b": [3, 4]}))
            self.assertIsNotNone(result)


class TestAssignOpPolars(PolarsTestCase):
    def test_polars_series(self):
        op = AssignOp(args=(), kwargs={"b": pl.Series([10, 20])})
        result = run_op(op, pl.DataFrame({"a": [1, 2]}))
        self.assertIn("b", result.columns)

    def test_literal_pandas_series_requires_alignment(self):
        from stratum.optimizer.physical._combine_execs import PolarsAssignOp
        op = AssignOp(kwargs={"b": pd.Series([10, 20], index=[1, 0])})
        self.assertFalse(PolarsAssignOp.supports(op, None))


class TestDatetimeConversionOp(unittest.TestCase):
    def test_polars_path(self):
        with force_polars():
            op = DatetimeConversionOp()
            result = run_op(op, pl.Series("dt", ["2025-01-01", "2025-06-15"]))
            self.assertEqual(pl.Datetime, result.dtype)

    def test_fields_on_both_backends(self):
        cases = [
            (dict(format="%d/%m/%Y"), ["01/02/2020"], [pd.Timestamp("2020-02-01")]),
            (dict(errors="coerce"), ["2025-01-01", "bad"], [pd.Timestamp("2025-01-01"), None]),
            # polars' non-exact match fails on a pattern that covers only part
            # of the string, where pandas parses it; this one agrees.
            (dict(format="%Y-%m-%d", exact=False), ["x2025-03-07y"], [pd.Timestamp("2025-03-07")]),
            (dict(cache=False), ["2025-01-01"], [pd.Timestamp("2025-01-01")]),
        ]
        for fields, values, expected in cases:
            for polars in (False, True):
                with self.subTest(fields=fields, polars=polars), force_polars(polars):
                    series = pl.Series("dt", values) if polars else pd.Series(values)
                    result = run_op(DatetimeConversionOp(**fields), series)
                    self.assertEqual(expected, [None if pd.isna(v) else v for v in result])

    def test_errors_is_raise_or_coerce(self):
        with self.assertRaises(ValueError):
            DatetimeConversionOp(errors="ignore")

    def test_graph_fed_fields_resolve(self):
        cases = [(dict(format=OperandRef(1)), "%Y-%m-%d", ["2025-01-06"]),
                 (dict(errors=OperandRef(1)), "coerce", ["2025-01-06", "bad"])]
        for fields, option, values in cases:
            for polars in (False, True):
                with self.subTest(fields=fields, polars=polars), force_polars(polars):
                    series = pl.Series("dt", values) if polars else pd.Series(values)
                    result = list(run_op(DatetimeConversionOp(**fields), series, option))
                    self.assertEqual(pd.Timestamp("2025-01-06"), result[0])
                    self.assertTrue(all(pd.isna(v) for v in result[1:]))


class TestDatetimeFieldOp(unittest.TestCase):
    def _run_polars(self, dt_values, field):
        with force_polars():
            s = pl.Series("dt", pd.to_datetime(dt_values))
            return run_op(DatetimeFieldOp(field=field), s)

    def test_polars_year(self):
        result = self._run_polars(["2025-01-15", "2025-06-20"], "year")
        self.assertEqual([2025, 2025], result.to_list())

    def test_polars_matches_pandas_values(self):
        values = _DATES["d"]
        for field in _DT_ATTRS_PANDAS_SEMANTICS:
            with self.subTest(field=field):
                result = self._run_polars(values, field)
                self.assertEqual(getattr(values.dt, field).tolist(), result.to_list())

    def test_polars_is_month_end(self):
        result = self._run_polars(["2025-01-31", "2025-01-15"], "is_month_end")
        self.assertEqual([True, False], result.to_list())

    def test_polars_refuses_untranslated_field(self):
        from stratum.optimizer.physical._map_execs import PolarsDatetimeFieldOp
        for _, field in _DT_ATTRS_PANDAS_ONLY:
            with self.subTest(field=field):
                op = DatetimeFieldOp(field=field)
                self.assertFalse(PolarsDatetimeFieldOp.supports(op, None))
                self.assertIn(f"dt.{field}", op._unsupported_reason)
        for field in ("dayofweek", "is_month_end"):
            with self.subTest(field=field):
                self.assertTrue(PolarsDatetimeFieldOp.supports(
                    DatetimeFieldOp(field=field), None))

    def test_only_dt_fields_become_datetime_field_ops(self):
        # Extraction sees any attribute read; everything but a `.dt` field (a
        # bare namespace, a `.dt` method read without its call, a read off a
        # field) stays an opaque attribute read with a pandas impl only.
        cases = {"dt.year": ("dt", "year"), "dt": ("dt",), "dt.round": ("dt", "round"),
                 "dt.year.values": ("dt", "year", "values"), "iloc": ("iloc",),
                 "T": ("T",), "cat.codes": ("cat", "codes"), "str": ("str",)}
        for name, chain in cases.items():
            with self.subTest(name):
                op = _series_input()
                for attr in chain:
                    read = _wire(GetAttrOp(attr_name=attr), op)
                    op = make_frame_get_attr(read)
                if name == "dt.year":
                    self.assertIsInstance(op, DatetimeFieldOp)
                    self.assertEqual("year", op.field)
                elif name == "dt.year.values":
                    self.assertIsInstance(op, AttributeReadOp)
                    self.assertEqual(["values"], op.attr_name)
                else:
                    self.assertIsInstance(op, AttributeReadOp)
                    self.assertEqual(list(chain), op.attr_name)
                self.assertIs(OutputType.SERIES, op.output_type)

    def test_shared_accessor_keeps_original_edge_when_fused(self):
        source = Op()
        source.output_type = OutputType.SERIES
        accessor = AttributeReadOp(attr_name=["dt"], inputs=[source], outputs=[])
        current = GetAttrOp(attr_name="year")
        current.inputs = [accessor]
        other_consumer = Op(inputs=[accessor])
        accessor.outputs = [current, other_consumer]
        source.outputs = [accessor]

        new_op = make_frame_get_attr(current)

        self.assertIsInstance(new_op, DatetimeFieldOp)
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
        pool.put(op, op.process([pool.pin(inp) for inp in op.inputs]))
    result = pool.pin(ops[-1])
    # Every op here has a native polars impl, so a pandas result would mean a
    # fallback hid the polars kernel.
    assert isinstance(result, (pl.Series, pl.DataFrame)) == polars
    return result.to_pandas() if polars else result


@pytest.mark.parametrize("polars", [False, True], ids=["pandas", "polars"])
@pytest.mark.parametrize("path", ["accessor", "assign", "selection"])
@pytest.mark.parametrize("attr", _DT_ATTRS_PANDAS_SEMANTICS)
def test_dt_attribute_matches_pandas(attr, path, polars):
    # The standalone accessor runs PolarsDatetimeFieldOp; assign and mask
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


# The `.dt` attributes polars has no method for: the period boundary flags
# (bar is_month_end), timetz, and the timedelta components, whose polars
# `total_*` counterparts are totals rather than components.
_PANDAS_ONLY_DT = pd.DataFrame({
    "d": pd.to_datetime(["2025-01-01 00:00", "2025-03-31 12:00", "2025-06-15 00:00",
                         "2025-12-31 00:00"]),
    "td": pd.to_timedelta(["1 days 02:03:04.000005006", "00:00:01",
                           "-1 days +00:00:01", "3 days"])})
_DT_ATTRS_PANDAS_ONLY = (
    *(("d", attr) for attr in ("is_month_start", "is_quarter_start",
                               "is_quarter_end", "is_year_start", "is_year_end",
                               "timetz")),
    *(("td", attr) for attr in ("days", "seconds", "microseconds", "nanoseconds")))


def _pandas_only_dt_plan(column, attr, path):
    """A plan reading ``.dt.<attr>`` off ``column``, and its eager pandas result."""
    values = getattr(_PANDAS_ONLY_DT[column].dt, attr)
    expected = {"accessor": values, "assign": _PANDAS_ONLY_DT.assign(v=values),
                "selection": _PANDAS_ONLY_DT[values == values.iloc[-1]]}[path]
    with skrub.config_context(eager_data_ops=False):
        src = skrub.var("x", _PANDAS_ONLY_DT)
        read = getattr(src[column].dt, attr)
        dag = {"accessor": read, "assign": src.assign(v=read),
               "selection": src[read == values.iloc[-1]]}[path]
    return dag, expected


@pytest.mark.parametrize("path", ["accessor", "assign", "selection"])
@pytest.mark.parametrize("column,attr", _DT_ATTRS_PANDAS_ONLY)
def test_untranslated_dt_attribute_falls_back_to_pandas(column, attr, path):
    # Greedy prefers polars wherever it is supported, so the read only runs if
    # polars refuses it at plan time.
    dag, expected = _pandas_only_dt_plan(column, attr, path)
    with st.config(scheduler=True, implementation_selector="greedy"):
        result = _api.evaluate(dag)
    if isinstance(result, (pl.Series, pl.DataFrame)):
        result = result.to_pandas()
    if path == "accessor":
        assert expected.tolist() == result.tolist()
    else:
        pd.testing.assert_frame_equal(expected.reset_index(drop=True),
                                      result.reset_index(drop=True),
                                      check_dtype=False)


@pytest.mark.parametrize("path", ["accessor", "assign", "selection"])
@pytest.mark.parametrize("column,attr", _DT_ATTRS_PANDAS_ONLY)
def test_untranslated_dt_attribute_fails_plan_pinned_to_polars(column, attr, path):
    # A strict polars pin has no pandas to fall back to, so planning fails
    # instead of the read raising AttributeError at run time.
    dag, _ = _pandas_only_dt_plan(column, attr, path)
    with force_polars(), pytest.raises(NotImplementedError):
        optimize(dag, env={"x": _PANDAS_ONLY_DT})


# `.dt` method calls, which run pandas' method on the namespace verbatim. polars
# lacks most of them or reads the arguments differently: `round` wants "1h"
# where pandas takes the offset alias "h", and `strftime` formats `%f` as
# nanoseconds.
_DT_METHOD_CALLS = (
    ("round", ("h",)), ("floor", ("D",)), ("ceil", ("15min",)),
    ("strftime", ("%Y-%m-%d %H:%M:%S.%f",)), ("normalize", ()),
    ("month_name", ()), ("day_name", ()), ("tz_localize", ("UTC",)))


def _dt_method_plan(method, args):
    """A plan calling ``.dt.<method>(*args)``, and its eager pandas result."""
    expected = getattr(_DATES["d"].dt, method)(*args)
    with skrub.config_context(eager_data_ops=False):
        dag = getattr(skrub.var("x", _DATES)["d"].dt, method)(*args)
    return dag, expected


@pytest.mark.parametrize("method,args", _DT_METHOD_CALLS)
def test_dt_method_call_falls_back_to_pandas(method, args):
    # Greedy prefers polars wherever it is supported, so the call only matches
    # pandas if polars refuses the bare `.dt` namespace at plan time.
    dag, expected = _dt_method_plan(method, args)
    with st.config(scheduler=True, implementation_selector="greedy"):
        result = _api.evaluate(dag)
    if isinstance(result, pl.Series):
        result = result.to_pandas()
    assert expected.tolist() == result.tolist()


@pytest.mark.parametrize("method,args", _DT_METHOD_CALLS)
def test_dt_method_call_fails_plan_pinned_to_polars(method, args):
    dag, _ = _dt_method_plan(method, args)
    with force_polars(), pytest.raises(NotImplementedError):
        optimize(dag, env={"x": _DATES})


# Pandas attribute reads with no polars translation. Extraction turns every
# attribute read on frame-like data that is not a `.dt` field into an
# AttributeReadOp, which only pandas runs.
_MIXED = pd.DataFrame({"c": pd.Series(["a", "b", "a"], dtype="category"),
                       "s": ["xy", "zw", "uv"], "n": [1, 2, 3]})
_PANDAS_ONLY_ATTR_READS = {
    "cat.codes": lambda x: x["c"].cat.codes,
    "cat.categories": lambda x: x["c"].cat.categories,
    "cat.rename_categories": lambda x: x["c"].cat.rename_categories(["A", "B"]),
    "str[0]": lambda x: x["s"].str[0],
    "iloc": lambda x: x.iloc[0:2],
    "loc": lambda x: x.loc[:, ["n"]],
    "T": lambda x: x.T,
    "values": lambda x: x["n"].values,
}


def _pandas_only_attr_plan(name):
    """A plan for one of :data:`_PANDAS_ONLY_ATTR_READS`, and its eager pandas result."""
    read = _PANDAS_ONLY_ATTR_READS[name]
    with skrub.config_context(eager_data_ops=False):
        dag = read(skrub.var("x", _MIXED))
    return dag, read(_MIXED)


def _assert_same_pandas_value(expected, result):
    assert type(expected) is type(result)
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(expected, result)
    elif isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(expected, result)
    elif isinstance(expected, pd.Index):
        pd.testing.assert_index_equal(expected, result)
    else:
        np.testing.assert_array_equal(expected, result)


@pytest.mark.parametrize("name", _PANDAS_ONLY_ATTR_READS)
def test_pandas_only_attribute_runs_on_pandas(name):
    dag, expected = _pandas_only_attr_plan(name)
    _assert_same_pandas_value(expected, _run_plan(dag, {"x": _MIXED}, polars=False))


@pytest.mark.parametrize("name", _PANDAS_ONLY_ATTR_READS)
def test_pandas_only_attribute_fails_plan_pinned_to_polars(name):
    # A strict polars pin has no pandas to fall back to, so planning fails
    # instead of the read crashing at run time.
    dag, _ = _pandas_only_attr_plan(name)
    with force_polars(), pytest.raises(NotImplementedError,
                                       match="No implementation was bound"):
        optimize(dag, env={"x": _MIXED})


@pytest.mark.parametrize("name", _PANDAS_ONLY_ATTR_READS)
def test_pandas_only_attribute_falls_back_to_pandas(name):
    # Greedy prefers polars wherever it is supported, so the read only runs if
    # polars refuses it at plan time.
    dag, expected = _pandas_only_attr_plan(name)
    with st.config(scheduler=True, implementation_selector="greedy"):
        result = _api.evaluate(dag)
    _assert_same_pandas_value(expected, result)


@pytest.mark.parametrize("polars", [False, True], ids=["pandas", "polars"])
def test_to_datetime_resolves_graph_fed_format(polars):
    with skrub.config_context(eager_data_ops=False):
        src = skrub.var("x", _DATES)
        fmt = skrub.deferred(lambda: "%Y-%m-%d")()
        dag = skrub.deferred(pd.to_datetime)(src["s"], format=fmt)
    result = _run_plan(dag, {"x": _DATES}, polars)
    assert pd.to_datetime(_DATES["s"]).tolist() == result.tolist()


class TestStringExpressionMaps(unittest.TestCase):
    """`col.str.<method>(...)` fuses the .str accessor and the call into one op."""

    def setUp(self):
        self.df = pd.DataFrame({"s": ["a1", "bb", "c1"], "x": [1, 2, 3]})

    def _one(self, ops, cls):
        found = [o for o in ops if isinstance(o, cls)]
        self.assertEqual(1, len(found), f"expected exactly one {cls.__name__}")
        return found[0]

    def test_str_method_fuses_accessor_away(self):
        # String extraction also runs when arithmetic Map extraction is disabled.
        # The column feeds the Map directly; the accessor is removed.
        data = st.as_data_op(self.df)
        with make_map_op(False):
            ops = optimize(data.assign(c=data["s"].str.upper()),
                           OptConfig(dataframe_ops=True))
        sm = self._one(ops, ExpressionMapOp)
        self.assertIsInstance(sm.expr, StrExpr)
        self.assertEqual("upper", sm.expr.method)
        self.assertIs(OutputType.SERIES, sm.output_type)
        self.assertIsInstance(sm.inputs[0], ColumnProjectionOp)  # the column, not the accessor
        self.assertEqual([], [o for o in ops if isinstance(o, AttributeReadOp)])

    def test_shared_accessor_detached_after_last_call(self):
        # When the same `.str` accessor feeds two calls, each folds to its own
        # expression Map and the accessor is removed after its last consumer.
        data = st.as_data_op(self.df)
        acc = data["s"].str
        with make_map_op(False):
            ops = optimize(data.assign(a=acc.count("1"), b=acc.upper()),
                           OptConfig(dataframe_ops=True))
        self.assertEqual(2, len([o for o in ops if isinstance(o, ExpressionMapOp)
                                and isinstance(o.expr, StrExpr)]))
        self.assertEqual([], [o for o in ops if isinstance(o, AttributeReadOp)])

    def test_arguments_bind_to_pandas_parameter_names(self):
        data = st.as_data_op(self.df)
        for call in (data["s"].str.replace("1", "2", regex=False),
                     data["s"].str.replace(pat="1", repl="2", regex=False),
                     data["s"].str.replace("1", repl="2", regex=False)):
            with self.subTest(call=call), make_map_op(False):
                sm = self._one(optimize(call, OptConfig(dataframe_ops=True)), ExpressionMapOp)
                self.assertEqual({"pat": "1", "repl": "2", "regex": False}, sm.expr.params)
                self.assertEqual(["pat", "repl", "regex"], list(sm.expr.params))

    def test_calls_pandas_rejects_stay_opaque(self):
        column = Op()
        column.output_type = OutputType.SERIES
        for method, args, kwargs in (("upper", (1,), {}), ("lower", (), {"x": 1}),
                                     ("no_such_method", (), {})):
            with self.subTest(method=method):
                accessor = AttributeReadOp(attr_name=["str"], inputs=[column], outputs=[])
                call = _wire(MethodCallOp(method, args=args, kwargs=kwargs), accessor)
                self.assertIsNone(make_string_method_op(call))


class TestMakeDatetimeConversionOp(unittest.TestCase):
    def _call(self, *args, output_type=OutputType.SERIES, **kwargs):
        op = CallOp(func=pd.to_datetime, args=args, kwargs=kwargs)
        op.inputs = [_inp(None)]
        op.inputs[0].output_type = output_type
        op.inputs[0].outputs = [op]
        return op

    def test_positional_and_keyword_options_parse_to_the_same_fields(self):
        for op in (self._call(OperandRef(0), "coerce", format="ISO8601"),
                   self._call(OperandRef(0), errors="coerce", format="ISO8601")):
            new_op = make_datetime_conversion_op(op)
            self.assertEqual(("ISO8601", None, True, "coerce"),
                             (new_op.format, new_op.exact, new_op.cache, new_op.errors))
            self.assertIs(OutputType.SERIES, new_op.output_type)

    def test_a_graph_fed_option_is_a_field(self):
        op = self._call(OperandRef(0), format=OperandRef(1))
        op.inputs.append(_inp(None))
        op.inputs[1].outputs = [op]
        self.assertEqual(OperandRef(1), make_datetime_conversion_op(op).format)

    def test_unmodelled_calls_stay_opaque(self):
        cases = {
            "dayfirst": self._call(OperandRef(0), dayfirst=True),
            "errors=ignore": self._call(OperandRef(0), errors="ignore"),
            "assembly from a frame": self._call(OperandRef(0), output_type=OutputType.FRAME),
            "unknown keyword": self._call(OperandRef(0), foo=1),
        }
        for name, op in cases.items():
            with self.subTest(name):
                self.assertIsNone(make_datetime_conversion_op(op))


if __name__ == "__main__":
    unittest.main()
