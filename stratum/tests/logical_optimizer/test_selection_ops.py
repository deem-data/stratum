import operator
import unittest
from contextlib import contextmanager

import pytest
import pandas as pd
import polars as pl

import stratum as st
from stratum._config import FLAGS
from stratum.optimizer.ir._ops import remap_operand_refs
from stratum.optimizer._optimize import OptConfig
from stratum.optimizer.ir._dataframe_ops import SelectionKind, SelectionOp
from stratum.optimizer.ir._ops import BinOp, GetItemOp, UnaryOp, Op, OperandRef, OutputType
from stratum.optimizer.ir._column_expr import Col, Const, BinOpExpr, UnaryOpExpr, OperandLeaf, StrExpr
from stratum.optimizer.ir._source_ops import rechunk_pl_frame
from stratum.tests.logical_optimizer.test_dataframe_ops import (
    optimize, run_op, force_polars)


@contextmanager
def pandas_query(enabled=True):
    """Temporarily set `FLAGS.pandas_query`."""
    orig = FLAGS.pandas_query
    FLAGS.pandas_query = enabled
    try:
        yield
    finally:
        FLAGS.pandas_query = orig


class TestSelectionExtraction(unittest.TestCase):
    """Row-selection method calls are rewritten to SelectionOp."""

    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 2, None], "y": [4, 5, 5, 6]})

    def _one_selection(self, ops):
        sels = [o for o in ops if isinstance(o, SelectionOp)]
        self.assertEqual(1, len(sels), "expected exactly one SelectionOp")
        return sels[0]

    def test_dropna_converts_to_selection(self):
        sel = self._one_selection(
            optimize(st.as_data_op(self.df).dropna(), OptConfig(dataframe_ops=True)))
        self.assertIs(SelectionKind.DROPNA, sel.kind)
        self.assertIs(OutputType.FRAME, sel.output_type)

    def test_drop_duplicates_converts_to_selection(self):
        sel = self._one_selection(
            optimize(st.as_data_op(self.df).drop_duplicates(), OptConfig(dataframe_ops=True)))
        self.assertIs(SelectionKind.DROP_DUPLICATES, sel.kind)

    def test_head_converts_to_selection_with_args(self):
        sel = self._one_selection(
            optimize(st.as_data_op(self.df).head(2), OptConfig(dataframe_ops=True)))
        self.assertIs(SelectionKind.HEAD, sel.kind)
        self.assertEqual((2,), tuple(sel.args))

    def test_tail_converts_to_selection(self):
        sel = self._one_selection(
            optimize(st.as_data_op(self.df).tail(1), OptConfig(dataframe_ops=True)))
        self.assertIs(SelectionKind.TAIL, sel.kind)

    def test_sample_converts_to_selection(self):
        sel = self._one_selection(
            optimize(st.as_data_op(self.df).sample(n=1, random_state=0),
                     OptConfig(dataframe_ops=True)))
        self.assertIs(SelectionKind.SAMPLE, sel.kind)

    def test_selection_on_column_is_series(self):
        # df["x"].dropna() selects rows of a column -> the selection stays a SERIES.
        sel = self._one_selection(
            optimize(st.as_data_op(self.df)["x"].dropna(), OptConfig(dataframe_ops=True)))
        self.assertIs(SelectionKind.DROPNA, sel.kind)
        self.assertIs(OutputType.SERIES, sel.output_type)


class TestSelectionProcess(unittest.TestCase):
    """SelectionOp.process executes the underlying frame method per backend."""

    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 2, None], "y": [4, 5, 5, 6]})

    def test_dropna_pandas(self):
        op = SelectionOp(kind=SelectionKind.DROPNA)
        result = run_op(op, self.df)
        self.assertEqual(3, len(result))

    def test_head_pandas(self):
        op = SelectionOp(kind=SelectionKind.HEAD, args=(2,))
        result = run_op(op, self.df)
        self.assertEqual(2, len(result))

    def test_drop_duplicates_pandas(self):
        op = SelectionOp(kind=SelectionKind.DROP_DUPLICATES)
        result = run_op(op, self.df.dropna())
        self.assertEqual(2, len(result))

    def test_dropna_polars_uses_drop_nulls(self):
        with force_polars():
            op = SelectionOp(kind=SelectionKind.DROPNA)
            result = run_op(op, pl.DataFrame({"x": [1, None, 3]}))
        self.assertEqual(2, len(result))

    def test_query_kind_not_yet_executable(self):
        # QUERY is produced by a later pass and has no method/predicate yet.
        op = SelectionOp(kind=SelectionKind.QUERY)
        with self.assertRaises(NotImplementedError):
            run_op(op, self.df)

    def test_mask_predicate_pandas(self):
        op = SelectionOp(kind=SelectionKind.MASK,
                         predicate=BinOpExpr(operator.gt, Col("x"), Const(1)))
        result = run_op(op, pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))
        self.assertEqual([2, 3], result["x"].tolist())

    def test_mask_predicate_boolop_pandas(self):
        op = SelectionOp(kind=SelectionKind.MASK,
                         predicate=BinOpExpr(operator.and_,
                                          BinOpExpr(operator.gt, Col("x"), Const(1)),
                                          BinOpExpr(operator.lt, Col("y"), Const(6))))
        result = run_op(op, pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))
        self.assertEqual([2], result["x"].tolist())

    def test_mask_predicate_polars(self):
        op = SelectionOp(kind=SelectionKind.MASK,
                         predicate=BinOpExpr(operator.gt, Col("x"), Const(1)))
        with force_polars():
            result = run_op(op, pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))
        self.assertEqual([2, 3], result["x"].to_list())

    def test_mask_predicate_with_operand_leaf_pandas(self):
        # OperandLeaf reads an external input (here input 1) at eval time.
        op = SelectionOp(kind=SelectionKind.MASK,
                         predicate=BinOpExpr(operator.gt, Col("x"),
                                             OperandLeaf(OperandRef(1))))
        result = run_op(op, pd.DataFrame({"x": [1, 2, 3]}), 1)
        self.assertEqual([2, 3], result["x"].tolist())

    def test_mask_str_predicate_pandas(self):
        op = SelectionOp(kind=SelectionKind.MASK,
                         predicate=BinOpExpr(operator.gt,
                                             StrExpr(Col("s"), "count", ("1",)), Const(0)))
        result = run_op(op, pd.DataFrame({"s": ["a1", "bb", "c1"]}))
        self.assertEqual(["a1", "c1"], result["s"].tolist())

    def test_mask_str_predicate_polars(self):
        # polars renames .str.count -> .str.count_matches (STR_POLARS_METHODS).
        op = SelectionOp(kind=SelectionKind.MASK,
                         predicate=BinOpExpr(operator.gt,
                                             StrExpr(Col("s"), "count", ("1",)), Const(0)))
        with force_polars():
            result = run_op(op, pl.DataFrame({"s": ["a1", "bb", "c1"]}))
        self.assertEqual(["a1", "c1"], result["s"].to_list())


class TestMaskFolding(unittest.TestCase):
    """df[bool_series] folds into a single SelectionOp(MASK, predicate)."""

    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6],
                                "flag": [True, False, True]})

    def _mask(self, ops):
        masks = [o for o in ops if isinstance(o, SelectionOp)
                 and o.kind is SelectionKind.MASK]
        self.assertEqual(1, len(masks), "expected exactly one mask SelectionOp")
        return masks[0]

    def test_simple_comparison_folds(self):
        data = st.as_data_op(self.df)
        ops = optimize(data[data["x"] > 1], OptConfig(dataframe_ops=True))
        sel = self._mask(ops)
        self.assertEqual(BinOpExpr(operator.gt, Col("x"), Const(1)), sel.predicate)
        self.assertIs(OutputType.FRAME, sel.output_type)
        # the column GetItem and comparison BinOp are absorbed into the predicate
        self.assertEqual([], [o for o in ops if isinstance(o, GetItemOp)])

    def test_chained_comparison_folds_to_boolop(self):
        data = st.as_data_op(self.df)
        ops = optimize(data[(data["x"] > 1) & (data["y"] < 6)],
                       OptConfig(dataframe_ops=True))
        sel = self._mask(ops)
        self.assertEqual(
            BinOpExpr(operator.and_,
                   BinOpExpr(operator.gt, Col("x"), Const(1)),
                   BinOpExpr(operator.lt, Col("y"), Const(6))),
            sel.predicate)

    def test_boolean_column_mask_folds_to_col(self):
        data = st.as_data_op(self.df)
        ops = optimize(data[data["flag"]], OptConfig(dataframe_ops=True))
        self.assertEqual(Col("flag"), self._mask(ops).predicate)

    def test_negated_mask_folds_to_unaryop(self):
        data = st.as_data_op(self.df)
        ops = optimize(data[~(data["x"] > 1)], OptConfig(dataframe_ops=True))
        self.assertEqual(
            UnaryOpExpr(operator.invert, BinOpExpr(operator.gt, Col("x"), Const(1))),
            self._mask(ops).predicate)

    def test_arithmetic_in_mask_folds(self):
        # df[df["x"] + df["y"] > 5] : arithmetic is still a BinOp during frame
        # extraction (before the numeric pass), so it folds into the predicate.
        data = st.as_data_op(self.df)
        ops = optimize(data[data["x"] + data["y"] > 5], OptConfig(dataframe_ops=True))
        self.assertEqual(
            BinOpExpr(operator.gt,
                      BinOpExpr(operator.add, Col("x"), Col("y")),
                      Const(5)),
            self._mask(ops).predicate)

    def test_string_filter_in_mask_folds(self):
        df = self.df.assign(s=["a1", "bb", "c1"])
        data = st.as_data_op(df)
        ops = optimize(data[data["s"].str.count("1") > 0], OptConfig(dataframe_ops=True))
        self.assertEqual(
            BinOpExpr(operator.gt, StrExpr(Col("s"), "count", ("1",)), Const(0)),
            self._mask(ops).predicate)

    def test_string_filter_in_mask_folds2(self):
        df = self.df.assign(s=["a1", "bb", "c1"])
        data = st.as_data_op(df)
        col_s_transformed = data["s"].apply(lambda s: "row_" + s)
        ops = optimize(data[col_s_transformed.str.count("1") > 0], OptConfig(dataframe_ops=True))
        self.assertEqual(
            BinOpExpr(operator.gt, StrExpr(OperandLeaf(OperandRef(1)), "count", ("1",)), Const(0)),
            self._mask(ops).predicate)

    def test_external_operand_folds_to_leaf(self):
        # df[df["x"] > thr] where `thr` is another data-op: the column folds to Col,
        # the external operand cannot, so it becomes an OperandLeaf input.
        data = st.as_data_op(self.df)
        thr = st.as_data_op(1)
        ops = optimize(data[data["x"] > thr], OptConfig(dataframe_ops=True))
        sel = self._mask(ops)
        self.assertEqual(
            BinOpExpr(operator.gt, Col("x"), OperandLeaf(OperandRef(1))),
            sel.predicate)
        self.assertEqual(2, len(sel.inputs))  # [src, thr]

    def test_duplicate_masks_dedup_after_cse(self):
        # Two independently-built identical masks fold to two SelectionOps, which
        # CSE (now running after extraction) merges back into one.
        data = st.as_data_op(self.df)
        root = data[data["x"] > 1].skb.concat([data[data["x"] > 1]], axis=0)
        ops = optimize(root, OptConfig(dataframe_ops=True))
        masks = [o for o in ops if isinstance(o, SelectionOp)
                 and o.kind is SelectionKind.MASK]
        self.assertEqual(1, len(masks))

    def test_shared_mask_folds_with_operand_leaf(self):
        # The comparison series feeds both the selection and an assign, so it cannot
        # be absorbed: it stays in the graph and the predicate references it via an
        # OperandLeaf (the selection takes it as an input after the source frame).
        data = st.as_data_op(self.df)
        mask = data["x"] > 1
        result = data[mask].assign(keep=mask)
        ops = optimize(result, OptConfig(dataframe_ops=True))
        sel = self._mask(ops)
        self.assertIsInstance(sel.predicate, OperandLeaf)
        self.assertEqual(2, len(sel.inputs))            # [src, shared comparison]
        self.assertTrue(any(isinstance(o, BinOp) for o in ops))  # comparison kept

    def test_get_item_multiple_consumers(self):
        # The shared column df["x"] feeds two comparisons, but both live inside the
        # mask -- every consumer of the column is absorbed -- so the column folds too
        # (appearing as Col("x") in both branches) rather than becoming a leaf.
        data = st.as_data_op(self.df)
        col = data["x"]
        mask = (col > 1) & (col < 4)
        ops = optimize(data[mask], OptConfig(dataframe_ops=True))
        sel = self._mask(ops)
        self.assertEqual(
            BinOpExpr(operator.and_,
                      BinOpExpr(operator.gt, Col("x"), Const(1)),
                      BinOpExpr(operator.lt, Col("x"), Const(4))),
            sel.predicate)
        self.assertEqual(1, len(sel.inputs))  # only the source frame; nothing kept
        self.assertEqual([], [o for o in ops if isinstance(o, (GetItemOp, BinOp))])

    def test_partially_shared_subgraph_folds_internal_keeps_external(self):
        # df[(c > 1) & (c < 4)] but the inner column c = df["x"] also feeds an assign.
        # c has a consumer outside the mask, so it stays as an OperandLeaf; the two
        # comparisons are consumed only by the mask and still fold around that leaf.
        data = st.as_data_op(self.df)
        col = data["x"]
        mask = (col > 1) & (col < 4)
        result = data[mask].assign(keep=col)
        ops = optimize(result, OptConfig(dataframe_ops=True))
        sel = self._mask(ops)
        self.assertEqual(
            BinOpExpr(operator.and_,
                      BinOpExpr(operator.gt, OperandLeaf(OperandRef(1)), Const(1)),
                      BinOpExpr(operator.lt, OperandLeaf(OperandRef(1)), Const(4))),
            sel.predicate)
        self.assertEqual(2, len(sel.inputs))  # [src, shared column]
        self.assertTrue(any(isinstance(o, GetItemOp) for o in ops))  # column kept


class TestColumnExprOperandRefs(unittest.TestCase):
    """The ref-traversal contract validate_dag / CSE rely on to descend into exprs."""

    def test_iter_operand_refs(self):
        expr = BinOpExpr(operator.and_,
                         OperandLeaf(OperandRef(1)),
                         UnaryOpExpr(operator.invert, OperandLeaf(OperandRef(3))))
        self.assertEqual([1, 3], [r.k for r in expr.iter_operand_refs()])

    def test_iter_operand_refs_none_for_pure_expr(self):
        expr = BinOpExpr(operator.gt, Col("x"), Const(1))
        self.assertEqual([], list(expr.iter_operand_refs()))

    def test_remap_operand_refs_rebuilds_tree(self):
        expr = BinOpExpr(operator.gt, Col("x"), OperandLeaf(OperandRef(2)))
        remapped = expr.remap_operand_refs({2: 1})
        self.assertEqual(
            BinOpExpr(operator.gt, Col("x"), OperandLeaf(OperandRef(1))), remapped)

class TestSelectionStructureKey(unittest.TestCase):
    """SelectionOp exposes a structural key so CSE can dedup identical selections."""

    def test_equal_for_identical_selections(self):
        src = Op()
        a = SelectionOp(kind=SelectionKind.HEAD, args=(2,), inputs=[src])
        b = SelectionOp(kind=SelectionKind.HEAD, args=(2,), inputs=[src])
        self.assertEqual(a.structure_key(), b.structure_key())

    def test_differs_on_args(self):
        src = Op()
        a = SelectionOp(kind=SelectionKind.HEAD, args=(2,), inputs=[src])
        b = SelectionOp(kind=SelectionKind.HEAD, args=(3,), inputs=[src])
        self.assertNotEqual(a.structure_key(), b.structure_key())

    def test_differs_on_kind(self):
        src = Op()
        a = SelectionOp(kind=SelectionKind.HEAD, args=(2,), inputs=[src])
        b = SelectionOp(kind=SelectionKind.TAIL, args=(2,), inputs=[src])
        self.assertNotEqual(a.structure_key(), b.structure_key())


class TestUnaryPredicateProcess(unittest.TestCase):
    """A UnaryOpExpr predicate (e.g. ~mask) evaluates on both backends."""

    def setUp(self):
        self.predicate = UnaryOpExpr(operator.invert,
                                     BinOpExpr(operator.gt, Col("x"), Const(1)))

    def test_unary_mask_pandas(self):
        op = SelectionOp(kind=SelectionKind.MASK, predicate=self.predicate)
        result = run_op(op, pd.DataFrame({"x": [1, 2, 3]}))
        self.assertEqual([1], result["x"].tolist())

    def test_unary_mask_polars(self):
        op = SelectionOp(kind=SelectionKind.MASK, predicate=self.predicate)
        with force_polars():
            result = run_op(op, pl.DataFrame({"x": [1, 2, 3]}))
        self.assertEqual([1], result["x"].to_list())


class TestPandasQuery(unittest.TestCase):
    """With FLAGS.pandas_query, an expressible MASK runs through DataFrame.query()."""

    def setUp(self):
        self.df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    def test_query_simple_comparison(self):
        op = SelectionOp(kind=SelectionKind.MASK,
                         predicate=BinOpExpr(operator.gt, Col("x"), Const(1)))
        with pandas_query():
            result = run_op(op, self.df)
        self.assertEqual([2, 3], result["x"].tolist())

    def test_query_boolop(self):
        op = SelectionOp(kind=SelectionKind.MASK,
                         predicate=BinOpExpr(operator.and_,
                                             BinOpExpr(operator.gt, Col("x"), Const(1)),
                                             BinOpExpr(operator.lt, Col("y"), Const(6))))
        with pandas_query():
            result = run_op(op, self.df)
        self.assertEqual([2], result["x"].tolist())

    def test_query_unaryop(self):
        op = SelectionOp(kind=SelectionKind.MASK,
                         predicate=UnaryOpExpr(operator.invert,
                                               BinOpExpr(operator.gt, Col("x"), Const(1))))
        with pandas_query():
            result = run_op(op, self.df)
        self.assertEqual([1], result["x"].tolist())

    def test_query_falls_back_on_operand_leaf(self):
        # An OperandLeaf isn't query-expressible (to_pandas_query -> None), so the
        # predicate as a whole yields None and process falls back to boolean masking.
        op = SelectionOp(kind=SelectionKind.MASK,
                         predicate=BinOpExpr(operator.gt, Col("x"),
                                             OperandLeaf(OperandRef(1))))
        with pandas_query():
            result = run_op(op, self.df, 1)
        self.assertEqual([2, 3], result["x"].tolist())


class TestColumnExprQueryStrings(unittest.TestCase):
    """to_pandas_query builds a query string, binding literals into params."""

    def test_col_backticks_name(self):
        self.assertEqual("`x`", Col("x").to_pandas_query({}))

    def test_const_binds_param(self):
        params = {}
        self.assertEqual("@p0", Const(7).to_pandas_query(params))
        self.assertEqual({"p0": 7}, params)

    def test_binop_unsupported_op_returns_none(self):
        # matmul isn't in BINARY_SYMBOLS, so it isn't query-expressible.
        expr = BinOpExpr(operator.matmul, Col("x"), Const(1))
        self.assertIsNone(expr.to_pandas_query({}))

    def test_unaryop_unsupported_op_returns_none(self):
        # abs isn't in UNARY_SYMBOLS.
        self.assertIsNone(UnaryOpExpr(operator.abs, Col("x")).to_pandas_query({}))

    def test_operand_leaf_and_str_not_query_expressible(self):
        self.assertIsNone(OperandLeaf(OperandRef(1)).to_pandas_query({}))
        self.assertIsNone(StrExpr(Col("s"), "count", ("1",)).to_pandas_query({}))

    def test_unaryop_over_non_expressible_operand_returns_none(self):
        # A supported unary op whose operand isn't query-expressible -> None.
        expr = UnaryOpExpr(operator.invert, OperandLeaf(OperandRef(1)))
        self.assertIsNone(expr.to_pandas_query({}))


class TestColumnExprMisc(unittest.TestCase):
    """Assorted ColumnExpr node behaviour."""

    def test_const_unhashable_value_key(self):
        # An unhashable literal falls back to an identity-based key.
        value = [1, 2]
        c = Const(value)
        self.assertEqual(("__id__", id(value)), c._key())
        self.assertEqual(c, Const(value))  # same object -> equal
        hash(c)  # does not raise

    def test_remap_unaryop_expr(self):
        expr = UnaryOpExpr(operator.invert, OperandLeaf(OperandRef(2)))
        self.assertEqual(
            UnaryOpExpr(operator.invert, OperandLeaf(OperandRef(1))),
            expr.remap_operand_refs({2: 1}))

    def test_remap_str_expr(self):
        expr = StrExpr(OperandLeaf(OperandRef(2)), "count", ("1",))
        self.assertEqual(
            StrExpr(OperandLeaf(OperandRef(1)), "count", ("1",)),
            expr.remap_operand_refs({2: 1}))

    def test_remap_refs_descends_into_column_expr(self):
        # remap_operand_refs recognises a ColumnExpr and rebuilds it with remapped refs.
        expr = BinOpExpr(operator.gt, Col("x"), OperandLeaf(OperandRef(2)))
        self.assertEqual(
            BinOpExpr(operator.gt, Col("x"), OperandLeaf(OperandRef(1))),
            remap_operand_refs(expr, {2: 1}))


class TestUnaryOpProcess(unittest.TestCase):
    """UnaryOp.process applies the callable to its (graph-fed) operand."""

    def test_invert_series(self):
        op = UnaryOp(operator.invert, OperandRef(0))
        result = run_op(op, pd.Series([True, False, True]))
        self.assertEqual([False, True, False], result.tolist())


class TestGetItemFilterFastPath(unittest.TestCase):
    """A boolean-mask GetItem uses polars .filter() when is_filter is set."""

    def test_filter_polars(self):
        op = GetItemOp(key=OperandRef(1), is_filter=True)
        with force_polars():
            result = run_op(op, pl.DataFrame({"x": [1, 2, 3]}),
                            pl.Series([True, False, True]))
        self.assertEqual([1, 3], result["x"].to_list())


class TestRechunkPlFrame(unittest.TestCase):
    """rechunk_pl_frame slices a frame into fixed-size chunks."""

    def test_splits_into_chunks(self):
        df = pl.DataFrame({"x": list(range(5))})
        out = rechunk_pl_frame(df, rows_per_chunk=2)
        self.assertEqual(5, len(out))
        self.assertEqual(list(range(5)), out["x"].to_list())

    def test_no_split_when_below_threshold(self):
        df = pl.DataFrame({"x": [1, 2]})
        self.assertIs(df, rechunk_pl_frame(df, rows_per_chunk=10))


"""End-to-end tests: A folded mask runs through the full optimize + execute path."""
@pytest.fixture
def df():
    return pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})

@pytest.fixture(params=[False, True])
def polars(request):
    with force_polars(request.param):
        yield request.param

def test_mask_pipeline_evaluates(df, polars):
        pred = st.as_data_op(df)
        pred = pred[pred["x"] > 2]
        result = st._api.evaluate(pred)
        assert all(result["x"] > 2)

def test_string_filter_pipeline_evaluates(df, polars):
    data = st.as_data_op(df)
    data = data.assign(s=["a1", "bb", "c2", "", "d1"])
    col_s_transformed = data["s"].apply(lambda s: "row_" + s)
    out = data[col_s_transformed.str.count("1") > 0]
    result = st._api.evaluate(out)
    assert len(result["x"]) == 2

def test_complex_filter(df, polars):
    with force_polars(polars):
        data = st.as_data_op(df)
        data = data.assign(s=["a1", "bb", "c2", "", "d1"])
        col_s_transformed = data["s"].apply(lambda s: "row_" + s)
        predicate = (col_s_transformed.str.count("1") > 0) | ((data["x"] > 2) & (data["x"] <= 4)) | (data["y"] == 50)
        out = data[predicate]
        result = st._api.evaluate(out)
        assert len(result["x"]) == 4

if __name__ == "__main__":
    unittest.main()
