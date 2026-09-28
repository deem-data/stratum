import operator
import unittest

import pandas as pd
import polars as pl

from stratum.optimizer.logical._column_expr import (
    AGG_PARAMS, AggExpr, BinOpExpr, Col, Const, DtExpr, EvalContext, OperandLeaf,
    StrExpr, UnaryOpExpr)
from stratum.optimizer.logical._ops import OperandRef

# Reductions polars can express; `sem` has no polars equivalent.
_POLARS_UNSUPPORTED = {"sem", "idxmin", "idxmax"}


def _pandas(expr, df, inputs=()):
    return expr.to_pandas(EvalContext(frame=df, inputs=list(inputs)))


def _polars(expr, df, inputs=()):
    return df.select(expr.to_polars(EvalContext(frame=df, inputs=list(inputs)))).item()


class TestAggExprValidation(unittest.TestCase):
    """Construction-time contract: canonical name, whitelisted params, no nesting."""

    def test_unknown_function_raises(self):
        with self.assertRaises(NotImplementedError):
            AggExpr("cumsum", Col("v"))

    def test_off_whitelist_param_raises_and_names_it(self):
        with self.assertRaises(NotImplementedError) as cm:
            AggExpr("sum", Col("v"), {"ddof": 1})
        self.assertIn("ddof", str(cm.exception))

    def test_nested_aggregate_raises(self):
        inner = AggExpr("sum", Col("v"))
        with self.assertRaises(ValueError):
            AggExpr("mean", inner)
        # ... including one buried inside a row-wise subtree.
        with self.assertRaises(ValueError):
            AggExpr("mean", BinOpExpr(operator.mul, Col("w"), inner))

    def test_params_are_copied_not_aliased(self):
        params = {"skipna": False}
        expr = AggExpr("sum", Col("v"), params)
        params["skipna"] = True
        self.assertEqual({"skipna": False}, expr.params)

    def test_every_whitelisted_param_is_accepted(self):
        for func, allowed in AGG_PARAMS.items():
            for param in allowed:
                with self.subTest(func=func, param=param):
                    AggExpr(func, Col("v"), {param: 1})


class TestAggExprValueSemantics(unittest.TestCase):
    """Structural equality and hashing -- the CSE contract."""

    def test_equal_shapes_compare_and_hash_equal(self):
        a = AggExpr("sum", BinOpExpr(operator.mul, Col("a"), Col("b")))
        b = AggExpr("sum", BinOpExpr(operator.mul, Col("a"), Col("b")))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_param_order_does_not_affect_equality(self):
        a = AggExpr("std", Col("v"), {"ddof": 2, "skipna": False})
        b = AggExpr("std", Col("v"), {"skipna": False, "ddof": 2})
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_different_func_child_or_params_differ(self):
        base = AggExpr("sum", Col("v"))
        self.assertNotEqual(base, AggExpr("mean", Col("v")))
        self.assertNotEqual(base, AggExpr("sum", Col("w")))
        self.assertNotEqual(base, AggExpr("sum", Col("v"), {"min_count": 1}))

    def test_repr_shows_func_child_and_params(self):
        r = repr(AggExpr("sum", Col("v"), {"min_count": 1}))
        self.assertIn("sum(", r)
        self.assertIn("Col('v')", r)
        self.assertIn("min_count=1", r)


class TestHasAggregate(unittest.TestCase):
    """`has_aggregate` marks the subtree so nesting can be refused."""

    def test_leaves_are_row_wise(self):
        for leaf in (Col("v"), Const(3), OperandLeaf(OperandRef(1))):
            with self.subTest(leaf=leaf):
                self.assertFalse(leaf.has_aggregate())

    def test_aggregate_marks_itself(self):
        self.assertTrue(AggExpr("sum", Col("v")).has_aggregate())

    def test_composites_recurse(self):
        agg = AggExpr("sum", Col("v"))
        self.assertTrue(BinOpExpr(operator.add, Col("w"), agg).has_aggregate())
        self.assertTrue(BinOpExpr(operator.add, agg, Col("w")).has_aggregate())
        self.assertTrue(UnaryOpExpr(operator.neg, agg).has_aggregate())
        self.assertTrue(StrExpr(agg, "strip").has_aggregate())
        self.assertTrue(DtExpr(agg, "day").has_aggregate())

    def test_row_wise_composite_is_not_an_aggregate(self):
        tree = BinOpExpr(operator.add, Col("a"),
                         UnaryOpExpr(operator.neg, Col("b")))
        self.assertFalse(tree.has_aggregate())


class TestAggExprOperandRefs(unittest.TestCase):
    """Refs buried in the child are walked and remapped."""

    def test_iter_operand_refs_reaches_into_child(self):
        expr = AggExpr("sum", BinOpExpr(operator.mul, Col("v"),
                                        OperandLeaf(OperandRef(2))))
        self.assertEqual([OperandRef(2)], list(expr.iter_operand_refs()))

    def test_no_refs_when_child_is_pure(self):
        self.assertEqual([], list(AggExpr("sum", Col("v")).iter_operand_refs()))

    def test_remap_rewrites_child_refs_and_keeps_config(self):
        expr = AggExpr("sum", OperandLeaf(OperandRef(1)), {"min_count": 1})
        out = expr.remap_operand_refs({1: 3})
        self.assertEqual([OperandRef(3)], list(out.iter_operand_refs()))
        self.assertEqual("sum", out.func)
        self.assertEqual({"min_count": 1}, out.params)
        # The original is untouched; expressions are immutable value types.
        self.assertEqual([OperandRef(1)], list(expr.iter_operand_refs()))


class TestAggExprEvaluation(unittest.TestCase):
    """Both backends agree with the plain pandas/polars reduction."""

    def setUp(self):
        # Deliberately asymmetric: symmetric data has zero skew on both the
        # biased and bias-corrected definitions, which would hide a mismatch.
        self.data = {"a": [1.0, 2.0, 3.0, 10.0], "b": [10.0, 20.0, 30.0, 40.0]}
        self.pdf = pd.DataFrame(self.data)
        self.pldf = pl.DataFrame(self.data)

    def test_backends_agree_on_every_reduction(self):
        # Index-label reductions are deliberately not lowered to arg_min/arg_max.
        for func in sorted(AGG_PARAMS):
            with self.subTest(func=func):
                expr = AggExpr(func, Col("a"))
                got = _pandas(expr, self.pdf)
                if func in _POLARS_UNSUPPORTED:
                    with self.assertRaises(NotImplementedError):
                        expr.to_polars(EvalContext(frame=self.pldf, inputs=[]))
                    continue
                self.assertAlmostEqual(float(got), float(_polars(expr, self.pldf)),
                                       msg=f"{func} disagrees across backends")

    def test_composite_child_folds_the_multiply_into_one_reduction(self):
        expr = AggExpr("sum", BinOpExpr(operator.mul, Col("a"), Col("b")))
        expected = (self.pdf["a"] * self.pdf["b"]).sum()
        self.assertAlmostEqual(expected, _pandas(expr, self.pdf))
        self.assertAlmostEqual(expected, _polars(expr, self.pldf))

    def test_operand_leaf_child_reads_the_inputs_list(self):
        expr = AggExpr("sum", OperandLeaf(OperandRef(1)))
        self.assertAlmostEqual(6.0, _pandas(expr, self.pdf,
                                            inputs=[None, pd.Series([1.0, 2.0, 3.0])]))

    def test_ddof_is_honoured_by_both_backends(self):
        for ddof in (0, 1, 2):
            with self.subTest(ddof=ddof):
                expr = AggExpr("std", Col("a"), {"ddof": ddof})
                self.assertAlmostEqual(self.pdf["a"].std(ddof=ddof),
                                       _pandas(expr, self.pdf))
                self.assertAlmostEqual(self.pdf["a"].std(ddof=ddof),
                                       _polars(expr, self.pldf))

    def test_quantile_matches_pandas_interpolation_default(self):
        # polars defaults to "nearest", pandas to "linear"; the node pins pandas'.
        expr = AggExpr("quantile", Col("a"), {"q": 0.3})
        self.assertAlmostEqual(self.pdf["a"].quantile(0.3), _pandas(expr, self.pdf))
        self.assertAlmostEqual(self.pdf["a"].quantile(0.3), _polars(expr, self.pldf))

    def test_size_counts_nulls_and_count_does_not(self):
        data = {"v": [1.0, None, 3.0]}
        pdf, pldf = pd.DataFrame(data), pl.DataFrame(data)
        self.assertEqual(3, _pandas(AggExpr("size", Col("v")), pdf))
        self.assertEqual(3, _polars(AggExpr("size", Col("v")), pldf))
        self.assertEqual(2, _pandas(AggExpr("count", Col("v")), pdf))
        self.assertEqual(2, _polars(AggExpr("count", Col("v")), pldf))

    def test_nunique_dropna_agrees_across_backends(self):
        # polars' n_unique counts null as a value; the node compensates.
        data = {"v": [1.0, None, 1.0, 2.0]}
        pdf, pldf = pd.DataFrame(data), pl.DataFrame(data)
        for dropna in (True, False):
            with self.subTest(dropna=dropna):
                expr = AggExpr("nunique", Col("v"), {"dropna": dropna})
                expected = pdf["v"].nunique(dropna=dropna)
                self.assertEqual(expected, _pandas(expr, pdf))
                self.assertEqual(expected, _polars(expr, pldf))

    def test_first_and_last_reduce_without_a_series_method(self):
        # pandas 3.0 has no Series.first/last; the node uses positional access.
        self.assertEqual(self.data["a"][0],
                         _pandas(AggExpr("first", Col("a")), self.pdf))
        self.assertEqual(self.data["a"][-1],
                         _pandas(AggExpr("last", Col("a")), self.pdf))

    def test_polars_refuses_parameters_it_cannot_express(self):
        expr = AggExpr("sum", Col("a"), {"min_count": 1})
        with self.assertRaises(NotImplementedError) as cm:
            expr.to_polars(EvalContext(frame=self.pldf, inputs=[]))
        self.assertIn("min_count", str(cm.exception))

    def test_polars_ignores_parameters_left_at_their_pandas_default(self):
        expr = AggExpr("sum", Col("a"), {"skipna": True, "numeric_only": False,
                                         "min_count": 0})
        self.assertAlmostEqual(self.pdf["a"].sum(), _polars(expr, self.pldf))

    def test_aggregate_is_not_a_query_predicate(self):
        self.assertIsNone(AggExpr("sum", Col("a")).to_pandas_query({}))


if __name__ == "__main__":
    unittest.main()
