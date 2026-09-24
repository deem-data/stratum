"""Unit tests for the plan-time map program and the linear ColumnExpr DAG walks."""
from __future__ import annotations

import operator
import unittest

from stratum.optimizer.logical._column_expr import (
    BinOpExpr, Col, ColumnExpr, ColumnMethodExpr, Const, OperandLeaf, StrExpr,
    has_static_polars_dtype, iter_postorder, substitute_cols)
from stratum.optimizer.logical._ops import OperandRef
from stratum.optimizer.physical._map_program import (
    MAX_INLINE_DEPTH, TempRef, plan_map_program)


def mul(a, b):
    return BinOpExpr(operator.mul, a, b)


def add(a, b):
    return BinOpExpr(operator.add, a, b)


def sub(a, b):
    return BinOpExpr(operator.sub, a, b)


def fused_tower(depth: int) -> dict:
    """Entries of a fused ``x_i = x_{i-1} * 2`` chain, prefixes shared by identity."""
    entries, prev = {}, Col("x")
    for i in range(1, depth + 1):
        prev = mul(prev, Const(2.0))
        entries[f"x{i}"] = prev
    return entries


def fused_fibonacci(depth: int) -> dict:
    """Entries of a fused ``f_i = f_{i-1} + f_{i-2}`` chain (diamond-shaped DAG)."""
    entries = {}
    prev2, prev1 = Col("a"), Col("b")
    for i in range(depth):
        prev2, prev1 = prev1, add(prev1, prev2)
        entries[f"f{i}"] = prev1
    return entries


def tree_size(expr) -> int:
    return 1 + sum(tree_size(c) for c in expr.children())


def nesting_depth(expr) -> int:
    """Nesting levels of ``expr``; leaves and ``TempRef`` count as 0."""
    kids = expr.children()
    return 1 + max(nesting_depth(c) for c in kids) if kids else 0


def overwrite_chain(depth: int, step=lambda e: add(e, Const(1))) -> dict:
    """Entries of a fused ``a = f(a)`` chain: one single-use nested expression."""
    expr = Col("a")
    for _ in range(depth):
        expr = step(expr)
    return {"a": expr}


class TestPlanMapProgram(unittest.TestCase):
    def test_independent_entries_have_no_steps(self):
        entries = {"y": add(Col("x"), Const(1)), "z": mul(Col("x"), Col("x"))}
        program = plan_map_program(entries)
        self.assertEqual((), program.steps)
        self.assertEqual(entries, program.outputs)

    def test_tower_materializes_each_prefix_once(self):
        program = plan_map_program(fused_tower(3))
        self.assertEqual(2, len(program.steps))
        s0, s1 = program.steps
        self.assertEqual(mul(Col("x"), Const(2.0)), s0.expr)
        self.assertEqual(mul(TempRef(0), Const(2.0)), s1.expr)
        self.assertEqual([0, 1], [s0.level, s1.level])
        self.assertEqual(
            {"x1": TempRef(0), "x2": TempRef(1), "x3": mul(TempRef(1), Const(2.0))},
            program.outputs)

    def test_tower_steps_are_stored_under_their_outputs(self):
        program = plan_map_program(fused_tower(3))
        self.assertEqual(["x1", "x2"], [s.output for s in program.steps])
        self.assertEqual(["x1", "x2"], [s.column for s in program.steps])
        self.assertEqual([], program.temp_names)
        self.assertEqual({"x3": mul(TempRef(1), Const(2.0))},
                         program.unstored_outputs)

    def test_output_read_from_input_frame_keeps_private_column(self):
        # "a" is overwritten by a shared step that other entries build on,
        # while the same program still reads the input's "a".
        s = add(Col("a"), Const(1))
        program = plan_map_program({"a": s, "b": mul(s, Col("a"))})
        (step,) = program.steps
        self.assertIsNone(step.output)
        self.assertEqual([step.name], program.temp_names)
        self.assertEqual({"a": TempRef(0), "b": mul(TempRef(0), Col("a"))},
                         program.unstored_outputs)

    def test_step_that_is_no_output_keeps_private_column(self):
        s = add(Col("x"), Const(1))
        program = plan_map_program({"p": mul(s, Const(2)), "q": sub(s, Const(2))})
        (step,) = program.steps
        self.assertIsNone(step.output)
        self.assertEqual(program.outputs, program.unstored_outputs)

    def test_diamond_chain_is_linear(self):
        depth = 40  # the unfolded tree of the last entry has ~2**40 nodes
        program = plan_map_program(fused_fibonacci(depth))
        self.assertEqual(depth - 1, len(program.steps))
        for step in program.steps:
            self.assertLessEqual(tree_size(step.expr), 3)
        for expr in program.outputs.values():
            self.assertLessEqual(tree_size(expr), 3)

    def test_single_use_subexpression_is_inlined(self):
        r = BinOpExpr(operator.truediv, Col("x"), add(Col("y"), Const(1)))
        z = mul(sub(r, Const(0.5)), Const(2))
        program = plan_map_program({"r": r, "z": z})
        self.assertEqual([r], [s.expr for s in program.steps])
        self.assertEqual(mul(sub(TempRef(0), Const(0.5)), Const(2)),
                         program.outputs["z"])

    def test_leaves_are_never_materialized(self):
        x = Col("x")
        program = plan_map_program({"a": x, "b": add(x, x), "c": sub(x, x)})
        self.assertEqual((), program.steps)
        self.assertIs(x, program.outputs["a"])

    def test_structurally_equal_subexpressions_are_merged(self):
        # Two distinct but equal objects, e.g. the same expression in two stages.
        p = add(mul(Col("a"), Const(3)), Const(1))
        q = sub(mul(Col("a"), Const(3)), Const(1))
        self.assertIsNot(p.left, q.left)
        program = plan_map_program({"p": p, "q": q})
        self.assertEqual([mul(Col("a"), Const(3))], [s.expr for s in program.steps])
        self.assertEqual(add(TempRef(0), Const(1)), program.outputs["p"])
        self.assertEqual(sub(TempRef(0), Const(1)), program.outputs["q"])

    def test_equal_outputs_share_one_step(self):
        program = plan_map_program({"p": add(Col("a"), Const(1)),
                                    "q": add(Col("a"), Const(1))})
        self.assertEqual(1, len(program.steps))
        self.assertEqual({"p": TempRef(0), "q": TempRef(0)}, program.outputs)
        self.assertEqual("p", program.steps[0].output)
        self.assertEqual({"q": TempRef(0)}, program.unstored_outputs)

    def test_constants_of_different_type_or_sign_are_not_merged(self):
        entries = {
            "i": mul(Col("x"), Const(1)),
            "f": mul(Col("x"), Const(1.0)),
            "b": mul(Col("x"), Const(True)),
            "pz": mul(Col("x"), Const(0.0)),
            "nz": mul(Col("x"), Const(-0.0)),
        }
        program = plan_map_program(entries)
        self.assertEqual((), program.steps)
        self.assertEqual(entries, program.outputs)

    def test_method_literals_of_different_type_are_not_merged(self):
        cond = ColumnMethodExpr(Col("x"), "notna")
        entries = {
            "fill_i": ColumnMethodExpr(Col("x"), "fillna", (0,)),
            "fill_f": ColumnMethodExpr(Col("x"), "fillna", (0.0,)),
            "where_i": ColumnMethodExpr(Col("x"), "where", (cond, 1)),
            "where_b": ColumnMethodExpr(Col("x"), "where", (cond, True)),
            "clip_kw_i": ColumnMethodExpr(Col("x"), "clip", (), {"lower": 0}),
            "clip_kw_f": ColumnMethodExpr(Col("x"), "clip", (), {"lower": -0.0}),
            "slice_i": StrExpr(Col("s"), "slice", (0, 1)),
            "slice_b": StrExpr(Col("s"), "slice", (False, True)),
        }
        program = plan_map_program(entries)
        self.assertEqual([cond], [s.expr for s in program.steps])
        self.assertEqual(len(entries), len({id(e) for e in program.outputs.values()}))

    def test_method_literals_of_same_type_are_merged(self):
        program = plan_map_program({
            "p": ColumnMethodExpr(Col("x"), "clip", (0.0, 1.0)),
            "q": ColumnMethodExpr(Col("x"), "clip", (0.0, 1.0)),
        })
        self.assertEqual(1, len(program.steps))
        self.assertEqual({"p": TempRef(0), "q": TempRef(0)}, program.outputs)

    def test_unhashable_literal_args_do_not_fail(self):
        shared = StrExpr(Col("s"), "translate", ({ord("a"): "b"},))
        program = plan_map_program({"t": shared, "u": StrExpr(shared, "upper")})
        self.assertEqual([shared], [s.expr for s in program.steps])

    def test_unhashable_non_container_literal_key_uses_identity(self):
        # Containers are keyed structurally; a bare unhashable leaf falls back
        # to identity so hashing the expression does not raise.
        blob = bytearray(b"abc")
        expr = StrExpr(Col("s"), "encode", (blob,))
        self.assertEqual(
            (Col("s"), "encode",
             ("tuple", (("__id__", id(blob)),)),
             ("__dict__", frozenset())),
            expr._key())
        hash(expr)  # does not raise

    def test_set_literal_args_are_keyed_by_elements(self):
        expr = StrExpr(Col("s"), "contains", kwargs={"pat": {"a", "b"}})
        self.assertEqual(
            (Col("s"), "contains",
             ("tuple", ()),
             ("__dict__", frozenset({
                 ("pat", ("__set__", frozenset({
                     (str, "a"), (str, "b")})))}))),
            expr._key())
    def test_dtype_reading_method_keeps_shared_operand_inline(self):
        cast = ColumnMethodExpr(Col("x"), "astype", ("float64",))
        filled = ColumnMethodExpr(cast, "fillna", (0.0,))
        program = plan_map_program({"p": filled, "q": add(cast, Const(1))})
        self.assertEqual([cast], [s.expr for s in program.steps])
        self.assertEqual(filled, program.outputs["p"])
        self.assertEqual(add(TempRef(0), Const(1)), program.outputs["q"])

    def test_dtype_agnostic_method_uses_shared_operand_step(self):
        cast = ColumnMethodExpr(Col("x"), "astype", ("float64",))
        clipped = ColumnMethodExpr(cast, "clip", (0.0, 1.0))
        program = plan_map_program({"p": clipped, "q": add(cast, Const(1))})
        self.assertEqual(ColumnMethodExpr(TempRef(0), "clip", (0.0, 1.0)),
                         program.outputs["p"])

    def test_levels_group_independent_steps(self):
        a = add(Col("x"), Const(1))
        b = add(Col("y"), Const(1))
        c = mul(a, b)
        program = plan_map_program({"a": a, "b": b, "c": c, "d": add(c, Const(1))})
        self.assertEqual([2, 1], [len(level) for level in program.levels])
        self.assertEqual(mul(TempRef(0), TempRef(1)), program.levels[1][0].expr)

    def test_chain_below_depth_limit_stays_inline(self):
        entries = overwrite_chain(MAX_INLINE_DEPTH - 1)
        program = plan_map_program(entries)
        self.assertEqual((), program.steps)
        self.assertEqual(entries["a"], program.outputs["a"])

    def test_deep_single_use_chain_is_cut_into_bounded_steps(self):
        depth = 10 * MAX_INLINE_DEPTH + 7
        program = plan_map_program(overwrite_chain(depth))
        self.assertEqual(depth // MAX_INLINE_DEPTH, len(program.steps))
        for step in program.steps:
            self.assertEqual(MAX_INLINE_DEPTH, nesting_depth(step.expr))
        self.assertEqual(list(range(len(program.steps))),
                         [s.level for s in program.steps])
        self.assertEqual(depth % MAX_INLINE_DEPTH,
                         nesting_depth(program.outputs["a"]))

    def test_deep_dtype_reading_chain_is_bounded(self):
        # The operand of each fillna has no static dtype, so it must not be
        # kept inline across a step boundary.
        chain = overwrite_chain(
            5 * MAX_INLINE_DEPTH,
            lambda e: ColumnMethodExpr(add(e, Const(1)), "fillna", (0.0,)))
        program = plan_map_program(chain)
        self.assertTrue(program.steps)
        for expr in [s.expr for s in program.steps] + list(program.outputs.values()):
            self.assertLessEqual(nesting_depth(expr), MAX_INLINE_DEPTH)

    def test_depth_step_keeps_static_dtype_operand_inline(self):
        # The astype reaches the depth limit and becomes a step; the fillna
        # above it still inlines the cast so the kernel resolves its dtype,
        # which puts the fillna over the limit as well.
        cast = ColumnMethodExpr(overwrite_chain(MAX_INLINE_DEPTH - 1)["a"],
                                "astype", ("float64",))
        filled = ColumnMethodExpr(cast, "fillna", (0.0,))
        program = plan_map_program({"p": filled})
        self.assertEqual([cast, filled], [s.expr for s in program.steps])
        self.assertEqual(cast, program.steps[1].expr.operand)
        self.assertEqual(MAX_INLINE_DEPTH + 1, nesting_depth(program.steps[1].expr))
        self.assertEqual(TempRef(1), program.outputs["p"])

    def test_temp_names_are_private(self):
        s = add(Col("x"), Const(1))
        program = plan_map_program({"p": mul(s, s)})
        self.assertEqual(1, len(program.temp_names))
        for name in program.temp_names:
            self.assertTrue(name.startswith("__stratum_map_tmp_"))

    def test_temp_ref_with_children_is_identity(self):
        ref = TempRef(0)
        self.assertIs(ref, ref.with_children(()))

    def test_temp_ref_rejects_children(self):
        with self.assertRaises(TypeError):
            TempRef(0).with_children((Col("x"),))


class TestColumnExprStructuralApi(unittest.TestCase):
    """Coverage for leaf / base structural hooks used by the DAG walks."""

    def test_base_column_expr_with_children_is_abstract(self):
        with self.assertRaises(TypeError):
            ColumnExpr().with_children(())

    def test_leaf_with_children_is_identity(self):
        col = Col("x")
        self.assertIs(col, col.with_children(()))

    def test_leaf_rejects_children(self):
        with self.assertRaises(TypeError):
            Col("x").with_children((Const(1),))

    def test_has_static_polars_dtype_for_bare_col(self):
        self.assertTrue(has_static_polars_dtype(Col("x")))
        self.assertFalse(has_static_polars_dtype(add(Col("x"), Const(1))))

    def test_has_static_polars_dtype_rejects_expr_dtype(self):
        # An astype whose dtype is itself an expression has no static dtype.
        cast = ColumnMethodExpr(Col("x"), "astype", (Col("y"),))
        self.assertFalse(has_static_polars_dtype(cast))
        cast_kw = ColumnMethodExpr(Col("x"), "astype", kwargs={"dtype": Col("y")})
        self.assertFalse(has_static_polars_dtype(cast_kw))

    def test_substitute_cols_rewrites_column_exprs_nested_in_list_args(self):
        # ``ColumnMethodExpr`` stores ``args`` as a tuple; a list nested inside
        # still carries ColumnExprs that rewrite_leaves must rebuild in place.
        expr = ColumnMethodExpr(Col("x"), "isin", ([Col("a"), "literal"],))
        out = substitute_cols(expr, {"a": Col("b")})
        self.assertEqual(
            ColumnMethodExpr(Col("x"), "isin", ([Col("b"), "literal"],)),
            out)


class TestLinearDagWalks(unittest.TestCase):
    """Walks over shared DAGs must visit every distinct node once."""

    def deep_diamond(self, depth: int):
        node = add(OperandLeaf(OperandRef(1)), Col("x"))
        for _ in range(depth):
            node = add(node, node)
        return node

    def test_iter_postorder_deduplicates_shared_nodes(self):
        expr = self.deep_diamond(60)
        self.assertEqual(3 + 60, len(iter_postorder([expr])))

    def test_iter_operand_refs_on_deep_diamond(self):
        refs = list(self.deep_diamond(60).iter_operand_refs())
        self.assertEqual([OperandRef(1)], refs)

    def test_remap_operand_refs_on_deep_diamond_keeps_sharing(self):
        out = self.deep_diamond(60).remap_operand_refs({1: 2})
        self.assertIs(out.left, out.right)
        self.assertEqual([OperandRef(2)], list(out.iter_operand_refs()))

    def test_remap_operand_refs_without_refs_returns_self(self):
        expr = add(Col("x"), Const(1))
        self.assertIs(expr, expr.remap_operand_refs({0: 0}))

    def test_hash_on_deep_diamond(self):
        left, right = self.deep_diamond(60), self.deep_diamond(60)
        self.assertEqual(hash(left), hash(right))

    def test_eq_on_shared_dag_short_circuits_on_identity(self):
        expr = self.deep_diamond(60)
        self.assertEqual(add(expr, Const(1)), add(expr, Const(1)))


if __name__ == "__main__":
    unittest.main()
