from __future__ import annotations

import operator
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import polars as pl
import pytest

import stratum as st
from stratum.optimizer._dataframe_rewrites import DataframeRewritesConfig
from stratum.optimizer._map_rewrites import fuse_assign_maps
from stratum.optimizer._optimize import OptConfig, optimize as optimize_
from stratum.optimizer.logical._column_expr import (
    BinOpExpr, Col, Const, DatetimeExpr, DtExpr, OperandLeaf, StrExpr,
    UnaryOpExpr, substitute_cols,
)
from stratum.optimizer.logical._map_ops import AssignMapOp
from stratum.optimizer.logical._ops import Op, OperandRef
from stratum.optimizer.logical._projection_ops import AssignOp
from stratum.optimizer.logical._selection_ops import SelectionOp
from tests.optimizer.logical.test_dataframe_ops import (
    force_polars, make_map_op, optimize,
)
from tests.optimizer.physical.test_source_execs import run_plan


def _cfg(*, fuse: bool = True, **kwargs) -> OptConfig:
    return OptConfig(
        dataframe_rewrite_config=DataframeRewritesConfig(fuse_assign_maps=fuse),
        **kwargs,
    )


def _run(dag, *, fuse: bool, polars_backend: bool = False):
    with force_polars(polars_backend):
        ops, *_ = optimize_(dag, _cfg(fuse=fuse))
        return run_plan(ops)


def _frames_equal(left, right):
    if isinstance(left, pl.DataFrame) or isinstance(right, pl.DataFrame):
        if isinstance(left, pd.DataFrame):
            left = pl.from_pandas(left)
        if isinstance(right, pd.DataFrame):
            right = pl.from_pandas(right)
        return left.equals(right)
    return left.equals(right)


def _assign_maps(ops):
    return [o for o in ops if isinstance(o, AssignMapOp)]


def assert_fused_equals_unfused(dag, *, polars_backend: bool,
                                expect_fusion: bool = True):
    """Fused and unfused plans must agree; optionally require fewer maps when fused."""
    with force_polars(polars_backend):
        fused_ops, *_ = optimize_(dag, _cfg(fuse=True))
        unfused_ops, *_ = optimize_(dag, _cfg(fuse=False))
        n_fused = len(_assign_maps(fused_ops))
        n_unfused = len(_assign_maps(unfused_ops))
        if expect_fusion:
            assert n_unfused >= 2
            assert n_fused < n_unfused, (n_fused, n_unfused)
        fused = run_plan(fused_ops)
        unfused = run_plan(unfused_ops)
    assert _frames_equal(fused, unfused), (
        f"fused/unfused mismatch (polars={polars_backend})\n"
        f"fused:\n{fused}\nunfused:\n{unfused}")


# --- substitute_cols ---------------------------------------------------------

class TestSubstituteCols(unittest.TestCase):
    def test_replaces_col_from_bindings(self):
        binding = BinOpExpr(operator.add, Col("a"), Const(1))
        out = substitute_cols(Col("a"), {"a": binding})
        # Immutable bindings are shared by identity (no deep clone).
        self.assertIs(binding, out)

    def test_does_not_rewalk_bindings(self):
        # Binding still contains Col("a"); substitution must not recurse into it.
        binding = BinOpExpr(operator.add, Col("a"), Const(1))
        result = substitute_cols(Col("a"), {"a": binding})
        self.assertIs(binding, result)
        self.assertEqual(Col("a"), result.left)

    def test_untouched_col_and_const(self):
        self.assertEqual(Col("b"), substitute_cols(Col("b"), {"a": Const(1)}))
        self.assertEqual(Const(3), substitute_cols(Const(3), {"a": Const(1)}))

    def test_binop_unary_str_dt_datetime(self):
        bindings = {"a": BinOpExpr(operator.add, Col("a"), Const(1))}
        cases = [
            BinOpExpr(operator.mul, Col("a"), Const(2)),
            UnaryOpExpr(operator.neg, Col("a")),
            StrExpr(Col("a"), "upper", ()),
            DtExpr(Col("a"), "year"),
            DatetimeExpr(Col("a")),
        ]
        for expr in cases:
            out = substitute_cols(expr, bindings)
            self.assertIsNot(out, expr)
            # The Col("a") child was replaced with the shared binding object.
            child = out.left if isinstance(out, BinOpExpr) else out.operand
            self.assertIs(bindings["a"], child)

    def test_operand_leaf_passthrough(self):
        leaf = OperandLeaf(OperandRef(1))
        self.assertIs(leaf, substitute_cols(leaf, {"a": Const(1)}))

    def test_identity_memo_shares_rewritten_subtrees(self):
        shared = Col("a")
        expr = BinOpExpr(operator.add, shared, shared)
        binding = Const(7)
        out = substitute_cols(expr, {"a": binding})
        self.assertIs(out.left, out.right)
        self.assertIs(binding, out.left)

    def test_fusion_shares_inlined_prefixes(self):
        """Tower fusion inlines source-relative DAGs; prefixes share identity."""
        src = Op(name="src")
        prev = src
        for i in range(3):
            name = f"x{i + 1}"
            src_col = "x" if i == 0 else f"x{i}"
            m = AssignMapOp(
                entries={name: BinOpExpr(operator.mul, Col(src_col), Const(2))},
                inputs=[prev], outputs=[])
            prev.outputs = [m]
            prev = m
        fused = fuse_assign_maps(prev)
        e1, e2, e3 = (fused.entries[f"x{i}"] for i in (1, 2, 3))
        self.assertEqual(BinOpExpr(operator.mul, Col("x"), Const(2)), e1)
        self.assertEqual(
            BinOpExpr(operator.mul,
                      BinOpExpr(operator.mul, Col("x"), Const(2)), Const(2)),
            e2)
        self.assertIs(e2.left, e1)
        self.assertIs(e3.left, e2)

    def test_unknown_node_fails_loudly(self):
        from stratum.optimizer.logical._column_expr import ColumnExpr

        class Unknown(ColumnExpr):
            def _key(self):
                return ()

        with self.assertRaises(TypeError):
            substitute_cols(Unknown(), {})


# --- plan / rewrite unit tests -----------------------------------------------

class TestFuseAssignMapsPlan(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "s": ["x", "y", "z"]})

    def test_maximal_chain_becomes_one_flat_map(self):
        src = st.as_data_op(self.df)
        m0 = src.assign(a=src["a"] + 1, snapshot=src["a"] * 2)
        m1 = m0.assign(b=m0["a"] * 2)
        m2 = m1.assign(a=-m1["a"], total=m1["a"] + m1["snapshot"])
        ops = optimize(m2, _cfg(fuse=True))
        maps = _assign_maps(ops)
        self.assertEqual(1, len(maps))
        entries = maps[0].entries
        self.assertEqual(
            {"a", "snapshot", "b", "total"}, set(entries))
        # Worked example from the design doc.
        self.assertEqual(
            UnaryOpExpr(operator.neg, BinOpExpr(operator.add, Col("a"), Const(1))),
            entries["a"])
        self.assertEqual(
            BinOpExpr(operator.mul, Col("a"), Const(2)),
            entries["snapshot"])
        self.assertEqual(
            BinOpExpr(operator.mul,
                      BinOpExpr(operator.add, Col("a"), Const(1)), Const(2)),
            entries["b"])
        # Within M2, total reads the pre-overwrite ``a`` (simultaneous siblings).
        self.assertEqual(
            BinOpExpr(
                operator.add,
                BinOpExpr(operator.add, Col("a"), Const(1)),
                BinOpExpr(operator.mul, Col("a"), Const(2))),
            entries["total"])
        # Single source input; no stage maps left.
        self.assertEqual(1, len(maps[0].inputs))

    def test_kill_switch_keeps_chain(self):
        src = st.as_data_op(self.df)
        m0 = src.assign(a=src["a"] + 1)
        m1 = m0.assign(b=m0["a"] * 2)
        ops = optimize(m1, _cfg(fuse=False))
        self.assertEqual(2, len(_assign_maps(ops)))

    def test_dataframe_rewrites_disabled(self):
        src = st.as_data_op(self.df)
        m0 = src.assign(a=src["a"] + 1)
        m1 = m0.assign(b=m0["a"] * 2)
        ops = optimize(m1, OptConfig(dataframe_rewrites=False))
        self.assertEqual(2, len(_assign_maps(ops)))

    def test_barrier_selection_in_middle(self):
        src = st.as_data_op(self.df)
        m0 = src.assign(a=src["a"] + 1)
        filtered = m0[m0["a"] > 1]
        m1 = filtered.assign(b=filtered["a"] * 2)
        ops = optimize(m1, _cfg(fuse=True))
        self.assertEqual(2, len(_assign_maps(ops)))
        self.assertEqual(1, len([o for o in ops if isinstance(o, SelectionOp)]))

    def test_barrier_branching_intermediate(self):
        src = st.as_data_op(self.df)
        m0 = src.assign(a=src["a"] + 1)
        # Branch: m0 feeds two consumers.
        left = m0.assign(b=m0["a"] * 2)
        right = m0.assign(c=m0["a"] * 3)
        out = left.skb.concat([right], axis=0)
        ops = optimize(out, _cfg(fuse=True))
        # m0 cannot fuse into either branch; three maps remain.
        self.assertEqual(3, len(_assign_maps(ops)))

    def test_barrier_external_operand(self):
        src = st.as_data_op(self.df)
        factor = st.as_data_op(2)
        m0 = src.assign(a=src["a"] * factor)
        m1 = m0.assign(b=m0["a"] + 1)
        ops = optimize(m1, _cfg(fuse=True))
        maps = _assign_maps(ops)
        self.assertEqual(2, len(maps))
        self.assertTrue(
            any(any(e.iter_operand_refs()) for e in maps[0].entries.values()))

    def test_barrier_opaque_aligning_assign(self):
        src = st.as_data_op(self.df)
        with make_map_op(False):
            opaque = src.assign(vals=[10.0, 20.0, 30.0])
        # Even with fusion on, the opaque AssignOp is not an AssignMapOp.
        m1 = opaque.assign(b=opaque["a"] * 2)
        ops = optimize(m1, _cfg(fuse=True))
        self.assertEqual(1, len([o for o in ops if isinstance(o, AssignOp)]))
        self.assertEqual(1, len(_assign_maps(ops)))

    def test_barrier_metadata_bearing_map(self):
        src = Op()
        m0 = AssignMapOp(entries={"a": BinOpExpr(operator.add, Col("a"), Const(1))},
                         inputs=[src])
        m1 = AssignMapOp(entries={"b": BinOpExpr(operator.mul, Col("a"), Const(2))},
                         inputs=[m0])
        src.outputs = [m0]
        m0.outputs = [m1]
        m0.is_X = True
        fused_root = fuse_assign_maps(m1)
        # Metadata on m0 blocks the chain; both maps remain.
        self.assertIs(m1, fused_root)
        self.assertEqual([m0], src.outputs)
        self.assertEqual([m1], m0.outputs)

    def test_barrier_output_edge_without_matching_input(self):
        src = Op()
        other = Op()
        m0 = AssignMapOp(
            entries={"a": BinOpExpr(operator.add, Col("a"), Const(1))},
            inputs=[src])
        other.inputs = [m0]
        m1 = AssignMapOp(
            entries={"b": BinOpExpr(operator.mul, Col("a"), Const(2))},
            inputs=[other])
        src.outputs = [m0]
        m0.outputs = [m1]
        other.outputs = [m1]
        fused_root = fuse_assign_maps(m1)
        self.assertIs(m1, fused_root)
        self.assertEqual([m0], src.outputs)
        self.assertEqual([m1], m0.outputs)
        self.assertEqual([src], m0.inputs)
        self.assertEqual([other], m1.inputs)
        self.assertEqual([m1], other.outputs)


class TestFusedMapEvalMemo(unittest.TestCase):
    """Physical CSE: shared tower prefixes evaluate once (pandas) / shallow prior cols (polars)."""

    @staticmethod
    def _tower_chain(n: int):
        src = Op(name="src")
        prev = src
        maps = []
        for i in range(n):
            name = f"x{i + 1}"
            src_col = "x" if i == 0 else f"x{i}"
            m = AssignMapOp(
                entries={name: BinOpExpr(operator.mul, Col(src_col), Const(2))},
                inputs=[prev], outputs=[])
            prev.outputs = [m]
            prev = m
            maps.append(m)
        return maps, prev

    def test_fused_tower_binop_evals_are_linear(self):
        from stratum.optimizer.physical._map_execs import PandasAssignMapOp

        n = 20
        _, root = self._tower_chain(n)
        fused = fuse_assign_maps(root)
        self.assertEqual(n, len(fused.entries))
        # Shared prefixes: O(n) IR nodes, not deep-cloned O(n²) trees.
        self.assertIs(fused.entries["x2"].left, fused.entries["x1"])

        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        op = PandasAssignMapOp(entries=fused.entries, inputs=[])
        calls = {"n": 0}
        orig = BinOpExpr._to_pandas

        def counting(self, ctx):
            calls["n"] += 1
            return orig(self, ctx)

        with mock.patch.object(BinOpExpr, "_to_pandas", counting):
            out = op.process("fit_transform", [df])
        self.assertEqual(n, calls["n"])
        self.assertEqual([2 ** n, 2 * 2 ** n, 3 * 2 ** n],
                         out[f"x{n}"].tolist())

    def test_polars_tower_lowers_to_prior_cols(self):
        from stratum.optimizer.logical._column_expr import replace_prior_cols
        from stratum.optimizer.physical._map_execs import PolarsAssignMapOp

        _, root = self._tower_chain(3)
        fused = fuse_assign_maps(root)
        # Shared IR prefixes; Polars lowering rewrites them to Col(prior).
        self.assertIs(fused.entries["x2"].left, fused.entries["x1"])
        priors = {}
        lowered = {}
        for name, expr in fused.entries.items():
            rewritten, used = replace_prior_cols(expr, priors)
            lowered[name] = rewritten
            priors[expr] = name
            if name != "x1":
                self.assertTrue(used)
        self.assertEqual(
            BinOpExpr(operator.mul, Col("x2"), Const(2)), lowered["x3"])

        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        op = PolarsAssignMapOp(entries=fused.entries, inputs=[])
        plans = []
        real_collect = pl.LazyFrame.collect

        def tracking_collect(self, *args, **kwargs):
            plans.append(self.explain(optimized=False))
            return real_collect(self, *args, **kwargs)

        with mock.patch.object(pl.LazyFrame, "collect", tracking_collect):
            out = op.process("fit_transform", [df])
        self.assertEqual(1, len(plans))
        # One lazy chain of shallow with_columns, collected once (no eager intermediates).
        self.assertGreaterEqual(plans[0].count("WITH_COLUMNS"), 3)
        self.assertEqual([8.0, 16.0, 24.0], out["x3"].to_list())

    def test_prior_cols_do_not_rewrite_pre_overwrite_exprs(self):
        """Priors key on *final* entry exprs: old ``a`` must not become ``Col("a")``.

        After fusion overwrites ``a``, a later-inserted column that still embeds
        the pre-overwrite ``a`` expression must stay fully inlined. Rewriting it
        to ``Col("a")`` would read the negated column and change results.
        """
        from stratum.optimizer.logical._column_expr import replace_prior_cols

        src = Op(name="src")
        m0 = AssignMapOp(
            entries={
                "a": BinOpExpr(operator.add, Col("a"), Const(1)),
                "snapshot": BinOpExpr(operator.mul, Col("a"), Const(2)),
            },
            inputs=[src], outputs=[])
        m1 = AssignMapOp(
            entries={"b": BinOpExpr(operator.mul, Col("a"), Const(2))},
            inputs=[m0], outputs=[])
        m2 = AssignMapOp(
            entries={
                "a": UnaryOpExpr(operator.neg, Col("a")),
                "total": BinOpExpr(operator.add, Col("a"), Col("snapshot")),
            },
            inputs=[m1], outputs=[])
        src.outputs = [m0]
        m0.outputs = [m1]
        m1.outputs = [m2]

        fused = fuse_assign_maps(m2)
        old_a = BinOpExpr(operator.add, Col("a"), Const(1))
        self.assertEqual(
            UnaryOpExpr(operator.neg, old_a), fused.entries["a"])
        self.assertEqual(
            BinOpExpr(operator.mul, old_a, Const(2)), fused.entries["b"])

        priors = {}
        lowered = {}
        for name, expr in fused.entries.items():
            rewritten, _ = replace_prior_cols(expr, priors)
            lowered[name] = rewritten
            priors[expr] = name

        # Final ``a`` is the prior key; ``b`` still embeds old ``a``, which must
        # not match that key (negated) or any other prior.
        self.assertEqual(
            BinOpExpr(operator.mul, old_a, Const(2)), lowered["b"])
        self.assertNotEqual(
            BinOpExpr(operator.mul, Col("a"), Const(2)), lowered["b"])
        # ``total`` may use Col("snapshot") but its left side is still old ``a``.
        self.assertEqual(
            BinOpExpr(operator.add, old_a, Col("snapshot")), lowered["total"])


# --- equivalence -------------------------------------------------------------

@pytest.fixture(params=[False, True], ids=["pandas", "polars"])
def polars(request):
    with force_polars(request.param):
        yield request.param


def _semantic_pipeline(df):
    src = st.as_data_op(df)
    m0 = src.assign(a=src["a"] + 1, snapshot=src["a"] * 2, tag=src["s"].str.upper())
    m1 = m0.assign(b=m0["a"] * 2, flag=1)
    m2 = m1.assign(a=-m1["a"], total=m1["a"] + m1["snapshot"])
    return m2


def test_fused_equals_unfused_semantic(polars):
    df = pd.DataFrame({
        "a": [1.0, np.nan, 3.0, np.inf],
        "s": ["ab", None, "cd", "ef"],
    })
    dag = _semantic_pipeline(df)
    fused = _run(dag, fuse=True, polars_backend=polars)
    unfused = _run(dag, fuse=False, polars_backend=polars)
    assert _frames_equal(fused, unfused)


def test_fused_equals_unfused_empty(polars):
    df = pd.DataFrame({"a": pd.Series([], dtype=float), "s": pd.Series([], dtype=object)})
    dag = _semantic_pipeline(df)
    assert _frames_equal(
        _run(dag, fuse=True, polars_backend=polars),
        _run(dag, fuse=False, polars_backend=polars))


def test_fused_equals_unfused_zero_column_then_assign(polars):
    df = pd.DataFrame(index=[0, 1, 2])
    src = st.as_data_op(df)
    m0 = src.assign(a=1)
    m1 = m0.assign(b=m0["a"] * 2)
    assert _frames_equal(
        _run(m1, fuse=True, polars_backend=polars),
        _run(m1, fuse=False, polars_backend=polars))


def test_fused_equals_unfused_nan_inf_across_boundary(polars):
    df = pd.DataFrame({"a": [np.nan, 1.0, -np.inf, np.inf]})
    src = st.as_data_op(df)
    m0 = src.assign(a=src["a"] * 2)
    m1 = m0.assign(b=m0["a"] + 1)
    assert _frames_equal(
        _run(m1, fuse=True, polars_backend=polars),
        _run(m1, fuse=False, polars_backend=polars))


def test_fused_equals_unfused_duplicate_index_pandas():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=[0, 0, 1])
    src = st.as_data_op(df)
    m0 = src.assign(a=src["a"] + 1)
    m1 = m0.assign(b=m0["a"] * 2)
    fused = _run(m1, fuse=True, polars_backend=False)
    unfused = _run(m1, fuse=False, polars_backend=False)
    pd.testing.assert_frame_equal(fused, unfused)


def test_fused_equals_unfused_mixed_dtypes(polars):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5], "s": ["x", "y", "z"]})
    src = st.as_data_op(df)
    m0 = src.assign(a=src["a"] + src["b"], label=src["s"].str.upper())
    m1 = m0.assign(score=m0["a"] * 10, n=m0["label"].str.len())
    assert _frames_equal(
        _run(m1, fuse=True, polars_backend=polars),
        _run(m1, fuse=False, polars_backend=polars))


def test_fused_equals_unfused_single_element(polars):
    df = pd.DataFrame({"a": [7.0], "s": ["q"]})
    dag = _semantic_pipeline(df)
    assert _frames_equal(
        _run(dag, fuse=True, polars_backend=polars),
        _run(dag, fuse=False, polars_backend=polars))


def test_evaluate_path_matches_kill_switch_off(polars):
    # Public evaluate uses the default OptConfig (fusion on).
    df = pd.DataFrame({"a": [1.0, 2.0], "s": ["a", "b"]})
    dag = _semantic_pipeline(df)
    via_api = st._api.evaluate(dag)
    via_off = _run(dag, fuse=False, polars_backend=polars)
    assert _frames_equal(via_api, via_off)


# --- additional fused == unfused coverage (not already covered above) --------

def test_fused_equals_unfused_tower_chain(polars):
    """Shared-spine chain x2=x*2, x3=x2*2, ... must match unfused results."""
    df = pd.DataFrame({"x": [0.0, -1.0, 2.0, np.nan]})
    src = st.as_data_op(df)
    cur = src.assign(x2=src["x"] * 2)
    cur = cur.assign(x3=cur["x2"] * 2)
    cur = cur.assign(x4=cur["x3"] * 2)
    dag = cur.assign(x5=cur["x4"] * 2)
    assert_fused_equals_unfused(dag, polars_backend=polars)


def test_fused_equals_unfused_long_overwrite_chain(polars):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    cur = st.as_data_op(df)
    for i in range(6):
        prev = cur
        cur = prev.assign(a=prev["a"] + i if i == 0 else prev["a"] * 2 - i)
    assert_fused_equals_unfused(cur, polars_backend=polars)


def test_fused_equals_unfused_string_and_datetime_chain(polars):
    df = pd.DataFrame({
        "s": [" a ", None, "Bb"],
        "ts": ["2020-01-31", "2021-06-15", "2024-02-29"],
        "a": [1.0, 2.0, 3.0],
    })
    src = st.as_data_op(df)
    date = src["ts"].skb.apply_func(pd.to_datetime)
    m0 = src.assign(up=src["s"].str.strip().str.upper(),
                    day=date.dt.day,
                    end=date.dt.is_month_end)
    dag = m0.assign(n=m0["up"].str.len() * 10,
                    day2=m0["day"] * 2,
                    score=m0["day"] + m0["a"])
    assert_fused_equals_unfused(dag, polars_backend=polars)


def test_fused_equals_unfused_boolean_and_deep_arithmetic(polars):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 1.0, 2.0]})
    src = st.as_data_op(df)
    a, b = src["a"], src["b"]
    m0 = src.assign(
        flag=~((a > 2) | (b < 4)),
        y=-((a + b) ** 2) / 3 + (a * b) % 5 + b // a,
    )
    dag = m0.assign(flipped=~m0["flag"], z=m0["y"] * 2 + m0["a"])
    assert_fused_equals_unfused(dag, polars_backend=polars)


def test_fused_equals_unfused_after_selection_barrier(polars):
    """Fusion of the post-filter chain must still match unfused execution."""
    df = pd.DataFrame({"a": [-1.0, 1.0, 2.0, 3.0], "s": ["a", "b", "c", "d"]})
    src = st.as_data_op(df)
    f = src[src["a"] > 0]
    m0 = f.assign(a2=f["a"] * 2)
    dag = m0.assign(a4=m0["a2"] * 2, label=m0["s"].str.upper())
    assert_fused_equals_unfused(dag, polars_backend=polars)


def test_fused_equals_unfused_two_segments_around_filter(polars):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    src = st.as_data_op(df)
    m0 = src.assign(a=src["a"] + 1)
    m1 = m0.assign(b=m0["a"] * 2)
    filtered = m1[m1["b"] > 4]
    m2 = filtered.assign(c=filtered["b"] + filtered["a"])
    dag = m2.assign(d=m2["c"] * 3)
    assert_fused_equals_unfused(dag, polars_backend=polars)


def test_fused_equals_unfused_two_chains_via_concat(polars):
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    src = st.as_data_op(df)
    left = src.assign(a=src["a"] + 1)
    left = left.assign(x=left["a"] * 2)
    right = src.assign(b=src["b"] + 1)
    right = right.assign(y=right["b"] * 3)
    dag = left.skb.concat([right], axis=0)
    assert_fused_equals_unfused(dag, polars_backend=polars)


def test_fused_equals_unfused_column_order_with_overwrite(polars):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    src = st.as_data_op(df)
    m0 = src.assign(a=src["a"] + 1, snapshot=src["a"] * 2, z=0)
    m1 = m0.assign(b=m0["a"] * 2)
    dag = m1.assign(a=-m1["a"], total=m1["a"] + m1["snapshot"])
    with force_polars(polars):
        fused = run_plan(optimize_(dag, _cfg(fuse=True))[0])
        unfused = run_plan(optimize_(dag, _cfg(fuse=False))[0])
    assert list(fused.columns) == list(unfused.columns)
    assert _frames_equal(fused, unfused)


def test_fused_equals_unfused_non_default_and_unsorted_index_pandas():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "s": ["x", "y", "z"]},
                      index=pd.Index([30, 10, 20], name="row_id"))
    src = st.as_data_op(df)
    m0 = src.assign(a=src["a"] + 1, tag=src["s"].str.upper())
    dag = m0.assign(b=m0["a"] * 2, n=m0["tag"].str.len())
    fused = _run(dag, fuse=True, polars_backend=False)
    unfused = _run(dag, fuse=False, polars_backend=False)
    pd.testing.assert_frame_equal(fused, unfused)


def test_fused_equals_unfused_shared_producer_and_weird_names(polars):
    df = pd.DataFrame({"col with space": [1, 2], "größe": [3, 4]})
    src = st.as_data_op(df)
    derived = src["col with space"] + 1
    m0 = src.assign(first=derived, second=derived,
                    **{"größe²": src["größe"] ** 2})
    dag = m0.assign(sum_=m0["first"] + m0["second"],
                    **{"new col": m0["first"] * m0["größe²"]})
    assert_fused_equals_unfused(dag, polars_backend=polars)


def test_barrier_branching_still_equals_unfused(polars):
    """Branching blocks fusion; fused and unfused plans are identical and agree."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    src = st.as_data_op(df)
    m0 = src.assign(a=src["a"] + 1)
    left = m0.assign(b=m0["a"] * 2)
    right = m0.assign(c=m0["a"] * 3)
    dag = left.skb.concat([right], axis=0)
    assert_fused_equals_unfused(dag, polars_backend=polars, expect_fusion=False)


def test_fused_equals_unfused_after_loc_row_col(polars):
    """Assign chains after ``df.loc[mask, cols]`` still fuse and match."""
    df = pd.DataFrame({"a": [1.0, -1.0, 2.0], "b": [10.0, 20.0, 30.0]})
    src = st.as_data_op(df)
    sub = src.loc[src["a"] > 0, ["a", "b"]]
    m0 = sub.assign(c=sub["a"] * 2)
    m1 = m0.assign(d=m0["c"] + m0["b"])
    assert_fused_equals_unfused(m1, polars_backend=polars)


def test_fused_equals_unfused_after_loc_single_column_series(polars):
    """``df.loc[mask, "col"]`` is a SERIES; later frame assigns still agree."""
    df = pd.DataFrame({"a": [1.0, -1.0, 2.0], "b": [10.0, 20.0, 30.0]})
    src = st.as_data_op(df)
    # Series-typed loc feeds assign on the filtered frame, so the SERIES output
    # type is in the graph.
    sub = src.loc[src["a"] > 0, ["a", "b"]]
    col = src.loc[src["a"] > 0, "a"]
    m0 = sub.assign(from_series=col)
    m1 = m0.assign(scaled=m0["from_series"] * 2)
    # from_series is an OperandLeaf (external ref), so the first map is not
    # fusible into a pure Col chain with the second — still must agree.
    assert_fused_equals_unfused(m1, polars_backend=polars, expect_fusion=False)


def test_search_plan_keeps_fusion_and_score_sink():
    """Fusion runs before ``install_candidate_set``: fewer maps + Score sink."""
    from sklearn.dummy import DummyRegressor

    from stratum.optimizer._optimize import SearchConfig
    from stratum.optimizer.logical._candidate_ops import ScoreCandidatesOp
    from stratum.optimizer.logical._scoring import resolve_scoring

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [0.0, 1.0, 0.0, 1.0]})
    data = st.as_data_op(df)
    X = data[["x"]].skb.mark_as_X()
    y = data["y"].skb.mark_as_y()
    X = X.assign(x2=X["x"] * 2)
    X = X.assign(x3=X["x2"] + 1)
    pred = X.skb.apply(DummyRegressor(), y=y)

    search = SearchConfig(metric=resolve_scoring("neg_mean_squared_error"))
    fused_ops, *_ = optimize_(pred, _cfg(fuse=True), search=search)
    unfused_ops, *_ = optimize_(pred, _cfg(fuse=False), search=search)

    assert isinstance(fused_ops[-1], ScoreCandidatesOp)
    assert isinstance(unfused_ops[-1], ScoreCandidatesOp)
    assert len(_assign_maps(fused_ops)) < len(_assign_maps(unfused_ops))


if __name__ == "__main__":
    unittest.main()
