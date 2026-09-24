from __future__ import annotations

import operator
import sys
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
    BinOpExpr, Col, ColumnMethodExpr, Const, DatetimeExpr, DtExpr, OperandLeaf,
    StrExpr, UnaryOpExpr, substitute_cols,
)
from stratum.optimizer.logical._map_ops import AssignMapOp
from stratum.optimizer.logical._ops import Op, OperandRef
from stratum.optimizer.logical._projection_ops import AssignOp
from stratum.optimizer.logical._selection_ops import SelectionOp
from stratum.optimizer.physical._map_execs import PandasAssignMapOp, PolarsAssignMapOp
from tests.optimizer.logical.test_dataframe_ops import (
    force_polars, make_map_op, optimize, run_op,
)
from tests.optimizer.physical.test_source_execs import run_plan
from tests._helpers import csv_file


def _cfg(*, fuse: bool = True, **kwargs) -> OptConfig:
    return OptConfig(
        dataframe_rewrite_config=DataframeRewritesConfig(fuse_assign_maps=fuse),
        **kwargs,
    )


def _map_kernel(polars_backend: bool):
    """Concrete AssignMapOp class the ``polars`` / ``pandas`` test arms must bind."""
    return PolarsAssignMapOp if polars_backend else PandasAssignMapOp


def _selector(polars_backend: bool) -> str:
    # Default prefers pandas regardless of force_polars; greedy prefers polars.
    return "greedy" if polars_backend else "default"


def _assert_map_kernel(ops, polars_backend: bool) -> list:
    """Require every AssignMapOp to be the backend kernel this arm claims to test."""
    kernel = _map_kernel(polars_backend)
    maps = _assign_maps(ops)
    assert maps, "expected at least one AssignMapOp"
    wrong = [type(m).__name__ for m in maps if type(m) is not kernel]
    assert not wrong, f"expected {kernel.__name__}, got {wrong}"
    return maps


def _run(dag, *, fuse: bool, polars_backend: bool = False):
    with st.config(implementation_selector=_selector(polars_backend)):
        ops, *_ = optimize_(dag, _cfg(fuse=fuse))
        _assert_map_kernel(ops, polars_backend)
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
    """Fused and unfused plans must agree on the claimed map kernel."""
    with st.config(implementation_selector=_selector(polars_backend)):
        fused_ops, *_ = optimize_(dag, _cfg(fuse=True))
        unfused_ops, *_ = optimize_(dag, _cfg(fuse=False))
        fused_maps = _assert_map_kernel(fused_ops, polars_backend)
        _assert_map_kernel(unfused_ops, polars_backend)
        if expect_fusion:
            n_unfused = len(_assign_maps(unfused_ops))
            assert n_unfused >= 2
            assert len(fused_maps) < n_unfused, (len(fused_maps), n_unfused)
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

    def test_column_method_operand_args_and_kwargs_are_substituted(self):
        expr = ColumnMethodExpr(Col("x"), "where", (Col("m"),), {"other": Col("y")})
        out = substitute_cols(expr, {"x": Const(1), "m": Col("k"), "y": Const(2)})
        self.assertEqual(
            ColumnMethodExpr(Const(1), "where", (Col("k"),), {"other": Const(2)}),
            out)

    def test_column_method_operand_refs_in_args(self):
        expr = ColumnMethodExpr(Col("x"), "where", (OperandLeaf(OperandRef(1)),))
        self.assertEqual([OperandRef(1)], list(expr.iter_operand_refs()))
        remapped = expr.remap_operand_refs({1: 3})
        self.assertEqual([OperandRef(3)], list(remapped.iter_operand_refs()))

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
        # Each entry is rewritten relative to the source frame, not intermediate stages.
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

    def test_barrier_dtype_reading_method_on_assigned_column(self):
        src = st.as_data_op(self.df)
        m0 = src.assign(r=src["a"] / 2)
        m1 = m0.assign(filled=m0["r"].fillna(0.0), present=m0["r"].notna())
        maps = _assign_maps(optimize(m1, _cfg(fuse=True)))
        self.assertEqual(2, len(maps))
        self.assertEqual(ColumnMethodExpr(Col("r"), "fillna", (0.0,)),
                         maps[1].entries["filled"])

    def test_dtype_reading_method_on_unassigned_column_fuses(self):
        src = st.as_data_op(self.df)
        m0 = src.assign(r=src["a"] / 2)
        m1 = m0.assign(filled=m0["a"].fillna(0.0), s=m0["r"] * 2)
        maps = _assign_maps(optimize(m1, _cfg(fuse=True)))
        self.assertEqual(1, len(maps))

    def test_barrier_read_of_constant_assigned_column(self):
        src = st.as_data_op(self.df)
        m0 = src.assign(flag=True)
        m1 = m0.assign(neg=~m0["flag"])
        maps = _assign_maps(optimize(m1, _cfg(fuse=True)))
        self.assertEqual(2, len(maps))
        self.assertEqual(UnaryOpExpr(operator.invert, Col("flag")),
                         maps[1].entries["neg"])

    def test_constant_assigned_column_not_read_fuses(self):
        src = st.as_data_op(self.df)
        m0 = src.assign(flag=True)
        m1 = m0.assign(b=m0["a"] * 2)
        maps = _assign_maps(optimize(m1, _cfg(fuse=True)))
        self.assertEqual(1, len(maps))
        self.assertEqual(Const(True), maps[0].entries["flag"])

    def test_constant_overwritten_by_expression_fuses(self):
        src = st.as_data_op(self.df)
        m0 = src.assign(flag=True)
        m1 = m0.assign(flag=m0["a"] > 1)
        m2 = m1.assign(neg=~m1["flag"])
        maps = _assign_maps(optimize(m2, _cfg(fuse=True)))
        self.assertEqual(1, len(maps))

    def test_barrier_constant_assigned_mid_chain(self):
        src = st.as_data_op(self.df)
        m0 = src.assign(b=src["a"] + 1)
        m1 = m0.assign(c=m0["b"] * 2, tag="x")
        m2 = m1.assign(u=m1["tag"].str.upper())
        maps = _assign_maps(optimize(m2, _cfg(fuse=True)))
        # m0 and m1 fuse; m2 reads the constant ``tag`` and stays separate.
        self.assertEqual(2, len(maps))
        self.assertEqual({"b", "c", "tag"}, set(maps[0].entries))

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


# --- equivalence -------------------------------------------------------------

@pytest.fixture(params=[False, True], ids=["pandas", "polars"])
def polars(request):
    """Whether the claimed map kernel is polars (``True``) or pandas (``False``).

    Selection uses the greedy / default selector (see :func:`_selector`); this
    fixture is only the boolean arm label.
    """
    return request.param


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


def test_polars_kernel_overwrite_after_dependent_column():
    """Fused inlined Cols must evaluate against the input frame on polars.

    Overwrite ``a``, then a later column that depends on an intermediate ``b``
    still embeds source-relative ``Col("a")``. One ``with_columns`` keeps that
    meaning; a prior-col chain would not.
    """
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    src = st.as_data_op(df)
    m0 = src.assign(a=src["a"] + 1)
    m1 = m0.assign(b=m0["a"] * 2)
    m2 = m1.assign(a=-m1["a"])
    dag = m2.assign(c=m2["b"] + 1)

    ops, *_ = optimize_(dag, _cfg(fuse=True))
    entries = _assign_maps(ops)[0].entries
    pandas_op = PandasAssignMapOp(entries=entries, inputs=[])
    polars_op = PolarsAssignMapOp(entries=entries, inputs=[])
    pandas_op.on_impl_selected(None)
    polars_op.on_impl_selected(None)
    pandas_out = pandas_op.process("fit_transform", [df])
    polars_out = polars_op.process("fit_transform", [pl.from_pandas(df)])
    assert _frames_equal(pandas_out, polars_out)
    assert [4.0, 6.0, 8.0] == polars_out["b"].to_list()
    assert [5.0, 7.0, 9.0] == polars_out["c"].to_list()


def test_polars_kernel_all_outputs_stored_as_steps():
    """When every output is staged under a step, skip the unstored with_columns.

    Needs more than one step so the polars kernel takes the lazy leveled path;
    a single step uses the eager ``with_columns`` shortcut and never reaches
    the unstored-outputs branch.
    """
    from stratum.optimizer.physical._map_program import (
        MAX_INLINE_DEPTH, plan_map_program)

    depth = MAX_INLINE_DEPTH * 2
    expr = Col("x")
    for _ in range(depth):
        expr = BinOpExpr(operator.add, expr, Const(1.0))
    entries = {"out": expr}
    program = plan_map_program(entries)
    assert program.unstored_outputs == {}
    assert len(program.steps) > 1

    op = PolarsAssignMapOp(entries=entries, inputs=[])
    op.on_impl_selected(None)
    out = op.process("fit_transform", [pl.DataFrame({"x": [1.0, 2.0]})])
    assert [1.0 + depth, 2.0 + depth] == out["out"].to_list()


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


def _invert_constant_bool(src):
    m0 = src.assign(flag=True)
    return m0.assign(neg=~m0["flag"])


def _str_method_on_constant(src):
    m0 = src.assign(tag="x")
    return m0.assign(u=m0["tag"].str.upper())


def _divide_constants_by_zero(src):
    m0 = src.assign(one=1, zero=0)
    return m0.assign(r=m0["one"] / m0["zero"])


def _column_method_on_constant(src):
    m0 = src.assign(c=1.5)
    return m0.assign(i=m0["c"].astype("int64"))


@pytest.mark.parametrize("build", [_invert_constant_bool, _str_method_on_constant,
                                   _divide_constants_by_zero,
                                   _column_method_on_constant])
def test_fused_equals_unfused_constant_assign_read_downstream(build, polars):
    """A constant assign read by a later stage keeps broadcast-Series semantics.

    On pandas, inlining the ``Const`` would evaluate the later stage on a plain
    Python scalar instead of a Series.
    """
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, 1.0]})
    dag = build(st.as_data_op(df))
    assert_fused_equals_unfused(dag, polars_backend=polars, expect_fusion=False)


def _fillna_int_and_float(src):
    return src.assign(i=src["o"].fillna(0), f=src["o"].fillna(0.0))


def _where_int_then_bool(src):
    m0 = src.assign(i=src["flag"].where(src["x"] > 1, 1))
    return m0.assign(b=m0["flag"].where(m0["x"] > 1, True))


@pytest.mark.parametrize("fuse", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize("build", [_fillna_int_and_float, _where_int_then_bool])
def test_method_literals_equal_in_python_keep_their_type(build, fuse):
    """``fillna(0)`` / ``fillna(0.0)`` and ``where(.., 1)`` / ``where(.., True)``
    compare equal in Python but give different values on pandas, so the map
    program must not evaluate them as one shared node."""
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "flag": [True, False, True],
        "o": pd.Series([1, None, "s"], dtype=object),
    })
    expected = build(df)
    got = _run(build(st.as_data_op(df)), fuse=fuse, polars_backend=False)
    pd.testing.assert_frame_equal(expected, got)
    # Object columns compare 0 == 0.0, so check the element types as well.
    for name in expected.columns:
        assert ([type(v) for v in expected[name]]
                == [type(v) for v in got[name]]), name


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


# --- chain-shape and barrier equivalence -------------------------------------

def test_fused_equals_unfused_tower_chain(polars):
    """Shared-spine chain x2=x*2, x3=x2*2, ... must match unfused results."""
    df = pd.DataFrame({"x": [0.0, -1.0, 2.0, np.nan]})
    src = st.as_data_op(df)
    cur = src.assign(x2=src["x"] * 2)
    cur = cur.assign(x3=cur["x2"] * 2)
    cur = cur.assign(x4=cur["x3"] * 2)
    dag = cur.assign(x5=cur["x4"] * 2)
    assert_fused_equals_unfused(dag, polars_backend=polars)


def _fibonacci_chain(src, depth=6):
    """Each column feeds the next two, so the fused DAG is diamond-shaped."""
    cur, names = src, ["a", "b"]
    for i in range(depth):
        name = f"f{i}"
        cur = cur.assign(**{name: cur[names[-1]] + cur[names[-2]]})
        names.append(name)
    return cur


def _fibonacci_pipeline(df, depth):
    return _fibonacci_chain(st.as_data_op(df), depth)


def _tower_chain(src):
    cur, prev = src, "a"
    for i in range(6):
        cur = cur.assign(**{f"x{i}": cur[prev] * 2})
        prev = f"x{i}"
    return cur


def _feature_chain(src):
    m0 = src.assign(r=src["a"] / (src["b"] + 1))
    m1 = m0.assign(z=(m0["r"] - 0.5) * 2 + m0["a"] * m0["b"])
    m2 = m1.assign(w=m1["z"] * m1["z"] - m1["r"])
    return m2.assign(f=(m2["w"] > 0) & (m2["a"] > m2["b"]))


def _overwrite_chain(src):
    """Shared steps read a column that a later stage overwrites."""
    m0 = src.assign(a=src["a"] + 1)
    m1 = m0.assign(b=m0["a"] * 2, keep=m0["a"] - m0["b"])
    m2 = m1.assign(a=-m1["a"], total=m1["a"] + m1["b"])
    return m2.assign(c=m2["b"] + m2["total"], a=m2["a"] * m2["keep"])


def _nan_method_chain(src):
    """NaN-sensitive column methods read a column assigned two stages earlier.

    ``0 / 0`` yields NaN on every row with finite inputs; missing CSV values
    read as null on polars, which would not exercise the NaN handling.
    """
    m0 = src.assign(r=(src["a"] * 0) / (src["b"] * 0))
    m1 = m0.assign(s=m0["r"] * 2)
    return m1.assign(filled=m1["r"].fillna(0.0), present=m1["r"].notna(),
                     t=m1["s"] + 1)


def _reordering_chain(src):
    """A shared step stored under its output name precedes, in write order, an
    output assigned before it."""
    m0 = src.assign(late=src["a"] * 0, s=src["a"] + 1)
    return m0.assign(late=m0["s"] * 2, t=m0["s"] + 1)


def _kernel_results(df, build, *, polars_backend):
    """Run ``build`` over a CSV read of ``df``, fused and unfused.

    File sources bind a polars frame under the greedy selector, which is the
    reliable way to exercise :class:`PolarsAssignMapOp` end to end (in-memory
    pandas sources can still feed a polars map after greedy source conversion,
    but CSV matches production read paths).
    """
    with csv_file(df) as path:
        dag = build(st.as_data_op(path).skb.apply_func(pd.read_csv))
        with st.config(implementation_selector=_selector(polars_backend)):
            fused_ops, *_ = optimize_(dag, _cfg(fuse=True))
            unfused_ops, *_ = optimize_(dag, _cfg(fuse=False))
            fused_maps = _assert_map_kernel(fused_ops, polars_backend)
            _assert_map_kernel(unfused_ops, polars_backend)
            assert len(fused_maps) < len(_assign_maps(unfused_ops))
            return run_plan(fused_ops), run_plan(unfused_ops)


@pytest.mark.parametrize("polars_backend", [False, True], ids=["pandas", "polars"])
@pytest.mark.parametrize("build", [_fibonacci_chain, _tower_chain, _feature_chain,
                                   _overwrite_chain, _reordering_chain,
                                   _nan_method_chain])
def test_kernel_fused_equals_unfused(build, polars_backend):
    df = pd.DataFrame({"a": [1.0, -2.0, np.nan, 0.25, np.inf],
                       "b": [0.5, 3.0, 1.0, -4.0, 2.0]})
    fused, unfused = _kernel_results(df, build, polars_backend=polars_backend)
    assert _frames_equal(fused, unfused), f"fused:\n{fused}\nunfused:\n{unfused}"


def test_fused_diamond_chain_evaluates_each_node_once():
    """Shared sub-expressions of a fused map are computed once per call."""
    depth = 6
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with force_polars(False):
        ops, *_ = optimize_(_fibonacci_pipeline(df, depth), _cfg(fuse=True))
    (fused,) = _assign_maps(ops)
    calls = {"n": 0}
    original = BinOpExpr.to_pandas

    def counting(self, ctx):
        calls["n"] += 1
        return original(self, ctx)

    with mock.patch.object(BinOpExpr, "to_pandas", counting):
        out = fused.process("fit_transform", [df])
    assert depth == calls["n"]
    assert [4.0, 6.0] == out["f0"].tolist()


def test_fusion_keeps_subexpressions_shared_within_a_stage():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    src = st.as_data_op(df)
    m0 = src.assign(t=src["a"] + 1)
    common = m0["t"] * 3
    dag = m0.assign(p=common + 1, q=common - 1)
    ops, *_ = optimize_(dag, _cfg(fuse=True))
    entries = _assign_maps(ops)[0].entries
    assert entries["p"].left is entries["q"].left


@pytest.mark.parametrize("polars_backend", [False, True], ids=["pandas", "polars"])
def test_shared_cast_feeding_nan_methods_keeps_nan_handling(polars_backend):
    # The cast has three uses, so the map program materializes it as a step.
    cast = ColumnMethodExpr(Col("x"), "astype", ("float64",))
    op = AssignMapOp(entries={
        "filled": ColumnMethodExpr(cast, "fillna", (0.0,)),
        "present": ColumnMethodExpr(cast, "notna"),
        "shifted": BinOpExpr(operator.add, cast, Const(1.0)),
    })
    values = [1.0, float("nan")]
    frame = pl.DataFrame({"x": values}) if polars_backend else pd.DataFrame({"x": values})
    with force_polars(polars_backend):
        out = run_op(op, frame)
    assert [1.0, 0.0] == list(out["filled"])
    assert [True, False] == list(out["present"])


def test_polars_kernel_keeps_assignment_column_order():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    fused, unfused = _kernel_results(df, _reordering_chain, polars_backend=True)
    assert ["a", "late", "s", "t"] == fused.columns == unfused.columns


def test_polars_kernel_rejects_reserved_column_names():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, 1.0],
                       "__stratum_map_tmp_0": [0.0, 0.0]})
    with pytest.raises(ValueError, match="reserved map column names"):
        _kernel_results(df, _overwrite_chain, polars_backend=True)


def test_polars_kernel_rejects_reserved_output_names():
    """An output named like a step's private column would be dropped with it."""
    shared = BinOpExpr(operator.add, Col("a"), Const(1.0))
    entries = {
        "p": BinOpExpr(operator.mul, shared, Const(2.0)),
        "__stratum_map_tmp_0": BinOpExpr(operator.sub, shared, Const(1.0)),
    }
    op = PolarsAssignMapOp(entries=entries, inputs=[])
    with pytest.raises(ValueError, match="reserved map column names"):
        op.on_impl_selected(None)


def test_polars_kernel_allows_prefixed_output_names_no_step_uses():
    shared = BinOpExpr(operator.add, Col("a"), Const(1.0))
    entries = {
        "p": BinOpExpr(operator.mul, shared, Const(2.0)),
        "q": BinOpExpr(operator.sub, shared, Const(1.0)),
        "__stratum_map_tmp_7": Col("a"),
    }
    op = PolarsAssignMapOp(entries=entries, inputs=[])
    op.on_impl_selected(None)
    out = op.process("fit_transform", [pl.DataFrame({"a": [1.0, 2.0]})])
    assert ["a", "p", "q", "__stratum_map_tmp_7"] == out.columns
    assert [1.0, 2.0] == out["__stratum_map_tmp_7"].to_list()


def test_polars_kernel_one_step_uses_eager_with_columns():
    """A single shared step skips the lazy leveled plan (same as zero steps).

    The one-step shared map used to pay lazy ``with_columns`` + ``collect``
    overhead; evaluating the source-relative entries in one eager
    ``with_columns`` matches that path and stays correct.
    """
    from stratum.optimizer.physical._map_program import plan_map_program

    shared = BinOpExpr(operator.add, Col("a"), Const(1.0))
    entries = {
        "p": BinOpExpr(operator.mul, shared, Const(2.0)),
        "q": BinOpExpr(operator.sub, shared, Const(1.0)),
    }
    assert 1 == len(plan_map_program(entries).steps)

    frame = pl.DataFrame({"a": [1.0, 2.0]})
    op = PolarsAssignMapOp(entries=entries, inputs=[])
    op.on_impl_selected(None)
    out = op.process("fit_transform", [frame])
    assert [4.0, 6.0] == out["p"].to_list()
    assert [1.0, 2.0] == out["q"].to_list()
    # No temp staging, so a reserved input name is harmless on this path.
    frame_tmp = frame.with_columns(pl.lit(0.0).alias("__stratum_map_tmp_0"))
    out_tmp = op.process("fit_transform", [frame_tmp])
    assert [4.0, 6.0] == out_tmp["p"].to_list()


def test_fused_equals_unfused_long_overwrite_chain(polars):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    cur = st.as_data_op(df)
    for i in range(6):
        prev = cur
        cur = prev.assign(a=prev["a"] + i if i == 0 else prev["a"] * 2 - i)
    assert_fused_equals_unfused(cur, polars_backend=polars)


@pytest.mark.parametrize("polars_backend", [False, True], ids=["pandas", "polars"])
def test_overwrite_chain_deeper_than_recursion_limit(polars_backend):
    """Fusing a long ``a = a + 1`` chain yields one single-use expression nested
    once per stage; the kernels must evaluate it without hitting the Python
    recursion limit."""
    depth = sys.getrecursionlimit() + 500
    src = Op()
    prev = src
    for _ in range(depth):
        stage = AssignMapOp(entries={"a": BinOpExpr(operator.add, Col("a"), Const(1.0))},
                            inputs=[prev])
        prev.outputs = [stage]
        prev = stage
    fused = fuse_assign_maps(prev)
    assert [src] == fused.inputs

    df = pd.DataFrame({"a": [1.0, -2.0, np.nan]})
    frame = pl.from_pandas(df) if polars_backend else df
    op = _map_kernel(polars_backend)(entries=fused.entries, inputs=[])
    op.on_impl_selected(None)
    out = op.process("fit_transform", [frame])
    np.testing.assert_array_equal([1.0 + depth, -2.0 + depth, np.nan],
                                  np.asarray(out["a"], dtype=float))


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
    with st.config(implementation_selector=_selector(polars)):
        fused_ops, *_ = optimize_(dag, _cfg(fuse=True))
        unfused_ops, *_ = optimize_(dag, _cfg(fuse=False))
        _assert_map_kernel(fused_ops, polars)
        _assert_map_kernel(unfused_ops, polars)
        fused = run_plan(fused_ops)
        unfused = run_plan(unfused_ops)
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


def test_fused_equals_unfused_after_loc_row_col():
    """Assign chains after ``df.loc[mask, cols]`` still fuse and match.

    Loc is pandas-only today, so this arm stays on the pandas map kernel.
    """
    df = pd.DataFrame({"a": [1.0, -1.0, 2.0], "b": [10.0, 20.0, 30.0]})
    src = st.as_data_op(df)
    sub = src.loc[src["a"] > 0, ["a", "b"]]
    m0 = sub.assign(c=sub["a"] * 2)
    m1 = m0.assign(d=m0["c"] + m0["b"])
    assert_fused_equals_unfused(m1, polars_backend=False)


def test_fused_equals_unfused_after_loc_single_column_series():
    """``df.loc[mask, "col"]`` is a SERIES; later frame assigns still agree.

    Loc is pandas-only today, so this arm stays on the pandas map kernel.
    """
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
    assert_fused_equals_unfused(m1, polars_backend=False, expect_fusion=False)


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
