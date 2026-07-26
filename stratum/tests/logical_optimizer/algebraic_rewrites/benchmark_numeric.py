"""
Benchmark: algebraic rewrite impact on DAG size **and runtime**.

Each rewrite was developed on a separate branch.  This script measures the
**marginal** impact of every individual rewrite (and their combined effect) so
you can quantify what each branch contributes — both in DAG node reduction and,
critically, in actual wall-clock execution time on real DataFrames.

Metrics collected per rewrite:
* DAG nodes before / after the rewrite pass
* Nodes eliminated (Δ) & reduction percentage
* Wall-clock **execution** time (ms) — runs the linearized plan through the
  SequentialScheduler so the data actually flows through every op
* Speedup factor (time_before / time_after)

Usage
-----
    # Run all benchmarks (default: 100k-row DataFrames)
    python -m pytest stratum/tests/logical_optimizer/algebraic_rewrites/benchmark_numeric.py -v -s

    # Run a single rewrite benchmark
    python -m pytest stratum/tests/logical_optimizer/algebraic_rewrites/benchmark_numeric.py \\
        -k "test_bench_log_exp" -v -s

    # Adjust data size via CLI
    python -m pytest stratum/tests/logical_optimizer/algebraic_rewrites/benchmark_numeric.py \\
        --bench-rows 1000000 -v -s

    # Visualize DAG before/after (needs Graphviz dot on PATH)
    python -m pytest stratum/tests/logical_optimizer/algebraic_rewrites/benchmark_numeric.py \\
        --bench-viz -k "test_bench_viz_combined" -v -s

Design
------
Each benchmark builds a DataOp DAG dominated by a rewrite pattern, then runs
the full ``optimize → SequentialScheduler.evaluate`` pipeline with the target
rewrite OFF (baseline) and ON (treatment).  Execution time is measured via
``perf_counter`` around the scheduler's ``evaluate()`` call, so it reflects the
real cost of computing (or skipping) the eliminated operations on DataFrame
columns.
"""

from __future__ import annotations

import os
import unittest
import time
from dataclasses import dataclass, asdict
from typing import ClassVar

import numpy as np
import pandas as pd

import stratum as st
from stratum.optimizer._optimize import (
    optimize,
    OptConfig,
    convert_to_ops,
    extract_frame_operators,
    extract_numeric_operators,
    run_op_cse_pass,
    choice_unrolling,
)
from stratum.optimizer._algebraic_rewrites import AlgebraicRewritesConfig, algebraic_rewrites
from stratum.optimizer.ir._dataframe_ops import add_splitting_op
from stratum.optimizer._op_utils import show_graph
from stratum.runtime._scheduler import SequentialScheduler
from stratum._config import FLAGS

# ---------------------------------------------------------------------------
# Global (overridden by conftest.py via pytest --bench-rows / --bench-viz)
# ---------------------------------------------------------------------------

BENCH_ROWS = 100_000
BENCH_VIZ = False  # set to True via --bench-viz to render before/after DAG PNGs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n_rows: int = None, seed: int = 42) -> pd.DataFrame:
    """Return a DataFrame with *n_rows* of numeric columns."""
    if n_rows is None:
        n_rows = BENCH_ROWS
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.uniform(0.1, 10.0, n_rows),
            "b": rng.uniform(0.1, 10.0, n_rows),
            "c": rng.normal(0.0, 2.0, n_rows),
        }
    )


def _plan_length(linearized_plan: list) -> int:
    """Number of ops in the linearized plan (excl. sentinels)."""
    return len(linearized_plan)


def _execute_plan(linearized_plan, split_pos, flagged_ops, warmup: bool = False) -> float:
    """Run the linearized plan through SequentialScheduler.evaluate() and return
    wall-clock time in seconds.

    Parameters
    ----------
    warmup : bool
        If True, run once before timing to warm caches / JIT.
    """
    if warmup:
        sched = SequentialScheduler(list(linearized_plan), split_pos, list(flagged_ops))
        sched.evaluate()

    sched = SequentialScheduler(list(linearized_plan), split_pos, list(flagged_ops))
    t0 = time.perf_counter()
    sched.evaluate()
    return time.perf_counter() - t0


def _all_off_config():
    """Config with every algebraic rewrite disabled."""
    return AlgebraicRewritesConfig(
        log_exp=False,
        exp_log=False,
        sqrt_square=False,
        log1p_expm1=False,
        expm1_log1p=False,
        identity_op=False,
        abs_abs=False,
        add_zero=False,
        exp_minus_one=False,
        identity_subtract=False,
        any_mul_zero=False,
        constant_folding=False,
    )


def _all_on_config():
    """Config with every algebraic rewrite enabled."""
    return AlgebraicRewritesConfig()


# ---------------------------------------------------------------------------
# DAG visualization
# ---------------------------------------------------------------------------

_VIZ_DIR = os.path.join(os.path.dirname(__file__), "viz")


def _dag_text_summary(root: "Op") -> str:
    """Return a human-readable text dump of the DAG rooted at *root*.

    Lists every node (type, name) and its edges.  Useful when graphviz ``dot``
    is not installed on the machine.
    """
    from stratum.optimizer._op_utils import topological_iterator

    lines = []
    for op in topological_iterator(root):
        op.update_name()
        label = f"{type(op).__name__}({op.name})"
        ins = ", ".join(type(i).__name__ for i in op.inputs)
        outs = ", ".join(type(o).__name__ for o in op.outputs)
        lines.append(f"  {label}")
        if ins:
            lines.append(f"    ← [{ins}]")
        if outs:
            lines.append(f"    → [{outs}]")
    return "\n".join(lines)


def _prepare_op_root(dag) -> "Op":
    """Replicate the optimize pipeline up to (but not including) algebraic rewrites.

    Returns the :class:`Op` root so it can be rendered with ``show_graph`` or
    passed directly to ``algebraic_rewrites``.
    """
    root = convert_to_ops(dag, None)
    root = add_splitting_op(root)
    root = extract_frame_operators(root)
    root = extract_numeric_operators(root)
    if FLAGS.cse:
        root = run_op_cse_pass(root)
    root = choice_unrolling(root)
    return root


def _render_dag(root: "Op", filename: str) -> str:
    """Render *root* via graphviz; fall back to a text dump if ``dot`` is missing.

    Returns the path to the output file (.png or .txt).
    """
    os.makedirs(_VIZ_DIR, exist_ok=True)
    try:
        show_graph(root, os.path.join("viz", filename))
        return os.path.abspath(os.path.join("graphs", "viz", filename + ".png"))
    except Exception:
        txt_path = os.path.join(_VIZ_DIR, filename + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(_dag_text_summary(root))
        return txt_path


def _render_rewrite_viz(
    name: str,
    dag,
    enable_flags: dict,
) -> tuple[str, str]:
    """Render before/after DAG graphs for a single rewrite.

    Returns (before_path, after_path).
    """
    root_before = _prepare_op_root(dag)
    before_path = _render_dag(root_before, f"{name}_before")

    on_cfg = AlgebraicRewritesConfig(**enable_flags)
    root_after = algebraic_rewrites(root_before, on_cfg)
    after_path = _render_dag(root_after, f"{name}_after")

    return before_path, after_path


@dataclass
class RewriteResult:
    name: str
    nodes_before: int
    nodes_after: int
    delta: int
    reduction_pct: float
    runtime_before_ms: float
    runtime_after_ms: float
    speedup: float


# ---------------------------------------------------------------------------
# Benchmark cases (one per rewrite)
# ---------------------------------------------------------------------------


class TestAlgebraicRewriteBenchmark(unittest.TestCase):
    """Each test method benchmarks a single algebraic rewrite pass.

    The naming convention ``test_bench_<rewrite_name>`` lets you filter with pytest's
    ``-k`` flag, e.g. ``-k "test_bench_log_exp or test_bench_sqrt_square"``.
    """

    results: ClassVar[list[RewriteResult]] = []

    @classmethod
    def setUpClass(cls):
        cls.results = []

    def _measure(
        self,
        name: str,
        build_dag,
        enable_flags: dict,
        *,
        disable_constant_folding: bool = True,
        skip_runtime: bool = False,
    ):
        """Build a DAG, optimize & execute with/without *enable_flags*, record result.

        Parameters
        ----------
        name : str
            Human-readable rewrite name.
        build_dag : () -> DataOp
            Callable that returns the DAG to benchmark.
        enable_flags : dict
            Kwargs to ``AlgebraicRewritesConfig`` that toggle just the rewrite
            under test ON.  All other rewrites are OFF.
        disable_constant_folding : bool
            When True, constant folding is also disabled in both runs so that
            identity/elimination rewrites are measured in isolation.
        """
        dag = build_dag()

        # -- baseline: target rewrite OFF -------------------------------------------
        off_cfg = _all_off_config()
        off_cfg = AlgebraicRewritesConfig(
            **{**asdict(off_cfg), "constant_folding": False}
        )
        off_plan, off_split, off_flagged = optimize(
            dag, config=OptConfig(algebraic_rewrite_config=off_cfg),
        )
        nodes_before = _plan_length(off_plan)

        # -- treatment: target rewrite ON --------------------------------------------
        on_flags = {**asdict(off_cfg), **enable_flags}
        if not disable_constant_folding:
            on_flags["constant_folding"] = True
        on_cfg = AlgebraicRewritesConfig(**on_flags)
        on_plan, on_split, on_flagged = optimize(
            dag, config=OptConfig(algebraic_rewrite_config=on_cfg),
        )
        nodes_after = _plan_length(on_plan)

        # -- execute & time both plans -----------------------------------------------
        if skip_runtime:
            runtime_before_s = runtime_after_s = 0.0
        else:
            _execute_plan(off_plan, off_split, off_flagged, warmup=True)
            runtime_before_s = _execute_plan(off_plan, off_split, off_flagged)
            runtime_after_s = _execute_plan(on_plan, on_split, on_flagged)

        delta = nodes_before - nodes_after
        pct = (delta / nodes_before * 100) if nodes_before else 0.0
        speedup = runtime_before_s / runtime_after_s if runtime_after_s > 0 else float("inf")

        result = RewriteResult(
            name=name,
            nodes_before=nodes_before,
            nodes_after=nodes_after,
            delta=delta,
            reduction_pct=pct,
            runtime_before_ms=runtime_before_s * 1000,
            runtime_after_ms=runtime_after_s * 1000,
            speedup=speedup,
        )
        TestAlgebraicRewriteBenchmark.results.append(result)

        if skip_runtime:
            print(
                f"  {name:30s} | "
                f"nodes: {nodes_before:2d}→{nodes_after:2d}  "
                f"(runtime skipped — scalar DAG)"
            )
        else:
            print(
                f"  {name:30s} | "
                f"nodes: {nodes_before:2d}→{nodes_after:2d}  "
                f"time: {result.runtime_before_ms:7.2f}→{result.runtime_after_ms:7.2f} ms  "
                f"({speedup:.1f}× faster)"
            )

        # Sanity: the rewrite must not increase node count or runtime.
        self.assertGreaterEqual(
            nodes_before, nodes_after,
            f"{name}: rewrite increased DAG size ({nodes_before} → {nodes_after})",
        )

    # ---- log/exp inverses ----------------------------------------------------------

    def test_bench_log_exp(self):
        """log(x) → exp → x  (2 ops eliminated)"""
        self._measure(
            "log_exp",
            lambda: st.as_data_op(_make_df()).skb.apply_func(np.log).skb.apply_func(np.exp),
            {"log_exp": True},
        )

    def test_bench_exp_log(self):
        """exp(x) → log → x  (2 ops eliminated)"""
        self._measure(
            "exp_log",
            lambda: st.as_data_op(_make_df()).skb.apply_func(np.exp).skb.apply_func(np.log),
            {"exp_log": True},
        )

    def test_bench_log1p_expm1(self):
        """log1p(x) → expm1 → x  (2 ops eliminated)"""
        self._measure(
            "log1p_expm1",
            lambda: st.as_data_op(_make_df()).skb.apply_func(np.log1p).skb.apply_func(np.expm1),
            {"log1p_expm1": True},
        )

    def test_bench_expm1_log1p(self):
        """expm1(x) → log1p → x  (2 ops eliminated)"""
        self._measure(
            "expm1_log1p",
            lambda: st.as_data_op(_make_df()).skb.apply_func(np.expm1).skb.apply_func(np.log1p),
            {"expm1_log1p": True},
        )

    # ---- sqrt / square --------------------------------------------------------------

    def test_bench_sqrt_square(self):
        """x² → sqrt → |x|  (2 ops → 1 abs op)"""
        self._measure(
            "sqrt_square",
            lambda: st.as_data_op(_make_df()).skb.apply_func(np.square).skb.apply_func(np.sqrt),
            {"sqrt_square": True},
        )

    # ---- identity operations --------------------------------------------------------

    def test_bench_identity_op(self):
        """x * 1 → x  (1 multiply eliminated)"""
        df = st.as_data_op(_make_df())
        self._measure(
            "identity_op",
            lambda: df * 1,
            {"identity_op": True},
        )

    def test_bench_add_zero(self):
        """x + 0 → x  (1 add eliminated)"""
        df = st.as_data_op(_make_df())
        self._measure(
            "add_zero",
            lambda: df + 0,
            {"add_zero": True},
        )

    def test_bench_identity_subtract(self):
        """x - 0 → x  (1 subtract eliminated)"""
        df = st.as_data_op(_make_df())
        self._measure(
            "identity_subtract",
            lambda: df - 0,
            {"identity_subtract": True},
        )

    # ---- abs/abs collapse -----------------------------------------------------------

    def test_bench_abs_abs(self):
        """abs(abs(x)) → abs(x)  (1 abs eliminated)"""
        self._measure(
            "abs_abs",
            lambda: st.as_data_op(_make_df()).skb.apply_func(np.abs).skb.apply_func(np.abs),
            {"abs_abs": True},
        )

    # ---- exp minus one → expm1 ------------------------------------------------------

    def test_bench_exp_minus_one(self):
        """exp(x) - 1 → expm1(x)  (2 ops → 1)"""
        df = st.as_data_op(_make_df())
        self._measure(
            "exp_minus_one",
            lambda: df.skb.apply_func(np.exp) - 1,
            {"exp_minus_one": True},
        )

    # ---- multiply-by-zero → constant-zero -------------------------------------------

    def test_bench_any_mul_zero(self):
        """x * 0 → ValueOp(0.0)  (multiply + dead source eliminated)"""
        df = st.as_data_op(_make_df())
        self._measure(
            "any_mul_zero",
            lambda: df * 0,
            {"any_mul_zero": True},
        )

    # ---- constant folding -----------------------------------------------------------

    def test_bench_constant_folding(self):
        """log(1) → 0, exp(0) → 1, etc. (NumericOp → ValueOp).

        Runtime measurement is skipped here — constant folding operates on
        compile-time scalars, not DataFrames, so the scheduler cannot execute
        the folded plan (ValueOp returns a raw scalar, not a DataFrame column)."""
        self._measure(
            "constant_folding",
            lambda: (
                st.as_data_op(1)
                .skb.apply_func(np.log)  # log(1)=0
                .skb.apply_func(np.exp)  # exp(0)=1
                .skb.apply_func(np.sqrt)  # sqrt(1)=1
            ),
            {"constant_folding": True},
            disable_constant_folding=False,
            skip_runtime=True,
        )

    # ---- visualization ---------------------------------------------------------------

    def test_bench_viz_combined(self):
        """Render before/after DAG graphs for the combined rewrite pass.

        Only active when ``--bench-viz`` is passed to pytest.  Outputs PNGs to
        ``stratum/tests/logical_optimizer/algebraic_rewrites/viz/``.
        """
        if not BENCH_VIZ:
            self.skipTest("--bench-viz not set (pass --bench-viz to pytest)")

        df = _make_df(n_rows=100)  # small DF for readable graph labels
        dag = (
            st.as_data_op(df).skb.apply_func(np.log).skb.apply_func(np.exp)  # log_exp
            * 1  # identity
            + 0  # add_zero
        )
        dag = dag.skb.apply_func(np.square).skb.apply_func(np.sqrt)  # sqrt_square

        before_path, after_path = _render_rewrite_viz(
            "combined",
            dag,
            asdict(_all_on_config()),
        )
        print(f"\n  Viz before: {before_path}")
        print(f"  Viz after:  {after_path}")

    # ---- combined: complex DAG ------------------------------------------------------

    def test_bench_combined_all_on(self):
        """All rewrites ON vs all OFF — measures cumulative runtime impact."""
        df = _make_df()
        dag = (
            st.as_data_op(df).skb.apply_func(np.log).skb.apply_func(np.exp)  # log_exp
            * 1  # identity
            + 0  # add_zero
            - 0  # identity_subtract
        )
        dag = dag.skb.apply_func(np.square).skb.apply_func(np.sqrt)  # sqrt_square
        dag = dag.skb.apply_func(np.exp) - 1  # exp_minus_one

        off_cfg = OptConfig(algebraic_rewrite_config=_all_off_config())
        on_cfg = OptConfig(algebraic_rewrite_config=_all_on_config())

        off_plan, off_split, off_flagged = optimize(dag, config=off_cfg)
        on_plan, on_split, on_flagged = optimize(dag, config=on_cfg)

        nodes_before = _plan_length(off_plan)
        nodes_after = _plan_length(on_plan)

        _execute_plan(off_plan, off_split, off_flagged, warmup=True)
        runtime_before_s = _execute_plan(off_plan, off_split, off_flagged)
        runtime_after_s = _execute_plan(on_plan, on_split, on_flagged)

        delta = nodes_before - nodes_after
        pct = (delta / nodes_before * 100) if nodes_before else 0.0
        speedup = runtime_before_s / runtime_after_s if runtime_after_s > 0 else float("inf")

        result = RewriteResult(
            name="★ COMBINED (all on vs all off)",
            nodes_before=nodes_before,
            nodes_after=nodes_after,
            delta=delta,
            reduction_pct=pct,
            runtime_before_ms=runtime_before_s * 1000,
            runtime_after_ms=runtime_after_s * 1000,
            speedup=speedup,
        )
        TestAlgebraicRewriteBenchmark.results.append(result)

        print(
            f"\n  {'★ COMBINED':30s} | "
            f"nodes: {nodes_before:2d}→{nodes_after:2d}  "
            f"time: {result.runtime_before_ms:7.2f}→{result.runtime_after_ms:7.2f} ms  "
            f"({speedup:.1f}× faster)"
        )


# ---------------------------------------------------------------------------
# Summary report (printed after all benchmarks)
# ---------------------------------------------------------------------------


def _print_summary(results: list[RewriteResult]) -> None:
    if not results:
        return
    print("\n" + "=" * 90)
    print("ALGEBRAIC REWRITE BENCHMARK SUMMARY")
    print(f"DataFrame rows: {BENCH_ROWS:,}")
    print("=" * 90)
    print(
        f"{'Rewrite':30s}  {'Nodes':>9s}  {'Time (ms)':>16s}  "
        f"{'Speedup':>8s}"
    )
    print(f"{'':30s}  {'Before After':>9s}  {'Before':>7s} {'After':>7s}  ")
    print("-" * 90)
    for r in results:
        if r.runtime_before_ms == 0 and r.runtime_after_ms == 0:
            time_col = "(scalar DAG)"
        else:
            time_col = f"{r.runtime_before_ms:7.2f} {r.runtime_after_ms:7.2f}  {r.speedup:6.1f}×"
        print(
            f"{r.name:30s}  {r.nodes_before:3d} → {r.nodes_after:<3d}  {time_col}"
        )
    print("=" * 90)

    total_before = sum(r.nodes_before for r in results if not r.name.startswith("★"))
    total_after = sum(r.nodes_after for r in results if not r.name.startswith("★"))
    if total_before:
        print(
            f"\nAggregate (excl. combined): "
            f"{total_before} → {total_after} nodes "
            f"({(total_before - total_after) / total_before * 100:.1f}% reduction)"
        )
    print()


# ---------------------------------------------------------------------------
# Module-level teardown (pytest)
# ---------------------------------------------------------------------------


def tearDownModule():
    """Print the summary table once, after all tests in this module finish."""
    _print_summary(TestAlgebraicRewriteBenchmark.results)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
