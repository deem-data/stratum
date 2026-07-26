"""Benchmark runner: orchestrates test cases, collects metrics, produces reports.

Usage
-----
    from stratum.tests.benchmarks.runner import BenchmarkRunner
    from stratum.tests.benchmarks.cases import ALL_CASES

    runner = BenchmarkRunner(data_rows=100_000)
    report = runner.run(ALL_CASES)
    runner.write_report(report, "benchmark_report.json")
"""

from __future__ import annotations

import time
import logging
from dataclasses import asdict

import numpy as np
import pandas as pd

from stratum.optimizer._optimize import optimize, OptConfig
from stratum.optimizer._algebraic_rewrites import AlgebraicRewritesConfig
from stratum.runtime._scheduler import SequentialScheduler

from .result import (
    BenchmarkReport,
    CaseResult,
    CaseMetrics,
    RuleCoverage,
    write_report,
)
from .coverage import RuleCoverageTracker
from .cases import BenchmarkCase

logger = logging.getLogger(__name__)


def _make_dataframe(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Create a DataFrame with positive-only columns safe for log/sqrt ops."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.uniform(0.5, 5.0, n_rows),
            "b": rng.uniform(0.5, 5.0, n_rows),
            "c": rng.uniform(0.5, 5.0, n_rows),
        }
    )


def _plan_length(plan: list) -> int:
    return len(plan)


def _execute_plan(plan, split_pos, flagged_ops) -> float:
    """Execute a linearized plan and return wall-clock seconds.

    Creates a fresh scheduler each time.  Important: the plan ops are shared
    objects, so a warmup run that consumes DataFrame data will leave ValueOps
    drained for subsequent runs.  Do NOT call this more than once per plan.
    """
    sched = SequentialScheduler(plan, split_pos, flagged_ops)
    t0 = time.perf_counter()
    try:
        sched.evaluate()
    except Exception:
        pass
    return time.perf_counter() - t0


def _outputs_equal(out1, out2, rtol: float = 1e-5) -> bool:
    """Compare two scheduler outputs for semantic equivalence.

    Handles DataFrames, numpy arrays, scalars, and None.
    """
    if out1 is None and out2 is None:
        return True
    if out1 is None or out2 is None:
        return False

    # DataFrame comparison
    if isinstance(out1, pd.DataFrame) and isinstance(out2, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(out1, out2, rtol=rtol, check_dtype=False)
            return True
        except AssertionError:
            return False

    # Array-like
    try:
        a1 = np.asarray(out1, dtype=float)
        a2 = np.asarray(out2, dtype=float)
        np.testing.assert_allclose(a1, a2, rtol=rtol)
        return True
    except (AssertionError, TypeError, ValueError):
        return False


def _all_off_config() -> AlgebraicRewritesConfig:
    """Every rewrite disabled."""
    return AlgebraicRewritesConfig(
        log_exp=False, exp_log=False, sqrt_square=False,
        log1p_expm1=False, expm1_log1p=False, identity_op=False,
        abs_abs=False, add_zero=False, exp_minus_one=False,
        identity_subtract=False, any_mul_zero=False, constant_folding=False,
    )


def _detect_triggered_rules(off_plan, on_plan) -> list[str]:
    """Detect which rewrite rules fired by comparing op type/name sets.

    Maps eliminated ``NumericOp`` names to specific rewrite rules based on
    the operation they perform.  Also checks for ``BinOp`` eliminations
    (``any_mul_zero``) and op-name transformations (``exp_minus_one``,
    ``abs_abs``).
    """
    off_ops = {(type(op).__name__, getattr(op, "name", "")) for op in off_plan}
    on_ops = {(type(op).__name__, getattr(op, "name", "")) for op in on_plan}
    eliminated = off_ops - on_ops
    added = on_ops - off_ops

    # NumericOp name → rule mapping
    _OPNAME_TO_RULE: dict[str, str] = {
        "log": "log_exp",
        "exp": "exp_log",
        "log1p": "log1p_expm1",
        "expm1": "expm1_log1p",
        "multiply": "identity_op",   # x * 1 → x  (also x * 0 → 0, handled below)
        "add": "add_zero",           # x + 0 → x
        "subtract": "identity_subtract",  # x - 0 → x
        "abs": "abs_abs",            # abs(abs(x)) → abs(x)
        "sqrt": "sqrt_square",       # sqrt(square(x)) → abs(x)
        "square": "sqrt_square",     # (eliminated as part of sqrt_square)
    }

    triggered: set[str] = set()

    # Detect from eliminated ops
    elim_abs_count = 0
    elim_multiply_count = 0
    for typ_name, op_name in eliminated:
        if typ_name == "NumericOp":
            rule = _OPNAME_TO_RULE.get(op_name)
            if rule:
                triggered.add(rule)
            if op_name == "abs":
                elim_abs_count += 1
            if op_name == "multiply":
                elim_multiply_count += 1
        elif typ_name == "BinOp":
            # BinOp eliminated — likely any_mul_zero (x * 0)
            triggered.add("any_mul_zero")

    # abs_abs: one abs eliminated but one remains → abs_abs fired
    has_abs_in_on = any(t == "NumericOp" and n == "abs" for t, n in on_ops)
    if elim_abs_count == 1 and has_abs_in_on:
        triggered.add("abs_abs")

    # any_mul_zero: multiply eliminated AND no multiply in ON → could be x*0
    # (identity_op also eliminates multiply but for x*1)
    has_multiply_in_on = any(t == "NumericOp" and n == "multiply" for t, n in on_ops)
    if elim_multiply_count >= 1 and not has_multiply_in_on:
        # Check if a ValueOp(0.0) was added
        if any(t == "ValueOp" for t, _ in added):
            triggered.add("any_mul_zero")

    # exp_minus_one: exp + subtract eliminated, expm1 added
    elim_has_exp = any(n == "exp" for _, n in eliminated)
    elim_has_subtract = any(n == "subtract" for _, n in eliminated)
    added_has_expm1 = any(t == "NumericOp" and n == "expm1" for t, n in added)
    if elim_has_exp and elim_has_subtract and added_has_expm1:
        triggered.add("exp_minus_one")
        triggered.discard("exp_log")
        triggered.discard("identity_subtract")

    # sqrt_square: square + sqrt eliminated, abs added
    elim_has_square = any(n == "square" for _, n in eliminated)
    elim_has_sqrt = any(n == "sqrt" for _, n in eliminated)
    added_has_abs = any(t == "NumericOp" and n == "abs" for t, n in added)
    if elim_has_square and elim_has_sqrt and added_has_abs:
        triggered.add("sqrt_square")

    # Constant folding: if NumericOps were replaced by ValueOps
    off_num_count = sum(1 for t, _ in off_ops if t == "NumericOp")
    on_num_count = sum(1 for t, _ in on_ops if t == "NumericOp")
    on_val_count = sum(1 for t, _ in on_ops if t == "ValueOp")
    off_val_count = sum(1 for t, _ in off_ops if t == "ValueOp")
    if off_num_count > 0 and on_num_count == 0 and on_val_count > off_val_count:
        triggered.add("constant_folding")

    return sorted(triggered)


class BenchmarkRunner:
    """Orchestrates benchmark execution.

    Parameters
    ----------
    data_rows : int
        Number of rows in the test DataFrames.
    output_json : str | None
        If set, write the JSON report to this path after :meth:`run`.
    """

    def __init__(self, data_rows: int = 100_000, output_json: str | None = None):
        self.data_rows = data_rows
        self.output_json = output_json
        self.coverage_tracker = RuleCoverageTracker()

    def run(self, cases: list[BenchmarkCase]) -> BenchmarkReport:
        """Execute all *cases* and return a :class:`BenchmarkReport`."""
        report = BenchmarkReport(
            data_rows=self.data_rows,
            total_cases=len(cases),
        )
        for case in cases:
            result = self._run_one(case)
            report.cases.append(result)
            if result.passed:
                report.passed += 1
            else:
                report.failed += 1
            logger.info(
                "  %-35s  %s  nodes=%d→%d  %.1f×",
                case.name,
                "✓" if result.passed else "✗",
                result.metrics.nodes_before if result.metrics else 0,
                result.metrics.nodes_after if result.metrics else 0,
                result.metrics.speedup if result.metrics else 0,
            )

        report.rule_coverage = self.coverage_tracker.snapshot()
        report.summary = self._build_summary(report)

        if self.output_json:
            write_report(report, self.output_json)

        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_one(self, case: BenchmarkCase) -> CaseResult:
        """Execute a single case and return its result."""
        df = _make_dataframe(self.data_rows, seed=hash(case.name) % (2**31))
        try:
            dag_off = case.dag_builder(df)
            dag_on = case.dag_builder(df.copy())  # independent copy
        except Exception as exc:
            return CaseResult(
                name=case.name, description=case.description,
                passed=False, error=f"DAG build failed: {exc}",
            )

        # --- optimize: OFF ------------------------------------------------
        off_cfg = OptConfig(algebraic_rewrite_config=_all_off_config())
        try:
            off_plan, off_split, off_flagged = optimize(dag_off, config=off_cfg)
        except Exception as exc:
            return CaseResult(
                name=case.name, description=case.description,
                passed=False, error=f"Optimize (OFF) failed: {exc}",
            )

        # --- optimize: ON -------------------------------------------------
        t_opt = time.perf_counter()
        on_cfg = OptConfig(algebraic_rewrite_config=AlgebraicRewritesConfig())
        try:
            on_plan, on_split, on_flagged = optimize(dag_on, config=on_cfg)
        except Exception as exc:
            return CaseResult(
                name=case.name, description=case.description,
                passed=False, error=f"Optimize (ON) failed: {exc}",
            )
        opt_time_ms = (time.perf_counter() - t_opt) * 1000

        # --- nodes ---------------------------------------------------------
        nodes_before = _plan_length(off_plan)
        nodes_after = _plan_length(on_plan)

        # --- execute & time (fresh DAGs so data isn't consumed) ------------
        df2 = _make_dataframe(self.data_rows, seed=hash(case.name) % (2**31))
        dag_ref = case.dag_builder(df2)
        ref_plan, ref_split, ref_flagged = optimize(
            dag_ref, config=off_cfg,
        )
        ref_sched = SequentialScheduler(ref_plan, ref_split, ref_flagged)
        try:
            ref_out = ref_sched.evaluate()
        except Exception:
            ref_out = None

        df3 = _make_dataframe(self.data_rows, seed=hash(case.name) % (2**31))
        dag_opt = case.dag_builder(df3)
        opt_plan2, opt_split2, opt_flagged2 = optimize(
            dag_opt, config=on_cfg,
        )
        opt_sched = SequentialScheduler(opt_plan2, opt_split2, opt_flagged2)
        try:
            opt_out = opt_sched.evaluate()
        except Exception:
            opt_out = None

        # --- correctness ---------------------------------------------------
        equal = _outputs_equal(ref_out, opt_out)
        diff = "" if equal else "Outputs differ (or could not be compared)"

        # --- runtime (separate fresh DAGs) ----------------------------------
        df4 = _make_dataframe(self.data_rows, seed=hash(case.name) % (2**31))
        dag_time_off = case.dag_builder(df4)
        time_off_plan, to_split, to_flagged = optimize(dag_time_off, config=off_cfg)
        runtime_before_s = _execute_plan(time_off_plan, to_split, to_flagged)

        df5 = _make_dataframe(self.data_rows, seed=hash(case.name) % (2**31))
        dag_time_on = case.dag_builder(df5)
        time_on_plan, tn_split, tn_flagged = optimize(dag_time_on, config=on_cfg)
        runtime_after_s = _execute_plan(time_on_plan, tn_split, tn_flagged)

        # --- rules ---------------------------------------------------------
        triggered = _detect_triggered_rules(off_plan, on_plan)
        self.coverage_tracker.record(triggered)

        # --- metrics -------------------------------------------------------
        delta = nodes_before - nodes_after
        pct = (delta / nodes_before * 100) if nodes_before else 0.0
        speedup = runtime_before_s / runtime_after_s if runtime_after_s > 0 else float("inf")

        metrics = CaseMetrics(
            nodes_before=nodes_before,
            nodes_after=nodes_after,
            nodes_delta=delta,
            reduction_pct=pct,
            runtime_before_ms=runtime_before_s * 1000,
            runtime_after_ms=runtime_after_s * 1000,
            speedup=speedup,
            optimize_time_ms=opt_time_ms,
            rules_triggered=triggered,
            rules_count=len(triggered),
        )

        return CaseResult(
            name=case.name,
            description=case.description,
            passed=equal,
            metrics=metrics,
            outputs_equal=equal,
            output_diff=diff,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(report: BenchmarkReport) -> dict:
        cases = report.cases
        if not cases:
            return {}

        total_nodes_before = sum(
            c.metrics.nodes_before for c in cases if c.metrics
        )
        total_nodes_after = sum(
            c.metrics.nodes_after for c in cases if c.metrics
        )
        total_rules = sum(
            c.metrics.rules_count for c in cases if c.metrics
        )
        avg_speedup = (
            sum(c.metrics.speedup for c in cases if c.metrics and c.metrics.speedup < float("inf"))
            / max(len([c for c in cases if c.metrics and c.metrics.speedup < float("inf")]), 1)
        )

        return {
            "total_nodes_before": total_nodes_before,
            "total_nodes_after": total_nodes_after,
            "overall_reduction_pct": (
                (total_nodes_before - total_nodes_after) / total_nodes_before * 100
                if total_nodes_before else 0
            ),
            "total_rules_triggered": total_rules,
            "avg_speedup": round(avg_speedup, 2),
            "correctness_pass_rate": (
                report.passed / report.total_cases * 100
                if report.total_cases else 0
            ),
        }


def write_report(report: BenchmarkReport, path: str) -> None:
    """Write *report* as JSON to *path* (module-level convenience)."""
    from .result import write_report as _wr
    _wr(report, path)
