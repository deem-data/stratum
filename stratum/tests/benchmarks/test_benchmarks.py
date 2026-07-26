"""pytest integration: run the full benchmark suite.

Usage
-----
    # Full suite (100k rows, summary to stdout)
    python -m pytest stratum/tests/benchmarks/test_benchmarks.py -v -s

    # Custom rows + JSON output
    python -m pytest stratum/tests/benchmarks/test_benchmarks.py -v -s \\
        --bench-rows 50000 --bench-json report.json

    # Single category
    python -m pytest stratum/tests/benchmarks/test_benchmarks.py -v -s \\
        -k "combined"

    # Standalone (no pytest)
    python stratum/tests/benchmarks/test_benchmarks.py --bench-rows 10000
"""

from __future__ import annotations

import sys
import unittest

from stratum.tests.benchmarks.runner import BenchmarkRunner
from stratum.tests.benchmarks.cases import ALL_CASES, NUMERIC_CASES, COMBINED_CASES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_option(option: str, default=None):
    """Read a pytest --bench-* option, with env-var fallback."""
    import os
    env_map = {
        "--bench-rows": ("STRATUM_BENCH_ROWS", int, 100_000),
        "--bench-json": ("STRATUM_BENCH_JSON", str, None),
    }
    if option in env_map:
        env_name, cast, fallback = env_map[option]
        env_val = os.environ.get(env_name)
        if env_val is not None:
            return cast(env_val)

    # Try reading from the last pytest invocation via conftest
    try:
        # conftest stores the config in the module
        from stratum.tests.benchmarks import conftest as _ct
        stored = getattr(_ct, "_pytest_config", None)
        if stored is not None:
            return stored.getoption(option, default=default if default is not None else fallback)
    except Exception:
        pass

    return default if default is not None else (fallback if 'fallback' in dir() else None)


# ---------------------------------------------------------------------------
# pytest test class
# ---------------------------------------------------------------------------


class TestBenchmarkSuite(unittest.TestCase):
    """Run the full benchmark suite and assert correctness."""

    @classmethod
    def setUpClass(cls):
        cls.rows = _get_option("--bench-rows", 100_000)
        cls.json_path = _get_option("--bench-json", None)

    def test_run_all_cases(self):
        """Execute all benchmark cases, verify correctness, print summary."""
        runner = BenchmarkRunner(
            data_rows=self.rows,
            output_json=self.json_path,
        )
        report = runner.run(ALL_CASES)

        # Print summary
        _print_report(report)

        # Assert all cases passed correctness check
        failed = [c for c in report.cases if not c.passed]
        if failed:
            names = ", ".join(c.name for c in failed)
            self.fail(f"Correctness failures: {names}")

    def test_run_numeric_cases(self):
        """Numeric rewrite cases only."""
        runner = BenchmarkRunner(data_rows=self.rows)
        report = runner.run(NUMERIC_CASES)
        _print_report(report)
        failed = [c for c in report.cases if not c.passed]
        self.assertFalse(failed, f"Correctness failures: {failed}")

    def test_run_combined_cases(self):
        """Combined multi-rewrite cases only."""
        runner = BenchmarkRunner(data_rows=self.rows)
        report = runner.run(COMBINED_CASES)
        _print_report(report)
        failed = [c for c in report.cases if not c.passed]
        self.assertFalse(failed, f"Correctness failures: {failed}")


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _print_report(report) -> None:
    """Print a human-readable benchmark report to stdout."""
    print("\n" + "=" * 95)
    print("STRATUM OPTIMIZER BENCHMARK REPORT")
    print(f"Data rows: {report.data_rows:,}  |  "
          f"Cases: {report.passed}/{report.total_cases} passed  |  "
          f"Timestamp: {report.timestamp}")
    print("=" * 95)
    print(f"{'Case':35s} {'OK':>3s}  {'Nodes':>9s}  "
          f"{'Time (ms)':>16s}  {'Speedup':>8s}  {'Rules':>6s}")
    print(f"{'':35s} {'':>3s}  {'Before After':>9s}  "
          f"{'Before':>7s} {'After':>7s}  {'':>8s}  {'':>6s}")
    print("-" * 95)
    for c in report.cases:
        m = c.metrics
        if m is None:
            print(f"  {c.name:33s}  {'ERR':>3s}  {c.error}")
            continue
        rules_str = ",".join(m.rules_triggered[:3])
        if len(m.rules_triggered) > 3:
            rules_str += f"+{len(m.rules_triggered)-3}"
        print(
            f"  {c.name:33s}  {'✓' if c.passed else '✗':>3s}  "
            f"{m.nodes_before:3d} → {m.nodes_after:<3d}  "
            f"{m.runtime_before_ms:7.2f} {m.runtime_after_ms:7.2f}  "
            f"{m.speedup:6.1f}×  {rules_str:>6s}"
        )
    print("=" * 95)

    # Rule coverage
    rc = report.rule_coverage
    if rc:
        print(f"\nRule coverage: {len(rc.triggered)}/{rc.total_rules} triggered")
        if rc.never_triggered:
            print(f"  Never triggered: {', '.join(rc.never_triggered)}")
        if rc.frequency:
            print("  Frequency:")
            for rule, count in sorted(rc.frequency.items(), key=lambda x: -x[1]):
                print(f"    {rule:25s} {count}")

    # Summary
    s = report.summary
    if s:
        print(f"\nSummary:")
        print(f"  Total nodes:         {s.get('total_nodes_before', 0)} → "
              f"{s.get('total_nodes_after', 0)} "
              f"({s.get('overall_reduction_pct', 0):.1f}% reduction)")
        print(f"  Avg speedup:         {s.get('avg_speedup', 0):.1f}×")
        print(f"  Correctness rate:    {s.get('correctness_pass_rate', 0):.1f}%")
        print(f"  Rules triggered:     {s.get('total_rules_triggered', 0)}")

    if report.data_rows and report.data_rows >= 100_000:
        print(f"\n  JSON report: {getattr(report, '_json_path', 'not written')}")

    print()


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Stratum optimizer benchmark")
    ap.add_argument("--bench-rows", type=int, default=100_000)
    ap.add_argument("--bench-json", type=str, default=None)
    ap.add_argument("--category", choices=["all", "numeric", "combined"], default="all")
    args = ap.parse_args()

    cases = {"all": ALL_CASES, "numeric": NUMERIC_CASES, "combined": COMBINED_CASES}[args.category]

    runner = BenchmarkRunner(data_rows=args.bench_rows, output_json=args.bench_json)
    report = runner.run(cases)
    _print_report(report)
