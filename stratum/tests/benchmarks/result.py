"""Stratum optimizer benchmark framework — result types and JSON serialization."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class CaseMetrics:
    """Per-case measurement data."""

    # ---- DAG size -----------------------------------------------------------
    nodes_before: int
    nodes_after: int
    nodes_delta: int
    reduction_pct: float

    # ---- runtime ------------------------------------------------------------
    runtime_before_ms: float
    runtime_after_ms: float
    speedup: float

    # ---- optimizer timing ---------------------------------------------------
    optimize_time_ms: float

    # ---- rules --------------------------------------------------------------
    rules_triggered: list[str] = field(default_factory=list)
    rules_count: int = 0


@dataclass
class CaseResult:
    """Complete result for a single benchmark case."""

    name: str
    description: str = ""
    passed: bool = True
    error: str = ""
    metrics: CaseMetrics | None = None

    # ---- correctness --------------------------------------------------------
    outputs_equal: bool = True
    output_diff: str = ""


@dataclass
class RuleCoverage:
    """Coverage report for rewrite rules."""

    total_rules: int
    triggered: list[str]
    never_triggered: list[str]
    frequency: dict[str, int]  # rule name → application count


@dataclass
class BenchmarkReport:
    """Top-level report written to JSON."""

    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    data_rows: int = 0
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    cases: list[CaseResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    rule_coverage: RuleCoverage | None = None


def report_to_json(report: BenchmarkReport, indent: int = 2) -> str:
    """Serialize *report* to a JSON string."""
    return json.dumps(asdict(report), indent=indent, default=str)


def write_report(report: BenchmarkReport, path: str) -> None:
    """Write *report* as JSON to *path*."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_to_json(report))
