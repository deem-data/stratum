"""Rule coverage analysis for the Stratum optimizer benchmark.

Tracks which rewrite rules are available, which fire during benchmarking,
and produces a coverage report.
"""

from __future__ import annotations

from dataclasses import fields

from stratum.optimizer._algebraic_rewrites import AlgebraicRewritesConfig

from .result import RuleCoverage


# All rule names derived from AlgebraicRewritesConfig boolean fields.
_ALL_RULE_NAMES: frozenset[str] = frozenset(
    f.name for f in fields(AlgebraicRewritesConfig)
)


def get_all_rules() -> frozenset[str]:
    """Return the set of all known rewrite rule names."""
    return _ALL_RULE_NAMES


class RuleCoverageTracker:
    """Tracks which rewrite rules are triggered across benchmark runs.

    Instantiate once per benchmark suite; call :meth:`record` after each
    case completes, then :meth:`snapshot` at the end.
    """

    def __init__(self) -> None:
        self._freq: dict[str, int] = {r: 0 for r in _ALL_RULE_NAMES}

    def record(self, triggered: list[str]) -> None:
        """Increment counts for every rule in *triggered*."""
        for rule in triggered:
            if rule in self._freq:
                self._freq[rule] += 1

    def snapshot(self) -> RuleCoverage:
        """Return a coverage report based on all :meth:`record` calls so far."""
        triggered = sorted(r for r, c in self._freq.items() if c > 0)
        never = sorted(r for r, c in self._freq.items() if c == 0)
        return RuleCoverage(
            total_rules=len(_ALL_RULE_NAMES),
            triggered=triggered,
            never_triggered=never,
            frequency={r: c for r, c in self._freq.items() if c > 0},
        )
