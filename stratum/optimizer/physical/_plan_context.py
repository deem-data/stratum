"""Snapshot configuration used by lowering and implementation selection.

- Read process-wide settings from ``FLAGS`` and plan-specific settings from
  ``OptConfig`` when constructing ``PlanContext``.
- Pass the immutable context through lowering and implementation selection.
- Store required settings on concrete operators for use during execution.

The implementation selector determines each operator's backend. ``PlanContext``
records the selection policy and its settings, rather than imposing a single
backend on the entire plan.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from stratum._config import FLAGS


@dataclass(frozen=True)
class PlanContext:
    """Immutable configuration used while building a physical plan.

    Attributes
    ----------
    pandas_query:
        Evaluate mask selections through ``DataFrame.query()`` where possible
        (pandas backend only).
    rechunk:
        Rechunk polars frames produced by in-memory sources.
    parallelism:
        Degree of parallelism to hand to estimator/transformer ops.
    rust_backend:
        Enable the legacy Rust preference used by ``FlagBasedSelector``.
    allow_patch:
        Combine with ``rust_backend`` to enable the legacy Rust preference
        (``allow_patch and rust_backend``).
    implementation_selector:
        Name of the implementation-selection policy to use during physical
        planning: ``"default"`` (pandas/sklearn-skrub-first) or ``"greedy"``
        (rust/polars-first).
    """

    pandas_query: bool
    rechunk: bool
    parallelism: int
    rust_backend: bool
    allow_patch: bool
    implementation_selector: str = "default"

    @property
    def prefer_rust(self) -> bool:
        return self.rust_backend and self.allow_patch

    @classmethod
    def from_flags(cls, config=None) -> "PlanContext":
        """Snapshot the global FLAGS, plus the plan-level settings on ``config``.

        ``config`` is the plan's ``OptConfig`` (untyped here: ``_optimize``
        imports this module, so naming the type would cycle). ``None`` means a
        context built outside a plan, which takes the plan-level defaults.
        """
        return cls(
            pandas_query=bool(config.pandas_query) if config is not None else False,
            rechunk=bool(FLAGS.rechunk),
            parallelism=int(FLAGS.num_threads) or (os.cpu_count() or 1),
            rust_backend=bool(FLAGS.rust_backend),
            allow_patch=bool(FLAGS.allow_patch),
            implementation_selector=FLAGS.implementation_selector,
        )
