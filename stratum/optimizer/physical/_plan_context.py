"""Plan-time snapshot of the configuration that drives operator selection.

The whole point of the logical/physical split is that no operator-selection
decision happens during execution. To honour that, every flag that used to be
read inside an op's ``process`` (``force_polars``, ``pandas_query``,
``rechunk``, ...) is read **once**, here, when the physical plan is built. The
resulting :class:`PlanContext` is threaded through lowering and implementation
selection; concrete ops fold whatever they need out of it into their own state.

Today this is a straight snapshot of :data:`~stratum._config.FLAGS`, which
reproduces the current flag-driven behaviour exactly. Later this is where
memory estimates, available backends and cost models feed the selector.
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
    backend:
        The dataframe backend every frame op should target: ``"polars"`` or
        ``"pandas"``.
    pandas_query:
        Evaluate mask selections through ``DataFrame.query()`` where possible
        (pandas backend only).
    rechunk:
        Rechunk polars frames produced by in-memory sources.
    parallelism:
        Degree of parallelism to hand to estimator/transformer ops.
    rust_backend:
        Prefer registered Rust kernel implementations where they support the op.
    allow_patch:
        Legacy soft kill-switch for non-sklearn backends; gates Rust selection
        together with ``rust_backend`` to preserve the pre-registry semantics
        (``allow_patch and rust_backend``).
    implementation_selector:
        Name of the implementation-selection policy to use during physical
        planning: ``"default"`` (pandas/sklearn-skrub-first) or ``"greedy"``
        (rust/polars-first).
    """

    backend: str
    pandas_query: bool
    rechunk: bool
    parallelism: int
    rust_backend: bool
    allow_patch: bool
    implementation_selector: str = "default"

    @property
    def is_polars(self) -> bool:
        return self.backend == "polars"

    @property
    def prefer_rust(self) -> bool:
        return self.rust_backend and self.allow_patch

    @classmethod
    def from_flags(cls) -> "PlanContext":
        """Snapshot the current global FLAGS into a plan context."""
        return cls(
            backend="polars" if FLAGS.force_polars else "pandas",
            pandas_query=bool(FLAGS.pandas_query),
            rechunk=bool(FLAGS.rechunk),
            parallelism=os.cpu_count(),
            rust_backend=bool(FLAGS.rust_backend),
            allow_patch=bool(FLAGS.allow_patch),
            implementation_selector=FLAGS.implementation_selector,
        )
