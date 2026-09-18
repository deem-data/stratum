"""Plan-time snapshot of the configuration that drives operator selection.

The whole point of the logical/physical split is that no operator-selection
decision happens during execution. To honour that, every setting that used to be
read inside an op's ``process`` (``pandas_query``, ``rechunk``, ...) is read
**once**, here, when the physical plan is built. The resulting
:class:`PlanContext` is threaded through lowering and implementation selection;
concrete ops fold whatever they need out of it into their own state.

It draws on two sources, because the settings have two lifetimes: process-global
runtime toggles come from :data:`~stratum._config.FLAGS`, while settings that
belong to one plan come from that plan's ``OptConfig``.

The dataframe backend is deliberately *not* here. Which backend an op runs on is
the outcome of implementation selection, so it belongs to the
:class:`~stratum.optimizer.physical._impl_selection.ImplementationSelector` that
makes the choice -- a backend field beside it could only ever disagree with the
impls actually bound, which is what a global ``force_polars`` flag used to do.

Today this is a straight snapshot of :data:`~stratum._config.FLAGS`. Later this
is where memory estimates, available backends and cost models feed the selector.
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
            parallelism=os.cpu_count(),
            rust_backend=bool(FLAGS.rust_backend),
            allow_patch=bool(FLAGS.allow_patch),
            implementation_selector=FLAGS.implementation_selector,
        )
