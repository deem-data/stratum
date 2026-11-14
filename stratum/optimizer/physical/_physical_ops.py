"""Physical operator hierarchy and implementation-selection interfaces.

Logical operators describe computations independently of a backend. Physical
operators implement execution after lowering or implementation selection.

- Abstract operators (``is_abstract = True``) hold configuration and require a
  concrete implementation before execution. Implementations are registered with
  ``@physical_impl`` and selected using the plan context.
- Concrete operators (``is_abstract = False``) implement ``process``.
- Selection changes ``op.__class__`` in place, preserving the identity used by
  DAG edges and buffer-pool keys. Implementations sharing the abstract
  operator's fields need no separate constructor; ``on_impl_selected`` initializes
  additional state from the plan configuration.

The ``supports``, ``cost`` and ``exec_mem`` methods supply feasibility checks and
cost/memory estimates through the registry's ``PhysicalImpl`` descriptors.
"""
from __future__ import annotations

from typing import Any

from stratum.optimizer.logical._base import IRNode, Fitted
from stratum.runtime._object_size import UNKNOWN_SIZE, get_size_estimator

import logging
logger = logging.getLogger(__name__)

class PhysicalOp(IRNode):
    """Base class for physical operators.

    Shares the DAG structure and ``process`` hook with the logical layer via
    :class:`~stratum.optimizer.logical._base.IRNode`, and adds the selection protocol
    (:meth:`supports`/:meth:`cost`/:meth:`exec_mem` for the selector,
    :meth:`on_impl_selected` for plan-time binding).
    """

    #: An abstract physical op carries configuration but cannot run; it must be
    #: replaced by a concrete impl before execution. Concrete ops set this False.
    is_abstract = False

    #: Use the concrete class name by default when rendering physical operators.
    #: Individual implementations can override their display labels.
    _is_physical = True

    # --- Selector-facing API (mirrors PhysicalImpl's fields) -----------------

    @classmethod
    def supports(cls, op: IRNode, ctx) -> bool:
        """Whether this implementation can run ``op`` under plan context ``ctx``.

        Feasibility may depend on the operator's configuration and the planning
        policy in ``ctx``. Implementations for the same backend can use these
        checks to restrict when they are candidates for selection.
        """
        return True

    @classmethod
    def cost(cls, op: IRNode, stats: Any) -> float:
        """Estimated compute cost of running ``op`` with this implementation."""
        # TODO: replace this placeholder with a compute-cost estimate.
        return 1.0

    @classmethod
    def exec_mem(cls, op: IRNode, stats: Any) -> int:
        """Estimated execution memory of running ``op`` with this implementation."""
        # TODO: replace this placeholder with an execution-memory estimate.
        return 0

    # --- Runtime state -----------------------------------------------------

    def retained_state_size(self) -> int:
        """Cached estimate of this operator's most recently fitted state.

        Excludes output buffers and temporary execution memory (``exec_mem``).
        The runtime state store owns explicit state and tracks its lifetime.
        Stateful implementations record their estimate when building the state,
        so reading it here does not traverse or serialize a fitted model.
        """
        return 0

    # --- Frame representation (#226) ------------------------------------------

    #: The frame library this impl reads and writes. ``None``: backend-agnostic
    #: (estimators, numpy, Rust kernels) -- it accepts either library and its
    #: output keeps the library of its input. ``@physical_impl`` sets it from the
    #: backend unless the class defines it. See ``_representation``.
    frame_lib = None

    @classmethod
    def accepts_input_lib(cls, op: IRNode, index: int, lib) -> bool:
        """Whether this impl of ``op`` reads its ``index``-th input as ``lib``.

        Overrides must match the input libraries supported by ``process`` and
        document any additional conversion or dispatch behavior.
        """
        return cls.frame_lib is None or lib == cls.frame_lib

    @classmethod
    def output_frame_lib(cls, op: IRNode, input_libs: list):
        """The frame library this impl of ``op`` produces."""
        if cls.frame_lib is not None:
            return cls.frame_lib
        return next((lib for lib in input_libs if lib is not None), None)

    # --- Plan-time binding ----------------------------------------------------

    def on_impl_selected(self, ctx) -> None:
        """Hook run after this op's concrete impl is fixed.

        Store planning settings, such as rechunking and parallelism, on the
        operator so execution uses the configuration selected for this plan.
        """
        pass

    def clone(self):
        # Logical cloning reconstructs an op from its fields. Physical unrolling
        # copies bound configuration instead, preserving selection-time setup.
        raise TypeError(
            f"{type(self).__name__} is a physical op; "
            f"copy its bound configuration instead of reconstructing it.")


class EstimatorExecution:
    """Shared execution and fit-time memory accounting for physical estimators.
    Store a retained-state estimate once per successful fitting pass.

    Concrete physical implementations can override ``_measure_retained_state``.
    The generic estimator walker remains the fallback for pass-through models
    and implementations without a specialized size estimate.
    """

    def process(self, inputs: list):
        task_data = self.extract_args_from_inputs(inputs)
        result, estimator = self.get_process_task()(task_data)
        if self.produces_state:
            # Update the estimate after fitting; inference calls do not recompute it.
            # Memory accounting must not turn a successful fit into a failure.
            self._retained_state_bytes = UNKNOWN_SIZE
            try:
                self._retained_state_bytes = int(self._measure_retained_state(estimator))
            except Exception:
                logger.debug("Could not size fitted state of %s", type(self).__name__,
                             exc_info=True)
            return Fitted(result, estimator, self._retained_state_bytes)
        return result

    def retained_state_size(self) -> int:
        """Estimated fitted-state size in bytes, or zero before fitting."""
        return getattr(self, "_retained_state_bytes", 0)

    def _measure_retained_state(self, estimator) -> int:
        return get_size_estimator(estimator)
