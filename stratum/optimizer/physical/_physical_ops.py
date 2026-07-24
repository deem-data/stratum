"""Physical operator layer.

A :class:`PhysicalOp` is a node in the IR *after* lowering. Where a logical
:class:`~stratum.optimizer.ir._ops.Op` says *what* to compute in a
backend-agnostic way, a physical op says *how*: a concrete physical op
(``PandasReadCSV``, ``PolarsReadCSV``, ...) runs on exactly one backend and its
``process`` contains **no backend-selection control flow** -- the choice was
made at plan time.

Two kinds of physical op exist:

* **abstract** (``is_abstract = True``): produced by lowering; carries the
  operation's configuration but cannot run. Its concrete implementations are
  registered in the :class:`~stratum.optimizer.physical._registry.PhysicalRegistry`
  (via the ``@physical_impl`` class decorator), and the implementation-selection
  pass picks one of them per the plan context.
* **concrete** (``is_abstract = False``): the default; runnable.

Selection swaps an abstract op to its concrete impl *in place*
(``op.__class__ = impl.impl_class``) so node identity is preserved -- the buffer
pool and every ``inputs``/``outputs`` edge key on identity. Concrete impls that
share the abstract op's fields therefore need no constructor of their own. An
impl that needs extra plan-time state precomputes it in :meth:`on_impl_selected`.

The selector-facing API (:meth:`supports` / :meth:`cost` / :meth:`exec_mem`)
mirrors the fields of :class:`~stratum.optimizer.physical._registry.PhysicalImpl`;
concrete classes override them as real feasibility checks and cost/memory
estimates land.
"""
from __future__ import annotations

from typing import Any

from stratum.optimizer.ir._base import IRNode


class PhysicalOp(IRNode):
    """Base for every node in the lowered (physical) IR.

    Shares the DAG structure and ``process`` hook with the logical layer via
    :class:`~stratum.optimizer.ir._base.IRNode`, and adds the selection protocol
    (:meth:`supports`/:meth:`cost`/:meth:`exec_mem` for the selector,
    :meth:`on_impl_selected` for plan-time binding).
    """

    #: An abstract physical op carries configuration but cannot run; it must be
    #: replaced by a concrete impl before execution. Concrete ops set this False.
    is_abstract = False

    #: Physical ops render as their concrete class name (the selected impl),
    #: never as the logical family they inherit from. See ``IRNode._is_physical``.
    _is_physical = True

    # --- Selector-facing API (mirrors PhysicalImpl's fields) -----------------

    @classmethod
    def supports(cls, op: IRNode, ctx) -> bool:
        """Whether this implementation can run ``op`` under plan context ``ctx``.

        Feasibility may depend both on the op's configuration and on plan-time
        policy carried by ``ctx`` (e.g. a fast path enabled by a flag). This is
        how two impls of the *same* backend stay mutually exclusive candidates
        instead of leaking the decision into a runtime branch.
        """
        return True

    @classmethod
    def cost(cls, op: IRNode, stats: Any) -> float:
        """Estimated compute cost of running ``op`` with this implementation."""
        # TODO: placeholder until the cost model lands.
        return 1.0

    @classmethod
    def exec_mem(cls, op: IRNode, stats: Any) -> int:
        """Estimated execution memory of running ``op`` with this implementation."""
        # TODO: placeholder until the memory model lands.
        return 0

    # --- Plan-time binding ----------------------------------------------------

    def on_impl_selected(self, ctx) -> None:
        """Hook run after this op's concrete impl is fixed.

        Lets a concrete op fold plan-time configuration (e.g. whether to rechunk,
        the degree of parallelism) into instance state, so ``process`` reads data,
        never flags.
        """
        pass

    def clone(self):
        # Cloning exists to duplicate logical sub-DAGs during choice unrolling,
        # which runs before lowering -- no physical op is ever cloned. Fail loudly
        # if that assumption is ever violated.
        raise TypeError(
            f"{type(self).__name__} is a physical op and must not be cloned; "
            f"cloning happens in the logical phase, before lowering.")


# FIXME: Why do we need this class?
class RustPhysicalOp(PhysicalOp):
    """Base for native Rust physical ops.

    Rust kernels run outside the GIL and parallelize internally (Rayon), so those
    are the defaults below; a specific kernel may override either as a class
    attribute. ``@rust_impl`` reads these off the op class into the
    :class:`~stratum.optimizer.physical._registry.RustPhysicalImpl` registry entry
    -- the same way ``supports``/``cost``/``exec_mem`` are captured from the class
    -- so the registry entry and the bound operator agree on the backend's
    scheduling capabilities and either can reason about them.
    """

    __slots__ = ()

    #: Kernel releases the GIL while running (safe to run concurrently with others).
    releases_gil = True
    #: Kernel already parallelizes internally, so the planner should not also fan
    #: it out (avoids oversubscription).
    data_parallel = True
