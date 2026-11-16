"""Recompile the unexecuted portion of a selected operator DAG.

Completed producers remain external inputs under their original storage keys.
Recompilation reuses cached results, prunes unnecessary dependencies and replans
lifetimes; it preserves the selected implementations, representations and
execution modes.
"""
from __future__ import annotations

import copy
from collections.abc import Callable

from .logical._base import IRNode
from .physical.rewrites._lifetime_planning import plan_input_removals
from .physical.rewrites._reuse_planning import plan_reuse


class Plan:
    """Selected operators in execution order and their required result.

    ``retained`` identifies producers whose values are needed by other executions.
    They may spill, but this plan must not release them at their last local use.
    ``profile`` holds optional observations for future implementation selection.
    Cache hits retain values only until their transfer to the runtime. Profiling
    metadata contains measurements, not intermediate values or fitted objects.
    """

    def __init__(self, ops: list[IRNode], *, retained=(), result=None, profile=None):
        self.ops = ops
        self.retained = set(retained)
        self.result = result if result is not None else ops[-1] if ops else None
        self.profile = profile

    def copy(self) -> Plan:
        """Instantiate an unexecuted plan, sharing configuration and external keys.

        Plans containing cache hits transfer each cached value once. Recompile
        them against the runtime's materialized values instead of copying their
        transfers.
        """
        if any(op.cache_hit is not None for op in self.ops):
            raise ValueError("transfer cached values to the runtime before recompiling this plan")
        return self._copy(self.ops)

    def _copy(self, ops: list[IRNode]) -> Plan:
        copies = {op: copy.copy(op) for op in ops}
        for op, new in copies.items():
            new.inputs = [copies.get(in_, in_) for in_ in op.inputs]
            new.outputs = []
            new.remove_after, new.release_state_after = [], []
            new.instance_of = op.instance_of or op
        members = set(copies.values())
        for new in copies.values():
            for in_ in new.inputs:
                if in_ in members:
                    in_.outputs.append(new)
        return Plan(list(copies.values()),
                    retained={copies.get(op, op) for op in self.retained},
                    result=copies.get(self.result, self.result), profile=self.profile)

    def recompile(self, position: int, *,
                  available: Callable[[IRNode, bool], bool], cache=None,
                  profile=None) -> Plan:
        """Compile the remaining work at an operator boundary.

        ``position`` is the next operator to execute. ``available(op, state)``
        checks for its materialized data or fitted state without loading it.
        Inputs outside the remaining region retain their original identities;
        missing inputs are rejected rather than recomputed.

        Already materialized cache hits become external inputs. New hits are
        transferred to the runtime before the returned plan starts execution.
        Removing nodes preserves topological order, so no reordering is needed.
        """
        if not 0 <= position <= len(self.ops):
            raise ValueError("position must be between zero and the plan length")
        pending = self.ops[position:]
        # The scheduler has already materialized these single-use transfers.
        new = self._copy([op for op in pending if op.cache_hit is None])
        new.profile = profile if profile is not None else self.profile
        try:
            if cache is not None and new.ops:
                roots = [new.result, *new.retained]
                new.ops = plan_reuse(new.ops, cache, roots)
            members = set(new.ops)
            data, states = new.required_values()
            for state, values in ((False, data), (True, states)):
                for op in values - members:
                    if not available(op, state):
                        kind = "fitted state" if state else "data"
                        raise ValueError(f"Cannot recompile: missing {kind} for {op}")
            plan_input_removals(new.ops, new.retained)
            return new
        except BaseException:
            for op in new.ops:
                if op.cache_hit is not None:
                    op.cache_hit.release()
            raise

    def required_values(self) -> tuple[set[IRNode], set[IRNode]]:
        """Data and state dependencies, including the result returned to the caller."""
        data, states = set(), set()
        for op in self.ops:
            for k, in_ in enumerate(op.inputs):
                (states if op.reads_state(k) else data).add(in_)
        if self.result is not None:
            data.add(self.result)
        return data, states
