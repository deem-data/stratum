"""Reuse cached outputs and fitted states, and flag eligible results for admission.

Run after implementation selection and before lifetime planning, both during
initial compilation and when replanning remaining work at execution boundaries.

- Keys cover the selected implementation, bound mode and ordered data/state
  inputs. Fold keys also include the selected rows.
- Starting at the root, a hit cuts off work needed only by that operator. Its
  values are retained by the plan so cache eviction cannot invalidate the plan.
- A fitting operator caches its output and fitted state separately. Reuse requires
  all values needed by the plan; the scheduler loads them into the buffer pool and
  state store.
- The scheduler attempts to cache remaining keyed results except constants and
  split outputs, subject to the cache budget. The cache and execution share
  values by reference; operators must treat them as read-only.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from stratum._profiler._diagnostics import current_diagnostics, record_reuse_plan
from stratum.optimizer._lineage import compute_lineage_keys, state_key
from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer.logical._base import FITTING_MODE, IRNode
from stratum.optimizer.logical._ops import ValueOp
from stratum.optimizer.logical._split_ops import SplitOutput
from stratum.utils._utils import start_time, log_time

if TYPE_CHECKING:
    from stratum.runtime._reuse_cache import ReuseCache

logger = logging.getLogger(__name__)

_MISS = object()


class CacheHit:
    """Cached outputs and fitted state retained until transfer to the runtime.

    A fitting operator's output and state are retained as required by the plan:
    its output if ``has_output``, its state if ``has_state``.
    """

    def __init__(self, key: str, value: Any,
                 state: Any = None, has_output: bool = True, has_state: bool = False,
                 state_size: int = 0):
        self.key = key
        self.has_output = has_output
        self.has_state = has_state
        self.state_size = state_size
        self._parts = {"output": value, "state": state}
        self._taken = set()

    def take(self) -> Any:
        """The value, handed over once: the plan must not keep it alive after the
        buffer pool has released it."""
        return self._take("output")

    def take_state(self) -> Any:
        """The state, handed over once, as ``take`` hands over the value."""
        return self._take("state")

    def release(self) -> None:
        """Drop cached objects left in the plan when execution fails to start."""
        self._taken.update(self._parts)
        self._parts.clear()

    def keep_read_parts(self, op: IRNode) -> None:
        """Release cached outputs or fitted state that no remaining consumer requires.

        Subsequent pruning may remove consumers used to determine the required
        values during cache lookup.
        """
        if self.has_state and not op.has_state_reader():
            self.has_state, self._parts["state"] = False, None
            self.state_size = 0
        if op.produces_state and self.has_output and not op.has_data_reader() and op.outputs:
            self.has_output, self._parts["output"] = False, None

    def _take(self, part: str) -> Any:
        if part in self._taken:
            raise RuntimeError("this cache hit was already consumed; "
                               "optimize the plan again to run it again")
        self._taken.add(part)
        value, self._parts[part] = self._parts[part], None
        return value


#: Constants and a fold's row selections are cheap to recompute.
_NEVER_ADMITTED = (ValueOp, SplitOutput)


def cacheable_region(ops: list[IRNode]) -> set[IRNode]:
    """Full-data fitting, preprocessing and inference with explicit fitted state."""
    post_split = post_split_region(ops)
    return {op for op in ops if op not in post_split
            and (op.mode in ("fit", FITTING_MODE) or op.state_ref is not None)}


def post_split_region(ops: list[IRNode]) -> set[IRNode]:
    """The split op and everything downstream of it: what runs once per fold."""
    post_split = set()
    stack = [op for op in ops if op.is_split_op]
    while stack:
        op = stack.pop()
        if op not in post_split:
            post_split.add(op)
            stack.extend(op.outputs)
    return post_split


def apply_reuse_cache(root: IRNode, cache: "ReuseCache") -> None:
    """Take the plan's cached intermediates from ``cache`` and flag what to admit.

    Changes the plan in place: the root stays the root, even when it is a hit.
    """
    ops = list(topological_iterator(root))
    region = cacheable_region(ops)
    reasons = {} if current_diagnostics() is not None else None
    remaining = plan_reuse(ops, cache, [root], cacheable=region, reasons=reasons)
    if reasons is not None:
        record_reuse_plan(remaining, region, {op: op.lineage for op in remaining}, reasons)


def _get(op: IRNode, key: str | None, cache: "ReuseCache") -> CacheHit | None:
    """Return cached values for ``op`` if all required values are available, else None.

    A fitting operator's consumers may require its output, its fitted state or
    both. Requirements are determined from the current consumers; subsequent
    pruning may remove some consumers (see ``CacheHit.keep_read_parts``).
    """
    if key is None:
        return None
    fit = op.produces_state
    has_output = not fit or op.has_data_reader() or not op.outputs
    has_state = fit and op.has_state_reader()
    if not has_output and not has_state:
        return None
    output = cache.get(key, _MISS) if has_output else None
    state = cache.get(state_key(key), _MISS, with_size=True) if has_state else (None, 0)
    if output is _MISS or state is _MISS:
        return None
    return CacheHit(key, output,
                    state=state[0], state_size=state[1],
                    has_output=has_output, has_state=has_state)


def plan_reuse(ops: list[IRNode], cache: ReuseCache,
               roots: list[IRNode], *, cacheable: set[IRNode] | None = None,
               reasons: dict | None = None) -> list[IRNode]:
    """Prune an execution region without modifying its external producers.

    ``ops`` contains the owned operators in topological order; it may be the
    whole DAG or a copy of its unexecuted region. ``roots`` are required results.
    ``cacheable`` optionally restricts lineage calculation, without excluding
    other operators from execution. External inputs retain their existing lineage.
    """
    start = start_time()
    if any(op.cache_hit is not None for op in ops):
        raise ValueError("reuse planning needs an unpruned graph; optimize the plan again")
    members = set(ops)
    known = {in_: in_.lineage for op in ops for in_ in op.lineage_inputs
             if in_ not in members}
    keys = compute_lineage_keys(ops, cacheable, reasons=reasons, known=known)
    for op in ops:
        op.lineage = keys[op]
        op.cache_key = None

    hits = []
    kept = set()
    required = set(roots)
    stack = list(roots)
    while stack:
        op = stack.pop()
        if op in kept or op not in members:
            continue
        kept.add(op)
        op.cache_hit = _get(op, keys[op], cache)
        if op.cache_hit is not None:
            hits.append(op)
            _cut_inputs(op, members, required)
        else:
            stack.extend(op.inputs)

    # Each visited operator has a path from a required result through non-hits.
    remaining = [op for op in ops if op in kept]
    for op in ops:
        if op in kept:
            op.outputs = [out for out in op.outputs if out in kept]
        else:
            op.outputs = []
            op.inputs = []
            op.state_ref = None
    for op in hits:
        op.cache_hit.keep_read_parts(op)

    eligible = {op for op in remaining if keys[op] is not None
                and op.cache_hit is None and not isinstance(op, _NEVER_ADMITTED)}
    for op in eligible:
        op.cache_key = keys[op]
    logger.debug(f"Reuse planning: {len(ops)} ops, "
                 f"{sum(keys[op] is not None for op in ops)} keyed, "
                 f"{len(hits)} taken from the cache, "
                 f"{len(eligible)} eligible for admission")
    log_time("reuse planning took", start)
    return remaining


def _cut_inputs(op: IRNode, members: set[IRNode], roots: set[IRNode]) -> None:
    """Prune a hit's dependencies within the owned region.

    Required results survive even without consumers. External producers retain
    their original edges. Updating internal consumers before the next lookup
    avoids requiring cached data or state whose readers have already been pruned.
    """
    stack = [op]
    while stack:
        consumer = stack.pop()
        producers = list(dict.fromkeys(consumer.inputs))
        consumer.inputs = []
        consumer.state_ref = None
        for producer in producers:
            if producer not in members:
                continue
            producer.outputs = [out for out in producer.outputs if out is not consumer]
            if not producer.outputs and producer not in roots:
                stack.append(producer)
