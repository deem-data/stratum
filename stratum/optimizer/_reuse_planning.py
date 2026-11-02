"""Reuse planning: take what the reuse cache has, decide what to add to it.

Runs on the physical plan, after implementation selection and before
linearization, so the plan it changes is the one that executes and everything
after it (linearization, removal planning) sees a cache hit as
an ordinary source. The plan before the split is planned when the plan is
compiled (:func:`apply_reuse_cache`); each fold's copy of the plan after it when
the copy is made, once the fold's rows are bound (:func:`plan_fold_reuse`). In
three steps:

1. **Keys.** Every op in the cacheable region gets a lineage key (see
   :mod:`stratum.optimizer._lineage`). Before the split, the region is what runs
   exactly once, in the fitting mode: every op that is not the split op or
   downstream of it, minus the ``eval_mode()`` sources the test rows read in their
   own mode. Without a split op, that is the whole plan. An entry an ``evaluate``
   wrote thus serves a ``grid_search`` reading the same prefix. In a fold, every
   op is in the region; a fold's keys follow from its rows, which the split's
   outputs digest, and an op's stage and mode are part of its key.
2. **Probe.** Walking from the root towards the sources, the first op on each path
   whose value the cache returns becomes a hit: it keeps its place in the plan,
   holds the value (a :class:`CacheHit`) and loses its inputs, and the sub-plan
   only it needed is pruned. The highest hit wins: it saves the most work. The
   value is taken with ``get``, so the plan holds its own reference and the cache
   stays free to evict the entry; the scheduler moves it into the plan's buffer
   pool before running.
3. **Admission.** An :class:`AdmissionPolicy` picks which of the remaining keyed
   ops to add to the cache. The scheduler hands such an op's buffer over to the
   cache once the plan no longer needs it, rather than freeing it.

A "fit" stage has two values: its output and the state it fitted, cached under
:func:`~stratum.optimizer._lineage.state_key` of its key. It is a hit only when
the cache holds each of the two that the plan reads.
"""
from __future__ import annotations

import logging
from time import perf_counter
from typing import TYPE_CHECKING, Any

from stratum._profiler._diagnostics import (diagnostics_enabled, record_fold_reuse,
                                            record_reuse_plan)
from stratum.optimizer._lineage import compute_lineage_keys, state_key
from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer.logical._base import IRNode
from stratum.optimizer.logical._ops import ValueOp
from stratum.optimizer.logical._split_ops import SplitOutput
from stratum.utils._utils import start_time, log_time

if TYPE_CHECKING:
    from stratum.runtime._reuse_cache import ReuseCache

logger = logging.getLogger(__name__)

_MISS = object()


class CacheHit:
    """The cached value of a hit op, until the scheduler puts it in its pool.

    ``cost`` is the seconds the entry took to compute, ``tier`` where the cache
    found it and ``load_seconds`` how long ``get`` took; they stay for inspection
    once the value is taken. A "fit" stage's hit holds the parts the plan reads:
    its output if ``has_output``, its state if ``has_state``.
    """

    def __init__(self, key: str, value: Any, cost: float, tier: str, load_seconds: float,
                 state: Any = None, has_output: bool = True, has_state: bool = False):
        self.key = key
        self.cost = cost
        self.tier = tier
        self.load_seconds = load_seconds
        self.has_output = has_output
        self.has_state = has_state
        self._parts = {"output": value, "state": state}
        self._taken = set()

    def take(self) -> Any:
        """The value, handed over once: the plan must not keep it alive after the
        buffer pool has released it."""
        return self._take("output")

    def take_state(self) -> Any:
        """The state, handed over once, as ``take`` hands over the value."""
        return self._take("state")

    def keep_read_parts(self, op: IRNode) -> None:
        """Drop the parts of ``op``'s hit that no reader left in the plan reads:
        the parts were chosen before other hits cut some of its readers."""
        if self.has_state and not op.has_state_reader():
            self.has_state, self._parts["state"] = False, None
        if op.stage == "fit" and self.has_output and not op.has_data_reader():
            self.has_output, self._parts["output"] = False, None

    def _take(self, part: str) -> Any:
        if part in self._taken:
            raise RuntimeError("this cache hit was handed to a scheduler already; "
                               "optimize the plan again to run it again")
        self._taken.add(part)
        value, self._parts[part] = self._parts[part], None
        return value


class AdmissionPolicy:
    """Strategy interface: which intermediates to add to the reuse cache.

    ``candidates`` are the keyed ops the probe left in the plan, in topological
    order. ``keys`` maps each to its lineage key.
    """

    def select(self, candidates: list[IRNode], keys: dict, cache: "ReuseCache") -> set[IRNode]:
        raise NotImplementedError


#: Ops cheaper to recompute than to read: constants, and a fold's rows of X or y.
_NEVER_ADMITTED = (ValueOp, SplitOutput)


class CacheAll(AdmissionPolicy):
    """Every candidate except constants."""

    def select(self, candidates, keys, cache):
        return {op for op in candidates if not isinstance(op, _NEVER_ADMITTED)}


class CacheNothing(AdmissionPolicy):
    """Read the cache, never add to it."""

    def select(self, candidates, keys, cache):
        return set()


ADMISSION_POLICIES = {"all": CacheAll, "none": CacheNothing}


def make_admission_policy(name: str) -> AdmissionPolicy:
    try:
        return ADMISSION_POLICIES[name]()
    except KeyError:
        raise ValueError(f"Unknown admission policy {name!r}; "
                         f"valid policies are {sorted(ADMISSION_POLICIES)}.") from None


def cacheable_region(ops: list[IRNode]) -> set[IRNode]:
    """The ops that run exactly once, in the fitting mode."""
    post_split = post_split_region(ops)
    return {op for op in ops if op not in post_split and op.pass_mode is None}


def post_split_region(ops: list[IRNode]) -> set[IRNode]:
    """The split op and everything downstream of it: what runs once per fold."""
    split = next((op for op in ops if op.is_split_op), None)
    post_split = set()
    stack = [split] if split is not None else []
    while stack:
        op = stack.pop()
        if op not in post_split:
            post_split.add(op)
            stack.extend(op.outputs)
    return post_split


def apply_reuse_cache(root: IRNode, cache: "ReuseCache", policy: AdmissionPolicy) -> None:
    """Take the plan's cached intermediates from ``cache`` and flag what to admit.

    Changes the plan in place: the root stays the root, even when it is a hit.
    """
    start = start_time()
    ops = list(topological_iterator(root))
    region = cacheable_region(ops)
    reasons = {} if diagnostics_enabled() else None
    keys = compute_lineage_keys(ops, region, reasons)
    for op in region:
        op.lineage = keys[op]
        if keys[op] is not None:
            cache.observe(keys[op], label=str(op))

    hits = _take_cached(root, keys, cache)
    for op in hits:
        op.cache_hit.keep_read_parts(op)

    candidates = [op for op in topological_iterator(root)
                  if keys.get(op) is not None and op.cache_hit is None]
    admitted = policy.select(candidates, keys, cache)
    for op in admitted:
        op.cache_key = keys[op]

    cache.stats.planned_hits += len(hits)
    cache.stats.planned_admissions += len(admitted)
    if reasons is not None:
        record_reuse_plan(ops, region, keys, reasons, hits, admitted)
    logger.debug(f"Reuse planning: {len(region)} ops in the cacheable region, "
                 f"{sum(k is not None for k in keys.values())} keyed, {len(hits)} taken "
                 f"from the cache, {len(admitted)} to admit")
    log_time("reuse planning took", start)


def _take_cached(root: IRNode, keys: dict, cache: "ReuseCache") -> list[IRNode]:
    """Make the highest cached op on every path from the root a hit."""
    hits = []
    seen = set()
    stack = [root]
    while stack:
        op = stack.pop()
        if op in seen:
            continue
        seen.add(op)
        op.cache_hit = _get(op, keys.get(op), cache)
        if op.cache_hit is not None:
            hits.append(op)
            _cut_inputs(op)
            # Whatever only `op` needed is pruned, so there is nothing below to probe.
            continue
        # Every op pushed here has a consumer on the path from the root that is not
        # a hit, so it keeps its edges while the hits below it are cut.
        stack.extend(op.inputs)
    return hits


def _get(op: IRNode, key: str | None, cache: "ReuseCache") -> CacheHit | None:
    """``op``'s values from the cache, if it holds all the plan reads, else None.

    A "fit" stage's readers may read its output, its state or both; which, is
    judged on the plan as it is before any hit is cut, so it may ask for a part
    a reader cut later would not have read (see ``CacheHit.keep_read_parts``).
    """
    if key is None:
        return None
    fit = op.stage == "fit"
    has_output = not fit or op.has_data_reader()
    has_state = fit and op.has_state_reader()
    parts = [k for k, wanted in ((key, has_output), (state_key(key), has_state)) if wanted]
    if not parts or not all(k in cache for k in parts):
        return None
    tier, cost = cache.tier(parts[0]), cache.cost(parts[0])
    t0 = perf_counter()
    values = [cache.get(k, _MISS) for k in parts]
    if any(v is _MISS for v in values):  # an entry could not be read
        return None
    output = values[0] if has_output else None
    state = values[-1] if has_state else None
    return CacheHit(key, output, cost, tier, perf_counter() - t0,
                    state=state, has_output=has_output, has_state=has_state)


def plan_fold_reuse(fold: list[IRNode], cache: "ReuseCache",
                    policy: AdmissionPolicy) -> list[IRNode]:
    """Plan reuse for ``fold``, a fold's copy of the plan after the split, in
    execution order with its rows bound. Returns what of it is left to run.

    Inputs from before the split keep the keys they were planned with. Ops before
    the split are never changed: a hit cuts its inputs, and what only it needed is
    left out of the fold.
    """
    start = start_time()
    members = set(fold)
    known = {in_: in_.lineage for op in fold for in_ in op.inputs if in_ not in members}
    reasons = {} if diagnostics_enabled() else None
    keys = compute_lineage_keys(fold, reasons=reasons, known=known)
    for op in fold:
        op.lineage = keys[op]
        if keys[op] is not None:
            cache.observe(keys[op], label=str(op))

    sink = fold[-1]
    hits = []
    seen = set()
    stack = [sink]
    while stack:
        op = stack.pop()
        if op in seen or op not in members:
            continue
        seen.add(op)
        op.cache_hit = _get(op, keys[op], cache)
        if op.cache_hit is not None:
            hits.append(op)
        else:
            stack.extend(op.inputs)
    for op in hits:
        op.inputs = []
        op.state_ref = None

    kept = _reachable(sink, members)
    fold = [op for op in fold if op in kept]
    for op in fold:
        op.outputs = [out for out in op.outputs if out in kept]
    # A hit only another hit read is left out with it.
    hits = [op for op in hits if op in kept]
    for op in hits:
        op.cache_hit.keep_read_parts(op)

    candidates = [op for op in fold if keys[op] is not None and op.cache_hit is None]
    admitted = policy.select(candidates, keys, cache)
    for op in admitted:
        op.cache_key = keys[op]

    cache.stats.planned_hits += len(hits)
    cache.stats.planned_admissions += len(admitted)
    if reasons is not None:
        record_fold_reuse(members, kept, keys, reasons, hits, admitted)
    log_time("fold reuse planning took", start)
    return fold


def _reachable(sink: IRNode, members: set) -> set:
    """The ops in ``members`` that ``sink`` reads, itself included."""
    reached = set()
    stack = [sink]
    while stack:
        op = stack.pop()
        if op in members and op not in reached:
            reached.add(op)
            stack.extend(op.inputs)
    return reached


def _cut_inputs(op: IRNode) -> None:
    """Take ``op``'s inputs away, and prune each producer left without a consumer."""
    producers = list(dict.fromkeys(op.inputs))
    for in_ in producers:
        in_.outputs = [out for out in in_.outputs if out is not op]
    op.inputs = []
    for in_ in producers:
        if not in_.outputs:
            in_.detach_and_prune()
