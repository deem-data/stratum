"""Reuse planning: take what the reuse cache has, decide what to add to it.

Runs on the physical plan, after implementation selection and before
linearization, so the plan it changes is the one that executes and everything
after it (linearization, removal planning) sees a cache hit as
an ordinary source. In three steps:

1. **Keys.** Every op in the cacheable region gets a lineage key (see
   :mod:`stratum.optimizer._lineage`). The region is what runs exactly once, in
   the fitting mode: every op that is not the split op or downstream of it, minus
   the ``eval_mode()`` sources the test rows read in their own mode. Without a
   split op, that is the whole plan.
   Mode is therefore never part of a key, and an entry an ``evaluate`` wrote serves
   a ``grid_search`` reading the same prefix.
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
"""
from __future__ import annotations

import logging
from time import perf_counter
from typing import TYPE_CHECKING, Any

from stratum._profiler._diagnostics import diagnostics_enabled, record_reuse_plan
from stratum.optimizer._fold_compilation import post_split_region
from stratum.optimizer._lineage import compute_lineage_keys
from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer.logical._base import IRNode
from stratum.optimizer.logical._ops import ValueOp
from stratum.utils._utils import start_time, log_time

if TYPE_CHECKING:
    from stratum.runtime._reuse_cache import ReuseCache

logger = logging.getLogger(__name__)

_MISS = object()


class CacheHit:
    """The cached value of a hit op, until the scheduler puts it in its pool.

    ``cost`` is the seconds the entry took to compute, ``tier`` where the cache
    found it and ``load_seconds`` how long ``get`` took; they stay for inspection
    once the value is taken.
    """

    def __init__(self, key: str, value: Any, cost: float, tier: str, load_seconds: float):
        self.key = key
        self.cost = cost
        self.tier = tier
        self.load_seconds = load_seconds
        self._value = value
        self._taken = False

    def take(self) -> Any:
        """The value, handed over once: the plan must not keep it alive after the
        buffer pool has released it."""
        if self._taken:
            raise RuntimeError("this cache hit was handed to a scheduler already; "
                               "optimize the plan again to run it again")
        value, self._value, self._taken = self._value, None, True
        return value


class AdmissionPolicy:
    """Strategy interface: which intermediates to add to the reuse cache.

    ``candidates`` are the keyed ops the probe left in the plan, in topological
    order. ``keys`` maps each to its lineage key.
    """

    def select(self, candidates: list[IRNode], keys: dict, cache: "ReuseCache") -> set[IRNode]:
        raise NotImplementedError


#: Ops cheaper to recompute than to read.
_NEVER_ADMITTED = (ValueOp,)


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
        if keys[op] is not None:
            cache.observe(keys[op], label=str(op))

    hits = _take_cached(root, keys, cache)

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
        key = keys.get(op)
        if key is not None and key in cache:
            tier, cost = cache.tier(key), cache.cost(key)
            t0 = perf_counter()
            value = cache.get(key, _MISS)
            if value is not _MISS:  # else the entry could not be read
                op.cache_hit = CacheHit(key, value, cost, tier, perf_counter() - t0)
                hits.append(op)
                _cut_inputs(op)
                # Whatever only `op` needed is pruned, so there is nothing below to probe.
                continue
        # Every op pushed here has a consumer on the path from the root that is not
        # a hit, so it keeps its edges while the hits below it are cut.
        stack.extend(op.inputs)
    return hits


def _cut_inputs(op: IRNode) -> None:
    """Take ``op``'s inputs away, and prune each producer left without a consumer."""
    producers = list(dict.fromkeys(op.inputs))
    for in_ in producers:
        in_.outputs = [out for out in in_.outputs if out is not op]
    op.inputs = []
    for in_ in producers:
        if not in_.outputs:
            in_.detach_and_prune()
