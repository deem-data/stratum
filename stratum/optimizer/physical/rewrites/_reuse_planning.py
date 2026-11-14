"""Reuse cached results before the split and flag eligible results for admission.

Runs after implementation selection and before linearization. Keys cover the
selected implementation, execution mode and ordered data/state inputs. Reuse
excludes the split and its descendants, plus sources outside the fitting mode.
Operators with live state consumers execute even if their data output is cached.

Starting at the root, a cache hit prunes dependencies with no other consumers.
The plan retains the cached value so eviction cannot invalidate the plan.
The scheduler attempts to cache each remaining keyed result except constants,
subject to the cache budget. The cache and execution share these values by
reference; operators must treat them as read-only.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from stratum._profiler._diagnostics import current_diagnostics, record_reuse_plan
from stratum.optimizer._lineage import compute_lineage_keys
from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer.logical._base import FITTING_MODE, IRNode
from stratum.optimizer.logical._ops import ValueOp

if TYPE_CHECKING:
    from stratum.runtime._reuse_cache import ReuseCache

logger = logging.getLogger(__name__)
_MISS = object()


class CacheHit:
    """A cached value whose reference transfers to the scheduler once."""

    def __init__(self, key: str, value: Any):
        self.key = key
        self._value = value
        self._taken = False

    def take(self) -> Any:
        if self._taken:
            raise RuntimeError("this cache hit was already consumed; "
                               "optimize the plan again to run it again")
        value, self._value, self._taken = self._value, None, True
        return value


def post_split_region(ops: list[IRNode]) -> set[IRNode]:
    """Every split and its descendants: work that depends on the current fold."""
    region = set()
    stack = [op for op in ops if op.is_split_op]
    while stack:
        op = stack.pop()
        if op not in region:
            region.add(op)
            stack.extend(op.outputs)
    return region


def cacheable_region(ops: list[IRNode]) -> set[IRNode]:
    """Full-data fitting, preprocessing and inference with explicit fitted state.

    Whether their configuration and inputs can be keyed is checked separately by
    compute_lineage_keys, which also blocks descendants of unkeyed inputs.
    """
    post_split = post_split_region(ops)
    return {op for op in ops if op not in post_split
            and (op.mode in ("fit", FITTING_MODE) or op.state_ref is not None)}


def apply_reuse_cache(root: IRNode, cache: ReuseCache) -> None:
    """Prune a selected physical DAG in place, retaining root and hit identities.

    Planning requires an unpruned graph: pruning a cache hit removes the input
    edges needed to compute lineage.
    """
    ops = list(topological_iterator(root))
    if any(op.cache_hit is not None for op in ops):
        raise ValueError("reuse planning needs an unpruned graph; optimize the plan again")
    region = cacheable_region(ops)
    reasons = {} if current_diagnostics() is not None else None
    keys = compute_lineage_keys(ops, region, reasons=reasons)
    for op in ops:
        op.cache_key = None

    hits = _take_cached(root, keys, cache)
    admitted = 0
    remaining = list(topological_iterator(root))
    for op in remaining:
        if (keys.get(op) is not None and op.cache_hit is None
                and not isinstance(op, ValueOp) and not op.has_state_reader()):
            op.cache_key = keys[op]
            admitted += 1
    if reasons is not None:
        record_reuse_plan(remaining, region, keys, reasons)
    logger.debug("Reuse planning: %d cache hits, %d admission candidates", hits, admitted)


def _take_cached(root: IRNode, keys: dict, cache: ReuseCache) -> int:
    """Take the nearest cached result on every path from the root."""
    hits = 0
    seen = set()
    stack = [root]
    while stack:
        op = stack.pop()
        if op in seen:
            continue
        seen.add(op)
        key = keys.get(op)
        # The cache contains data outputs, not fitted state. State producers must
        # execute unless a downstream cache hit prunes their entire computation.
        if key is not None and not op.has_state_reader():
            # One atomic lookup: an entry can be evicted at any time, and None
            # is a valid cached result. The plan retains the returned reference.
            value = cache.get(key, _MISS)
            if value is not _MISS:
                op.cache_hit = CacheHit(key, value)
                hits += 1
                _cut_inputs(op)
                continue
        stack.extend(op.inputs)
    return hits


def _cut_inputs(op: IRNode) -> None:
    """Remove a cache hit's input edges and prune producers without consumers.

    Walk iteratively so a hit can prune a long pipeline without recursion.
    Retain shared producers that have remaining consumers.
    """
    stack = [op]
    while stack:
        consumer = stack.pop()
        producers = list(dict.fromkeys(consumer.inputs))
        consumer.inputs = []
        for producer in producers:
            producer.outputs = [out for out in producer.outputs if out is not consumer]
            if not producer.outputs:
                stack.append(producer)
