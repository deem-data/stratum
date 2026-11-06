"""Reuse cached results before the split and flag eligible results for admission.

Runs after implementation selection and before linearization. Keys cover the
selected implementation and its ordered inputs. The split and its descendants,
plus operators recomputed each pass, cannot supply reusable results here.

Starting at the root, a hit cuts off the work needed only by that operator. Its
value is retained by the plan so cache eviction cannot invalidate the plan.
Remaining keyed operations except constants are admitted by the fixed allow-all
oracle policy when their results are computed by the scheduler. The cache and
execution share these values by reference; operators must treat them as read-only.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from stratum.optimizer._lineage import compute_lineage_keys
from stratum.optimizer._linearization import is_recomputed
from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer.logical._base import IRNode
from stratum.optimizer.logical._ops import ValueOp

if TYPE_CHECKING:
    from stratum.runtime._reuse_cache import ReuseCache

logger = logging.getLogger(__name__)
_MISS = object()


class CacheHit:
    """A retained value, handed to the scheduler once and then released here."""

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
    """Operators outside the split region that are not recomputed each pass.

    Whether their configuration and inputs can be keyed is checked separately by
    compute_lineage_keys, which also blocks descendants of unkeyed inputs.
    """
    post_split = post_split_region(ops)
    return {op for op in ops if op not in post_split and not is_recomputed(op)}


def apply_reuse_cache(root: IRNode, cache: ReuseCache) -> None:
    """Prune a selected physical DAG in place, retaining root and hit identities.

    Planning must start from an unpruned graph: a previous hit has already lost
    the input edges needed to compute lineage.
    """
    ops = list(topological_iterator(root))
    if any(op.cache_hit is not None for op in ops):
        raise ValueError("reuse planning needs an unpruned graph; optimize the plan again")
    keys = compute_lineage_keys(ops, cacheable_region(ops))
    for op in ops:
        op.cache_key = None

    hits = _take_cached(root, keys, cache)
    admitted = 0
    for op in topological_iterator(root):
        if (keys.get(op) is not None and op.cache_hit is None
                and not isinstance(op, ValueOp)):
            op.cache_key = keys[op]
            admitted += 1
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
        if key is not None:
            # One atomic lookup: an entry can be evicted at any time, and None
            # is a valid cached result. The returned reference belongs to us.
            value = cache.get(key, _MISS)
            if value is not _MISS:
                op.cache_hit = CacheHit(key, value)
                hits += 1
                _cut_inputs(op)
                continue
        stack.extend(op.inputs)
    return hits


def _cut_inputs(op: IRNode) -> None:
    """Cut this hit's inputs and prune producers left without consumers.

    Walk iteratively so a hit can prune a long pipeline without recursion.
    Shared producers stay alive while any surviving consumer needs them.
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
