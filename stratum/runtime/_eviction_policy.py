"""Rank entries for eviction from the buffer pool or reuse cache.

- The store determines when to evict and how many bytes to release.
- The policy tracks insertions, accesses and removals, then yields candidates
  in eviction order through ``victims()``.
- The store skips ineligible candidates (pinned, already spilled or unspillable)
  and stops when it has released enough bytes.

Separating ranking from eviction allows a shared memory manager to allocate
budgets while each store retains its own eviction policy.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Hashable, Iterator


class EvictionPolicy:
    """Ranks a store's entries for eviction. Subclasses implement the ranking."""

    def on_insert(self, key: Hashable, size: int, cost: float = 0.0) -> None:
        """``key`` entered the store. ``size`` is its in-memory cost in bytes,
        ``cost`` the seconds it would take to compute it again (0 if unknown)."""
        raise NotImplementedError

    def on_access(self, key: Hashable) -> None:
        """``key`` was read."""
        raise NotImplementedError

    def on_remove(self, key: Hashable) -> None:
        """Remove ``key`` from the policy's tracked entries."""
        raise NotImplementedError

    def victims(self) -> Iterator[Hashable]:
        """Every tracked key, best eviction candidate first."""
        raise NotImplementedError


class LRUPolicy(EvictionPolicy):
    """Least recently inserted or accessed first."""

    def __init__(self):
        self._order: OrderedDict[Hashable, None] = OrderedDict()

    def on_insert(self, key: Hashable, size: int, cost: float = 0.0) -> None:
        self._order[key] = None
        self._order.move_to_end(key)

    def on_access(self, key: Hashable) -> None:
        if key in self._order:
            self._order.move_to_end(key)

    def on_remove(self, key: Hashable) -> None:
        self._order.pop(key, None)

    def victims(self) -> Iterator[Hashable]:
        # A snapshot, so the store may report accesses or removals while iterating.
        # FIXME(perf): this copies every key even when only a few victims are
        # needed. Iterate in bounded chunks once stores hold thousands of entries.
        return iter(list(self._order))
