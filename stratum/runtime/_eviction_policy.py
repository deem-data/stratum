"""Eviction policies: *which* entries a memory store gives up first.

- A store (BufferPool, reuse cache) decides *when* and *how much* to evict; a
  policy only ranks entries. A future memory manager can then set how much each
  store frees while every store keeps its own notion of what is worth keeping.
- The store reports each entry's lifecycle (``on_insert``, ``on_access``,
  ``on_remove``) and asks for ``victims()``, best victim first, when it needs room.
- The store skips victims it cannot evict right now (pinned, spilled,
  unspillable), so a policy never knows about pins or tiers.
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
        """``key`` left the store for good."""
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


class CostSizePolicy(EvictionPolicy):
    """Cheapest to recompute per byte first: the entry whose eviction loses least.

    An entry's value is ``cost / size``, the seconds of work its bytes save. Ties,
    including the common case of entries without a measured cost, fall back to
    least recently used.
    """

    def __init__(self):
        self._entries: dict = {}  # key -> [cost / size, last access]
        self._clock = 0

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def on_insert(self, key: Hashable, size: int, cost: float = 0.0) -> None:
        self._entries[key] = [cost / max(size, 1), self._tick()]

    def on_access(self, key: Hashable) -> None:
        entry = self._entries.get(key)
        if entry is not None:
            entry[1] = self._tick()

    def on_remove(self, key: Hashable) -> None:
        self._entries.pop(key, None)

    def victims(self) -> Iterator[Hashable]:
        return iter(sorted(self._entries, key=lambda k: tuple(self._entries[k])))


EVICTION_POLICIES = {"lru": LRUPolicy, "cost_size": CostSizePolicy}


def make_eviction_policy(name: str) -> EvictionPolicy:
    try:
        return EVICTION_POLICIES[name]()
    except KeyError:
        raise ValueError(f"Unknown eviction policy {name!r}; "
                         f"valid policies are {sorted(EVICTION_POLICIES)}.") from None
