"""Eviction policies: *which* entries a memory store gives up first.

A store (the BufferPool, the reuse cache) decides *when* it has to evict and *how
much*; a policy only ranks its entries. Keeping the two apart is what lets one
memory manager later decide how much each store must free while every store keeps
its own notion of what is worth keeping.

A store reports every entry's lifecycle to its policy (``on_insert``,
``on_access``, ``on_remove``) and asks for ``victims()`` when it needs room. The
policy yields candidates best-victim-first; the store skips the ones it cannot
evict right now (pinned, already spilled, unspillable) and stops once it has freed
enough, so a policy never needs to know about pins or tiers.
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
