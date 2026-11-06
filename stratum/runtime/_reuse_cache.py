"""Intermediates kept in memory across plans, keyed by Merkle lineage keys.

The planner decides which results have reliable keys. This store accepts every
such result that fits its budget, without cost-based admission. It owns a reference
to each cached object; a reader keeps its own reference even after eviction.
Outputs are shared with execution without copying. Callers must not mutate values
after insertion or retrieval, including nested objects.

The budget accounts for estimated object sizes using the same sizing helper as
the BufferPool. It is not a bound on total process memory: readers can keep evicted
objects alive, and shared buffers can be counted more than once. Entries live
only as long as this cache instance and are never serialized.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from operator import index
from threading import Lock
from typing import Any

from stratum.runtime._eviction_policy import LRUPolicy
from stratum.runtime._object_size import get_size


@dataclass(frozen=True)
class _CacheEntry:
    data: Any
    size: int


@dataclass
class ReuseCacheStats:
    """Lifetime counters; clearing entries does not reset them."""

    puts: int = 0
    rejected: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0


def _budget_bytes(value: int) -> int:
    value = index(value)
    if value < 0:
        raise ValueError("memory_budget must be non-negative")
    return value


class ReuseCache:
    """Thread-safe memory cache with fixed LRU eviction.

    ``memory_budget`` is an estimated byte limit; zero means unlimited. Lowering
    the budget evicts entries immediately. Each operation is atomic, but a
    membership check does not reserve an entry: use ``get(key, sentinel)`` to
    distinguish a miss from a cached ``None`` in a single operation.
    """

    def __init__(self, *, memory_budget: int = 0):
        self._memory_budget = _budget_bytes(memory_budget)
        self._memory_usage = 0
        self._entries: dict[str, _CacheEntry] = {}
        self._policy = LRUPolicy()
        self._stats = ReuseCacheStats()
        self._lock = Lock()

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def memory_usage(self) -> int:
        with self._lock:
            return self._memory_usage

    @property
    def memory_budget(self) -> int:
        with self._lock:
            return self._memory_budget

    @memory_budget.setter
    def memory_budget(self, value: int) -> None:
        value = _budget_bytes(value)
        with self._lock:
            self._memory_budget = value
            if value:
                self._evict(self._memory_usage - value)

    @property
    def stats(self) -> ReuseCacheStats:
        """A consistent snapshot of the counters, independent of future updates."""
        with self._lock:
            return replace(self._stats)

    def put(self, key: str, data: Any) -> bool:
        """Keep ``data`` if it fits, evicting older entries as needed.

        A duplicate key keeps the existing value. An oversized entry is refused
        without evicting anything. False indicates either rejection.
        """
        with self._lock:
            if key in self._entries:
                self._stats.rejected += 1
                return False
            size = int(get_size(data))
            if 0 < self._memory_budget < size:
                self._stats.rejected += 1
                return False
            if self._memory_budget:
                self._evict(self._memory_usage + size - self._memory_budget)
            self._entries[key] = _CacheEntry(data, size)
            self._memory_usage += size
            self._policy.on_insert(key, size)
            self._stats.puts += 1
            return True

    def get(self, key: str, default: Any = None) -> Any:
        """Return the cached object and refresh its LRU position, or ``default``."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._stats.misses += 1
                return default
            self._policy.on_access(key)
            self._stats.hits += 1
            return entry.data

    def evict(self, bytes_needed: int) -> int:
        """Drop least recently used entries until enough estimated bytes are freed.

        Return the bytes removed from this cache, which can be less than requested
        if the cache is exhausted. Existing readers retain their references.
        """
        with self._lock:
            return self._evict(bytes_needed)

    def _evict(self, bytes_needed: int) -> int:
        """Evict while the caller holds the lock."""
        freed = 0
        if bytes_needed <= 0:
            return freed
        for key in self._policy.victims():
            entry = self._entries.pop(key)
            self._policy.on_remove(key)
            self._memory_usage -= entry.size
            self._stats.evictions += 1
            freed += entry.size
            if freed >= bytes_needed:
                break
        return freed

    def clear(self) -> None:
        """Release all cache references and LRU state, retaining the counters."""
        with self._lock:
            self._entries.clear()
            self._policy = LRUPolicy()
            self._memory_usage = 0
