"""In-memory reuse of materialized intermediates, keyed by Merkle lineage keys.

- The planner decides which results have reliable keys. This store accepts every
  such result that fits its budget, without cost-based admission.
- All intermediates are immutable, including nested objects. No operator may
  modify an intermediate in place. The cache and BufferPool share materialized
  results without copying.
- The runtime owns intermediate lifetimes through the BufferPool and reuse
  cache. Operators borrow intermediates only for the duration of their
  execution; retaining references to inputs or outputs afterward is a memory
  leak and violates the operator contract.
- The cache budget (> 0) bounds the sum of estimated sizes of resident cache entries,
  using the same sizing helper as the BufferPool. Eviction releases cache
  ownership; the BufferPool manages intermediates still live in an executing
  plan.
- Entries live only as long as this cache instance and are never serialized (currently).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import logging
from operator import index
from threading import Lock
import time
from typing import Any, Callable

from stratum._config import FLAGS
from stratum.runtime._buffer_pool import _resolve_memory_budget
from stratum.runtime._eviction_policy import LRUPolicy
from stratum.runtime._object_size import get_size

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    data: Any
    size: int
    admission_id: int
    label: str | None = None
    hits: int = 0
    last_access: float = field(default_factory=time.time)


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
    the budget evicts entries immediately. Each operation is atomic, but an entry can be evicted between checking
    whether its key exists and retrieving its value: use ``get(key, sentinel)`` to
    distinguish a miss (returns ``None``) from a cached ``None`` in a single operation.
    """

    def __init__(self, *, memory_budget: int = 0):
        self._memory_budget = _budget_bytes(memory_budget)
        self._memory_usage = 0
        self._entries: dict[str, _CacheEntry] = {}
        self._policy = LRUPolicy()
        self._stats = ReuseCacheStats()
        self._lock = Lock()
        #: Optional diagnostic observers, called outside the cache lock.
        self.listeners: list[Callable[[str], None]] = []

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
            freed = self._evict(self._memory_usage - value) if value else 0
        if freed:
            self._notify("evict")

    @property
    def stats(self) -> ReuseCacheStats:
        """A consistent snapshot of the counters, independent of future updates."""
        with self._lock:
            return replace(self._stats)

    def snapshot(self, *, include_entries: bool = False) -> dict:
        """Copy counters and memory state atomically, optionally with entry metadata.

        Entry details contain no cached values and do not affect lookup counters
        or LRU order. Evicted entries leave no per-key history behind.
        """
        with self._lock:
            report = {
                "stats": asdict(self._stats),
                "entries": len(self._entries),
                "memory_usage": self._memory_usage,
                "memory_budget": self._memory_budget,
            }
            if include_entries:
                report["entry_details"] = [{
                    "key": key, "label": entry.label, "size": entry.size,
                    "admission_id": entry.admission_id,
                    "hits": entry.hits, "last_access": entry.last_access,
                } for key, entry in self._entries.items()]
            return report

    def put(self, key: str, data: Any, *, label: str | None = None) -> bool:
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
            self._entries[key] = _CacheEntry(data, size, self._stats.puts + 1, label=label)
            self._memory_usage += size
            self._policy.on_insert(key, size)
            self._stats.puts += 1
        self._notify("admit")
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
            entry.hits += 1
            entry.last_access = time.time()
            data = entry.data
        self._notify("read")
        return data

    def evict(self, bytes_needed: int) -> int:
        """Drop least recently used entries until enough estimated bytes are freed.

        Return the bytes removed from this cache, which can be less than requested
        if the cache is exhausted. Eviction releases cache ownership without
        changing the lifetime of intermediates managed by an executing plan.
        """
        with self._lock:
            freed = self._evict(bytes_needed)
        if freed:
            self._notify("evict")
        return freed

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
        self._notify("clear")

    def _notify(self, event: str) -> None:
        """Observers may inspect the cache; failures must not affect execution."""
        for listener in tuple(self.listeners):
            try:
                listener(event)
            except Exception:
                logger.warning("Could not record reuse cache event %s", event, exc_info=True)


_active: ReuseCache | None = None
_active_lock = Lock()


def active_reuse_cache() -> ReuseCache | None:
    """One lazily created memory cache shared by enabled runs in this process.

    Disabled runs neither create nor discard it. The configured budget takes
    effect on each enabled call; reducing it can evict existing entries. The
    lock ensures concurrent first callers receive the same cache instance.
    """
    global _active
    if not FLAGS.reuse_cache:
        return None
    with _active_lock:
        budget = _resolve_memory_budget(FLAGS.reuse_cache_memory_fraction, name="reuse cache")
        if _active is None:
            _active = ReuseCache(memory_budget=budget)
        else:
            _active.memory_budget = budget
        return _active
