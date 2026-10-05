from __future__ import annotations
import logging
import os
import sys
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Hashable

from stratum._config import FLAGS
from stratum.runtime._object_serialization import (
    SpilledObject,
    UnsupportedSpillType,
    delete_object,
    deserialize_object,
    serialize_object,
)
from stratum.runtime._object_size import get_size, prettify_bytes

logger = logging.getLogger(__name__)

_DEFAULT_SPILL_ROOT = Path(".stratum/bufferpool")

# Fallback budget used when the host's memory cannot be read.
_FALLBACK_MEMORY_BUDGET = 2 * 1024**3  # 2 GiB


def _host_memory_bytes() -> int | None:
    """Total physical RAM of the host, or None when it cannot be read."""
    # POSIX (Linux/macOS): total pages times page size.
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return int(pages) * int(page_size)
        except (ValueError, OSError, AttributeError):
            pass
    # Windows has no os.sysconf; GlobalMemoryStatusEx reports total physical memory.
    if sys.platform == "win32":
        try:
            # Imported lazily: only the Windows path needs ctypes.
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except (OSError, AttributeError, ValueError):
            pass
    return None


def _resolve_memory_budget(fraction: float) -> int:
    """Memory budget for the BufferPool: ``fraction`` of the detected memory.
    A failed detection logs a warning and falls back to _FALLBACK_MEMORY_BUDGET
    """
    host_memory = _host_memory_bytes()
    if host_memory is None:
        logger.warning(
            "Could not detect system memory; falling back to a "
            f"{prettify_bytes(_FALLBACK_MEMORY_BUDGET)} buffer pool budget."
        )
        return _FALLBACK_MEMORY_BUDGET
    # Clamp to at least one byte: budget 0 would silently disable eviction.
    return max(1, int(fraction * host_memory))


@dataclass
class _Entry:
    """Internal record. `data` is None iff the entry has been spilled to disk."""
    size: int  # in-memory cost; constant after put
    data: Any = None
    handle: SpilledObject | None = None
    pin_count: int = 0
    spillable: bool = True  # False once serialization refused the data

    @property
    def is_spilled(self) -> bool:
        return self.handle is not None


@dataclass
class BufferPoolStats:
    """Counters/timers maintained over a BufferPool's lifetime.

    Hit = `pin` served the entry directly from memory.
    Miss = `pin` had to deserialize the entry back from disk.
    """
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    serialize_time: float = 0.0  # cumulative seconds spent spilling
    deserialize_time: float = 0.0  # cumulative seconds spent reloading
    bytes_spilled: int = 0
    bytes_loaded: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __str__(self) -> str:
        return (
            "BufferPool stats:\n"
            f"  Hits:             {self.hits}\n"
            f"  Misses:           {self.misses}\n"
            f"  Hit rate:         {self.hit_rate * 100:.1f}%\n"
            f"  Evictions:        {self.evictions}\n"
            f"  Serialize time:   {self.serialize_time:.3f}s\n"
            f"  Deserialize time: {self.deserialize_time:.3f}s\n"
            f"  Bytes spilled:    {prettify_bytes(self.bytes_spilled)}\n"
            f"  Bytes loaded:     {prettify_bytes(self.bytes_loaded)}"
        )


class BufferPool:
    """Cache for intermediates with LRU spill-to-disk eviction.

    When `memory_usage` exceeds `memory_budget`, the least-recently-pinned
    unpinned entries are spilled to disk under `spill_root`. A spilled entry
    stays in `live_variable_map` so subsequent `pin` calls can transparently
    re-load it; only the in-memory bytes are released.

    `memory_budget` defaults to `FLAGS.buffer_pool_memory_fraction` (0.7) of the
    host's total RAM, with a fixed 2 GiB fallback when memory cannot be detected.
    No config disables eviction; only direct assignment of a non-positive budget
    does.
    """

    def __init__(self, spill_root: Path | str = _DEFAULT_SPILL_ROOT):
        # TODO:
        # right now our buffer pool is basically a symbol table and (main memory) puffer pool in one thing
        # once we have multiple backends, we will have to separate both
        self.live_variable_map: OrderedDict[Hashable, _Entry] = OrderedDict()
        self._removed_count: int = 0
        self.memory_usage = 0
        # The spill root is created lazily on the first eviction. Each spill
        # gets a uuid4 filename prefix, so files never collide across entries,
        # BufferPool instances, or processes sharing the same root.
        # TODO: stale spill files from a crashed run are never reclaimed; add
        # startup/teardown cleanup of the spill root.
        self._spill_root = Path(spill_root)
        self.memory_budget = _resolve_memory_budget(FLAGS.buffer_pool_memory_fraction)
        self.stats = BufferPoolStats()

    def put(self, key: Hashable, data: Any):
        """Store data for a key. Overwrites any existing entry."""
        if key in self.live_variable_map:
            self._drop(key)
        size = get_size(data)
        self.live_variable_map[key] = _Entry(size=size, data=data)
        self.memory_usage += size
        self.check_for_eviction()

    def pin(self, key: Hashable) -> Any:
        """Retrieve stored data for a key, or None if absent.

        Locks the entry against eviction (refcounted; balance every `pin` with
        an `unpin`). If the entry was previously spilled to disk, it is
        deserialized back into memory.

        TODO: `pin`/`unpin` (and the pin_count mutation) are not thread-safe;
        guard the refcount and the spill/reload transition with a lock before
        the pool is accessed from multiple threads.
        """
        entry = self.live_variable_map.get(key)
        if entry is None:
            return None
        if entry.is_spilled:
            logger.debug(f"Deserializing {key} into memory: {prettify_bytes(entry.size)}")
            t0 = perf_counter()
            entry.data = deserialize_object(entry.handle)
            delete_object(entry.handle)
            self.stats.deserialize_time += perf_counter() - t0
            self.stats.misses += 1
            self.stats.bytes_loaded += entry.size
            entry.handle = None
            self.memory_usage += entry.size
        else:
            self.stats.hits += 1
        entry.pin_count += 1
        self.live_variable_map.move_to_end(key)
        return entry.data

    def unpin(self, key: Hashable) -> None:
        """Release a pinned buffer, allowing it to be evicted."""
        entry = self.live_variable_map.get(key)
        if entry is None:
            return
        if entry.pin_count > 0:
            entry.pin_count -= 1

    def remove(self, key: Hashable) -> bool:
        """Remove a single buffer, dropping its data (and any spilled file)."""
        if key not in self.live_variable_map:
            return False
        self._drop(key)
        self._removed_count += 1
        logger.debug(f"Removing buffer for {key}")
        assert self.memory_usage >= 0, "Memory usage is negative"
        return True

    def _drop(self, key: Hashable) -> None:
        """Internal: free in-memory bytes and any spilled file, then unlink the entry."""
        entry = self.live_variable_map.pop(key)
        if entry.is_spilled:
            delete_object(entry.handle)
        else:
            self.memory_usage -= entry.size

    def remove_all(self) -> list:
        """Remove everything, including pinned. Used at end of execution.

        Returns a list of removed keys.
        """
        removed = list(self.live_variable_map.keys())
        for key in removed:
            self.remove(key)
        return removed

    @property
    def active_count(self) -> int:
        return len(self.live_variable_map)

    @property
    def total_removed(self) -> int:
        return self._removed_count

    @property
    def total_size(self) -> str:
        """Pretty print the total size of the buffer pool."""
        return prettify_bytes(self.memory_usage)

    def check_for_eviction(self) -> bool:
        """Spill LRU unpinned entries until under budget. Returns True if any evicted."""
        if self.memory_budget <= 0 or self.memory_usage <= self.memory_budget:
            return False
        evicted = False
        # OrderedDict iterates oldest → newest; oldest = LRU.
        # FIXME(perf): `list(...)` snapshots every key up front, allocating O(n)
        # even when only a few LRU entries need spilling. With thousands of live
        # entries this is wasteful. Iterate in bounded chunks (e.g. 64 keys) and
        # stop once back under budget, re-snapshotting only if more are needed.
        for key in list(self.live_variable_map.keys()):
            if self.memory_usage <= self.memory_budget:
                break
            entry = self.live_variable_map[key]
            if entry.pin_count > 0 or entry.is_spilled or not entry.spillable:
                continue
            try:
                self._evict(key, entry)
            except UnsupportedSpillType as exc:
                # Admitted but not spillable (e.g. a CV splitter object): keep it
                # resident and try the next LRU entry instead of failing the plan.
                entry.spillable = False
                logger.debug(f"Keeping {key} in memory, it cannot be spilled: {exc}")
                continue
            evicted = True
        if self.memory_usage > self.memory_budget:
            logger.warning(
                f"Budget {prettify_bytes(self.memory_budget)} still exceeded after eviction "
                f"(usage {prettify_bytes(self.memory_usage)}); all remaining entries are "
                f"pinned or cannot be spilled."
            )
        return evicted

    def _evict(self, key: Hashable, entry: _Entry) -> None:
        self._spill_root.mkdir(parents=True, exist_ok=True)
        stem = self._spill_root / uuid.uuid4().hex
        t0 = perf_counter()
        handle = serialize_object(entry.data, stem)
        self.stats.serialize_time += perf_counter() - t0
        self.stats.evictions += 1
        self.stats.bytes_spilled += handle.size_on_disk
        self.memory_usage -= entry.size
        entry.data = None
        entry.handle = handle
        logger.debug(
            f"Evicted {key}: freed {prettify_bytes(entry.size)} "
            f"(spilled {prettify_bytes(handle.size_on_disk)} to disk)"
        )
