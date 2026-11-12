"""Store for resident execution state, like scikit-learn estimators

- Values are never spilled or evicted
- Callers supply a size estimate at insertion and release entries after their last consumer finishes
"""
from __future__ import annotations

from dataclasses import dataclass
from operator import index
from threading import Lock
from typing import Any, Hashable


def _bytes(value: int) -> int:
    value = index(value)
    if value < 0:
        raise ValueError("byte counts must be non-negative")
    return value


@dataclass(frozen=True)
class _Entry:
    # Retain the key too, so removing it from the dict cannot run its destructor
    # under the lock. Keys must have stable, non-reentrant hash/equality methods.
    key: Hashable
    value: Any
    size: int


class StateStore:
    """Thread-safe storage for arbitrary objects under hashable keys.

    Each operation, including its accounting, is atomic. Values are borrowed by
    reference; the lock does not protect their contents. Treat stored values as
    read-only and replace them to update their size. A missing ``get`` raises
    ``KeyError``.

    Size estimates are tracked without enforcing a memory limit.
    """

    def __init__(self):
        self._memory_usage = 0
        self._entries: dict[Hashable, _Entry] = {}
        self._lock = Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            return key in self._entries

    @property
    def memory_usage(self) -> int:
        """Sum of caller-supplied size estimates for stored entries."""
        with self._lock:
            return self._memory_usage

    def put(self, key: Hashable, value: Any, *, size: int) -> None:
        """Insert or replace an entry atomically using its caller-supplied size."""
        size = _bytes(size)
        entry = _Entry(key, value, size)
        with self._lock:
            previous = self._entries.get(key)
            usage = self._memory_usage + size - (previous.size if previous else 0)
            # A dict retains the original key when an equal key replaces it.
            # Keep that key alive in the replacement entry as well.
            if previous is not None:
                entry = _Entry(previous.key, value, size)
            self._entries[key] = entry
            self._memory_usage = usage
        # The previous value's destructor may inspect or write to the store.
        del previous

    def get(self, key: Hashable) -> Any:
        """Return the stored object without copying."""
        with self._lock:
            return self._entries[key].value

    def remove(self, key: Hashable) -> bool:
        """Release the store's ownership, returning whether the key existed."""
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._memory_usage -= entry.size
        # Destruction happens only after releasing the lock.
        return entry is not None

    def clear(self) -> None:
        """Release all entries atomically; borrowed references remain valid."""
        with self._lock:
            entries, self._entries = self._entries, {}
            self._memory_usage = 0
        entries.clear()

    def snapshot(self) -> dict[str, int]:
        """An internally consistent accounting snapshot, without object references."""
        with self._lock:
            return {"entries": len(self._entries),
                    "memory_usage": self._memory_usage}
