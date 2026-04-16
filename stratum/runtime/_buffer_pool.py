from __future__ import annotations

import logging
from typing import Any, Hashable
logger = logging.getLogger(__name__)

class BufferPool:
    """Simple cache for intermediate buffers, will be replaced by a proper buffer in future. """

    def __init__(self):
        self._entries: dict[Hashable, Any] = {}  # key -> data
        self._released_count: int = 0

    def put(self, key: Hashable, data: Any):
        """Store data for a key. Overwrites any existing entry."""
        self._entries[key] = data

    def get(self, key: Hashable) -> Any:
        """Retrieve stored data for a key, or None if not present."""
        return self._entries.get(key)

    def release(self, key: Hashable) -> bool:
        """Release a single buffer, dropping its data."""
        entry = self._entries.pop(key, None)
        if entry is not None:
            logger.debug(f"Releasing buffer for {key}")
            self._released_count += 1
            return True
        return False

    def release_all(self) -> list:
        """Release everything, including pinned. Used at end of execution.

        Returns list of released keys.
        """
        released = list(self._entries.keys())
        for key in released:
            self.release(key)
        self._entries.clear()
        return released

    @property
    def active_count(self) -> int:
        return len(self._entries)

    @property
    def total_released(self) -> int:
        return self._released_count
