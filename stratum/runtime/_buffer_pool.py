from __future__ import annotations

import logging
from typing import Any, Hashable
logger = logging.getLogger(__name__)

class BufferPool:
    """Simple cache for intermediate buffers, will be replaced by a proper buffer in future. """

    def __init__(self):
        # TODO:
        # right now our buffer pool is basically a symbol table and (main memory) puffer pool in one thing
        # once we have multiple backends, we will have to separate both
        self.live_variable_map: dict[Hashable, Any] = {}  # key -> data
        self._removed_count: int = 0

    def put(self, key: Hashable, data: Any):
        """Store data for a key. Overwrites any existing entry."""
        self.live_variable_map[key] = data

    def pin(self, key: Hashable) -> Any:
        """Retrieve stored data for a key, or None if not present and lock s.t. the intermediate is not evicted"""
        return self.live_variable_map.get(key)

    def unpin(self, key: Hashable) -> None:
        """Release a pinned buffer, allowing it to be evicted."""
        return

    def remove(self, key: Hashable) -> bool:
        """Remove a single buffer, dropping its data."""
        entry = self.live_variable_map.pop(key, None)
        if entry is not None:
            logger.debug(f"Removing buffer for {key}")
            self._removed_count += 1
            return True
        return False

    def remove_all(self) -> list:
        """Remove everything, including pinned. Used at end of execution.

        Returns a list of removed keys.
        """
        removed = list(self.live_variable_map.keys())
        for key in removed:
            self.remove(key)
        self.live_variable_map.clear()
        return removed

    @property
    def active_count(self) -> int:
        return len(self.live_variable_map)

    @property
    def total_removed(self) -> int:
        return self._removed_count
