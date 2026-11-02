"""Persistent reuse cache: intermediates kept across plans, keyed by lineage key.

The BufferPool holds one plan's intermediates; this cache holds those *other* plans
may need, under the keys of :mod:`stratum.optimizer._lineage`. Reuse planning
``get``s the entries a plan can use and hands them to the plan's pool, which from
then on holds its own reference: the cache never waits for a plan, and may evict
any entry at any time.

- **Tiers**, each with its own budget and eviction policy:
  - *memory*: the object. Evicting it spills to disk, or just drops the object
    when a disk copy exists.
  - *disk*: the serialized object under ``root``. Evicting it deletes the file;
    an entry with neither copy is gone.
- **Immutable entries**: one loaded back from disk keeps its file, so it can leave
  memory again for free. Nobody may mutate an object ``put`` in or returned by
  ``get``; the runtime already relies on operators never mutating their inputs.
- **History**: a record per key, kept after its entry is evicted: how often a plan
  needed it, read it, admitted it and lost it, and its last known size and
  recompute cost. It is what tells an entry evicted and admitted over and over
  from one needed once. Kept in ``history.json`` and never pruned for now.
- **Persistence**: ``manifest.json`` lists the disk tier and is rewritten whenever
  that tier changes, so a restarted process finds every entry on disk. Writes are
  write-back (on memory eviction or ``flush``): a crash loses the memory tier,
  never correctness. Unlisted cache files are deleted on load; other files under
  ``root`` are left alone.
- **One process per root**: no file locking; a second process would overwrite the
  manifest.
- **Observation**: every change is logged at INFO (``stratum.runtime._reuse_cache``)
  and reported to ``listeners``, callables taking the event name, which is how a
  server records the cache's size over time.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from stratum._config import FLAGS
from stratum.runtime._buffer_pool import _resolve_memory_budget
from stratum.runtime._eviction_policy import (EvictionPolicy, LRUPolicy,
                                              make_eviction_policy)
from stratum.runtime._object_serialization import (
    SpilledObject, _LeafRef, delete_object, deserialize_object, serialize_object)
from stratum.runtime._object_size import get_size, prettify_bytes

logger = logging.getLogger(__name__)

DEFAULT_REUSE_CACHE_ROOT = Path(".stratum/reuse_cache")
_MANIFEST = "manifest.json"
_MANIFEST_VERSION = 1
_HISTORY = "history.json"
_HISTORY_VERSION = 1
#: Files the cache writes: a uuid4 stem, a leaf index for a skeleton spill's leaves,
#: and the format's extension. Only these are ever deleted as orphans.
_CACHE_FILE = re.compile(r"[0-9a-f]{32}(_\d+)?(\.\w+)*")


@dataclass
class _CacheEntry:
    key: str
    size: int  # in-memory cost, bytes
    cost: float  # seconds to recompute
    data: Any = None
    in_memory: bool = False  # `data` may legitimately be None, so tracked apart
    handle: SpilledObject | None = None
    hits: int = 0
    last_access: float = field(default_factory=time.time)
    label: str | None = None  # the op that computed it, for inspection only

    @property
    def memory_bytes(self) -> int:
        return self.size if self.in_memory else 0

    @property
    def disk_bytes(self) -> int:
        return self.handle.size_on_disk if self.handle is not None else 0


@dataclass
class KeyHistory:
    """What the cache has seen of one key, whether or not it holds an entry now."""
    label: str | None = None  # the op that computes it, for inspection only
    size: int | None = None  # bytes in memory, when last admitted
    cost: float | None = None  # seconds to recompute, when last admitted
    seen: int = 0  # plans that needed it (reuse planning keyed it)
    hits: int = 0  # reads that found it
    admissions: int = 0  # entries made for it
    spills: int = 0  # times its entry left memory for disk
    evictions: int = 0  # times its entry left the cache altogether
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


@dataclass
class ReuseCacheStats:
    planned_hits: int = 0  # ops reuse planning took from the cache instead of computing
    planned_admissions: int = 0  # intermediates reuse planning flagged for admission
    puts: int = 0
    rejected: int = 0  # admissions refused (already cached, or too large)
    memory_hits: int = 0
    disk_hits: int = 0  # reads that loaded the entry back from disk
    misses: int = 0  # reads of a key the cache does not hold
    spills: int = 0  # memory evictions that wrote the entry to disk
    drops: int = 0  # memory evictions that lost the entry (it cannot be serialized)
    disk_evictions: int = 0
    discards: int = 0  # entries deleted because they could not be read
    bytes_written: int = 0
    bytes_loaded: int = 0
    serialize_time: float = 0.0
    deserialize_time: float = 0.0

    def __str__(self) -> str:
        return (
            "ReuseCache stats:\n"
            f"  Planned:          {self.planned_hits} hits, "
            f"{self.planned_admissions} admissions\n"
            f"  Puts:             {self.puts} ({self.rejected} rejected)\n"
            f"  Hits:             {self.memory_hits} memory, {self.disk_hits} disk "
            f"({self.misses} misses, {self.discards} unreadable)\n"
            f"  Spills:           {self.spills} ({self.drops} dropped)\n"
            f"  Disk evictions:   {self.disk_evictions}\n"
            f"  Bytes written:    {prettify_bytes(self.bytes_written)}\n"
            f"  Bytes loaded:     {prettify_bytes(self.bytes_loaded)}\n"
            f"  Serialize time:   {self.serialize_time:.3f}s\n"
            f"  Deserialize time: {self.deserialize_time:.3f}s"
        )

    def since(self, before: dict) -> dict:
        """The counts accrued since ``before``, an earlier ``asdict`` of these stats."""
        return {name: value - before[name] for name, value in asdict(self).items()}


class ReuseCache:
    """Lineage-keyed cache of intermediates with a memory and a disk tier.

    ``memory_budget`` / ``disk_budget`` are in bytes; a non-positive budget means
    unlimited. Each tier ranks its own entries with its own policy (LRU unless
    given), so the policies can be compared independently.
    """

    def __init__(self, root: Path | str = DEFAULT_REUSE_CACHE_ROOT, *,
                 memory_budget: int = 0, disk_budget: int = 0,
                 memory_policy: EvictionPolicy | None = None,
                 disk_policy: EvictionPolicy | None = None):
        # Absolute, because a skeleton spill records its leaves' paths as written
        # and a server's working directory may change.
        self.root = Path(root).resolve()
        self.memory_budget = memory_budget
        self.disk_budget = disk_budget
        self.memory_policy = memory_policy if memory_policy is not None else LRUPolicy()
        self.disk_policy = disk_policy if disk_policy is not None else LRUPolicy()
        self.entries: dict[str, _CacheEntry] = {}
        self.history: dict[str, KeyHistory] = {}
        self.memory_usage = 0
        self.disk_usage = 0
        self.stats = ReuseCacheStats()
        #: Called with an event name ("admit", "read", "load", "spill", "drop",
        #: "delete", "discard", "clear") after the cache changed or was read.
        self.listeners: list[Callable[[str], None]] = []
        # Set when the disk tier (the history) changes, cleared when the manifest
        # (the history file) is written.
        self._manifest_dirty = False
        self._history_dirty = False
        self._load_manifest()
        self._load_history()

    # --- lookup -----------------------------------------------------------------

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def cost(self, key: str) -> float:
        return self.entries[key].cost

    def tier(self, key: str) -> str | None:
        """Where a read of ``key`` would find it: "memory", "disk", or None."""
        entry = self.entries.get(key)
        if entry is None:
            return None
        return "memory" if entry.in_memory else "disk"

    def observe(self, key: str, label: str | None = None) -> None:
        """Record that a plan needs ``key``'s value, cached or not."""
        record = self._record(key, label)
        record.seen += 1
        record.last_seen = time.time()

    # --- admission ----------------------------------------------------------------

    def put(self, key: str, data: Any, cost: float = 0.0, label: str | None = None) -> bool:
        """Admit ``data`` under ``key``. Returns False if it was not admitted.

        Entries are immutable: a key already cached keeps its entry. ``data`` must
        not be mutated afterwards.
        """
        if key in self.entries:
            self.stats.rejected += 1
            return False
        size = int(get_size(data))
        if 0 < self.memory_budget < size:
            # It would evict everything else and still not fit.
            self.stats.rejected += 1
            return False
        entry = _CacheEntry(key=key, size=size, cost=cost, data=data, in_memory=True,
                            label=label)
        self.entries[key] = entry
        self._admitted(entry)
        self._enter_memory(entry)
        self._changed("admit", entry, "memory")
        self.check_for_eviction()
        return True

    def put_spilled(self, key: str, handle: SpilledObject, size: int,
                    cost: float = 0.0, label: str | None = None) -> bool:
        """Admit an object some other store already spilled, by taking its files.

        Only single-file spills can move: a skeleton spill records its leaves'
        absolute paths inside its pickle. Returns False (and leaves the files to
        their owner) when the handle cannot be taken.
        """
        if (key in self.entries or handle.format == "skeleton"
                or 0 < self.disk_budget < handle.size_on_disk):
            self.stats.rejected += 1
            return False
        self._ensure_root()
        target = self.root / f"{uuid.uuid4().hex}{handle.path.suffix}"
        shutil.move(handle.path, target)
        moved = SpilledObject(path=target, format=handle.format, leaves=(),
                              size_on_disk=handle.size_on_disk)
        entry = _CacheEntry(key=key, size=int(size), cost=cost, label=label)
        self.entries[key] = entry
        self._admitted(entry)
        self._enter_disk(entry, moved)
        self._changed("admit", entry, "disk")
        self.check_for_eviction()
        return True

    # --- reads ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """The entry's object, loaded from disk if needed, or ``default`` if the
        cache does not hold ``key``.

        The caller keeps its own reference: the entry may be evicted right after,
        as any other. An entry that cannot be loaded (a damaged file, a pickle a
        library upgrade cannot read) is discarded and reads as a miss.
        """
        entry = self.entries.get(key)
        if entry is None:
            self.stats.misses += 1
            return default
        if entry.in_memory:
            self.stats.memory_hits += 1
            self._changed("read", entry, "memory")
        else:
            t0 = perf_counter()
            try:
                entry.data = deserialize_object(entry.handle)
            except Exception as exc:  # noqa: BLE001 -- any load failure poisons the entry
                logger.warning(f"Discarding unreadable reuse cache entry {key}: {exc}")
                self.stats.discards += 1
                self._discard(entry)
                return default
            self.stats.deserialize_time += perf_counter() - t0
            self.stats.disk_hits += 1
            self.stats.bytes_loaded += entry.size
            entry.in_memory = True
            self._enter_memory(entry)
            self._changed("load", entry, "disk")
        entry.hits += 1
        entry.last_access = time.time()
        self._record(key).hits += 1
        self.memory_policy.on_access(key)
        if entry.handle is not None:
            self.disk_policy.on_access(key)
        data = entry.data
        self.check_for_eviction()
        return data

    # --- eviction ----------------------------------------------------------------

    def check_for_eviction(self) -> None:
        """Evict from each tier until it is within its budget."""
        if 0 < self.memory_budget < self.memory_usage:
            self.evict(self.memory_usage - self.memory_budget)
        if 0 < self.disk_budget < self.disk_usage:
            self.evict_disk(self.disk_usage - self.disk_budget)
        self._sync_manifest()

    def evict(self, nbytes: int) -> int:
        """Free ``nbytes`` of memory, in the memory policy's order. Returns bytes freed.

        An entry with a disk copy just drops its object; one without is spilled
        first, or lost if it cannot be.
        """
        freed = 0
        for key in self.memory_policy.victims():
            if freed >= nbytes:
                break
            entry = self.entries.get(key)
            if entry is None or not entry.in_memory:
                continue
            held = entry.memory_bytes
            if entry.handle is None and not self._spill(entry):
                self._leave_memory(entry)
                self._gone(entry)
                self.stats.drops += 1
                self._changed("drop", entry, "memory")
            else:
                self._leave_memory(entry)
                self._record(key).spills += 1
                self._changed("spill", entry, "memory")
            freed += held
        self._sync_manifest()
        return freed

    def evict_disk(self, nbytes: int) -> int:
        """Free ``nbytes`` of disk, in the disk policy's order. Returns bytes freed.

        An entry still in memory keeps living there; one that is not is gone.
        """
        freed = 0
        for key in self.disk_policy.victims():
            if freed >= nbytes:
                break
            entry = self.entries.get(key)
            if entry is None or entry.handle is None:
                continue
            freed += self._leave_disk(entry)
            self.stats.disk_evictions += 1
            if not entry.in_memory:
                self._gone(entry)
            self._changed("delete", entry, "disk")
        self._sync_manifest()
        return freed

    def _discard(self, entry: _CacheEntry) -> None:
        """Delete one entry, files included, whatever holds it."""
        if entry.in_memory:
            self._leave_memory(entry)
        if entry.handle is not None:
            self._leave_disk(entry)
        self._gone(entry)
        self._changed("discard", entry)
        self._sync_manifest()

    def clear(self) -> None:
        """Delete every entry, files included, and the history."""
        for entry in list(self.entries.values()):
            if entry.in_memory:
                self._leave_memory(entry)
            if entry.handle is not None:
                self._leave_disk(entry)
        self.entries.clear()
        self.history.clear()
        self._history_dirty = True
        self._changed("clear")
        self.save_manifest()

    # --- persistence -----------------------------------------------------------

    def flush(self) -> None:
        """Write every entry that is only in memory to disk, then the manifest and
        the history.

        What a server does on shutdown, so the next one starts with them. Entries
        that cannot be serialized stay memory-only and are lost with the process.
        """
        for entry in list(self.entries.values()):
            if entry.handle is None:
                self._spill(entry)
        self.check_for_eviction()
        self.save_manifest()

    def _sync_manifest(self) -> None:
        if self._manifest_dirty:
            self._write_manifest()

    def save_manifest(self) -> None:
        """Record the disk tier (and each entry's hits and last access) and the
        history for the next process to find."""
        self._write_manifest()
        if self._history_dirty:
            self._write_history()

    def _write_manifest(self) -> None:
        self._manifest_dirty = False
        if not self.root.exists():
            return
        entries = []
        for entry in self.entries.values():
            if entry.handle is None:
                continue
            entries.append({
                "key": entry.key, "size": entry.size, "cost": entry.cost,
                "hits": entry.hits, "last_access": entry.last_access,
                "label": entry.label,
                "handle": _handle_to_json(entry.handle, self.root),
            })
        entries.sort(key=lambda e: e["last_access"])
        _write_json(self.root / _MANIFEST, {"version": _MANIFEST_VERSION, "entries": entries})

    def _write_history(self) -> None:
        self._history_dirty = False
        self._ensure_root()
        records = {key: asdict(record) for key, record in self.history.items()}
        _write_json(self.root / _HISTORY, {"version": _HISTORY_VERSION, "keys": records})

    def _load_manifest(self) -> None:
        records = _read_json(self.root / _MANIFEST, _MANIFEST_VERSION, "entries")
        if records is None:
            return
        for record in records:  # oldest access first, so LRU order is restored
            handle = _handle_from_json(record["handle"], self.root)
            files = (handle.path, *(Path(leaf.path) for leaf in handle.leaves))
            if not all(f.exists() for f in files):
                continue
            entry = _CacheEntry(key=record["key"], size=record["size"],
                                cost=record["cost"], hits=record["hits"],
                                last_access=record["last_access"],
                                label=record.get("label"))
            self.entries[entry.key] = entry
            self._enter_disk(entry, handle)
        self._remove_orphans()

    def _load_history(self) -> None:
        records = _read_json(self.root / _HISTORY, _HISTORY_VERSION, "keys")
        if records is None:
            return
        for key, record in records.items():
            self.history[key] = KeyHistory(**record)

    def _remove_orphans(self) -> None:
        """Delete cache files under ``root`` no entry owns (from a crashed process)."""
        owned = set()
        for entry in self.entries.values():
            if entry.handle is not None:
                owned.add(entry.handle.path.name)
                owned.update(Path(leaf.path).name for leaf in entry.handle.leaves)
        for path in self.root.iterdir():
            if (path.is_file() and path.name not in owned
                    and _CACHE_FILE.fullmatch(path.name)):
                path.unlink(missing_ok=True)

    # --- observation ------------------------------------------------------------

    def _changed(self, event: str, entry: _CacheEntry | None = None,
                 tier: str | None = None) -> None:
        if logger.isEnabledFor(logging.INFO):
            what = "" if entry is None else (
                f" {entry.label or entry.key[:12]} ({prettify_bytes(entry.size)})")
            where = f" [{tier}]" if tier else ""
            logger.info(f"{event}{what}{where}; memory {prettify_bytes(self.memory_usage)}, "
                        f"disk {prettify_bytes(self.disk_usage)}, {len(self.entries)} entries")
        for listener in self.listeners:
            try:
                listener(event)
            except Exception:  # noqa: BLE001 -- an observer must never fail a plan
                logger.warning("Reuse cache listener failed", exc_info=True)

    # --- history ------------------------------------------------------------------

    def _record(self, key: str, label: str | None = None) -> KeyHistory:
        record = self.history.get(key)
        if record is None:
            record = self.history[key] = KeyHistory(label=label)
        elif label is not None:
            record.label = label
        self._history_dirty = True
        return record

    def _admitted(self, entry: _CacheEntry) -> None:
        self.stats.puts += 1
        record = self._record(entry.key, entry.label)
        record.admissions += 1
        record.size = entry.size
        record.cost = entry.cost

    def _gone(self, entry: _CacheEntry) -> None:
        """``entry`` left the cache; its key's history stays."""
        del self.entries[entry.key]
        self._record(entry.key).evictions += 1

    # --- tier bookkeeping --------------------------------------------------------

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _enter_memory(self, entry: _CacheEntry) -> None:
        self.memory_usage += entry.size
        self.memory_policy.on_insert(entry.key, entry.size, entry.cost)

    def _leave_memory(self, entry: _CacheEntry) -> None:
        self.memory_usage -= entry.memory_bytes
        entry.data = None
        entry.in_memory = False
        self.memory_policy.on_remove(entry.key)

    def _enter_disk(self, entry: _CacheEntry, handle: SpilledObject) -> None:
        entry.handle = handle
        self._manifest_dirty = True
        self.disk_usage += entry.disk_bytes
        self.disk_policy.on_insert(entry.key, entry.disk_bytes, entry.cost)

    def _leave_disk(self, entry: _CacheEntry) -> int:
        size = entry.disk_bytes
        delete_object(entry.handle)
        entry.handle = None
        self._manifest_dirty = True
        self.disk_usage -= size
        self.disk_policy.on_remove(entry.key)
        return size

    def _spill(self, entry: _CacheEntry) -> bool:
        """Write ``entry`` to disk. False if its object cannot be serialized, or
        alone exceeds the disk budget (keeping it would evict every other entry
        from disk before itself)."""
        self._ensure_root()
        t0 = perf_counter()
        try:
            handle = serialize_object(entry.data, self.root / uuid.uuid4().hex)
        except Exception as exc:  # noqa: BLE001 -- a cache must never fail a plan
            logger.debug(f"Reuse cache cannot serialize {entry.key}: {exc}")
            return False
        self.stats.serialize_time += perf_counter() - t0
        if 0 < self.disk_budget < handle.size_on_disk:
            delete_object(handle)
            logger.debug(f"Reuse cache entry {entry.key} exceeds the disk budget "
                         f"({prettify_bytes(handle.size_on_disk)})")
            return False
        self.stats.spills += 1
        self.stats.bytes_written += handle.size_on_disk
        self._enter_disk(entry, handle)
        return True


def _write_json(path: Path, document: dict) -> None:
    """Replace ``path`` atomically, so a crash leaves the old or the new file. The
    temporary file is unique, so two processes sharing a root (unsupported, but
    possible) overwrite each other's file instead of failing."""
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(document))
    os.replace(tmp, path)


def _read_json(path: Path, version: int, field_name: str):
    """``field_name`` of the cache file at ``path``, or None if it is missing,
    unreadable or of another version."""
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text())
        if document.get("version") != version:
            raise ValueError(f"version {document.get('version')}")
        return document[field_name]
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        logger.warning(f"Ignoring unreadable reuse cache file {path}: {exc}")
        return None


_active: ReuseCache | None = None


def active_reuse_cache() -> ReuseCache | None:
    """The process's reuse cache, or None while ``FLAGS.reuse_cache`` is off.

    One cache per process, so every plan (every script a server runs) shares it.
    It is created on first use, re-created when the configured directory changes,
    and flushed to disk when the process exits, so separate processes using the
    same directory share it too. Budgets follow the config on every call; the
    eviction policy is fixed when the cache is created.
    """
    global _active
    if not FLAGS.reuse_cache:
        return None
    root = Path(FLAGS.reuse_cache_dir).resolve()
    if _active is None or _active.root != root:
        if _active is not None:
            _active.flush()
        else:
            atexit.register(_flush_active)
        _active = ReuseCache(root,
                             memory_policy=make_eviction_policy(FLAGS.reuse_cache_eviction),
                             disk_policy=make_eviction_policy(FLAGS.reuse_cache_eviction))
    _active.memory_budget = _resolve_memory_budget(FLAGS.reuse_cache_memory_fraction)
    _active.disk_budget = FLAGS.reuse_cache_disk_budget
    return _active


def _flush_active() -> None:
    if _active is not None:
        _active.flush()


def _handle_to_json(handle: SpilledObject, root: Path) -> dict:
    return {
        "path": os.path.relpath(handle.path, root),
        "format": handle.format,
        "leaves": [[os.path.relpath(leaf.path, root), leaf.format] for leaf in handle.leaves],
        "size_on_disk": handle.size_on_disk,
    }


def _handle_from_json(record: dict, root: Path) -> SpilledObject:
    # A skeleton pickle records its leaves' paths as written, so the cache root
    # must not move between processes; leaves are listed for cleanup only.
    return SpilledObject(
        path=root / record["path"], format=record["format"],
        leaves=tuple(_LeafRef(path=str(root / p), format=f) for p, f in record["leaves"]),
        size_on_disk=record["size_on_disk"])
