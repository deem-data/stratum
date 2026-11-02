"""What a stratum server did, written to files the inspector (stratum-gui) reads.

Started with ``python -m stratum serve --diagnostics-dir DIR``, a server records one
session under ``DIR/server/``, next to the run artifacts its jobs export to ``DIR``:

* ``<session>.json``: the session, its jobs (each with the reuse-cache counts it
  accrued and the runs it exported) and every cache entry. Rewritten atomically
  when a job starts and finishes, and when the server stops.
* ``<session>.timeline.jsonl``: the cache's memory and disk usage over time, one
  line per cache change, at most one a second (plus one at each job boundary).
* ``<session>.log``: the ``stratum`` logs at the chosen level (cache events are
  INFO) and every job's output, line by line.

Files only, so the inspector never connects to the server; a stopped server's
session stays readable.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from stratum._config import FLAGS
from stratum.runtime._reuse_cache import ReuseCache

logger = logging.getLogger(__name__)
job_logger = logging.getLogger("stratum.job")

SCHEMA_VERSION = 1
#: Seconds between timeline samples taken on cache changes.
SAMPLE_INTERVAL = 1.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionRecorder:
    """Records one server session under ``directory / "server"``."""

    def __init__(self, directory: str | os.PathLike, cache: ReuseCache,
                 server_dir: str | os.PathLike, log_level: str = "INFO"):
        self.directory = Path(directory).expanduser().resolve() / "server"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session_id = uuid4().hex
        self.path = self.directory / f"{self.session_id}.json"
        self.timeline_path = self.directory / f"{self.session_id}.timeline.jsonl"
        self.log_path = self.directory / f"{self.session_id}.log"
        self.cache = cache
        self.server_dir = str(Path(server_dir).resolve())
        self.started_at = _now()
        self.status = "running"
        self.jobs: list[dict] = []
        self.current: dict | None = None
        # Which job added each entry; entries loaded from disk at startup have none.
        self.entry_jobs: dict[str, str] = {}
        self._job_before: tuple[dict, set] | None = None
        self._last_sample = 0.0
        self._partial = {"stdout": "", "stderr": ""}

        self._handler = logging.FileHandler(self.log_path, encoding="utf-8")
        self._handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"))
        level = logging.getLevelName(log_level.upper())
        self._handler.setLevel(level)
        self._stratum_logger = logging.getLogger("stratum")
        self._previous_level = self._stratum_logger.level
        if self._stratum_logger.getEffectiveLevel() > level:
            self._stratum_logger.setLevel(level)
        self._stratum_logger.addHandler(self._handler)
        # Job output goes to the session log only: through the root logger it could
        # reach the job's own (redirected) stderr and be logged again.
        job_logger.propagate = False
        job_logger.addHandler(self._handler)

        cache.listeners.append(self._on_cache_event)
        self.sample("start", force=True)
        self.write()

    # --- jobs ----------------------------------------------------------------------

    def job_started(self, number: int, script: str, argv: list[str], cwd: str) -> str:
        """Record a job about to run. Returns its id, unique across sessions."""
        job_id = f"{self.session_id}-{number}"
        self.current = {
            "job_id": job_id, "number": number, "script": script, "argv": list(argv),
            "cwd": cwd, "started_at": _now(), "finished_at": None, "seconds": None,
            "exit_code": None, "run_ids": [], "stats": None, "entries": None,
            "added_entries": None, "memory_usage": None, "disk_usage": None,
        }
        self.jobs.append(self.current)
        self._job_before = (asdict(self.cache.stats), set(self.cache.entries))
        logger.info(f"job {number} started: {script} {' '.join(argv)}".rstrip())
        self.sample("job_start", force=True)
        self.write()
        return job_id

    def job_output(self, stream: str, data: str) -> None:
        """Log a job's output, a line at a time."""
        text = self._partial[stream] + data
        *lines, self._partial[stream] = text.split("\n")
        for line in lines:
            self._log_job_line(stream, line)

    def job_finished(self, exit_code: int, seconds: float, run_ids: list[str]) -> None:
        job, (stats_before, keys_before) = self.current, self._job_before
        for stream, rest in self._partial.items():
            if rest:
                self._log_job_line(stream, rest)
            self._partial[stream] = ""
        added = set(self.cache.entries) - keys_before
        for key in added:
            self.entry_jobs[key] = job["job_id"]
        job.update(finished_at=_now(), seconds=seconds, exit_code=exit_code,
                   run_ids=list(run_ids), stats=self.cache.stats.since(stats_before),
                   entries=len(self.cache), added_entries=len(added),
                   memory_usage=self.cache.memory_usage, disk_usage=self.cache.disk_usage)
        stats = job["stats"]
        logger.info(
            f"job {job['number']} finished: exit {exit_code} in {seconds:.1f}s; "
            f"{stats['planned_hits']} hits ({stats['memory_hits']} memory, "
            f"{stats['disk_hits']} disk), {stats['puts']} admitted, {stats['spills']} "
            f"spilled; cache {len(self.cache)} entries, memory "
            f"{self.cache.memory_usage:,} B, disk {self.cache.disk_usage:,} B")
        self.current = self._job_before = None
        self.sample("job_end", force=True)
        self.write()

    def _log_job_line(self, stream: str, line: str) -> None:
        level = logging.INFO if stream == "stdout" else logging.WARNING
        job_logger.log(level, f"[{self.current['number']}] {line}")

    # --- the cache over time ----------------------------------------------------------

    def _on_cache_event(self, event: str) -> None:
        self.sample(event)

    def sample(self, event: str, force: bool = False) -> None:
        """Append the cache's current size to the timeline, throttled unless ``force``."""
        now = time.time()
        if not force and now - self._last_sample < SAMPLE_INTERVAL:
            return
        self._last_sample = now
        record = {
            "t": now, "event": event,
            "job_id": self.current["job_id"] if self.current is not None else None,
            "memory_usage": self.cache.memory_usage, "disk_usage": self.cache.disk_usage,
            "entries": len(self.cache),
        }
        with self.timeline_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    # --- the session file ---------------------------------------------------------------

    def write(self) -> None:
        cache = self.cache
        entries = [{
            "key": e.key, "label": e.label, "size": e.size, "cost": e.cost,
            "hits": e.hits, "last_access": e.last_access, "in_memory": e.in_memory,
            "size_on_disk": e.handle.size_on_disk if e.handle is not None else None,
            "job_id": self.entry_jobs.get(e.key),
        } for e in cache.entries.values()]
        document = {
            "schema_version": SCHEMA_VERSION, "session_id": self.session_id,
            "pid": os.getpid(), "started_at": self.started_at, "updated_at": _now(),
            "status": self.status, "server_dir": self.server_dir,
            "log": self.log_path.name, "timeline": self.timeline_path.name,
            "cache": {
                "dir": str(cache.root), "memory_budget": cache.memory_budget,
                "disk_budget": cache.disk_budget, "eviction": FLAGS.reuse_cache_eviction,
                "admission": FLAGS.reuse_cache_admission,
                "entries": len(cache), "memory_usage": cache.memory_usage,
                "disk_usage": cache.disk_usage, "stats": asdict(cache.stats),
            },
            "jobs": self.jobs, "entries": entries,
        }
        temporary = self.directory / f".{self.session_id}.tmp"
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(document, stream, separators=(",", ":"), allow_nan=False)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def close(self) -> None:
        """Mark the session stopped, and stop logging to it."""
        self.status = "stopped"
        self.sample("stop", force=True)
        self.write()
        self.cache.listeners.remove(self._on_cache_event)
        self._stratum_logger.removeHandler(self._handler)
        self._stratum_logger.setLevel(self._previous_level)
        job_logger.removeHandler(self._handler)
        job_logger.propagate = True
        self._handler.close()
