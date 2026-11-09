"""What a stratum server did, written to files the inspector (stratum-gui) reads.

Started with ``python -m stratum serve --diagnostics-dir DIR``, a server records one
session under ``DIR/server/``, next to the run artifacts its jobs export to ``DIR``:

* ``<session>.json``: the session, its jobs (each with cache counter changes and
  the runs it exported), aggregate cache state and live entry metadata. Rewritten
  atomically when a job starts and finishes, and when the server stops.
* ``<session>.timeline.jsonl``: the cache's memory usage on cache events, sampled
  at most once a second, plus job boundaries, startup, shutdown and clearing.
* ``<session>.log``: the ``stratum`` logs at the chosen level and every job's
  output, line by line.

Files only, so the inspector never connects to the server; a stopped server's
session stays readable.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from stratum.runtime._reuse_cache import ReuseCache

logger = logging.getLogger(__name__)
job_logger = logging.getLogger("stratum.job")

SCHEMA_VERSION = 1
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
        self.entry_jobs: dict[str, str] = {}
        self._job_before: tuple[dict, dict] | None = None
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

        self.cache.listeners.append(self._on_cache_event)
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
            "stats_scope": "job_interval", "memory_usage": None, "added_entries": None,
        }
        self.jobs.append(self.current)
        snapshot = self.cache.snapshot(include_entries=True)
        self._job_before = (snapshot["stats"],
                            {entry["key"]: entry["admission_id"]
                             for entry in snapshot["entry_details"]})
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
        snapshot = self.cache.snapshot(include_entries=True)
        # A key can be evicted and inserted again during the same job.
        added = {entry["key"] for entry in snapshot["entry_details"]
                 if keys_before.get(entry["key"]) != entry["admission_id"]}
        for key in added:
            self.entry_jobs[key] = job["job_id"]
        stats = {name: value - stats_before[name] for name, value in snapshot["stats"].items()}
        job.update(finished_at=_now(), seconds=seconds, exit_code=exit_code,
                   run_ids=list(run_ids), stats=stats, entries=snapshot["entries"],
                   memory_usage=snapshot["memory_usage"], added_entries=len(added))
        logger.info(
            f"job {job['number']} finished: exit {exit_code} in {seconds:.1f}s; "
            f"{stats['hits']} hits, {stats['puts']} inserted, "
            f"{stats['rejected']} rejected, {stats['evictions']} evicted; "
            f"cache {snapshot['entries']} entries, memory {snapshot['memory_usage']:,} B")
        self.current = self._job_before = None
        self.sample("job_end", force=True)
        self.write()

    def _log_job_line(self, stream: str, line: str) -> None:
        level = logging.INFO if stream == "stdout" else logging.WARNING
        job_logger.log(level, f"[{self.current['number']}] {line}")

    # --- the cache over time ----------------------------------------------------------

    def _on_cache_event(self, event: str) -> None:
        self.sample(event, force=event == "clear")

    def sample(self, event: str, force: bool = False) -> None:
        """Sample cache changes, always retaining lifecycle boundaries."""
        now = time.time()
        if not force and now - self._last_sample < SAMPLE_INTERVAL:
            return
        self._last_sample = now
        snapshot = self.cache.snapshot()
        record = {
            "t": now, "event": event,
            "job_id": self.current["job_id"] if self.current is not None else None,
            "memory_usage": snapshot["memory_usage"],
            "memory_budget": snapshot["memory_budget"], "entries": snapshot["entries"],
        }
        with self.timeline_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    # --- the session file ---------------------------------------------------------------

    def write(self) -> None:
        snapshot = self.cache.snapshot(include_entries=True)
        entries = snapshot.pop("entry_details")
        live = {entry["key"] for entry in entries}
        self.entry_jobs = {key: job for key, job in self.entry_jobs.items() if key in live}
        for entry in entries:
            entry["job_id"] = self.entry_jobs.get(entry["key"])
        document = {
            "schema_version": SCHEMA_VERSION, "session_id": self.session_id,
            "pid": os.getpid(), "started_at": self.started_at, "updated_at": _now(),
            "status": self.status, "server_dir": self.server_dir,
            "log": self.log_path.name, "timeline": self.timeline_path.name,
            "cache": {**snapshot, "stats_scope": "cache_lifetime"},
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
