"""Run scripts or ``grid_search`` / ``evaluate`` calls through an HTTP API.

Two submission endpoints share a job queue::

    StratumClient / CLI                       Stratum server
    -------------------                       --------------
    script + argv ---- POST /v1/jobs -----+
                                          +--> queue --> sequential execution
    DAG + kwargs ----- POST /v1/calls ----+              in the server process
    (grid_search / evaluate)                                  |
                                                              v
    job status        <-- GET /v1/jobs/{id} ------------ job state
    stdout / stderr   <-- GET /v1/jobs/{id}/output ------ output
    call return value <-- GET /v1/jobs/{id}/result ------ result

* Scripts: run a shared file or uploaded source with the supplied arguments and
  working directory when available. Data must be accessible to the server.
* DAG calls: send a serialized DAG and function arguments; Python, skrub and
  Stratum versions must match between client and server.
* Jobs: submission returns an id immediately; output streaming supports resuming
  after disconnects. Status requests remain responsive while jobs run.
* Access: bearer-token authentication, except for ``GET /v1/health``. Local clients
  discover the URL and token under ``server_dir``. Submitted code must be trusted.
* Cache and lifecycle: jobs share an in-memory cache. ``GET /v1/stats`` inspects it;
  ``POST /v1/cache/clear`` clears it; ``POST /v1/shutdown`` stops after queued work.
* Diagnostics: optional session logs, job records and run artifacts; see
  :mod:`stratum.server._session`.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import functools
import importlib
import inspect
import io
import json
import logging
import os
import pickle
import queue
import runpy
import secrets
import socket
import sys
import tempfile
import threading
import time
import traceback
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from pathlib import Path
from typing import Literal

import cloudpickle
import skrub
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import stratum
from stratum._api import evaluate, grid_search
from stratum._config import DEV_FLAGS, FLAGS, dev_config, set_config
from stratum._profiler._diagnostics import job_context
from stratum.runtime._reuse_cache import active_reuse_cache

from ._client import (DEFAULT_SERVER_DIR, SERVER_INFO, StratumClient, is_library_module,
                      read_server_info, runtime)
from ._session import SessionRecorder

logger = logging.getLogger(__name__)

#: Seconds a job's output stream may stay quiet before the server sends a blank
#: line, so neither side mistakes a quiet job for a dead connection.
_HEARTBEAT = 5.0


# --- running one job ----------------------------------------------------------


class _Stream(io.TextIOBase):
    """A text stream that appends everything written to a job's output."""

    def __init__(self, job: _Job, name: str, tee=None):
        self._job = job
        self._name = name
        self._tee = tee

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        if s:
            self._job.emit(self._name, s, self._tee)
        return len(s)


@contextlib.contextmanager
def _isolated_job(cwd: str, argv: list[str] | None = None, import_dir: str | None = None):
    """Run the body from ``cwd`` (with ``sys.argv`` set to ``argv`` and ``import_dir``
    first on ``sys.path``, when given), then restore the process state a job may
    change."""
    flags, dev_flags = vars(FLAGS).copy(), vars(DEV_FLAGS).copy()
    skrub_config = skrub.get_config()
    environ = dict(os.environ)
    filters = warnings.filters[:]
    root_logger = logging.getLogger()
    log_handlers, log_level = root_logger.handlers[:], root_logger.level
    old_argv, old_path, old_cwd = sys.argv, sys.path[:], os.getcwd()
    modules = set(sys.modules)
    try:
        importlib.invalidate_caches()  # find modules created since the last job
        os.chdir(cwd)
        if argv is not None:
            sys.argv = argv
        if import_dir is not None:
            sys.path.insert(0, import_dir)
        yield
    finally:
        os.chdir(old_cwd)
        sys.argv, sys.path[:] = old_argv, old_path
        warnings.filters[:] = filters
        root_logger.handlers[:], root_logger.level = log_handlers, log_level
        os.environ.clear()
        os.environ.update(environ)
        skrub.set_config(**skrub_config)
        vars(FLAGS).update(flags)
        vars(DEV_FLAGS).update(dev_flags)
        for name in set(sys.modules) - modules:
            if not is_library_module(name):
                del sys.modules[name]


def run_script(script: str | os.PathLike, argv: list[str] = (), cwd: str | None = None,
               stdout=None, stderr=None) -> int:
    """Run ``script`` in this process as ``__main__``. Returns its exit code.

    Output goes to ``stdout`` / ``stderr`` (the process's own by default). An
    uncaught exception is printed to ``stderr`` and gives exit code 1, as it would
    for ``python script.py``.
    """
    cwd = cwd or os.getcwd()
    script = Path(cwd, script).resolve()
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    with _isolated_job(cwd, [str(script), *argv], str(script.parent)):
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                runpy.run_path(str(script), run_name="__main__")
                return 0
            except SystemExit as exc:
                if exc.code is None or isinstance(exc.code, int):
                    return exc.code or 0
                print(exc.code, file=stderr)
                return 1
            except BaseException:  # noqa: BLE001 -- report it like the interpreter would
                traceback.print_exc(file=stderr)
                return 1


def _grid_search(dag, return_predictions=False, **kwargs):
    # The scheduler stays here; the client gets the table it filled.
    if return_predictions:
        sched, predictions = grid_search(dag, return_predictions=True, **kwargs)
        return sched.results_, predictions
    return grid_search(dag, **kwargs).results_


#: What a call may run, by the name the client gives.
CALLS = {"grid_search": _grid_search, "evaluate": evaluate}

#: The settings a call may change for itself. The server owns its reuse cache (its
#: budget); a call may run without it, with
#: ``reuse_cache=False``: it then reads nothing from the cache and adds nothing.
CALL_SETTINGS = frozenset(name for name in inspect.signature(set_config).parameters
                          if not name.startswith("reuse_cache_"))


def run_call(function: str, payload: bytes, config: dict | None = None,
             cwd: str | None = None, stdout=None, stderr=None) -> tuple[int, bytes | None]:
    """Run ``CALLS[function]`` on the DAG pickled in ``payload`` (see
    :func:`stratum.server._client.dump_call`) in this process, with the stratum settings
    ``config``, from ``cwd``. Returns its exit code and what it returned, pickled
    (None when it raised).

    Output goes to ``stdout`` / ``stderr`` as for :func:`run_script`; an exception,
    raised while loading the DAG or running it, is printed to ``stderr`` and gives
    exit code 1.
    """
    stdout = stdout if stdout is not None else sys.stdout
    stderr = stderr if stderr is not None else sys.stderr
    with _isolated_job(cwd or os.getcwd()):
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                set_config(**(config or {}))
                # Loading imports what the DAG names by reference; the job's isolation
                # drops the user modules among them afterwards.
                call = pickle.loads(payload)
                value = CALLS[function](call["dag"], **call["kwargs"])
                # cloudpickle: the value may hold a class the client sent by value.
                return 0, cloudpickle.dumps(value)
            except BaseException:  # noqa: BLE001 -- report it like a script's
                traceback.print_exc(file=stderr)
                return 1, None


@contextlib.contextmanager
def _job_files(request: JobRequest | CallRequest) -> Iterator[tuple[str | None, str]]:
    """The script to run for ``request`` (None for a call) and the directory to run
    it in.

    The client's file and working directory when this machine sees them too;
    otherwise the uploaded source, from a temporary directory.
    """
    if isinstance(request, CallRequest):
        if os.path.isdir(request.cwd):
            yield None, request.cwd
            return
        with tempfile.TemporaryDirectory(prefix="stratum-job-") as tmp:
            yield None, tmp
        return
    script = Path(request.script)
    try:
        in_place = script.read_text(encoding="utf-8") == request.source
    except (OSError, UnicodeDecodeError):
        in_place = False
    if in_place and os.path.isdir(request.cwd):
        yield str(script), request.cwd
        return
    # TODO(multi-tenant): only the script travels. Helper modules next to it, and
    # data at relative paths, need a file system the server shares with the client.
    with tempfile.TemporaryDirectory(prefix="stratum-job-") as tmp:
        copy = Path(tmp, script.name)
        copy.write_text(request.source, encoding="utf-8")
        yield str(copy), request.cwd if os.path.isdir(request.cwd) else tmp


# --- jobs ------------------------------------------------------------------------


class JobRequest(BaseModel):
    script: str  #: the script's absolute path on the client
    source: str
    argv: list[str] = []
    cwd: str  #: the client's working directory

    @property
    def label(self) -> str:
        return self.script

    def describe(self) -> dict:
        return {"script": self.script, "argv": self.argv}


class CallRequest(BaseModel):
    function: Literal["grid_search", "evaluate"]
    payload: str  #: base64 of the pickled DAG and kwargs (stratum.server._client.dump_call)
    config: dict = {}  #: stratum settings for the call, as for set_config
    runtime: dict  #: the client's Python, skrub and stratum versions
    cwd: str  #: the client's working directory

    @property
    def label(self) -> str:
        return f"<{self.function}>"

    @property
    def argv(self) -> list[str]:
        return []

    def describe(self) -> dict:
        return {"function": self.function}


class _Job:
    """A submitted script or call: its request, its state, the output it printed so
    far and, for a call, what it returned.

    The main thread runs it and appends its output; API threads read it.
    """

    def __init__(self, number: int, request: JobRequest | CallRequest):
        self.id = secrets.token_hex(8)
        self.number = number
        self.request = request
        self.status = "queued"
        self.exit_code: int | None = None
        self.seconds: float | None = None
        # TODO(multi-tenant): finished jobs, their output and results stay in memory
        # until the server stops.
        self.output: list[dict] = []
        self.result: bytes | None = None  #: a call's value, pickled
        self.changed = threading.Condition()

    @property
    def done(self) -> bool:
        return self.status in ("finished", "cancelled")

    def info(self) -> dict:
        return {"id": self.id, "number": self.number, **self.request.describe(),
                "status": self.status, "exit_code": self.exit_code, "seconds": self.seconds}

    def update(self, status: str, **fields) -> None:
        with self.changed:
            self.status = status
            for name, value in fields.items():
                setattr(self, name, value)
            self.changed.notify_all()

    def emit(self, stream: str, data: str, tee=None) -> None:
        with self.changed:
            self.output.append({"type": "output", "stream": stream, "data": data})
            if tee is not None:
                tee(stream, data)
            self.changed.notify_all()

    def follow(self, offset: int) -> Iterator[str]:
        """The job's output from event ``offset`` on, then its result, as lines of
        JSON. A blank line stands for a quiet ``_HEARTBEAT``."""
        while True:
            with self.changed:
                self.changed.wait_for(lambda: len(self.output) > offset or self.done,
                                      timeout=_HEARTBEAT)
                new, done = self.output[offset:], self.done
            offset += len(new)
            if new:
                yield "".join(json.dumps(event) + "\n" for event in new)
            elif not done:
                yield "\n"
            if done:
                yield json.dumps({"type": "result", **self.info()}) + "\n"
                return


# --- the server ------------------------------------------------------------------


class StratumServer:
    """Serves the HTTP API on ``host``:``port`` (a free port by default) and runs the
    submitted jobs on the thread that calls :meth:`serve_forever`."""

    def __init__(self, server_dir: str | os.PathLike = DEFAULT_SERVER_DIR,
                 diagnostics_dir: str | os.PathLike | None = None, log_level: str = "INFO",
                 host: str = "127.0.0.1", port: int = 0):
        self.server_dir = Path(server_dir)
        self.diagnostics_dir = diagnostics_dir
        self.log_level = log_level
        self.host = host
        self.port = port
        # TODO(multi-tenant): one token for everyone. Per-user tokens would let jobs,
        # stats and the admin requests (clear, shutdown) be scoped to their user.
        self.token = os.getenv("STRATUM_TOKEN") or secrets.token_urlsafe(32)
        self.recorder: SessionRecorder | None = None
        self.cache = None
        self.jobs: dict[str, _Job] = {}
        # TODO(multi-tenant): every job runs in this process, one after the other, so
        # one user's job waits for another's and can break the process for everyone.
        self._tasks: queue.SimpleQueue[Callable[[], None] | None] = queue.SimpleQueue()

    def serve_forever(self) -> None:
        self.server_dir.mkdir(parents=True, exist_ok=True)
        info_path = self.server_dir / SERVER_INFO
        if _alive(self.server_dir):
            raise RuntimeError(f"A stratum server is already running for {self.server_dir}.")
        FLAGS.reuse_cache = True
        cache = self.cache = active_reuse_cache()
        if self.diagnostics_dir is not None:
            self.recorder = SessionRecorder(self.diagnostics_dir, cache, self.server_dir,
                                            self.log_level)
            print(f"recording session {self.recorder.path}", flush=True)
        family = socket.AF_INET6 if ":" in self.host else socket.AF_INET
        sock = socket.create_server((self.host, self.port), family=family)
        port = sock.getsockname()[1]
        api = uvicorn.Server(uvicorn.Config(self._app(), lifespan="off", access_log=False,
                                            log_level="warning", timeout_graceful_shutdown=5))
        thread = threading.Thread(target=api.run, kwargs={"sockets": [sock]},
                                  name="stratum-api", daemon=True)
        try:
            thread.start()
            while not api.started:
                if not thread.is_alive():
                    raise RuntimeError("The stratum server's API failed to start.")
                time.sleep(0.01)
            # The address is published only once the API accepts requests.
            local = {"0.0.0.0": "127.0.0.1", "": "127.0.0.1", "::": "::1"}.get(self.host, self.host)
            url = f"http://[{local}]:{port}" if ":" in local else f"http://{local}:{port}"
            fd = os.open(info_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump({"url": url, "token": self.token, "pid": os.getpid()}, f)
            print(f"stratum server listening on {self.host}:{port} "
                  f"(in-memory reuse cache, {len(cache)} entries)", flush=True)
            while True:
                try:
                    # A wait without timeout misses Ctrl-C when the API thread gets it.
                    task = self._tasks.get(timeout=0.5)
                except queue.Empty:
                    continue
                if task is None:
                    break
                task()
        finally:
            info_path.unlink(missing_ok=True)
            api.should_exit = True
            for job in list(self.jobs.values()):
                if job.status == "queued":
                    job.update("cancelled")
            thread.join(timeout=10)
            sock.close()
            if self.recorder is not None:
                self.recorder.close()
            print("stratum server stopped", flush=True)

    def _app(self) -> FastAPI:
        app = FastAPI(title="stratum server", version=stratum.__version__)

        @app.get("/v1/health")
        def health() -> dict:
            return {"ok": True, "version": stratum.__version__, "jobs": len(self.jobs),
                    "queued": sum(job.status == "queued" for job in list(self.jobs.values()))}

        api = APIRouter(prefix="/v1", dependencies=[Depends(self._authorize)])

        @api.post("/jobs")
        async def submit(request: JobRequest) -> dict:
            return self._queue(request)

        @api.post("/calls")
        async def call(request: CallRequest) -> dict:
            ours = runtime()
            differ = {k: (request.runtime.get(k), v) for k, v in ours.items()
                      if request.runtime.get(k) != v}
            if differ:
                detail = ", ".join(f"{k} {client} (server: {server})"
                                   for k, (client, server) in differ.items())
                raise HTTPException(409, f"the call was pickled under {detail}; "
                                         "a pickled DAG loads only under the same versions")
            if unknown := sorted(set(request.config) - CALL_SETTINGS):
                raise HTTPException(400, f"a call cannot change {', '.join(unknown)}")
            return self._queue(request)

        @api.get("/jobs/{job_id}")
        def job_info(job_id: str) -> dict:
            return self._job(job_id).info()

        @api.get("/jobs/{job_id}/result")
        def job_result(job_id: str) -> dict:
            job = self._job(job_id)
            if not isinstance(job.request, CallRequest):
                raise HTTPException(404, f"job {job_id} ran a script, which returns nothing")
            if not job.done:
                raise HTTPException(409, f"job {job_id} is {job.status}")
            if job.result is None:
                raise HTTPException(409, f"job {job_id} failed with exit code {job.exit_code}")
            return {"value": base64.b64encode(job.result).decode("ascii")}

        @api.get("/jobs/{job_id}/output")
        def job_output(job_id: str, offset: int = 0) -> StreamingResponse:
            return StreamingResponse(self._job(job_id).follow(offset),
                                     media_type="application/x-ndjson")

        @api.get("/stats")
        def stats() -> dict:
            return self._stats()

        @api.post("/cache/clear")
        async def clear() -> dict:
            # TODO(multi-tenant): any user may clear the cache everyone shares.
            return await asyncio.wrap_future(self._on_main_thread(self._clear))

        @api.post("/shutdown")
        def shutdown() -> dict:
            self._tasks.put(None)
            return {"ok": True}

        app.include_router(api)
        return app

    def _authorize(self, authorization: str = Header("")) -> None:
        expected = f"Bearer {self.token}".encode()
        if not secrets.compare_digest(authorization.encode(), expected):
            raise HTTPException(401, "missing or wrong token")

    def _job(self, job_id: str) -> _Job:
        try:
            return self.jobs[job_id]
        except KeyError:
            raise HTTPException(404, f"no job {job_id}") from None

    def _queue(self, request: JobRequest | CallRequest) -> dict:
        # Runs on the event loop's thread, so job numbers do not race.
        job = _Job(len(self.jobs) + 1, request)
        self.jobs[job.id] = job
        self._tasks.put(functools.partial(self._run, job))
        return job.info()

    def _on_main_thread(self, fn: Callable[[], dict]) -> Future:
        """Run ``fn`` after the queued jobs, on the thread that runs them."""
        future: Future = Future()

        def task():
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(fn())
                except BaseException as exc:  # noqa: BLE001 -- the caller gets it
                    future.set_exception(exc)
        self._tasks.put(task)
        return future

    def _run(self, job: _Job) -> None:
        recorder = self.recorder
        tee = recorder.job_output if recorder is not None else None
        out, err = _Stream(job, "stdout", tee), _Stream(job, "stderr", tee)
        request = job.request
        job.update("running")
        start = time.perf_counter()
        context = {"run_ids": []}
        try:
            with _job_files(request) as (script, cwd):
                if recorder is None:
                    code = self._execute(job, script, cwd, out, err)
                else:
                    job_id = recorder.job_started(job.number, request.label, request.argv, cwd)
                    # Its runs are exported beside the session (a script may still
                    # choose another directory).
                    with job_context(job_id) as context, \
                            dev_config(diagnostics_dir=str(recorder.directory.parent)):
                        code = self._execute(job, script, cwd, out, err)
        except Exception:  # noqa: BLE001 -- the server outlives a job it could not start
            traceback.print_exc(file=err)
            code = 1
        seconds = time.perf_counter() - start
        if recorder is not None:
            recorder.job_finished(code, seconds, context["run_ids"])
        job.update("finished", exit_code=code, seconds=seconds)

    @staticmethod
    def _execute(job: _Job, script: str | None, cwd: str, out, err) -> int:
        request = job.request
        if isinstance(request, CallRequest):
            code, job.result = run_call(request.function, base64.b64decode(request.payload),
                                        request.config, cwd, stdout=out, stderr=err)
            return code
        return run_script(script, request.argv, cwd, stdout=out, stderr=err)

    def _clear(self) -> dict:
        self.cache.clear()
        if self.recorder is not None:
            self.recorder.write()
        return self._stats()

    def _stats(self) -> dict:
        # TODO(multi-tenant): the cache is shared, so a user is served entries derived
        # from data they may not be allowed to read, and sees everyone's statistics.
        # A job may temporarily disable reuse. Read the server-owned cache
        # directly, without applying that job's transient configuration.
        return {"jobs": len(self.jobs), **self.cache.snapshot()}


def _alive(server_dir: Path) -> bool:
    info = read_server_info(server_dir)
    if info is None:
        return False
    try:
        with StratumClient(info["url"], info["token"], timeout=5.0, check=False) as client:
            client.health()
            return True
    except Exception:  # noqa: BLE001 -- a stale address: nothing (or something else) answers
        return False
