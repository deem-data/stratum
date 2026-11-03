"""A client for the stratum server (:mod:`stratum._server`).

    client = StratumClient()                 # the local server, found under ~/.stratum/server
    client = StratumClient("http://host:8787", token=...)   # or a remote one
    exit_code = client.submit("train.py", ["--alpha", "1.0"])
    results = client.grid_search(pred, cv=3, scoring="r2")   # a DAG built here
    table = client.evaluate(query)

The client checks once, when it is created, that the server is reachable and runs
the same stratum version. Each call is a plain HTTP request, so one client can be
kept for a whole session.

A submitted script or call becomes a job on the server: ``start`` / ``start_call``
queue it and return its id, ``follow`` streams its output until it finishes
(reconnecting if the connection drops), and ``submit`` / ``grid_search`` /
``evaluate`` do both. A call sends the DAG pickled (``dump_call``), so it runs on
the server as it would here; ``result`` returns what it computed. Data the DAG or
script reads should be on shared storage (GCS) or at absolute paths the server can
see.
"""
from __future__ import annotations

import base64
import functools
import json
import os
import pickle
import site
import sys
import sysconfig
import time
import types
import warnings
from pathlib import Path

import httpx

DEFAULT_SERVER_DIR = Path(os.getenv("STRATUM_SERVER_DIR", Path.home() / ".stratum" / "server"))
#: Written by a running server: its URL and token, readable only by its user.
SERVER_INFO = "server.json"

#: Times ``follow`` reconnects to a job's output before giving up.
_RECONNECTS = 3


class ServerNotRunning(ConnectionError):
    """No stratum server answers at the configured address."""


class ServerError(RuntimeError):
    """The server refused or failed a request."""


class JobFailed(ServerError):
    """A call raised on the server; its traceback went to the job's stderr."""


def read_server_info(server_dir: str | os.PathLike) -> dict | None:
    try:
        return json.loads((Path(server_dir) / SERVER_INFO).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# --- what a call takes along --------------------------------------------------


def runtime() -> dict:
    """What a pickled call depends on: a function sent by value is bytecode of this
    Python, and the DAG names skrub's and stratum's classes."""
    import stratum

    return {"python": "%d.%d" % sys.version_info[:2], "skrub": stratum.__skrub_version__,
            "stratum": stratum.__version__}


@functools.cache
def _library_roots() -> tuple[str, ...]:
    paths = sysconfig.get_paths()
    roots = {paths.get(k) for k in ("stdlib", "platstdlib", "purelib", "platlib")}
    try:
        roots.update(site.getsitepackages())
    except AttributeError:  # some virtualenv flavours lack it
        pass
    roots.add(site.getusersitepackages())
    return tuple(os.path.realpath(r) + os.sep for r in roots if r)


def is_library_module(module_name: str) -> bool:
    """Whether ``module_name`` belongs to an installed library rather than user code.

    The server keeps library modules loaded between jobs (extension modules cannot be
    imported twice), and a call sends user modules by value, as the server need not
    have them. Stratum counts as a library, since the server itself runs on it.
    """
    top = module_name.partition(".")[0]
    if top in ("builtins", "stratum") or top in sys.stdlib_module_names:
        return True
    if module_name == "__main__":
        return False
    module = sys.modules.get(module_name)
    path = getattr(module, "__file__", None)
    if path is None:
        # A built-in or namespace module: not something a user edits.
        return module is not None
    return os.path.realpath(path).startswith(_library_roots())


def dump_call(dag, kwargs: dict) -> bytes:
    """Pickle ``dag`` and the keyword arguments of the call that runs it.

    An installed library's functions and classes travel by name, so the server
    imports its own pandas. User code travels by value: what ``__main__`` or a
    notebook defines, lambdas and closures, and every module that is not an installed
    library, such as a helper next to the script.
    """
    import cloudpickle

    registered = cloudpickle.list_registry_pickle_by_value()
    user = [module for name, module in list(sys.modules.items())
            if name != "__main__" and name not in registered
            and isinstance(module, types.ModuleType) and not is_library_module(name)]
    for module in user:
        cloudpickle.register_pickle_by_value(module)
    try:
        return cloudpickle.dumps({"dag": dag, "kwargs": kwargs})
    finally:
        for module in user:
            cloudpickle.unregister_pickle_by_value(module)


class StratumClient:
    """A connection to a stratum server.

    The server is ``url`` with ``token``, else ``$STRATUM_URL`` / ``$STRATUM_TOKEN``,
    else the local server that keeps its address under ``server_dir``. With
    ``check``, the server is asked for its health right away, so an unreachable
    server fails here rather than on the first request.
    """

    def __init__(self, url: str | None = None, token: str | None = None, *,
                 server_dir: str | os.PathLike = DEFAULT_SERVER_DIR, timeout: float = 30.0,
                 check: bool = True):
        url = url or os.getenv("STRATUM_URL")
        token = token or os.getenv("STRATUM_TOKEN")
        if url is None:
            info = read_server_info(server_dir)
            if info is None:
                raise ServerNotRunning(f"No stratum server is running for {server_dir}.")
            url, token = info["url"], token or info["token"]
        self.url = url
        # The server sends a heartbeat while a job is quiet, well within ``timeout``.
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._http = httpx.Client(base_url=url, headers=headers, timeout=timeout)
        self.server_version: str | None = None
        if check:
            try:
                self.server_version = self.health()["version"]
            except BaseException:
                self.close()
                raise
            from stratum import __version__
            if self.server_version != __version__:
                warnings.warn(f"The stratum server at {url} runs stratum {self.server_version}, "
                              f"this client {__version__}.", stacklevel=2)

    # --- requests ---------------------------------------------------------------

    def health(self) -> dict:
        """The server's version and how many jobs it has seen and queued."""
        return self._request("GET", "/v1/health")

    def submit(self, script: str | os.PathLike, argv: list[str] = (), *,
               cwd: str | os.PathLike | None = None, stdout=None, stderr=None) -> int:
        """Run ``script`` on the server, as if started here. Returns its exit code.

        Its output is written to ``stdout`` / ``stderr`` (this process's own by
        default) as it arrives.
        """
        return self.follow(self.start(script, argv, cwd=cwd), stdout=stdout, stderr=stderr)

    def start(self, script: str | os.PathLike, argv: list[str] = (), *,
              cwd: str | os.PathLike | None = None) -> str:
        """Queue ``script`` (relative to ``cwd``, by default this process's working
        directory) with ``argv`` on the server. Returns the job's id."""
        cwd = os.path.abspath(cwd or os.getcwd())
        path = Path(cwd, script).resolve()
        source = path.read_text(encoding="utf-8")
        reply = self._request("POST", "/v1/jobs", json={
            "script": str(path), "source": source, "argv": [str(a) for a in argv], "cwd": cwd})
        return reply["id"]

    def follow(self, job_id: str, *, stdout=None, stderr=None) -> int:
        """Write the job's output to ``stdout`` / ``stderr`` until it finishes.
        Returns its exit code."""
        stdout = stdout if stdout is not None else sys.stdout
        stderr = stderr if stderr is not None else sys.stderr
        offset, reconnects = 0, 0
        while True:
            try:
                with self._http.stream("GET", f"/v1/jobs/{job_id}/output",
                                       params={"offset": offset}) as response:
                    if response.is_error:
                        response.read()
                        raise _error(response)
                    for line in response.iter_lines():
                        if not line:  # a heartbeat
                            continue
                        event = json.loads(line)
                        if event["type"] == "result":
                            if event["status"] != "finished":
                                raise ServerError(f"Job {job_id} was {event['status']}.")
                            return event["exit_code"]
                        stream = stdout if event["stream"] == "stdout" else stderr
                        stream.write(event["data"])
                        stream.flush()
                        offset += 1
                        reconnects = 0
            except httpx.TransportError as exc:
                reconnects += 1
                if reconnects > _RECONNECTS:
                    raise ServerNotRunning(f"Lost the stratum server at {self.url} while "
                                           f"following job {job_id}: {exc}") from exc
                time.sleep(1)

    def grid_search(self, dag, cv=None, scoring=None, return_predictions=False, *,
                    config: dict | None = None, cwd: str | os.PathLike | None = None,
                    stdout=None, stderr=None):
        """Grid-search ``dag`` on the server, as :func:`stratum._api.grid_search` would
        here. Returns its results table, and with ``return_predictions`` the
        predictions too.

        ``config`` holds stratum settings for the call (as for ``set_config``); the
        server's own apply otherwise, never this process's. The server's reuse cache
        is its own, but ``{"reuse_cache": False}`` runs the call without it. Relative paths in the DAG
        are read from ``cwd``, by default this process's working directory. Its output
        is written to ``stdout`` / ``stderr`` as it arrives.
        """
        kwargs = {"cv": cv, "scoring": scoring, "return_predictions": return_predictions}
        return self._call("grid_search", dag, kwargs, config, cwd, stdout, stderr)

    def evaluate(self, dag, seed: int = 42, test_size=0.2, *, config: dict | None = None,
                 cwd: str | os.PathLike | None = None, stdout=None, stderr=None):
        """Evaluate ``dag`` on the server, as :func:`stratum._api.evaluate` would here:
        what a DAG without X and y computes (a query's table), else the predictions on
        a held-out split. ``config``, ``cwd``, ``stdout`` and ``stderr`` are as for
        :meth:`grid_search`."""
        kwargs = {"seed": seed, "test_size": test_size}
        return self._call("evaluate", dag, kwargs, config, cwd, stdout, stderr)

    def start_call(self, function: str, dag, kwargs: dict | None = None, *,
                   config: dict | None = None, cwd: str | os.PathLike | None = None) -> str:
        """Queue ``function`` ("grid_search" or "evaluate") of ``dag`` with ``kwargs``
        on the server. Returns the job's id."""
        payload = base64.b64encode(dump_call(dag, kwargs or {})).decode("ascii")
        reply = self._request("POST", "/v1/calls", json={
            "function": function, "payload": payload, "config": config or {},
            "runtime": runtime(), "cwd": os.path.abspath(cwd or os.getcwd())})
        return reply["id"]

    def result(self, job_id: str):
        """What the finished call ``job_id`` returned."""
        reply = self._request("GET", f"/v1/jobs/{job_id}/result")
        # The server's pickle: this client trusts the server it sends its code to.
        return pickle.loads(base64.b64decode(reply["value"]))

    def _call(self, function, dag, kwargs, config, cwd, stdout, stderr):
        job = self.start_call(function, dag, kwargs, config=config, cwd=cwd)
        code = self.follow(job, stdout=stdout, stderr=stderr)
        if code != 0:
            raise JobFailed(f"{function} failed on the server (job {job}, exit code {code}); "
                            "its traceback is in the job's stderr.")
        return self.result(job)

    def job(self, job_id: str) -> dict:
        """The job's state: ``queued``, ``running``, ``finished`` or ``cancelled``,
        with its exit code and duration once finished."""
        return self._request("GET", f"/v1/jobs/{job_id}")

    def stats(self) -> dict:
        """The reuse cache's size and statistics."""
        return self._request("GET", "/v1/stats")

    def clear(self) -> dict:
        """Delete every reuse cache entry, once the queued jobs ran. Returns the stats."""
        return self._request("POST", "/v1/cache/clear")

    def shutdown(self) -> None:
        """Stop the server once the queued jobs ran; it flushes the reuse cache."""
        self._request("POST", "/v1/shutdown")

    # --- plumbing ---------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise ServerNotRunning(f"No stratum server answers at {self.url}: {exc}") from exc
        if response.is_error:
            raise _error(response)
        return response.json()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> StratumClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"StratumClient({self.url!r})"


def _error(response: httpx.Response) -> ServerError:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    return ServerError(f"{response.status_code} {response.reason_phrase}: {detail}")
