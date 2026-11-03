"""Stratum as a server: scripts and DAGs run in one process and share its reuse cache."""
import io
import json
import os
import pickle
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pandas as pd
import polars as pl
import polars.testing
import pytest
import skrub
from sklearn.linear_model import Ridge

import stratum as st
from stratum.__main__ import main
from stratum._api import evaluate, grid_search
from stratum._client import (SERVER_INFO, JobFailed, ServerError, ServerNotRunning,
                             StratumClient, dump_call)
from stratum._config import FLAGS
from stratum._server import run_call, run_script

REPO = Path(__file__).resolve().parents[1]

JOB = textwrap.dedent("""
    import sys, pathlib
    import numpy as np, pandas as pd
    import stratum as st
    from stratum._api import evaluate
    from stratum.runtime._reuse_cache import active_reuse_cache
    from sklearn.linear_model import Ridge
    import helper

    csv = pathlib.Path("data.csv")
    if not csv.exists():
        rng = np.random.default_rng(0)
        pd.DataFrame({"a": rng.random(50), "b": rng.random(50),
                      "y": rng.random(50)}).to_csv(csv, index=False)
    data = st.as_data_op(str(csv)).skb.apply_func(pd.read_csv)
    data = data.assign(c=data["a"] * helper.OFFSET + data["b"])
    X = data[["a", "b", "c"]].skb.mark_as_X()
    y = data["y"].skb.mark_as_y()
    pred = evaluate(X.skb.apply(Ridge(alpha=float(sys.argv[1])), y=y))
    print("pred", round(float(pred[0]), 6))
    print("reads", active_reuse_cache().stats.planned_hits)
""")


def write(path: Path, text: str) -> Path:
    path.write_text(textwrap.dedent(text))
    return path


# --- running a script in-process --------------------------------------------------

def test_run_script_behaves_like_python(tmp_path):
    script = write(tmp_path / "s.py", """
        import os, sys
        print("args", sys.argv[1:], "cwd", os.getcwd())
        print("oops", file=sys.stderr)
    """)
    out, err = io.StringIO(), io.StringIO()
    assert run_script(script, ["a", "b"], cwd=str(tmp_path), stdout=out, stderr=err) == 0
    assert out.getvalue() == f"args ['a', 'b'] cwd {tmp_path}\n"
    assert err.getvalue() == "oops\n"

    assert run_script(write(tmp_path / "e.py", "raise SystemExit(3)"), stdout=out) == 3
    err = io.StringIO()
    assert run_script(write(tmp_path / "f.py", "1 / 0"), stderr=err) == 1
    assert "ZeroDivisionError" in err.getvalue()


def test_run_script_restores_process_state(tmp_path):
    write(tmp_path / "helper_mod.py", "VALUE = 1\n")
    script = write(tmp_path / "s.py", """
        import os, sys, stratum as st, skrub
        import helper_mod
        print(helper_mod.VALUE)
        st.set_config(stats=True, reuse_cache_admission="none")
        skrub.set_config(eager_data_ops=False)
        os.environ["STRATUM_TEST_VAR"] = "1"
        sys.argv.append("x")
    """)
    import skrub
    before = (dict(vars(FLAGS)), skrub.get_config(), dict(os.environ), list(sys.argv),
              os.getcwd())
    out = io.StringIO()
    assert run_script(script, cwd=str(tmp_path), stdout=out) == 0
    after = (dict(vars(FLAGS)), skrub.get_config(), dict(os.environ), list(sys.argv),
             os.getcwd())
    assert after == before
    assert "helper_mod" not in sys.modules, "user modules are dropped after a job"
    # A different size: the bytecode cache checks size and whole-second mtime.
    write(tmp_path / "helper_mod.py", "VALUE = 22\n")
    assert run_script(script, cwd=str(tmp_path), stdout=out) == 0
    assert out.getvalue().split() == ["1", "22"], "an edited helper is re-imported"


# --- running a call in-process ----------------------------------------------------

SCALE = 2.0


def with_ratio(df):
    # Defined in a module the server need not have, reading one of its globals.
    return df.assign(r=df["a"] / (df["b"] + 1) * SCALE)


def sales_dag(csv: Path, net):
    """A grid search over a CSV read through user functions: ``with_ratio``, ``net``
    (from a helper module) and a lambda with a closure."""
    offset = 0.5
    with skrub.config_context(eager_data_ops=False):
        data = st.as_data_op(str(csv)).skb.apply_func(pd.read_csv)
        data = data.skb.apply_func(with_ratio).skb.apply_func(net)
        data = data.assign(s=data["a"].skb.apply_func(lambda a: a + offset))
        X = data[["a", "b", "r", "n", "s"]].skb.mark_as_X()
        y = data["y"].skb.mark_as_y()
        return X.skb.apply(Ridge(alpha=skrub.choose_from([0.1, 10.0], name="alpha")), y=y)


@pytest.fixture
def sales_csv(tmp_path):
    rng = __import__("numpy").random.default_rng(0)
    path = tmp_path / "sales.csv"
    pd.DataFrame({"a": rng.random(60), "b": rng.random(60),
                  "y": rng.random(60)}).to_csv(path, index=False)
    return path


@pytest.fixture
def pricing(tmp_path, monkeypatch):
    """A helper module only this process can import: the server's ``sys.path`` and
    working directory hold no ``pricing``."""
    lib = tmp_path / "lib"
    lib.mkdir()
    write(lib / "pricing.py", """
        def net(df):
            return df.assign(n=df["a"] * (1 - df["b"]))
    """)
    monkeypatch.syspath_prepend(str(lib))
    import pricing
    yield pricing
    sys.modules.pop("pricing", None)


def query(frame):
    with skrub.config_context(eager_data_ops=False):
        t = st.var("t", frame)
        return t.assign(w=t["v"] * 10)


def test_a_call_travels_by_name_for_libraries_and_by_value_for_user_code(sales_csv, pricing):
    dag = sales_dag(sales_csv, pricing.net)
    blob = dump_call(dag, {"cv": 3})
    assert b"pricing" in blob and b"read_csv" in blob
    # A process without the helper module loads it, and computes what this one does.
    code = ("import pickle, sys; from stratum._api import grid_search;"
            "call = pickle.loads(sys.stdin.buffer.read());"
            "print(grid_search(call['dag'], scoring='r2', **call['kwargs']).results_"
            ".sort('id')['scores'].to_list())")
    out = subprocess.run([sys.executable, "-c", code], input=blob, capture_output=True,
                         cwd=REPO, check=True).stdout.decode()
    local = grid_search(dag, cv=3, scoring="r2").results_.sort("id")["scores"].to_list()
    assert out.strip().splitlines()[-1] == str(local)
    import cloudpickle
    assert not cloudpickle.list_registry_pickle_by_value(), "registrations are undone"


def test_run_call_applies_its_settings_only_while_it_runs(tmp_path):
    frame = pd.DataFrame({"v": [0, 1, 2, 3]})
    before = dict(vars(FLAGS))
    code, value = run_call("evaluate", dump_call(query(frame), {}),
                           {"implementation_selector": "greedy"}, str(tmp_path))
    assert code == 0 and vars(FLAGS) == before
    result = pickle.loads(value)
    assert isinstance(result, pl.DataFrame), "the call's selector ran it on polars"
    pd.testing.assert_frame_equal(result.to_pandas(), evaluate(query(frame)))

    def boom(df):
        raise ValueError("no such column")
    err = io.StringIO()
    with skrub.config_context(eager_data_ops=False):
        failing = st.var("t", frame).skb.apply_func(boom)
    code, value = run_call("evaluate", dump_call(failing, {}), stderr=err, cwd=str(tmp_path))
    assert (code, value) == (1, None)
    assert "ValueError: no such column" in err.getvalue()


# --- the server -------------------------------------------------------------------

def start_server(server_dir: Path, cache_dir: Path, *serve_args: str, cwd=REPO) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "stratum", "--server-dir", str(server_dir), "serve",
         "--cache-dir", str(cache_dir), *serve_args],
        cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + 60
    while not (server_dir / SERVER_INFO).exists():
        if proc.poll() is not None or time.monotonic() > deadline:
            proc.kill()
            raise RuntimeError(f"server did not start:\n{proc.stdout.read()}")
        time.sleep(0.1)
    return proc


def stop_server(server_dir: Path, proc: subprocess.Popen) -> str:
    with StratumClient(server_dir=server_dir) as client:
        client.shutdown()
    out, _ = proc.communicate(timeout=60)
    return out


@pytest.fixture(autouse=True)
def no_remote_server(monkeypatch):
    monkeypatch.delenv("STRATUM_URL", raising=False)
    monkeypatch.delenv("STRATUM_TOKEN", raising=False)


@pytest.fixture
def job_dir(tmp_path, monkeypatch):
    job = tmp_path / "job"
    job.mkdir()
    write(job / "job.py", JOB)
    write(job / "helper.py", "OFFSET = 2\n")
    monkeypatch.chdir(job)
    return job


def run(server_dir, *argv):
    out, err = io.StringIO(), io.StringIO()
    with StratumClient(server_dir=server_dir) as client:
        code = client.submit("job.py", list(argv), stdout=out, stderr=err)
    assert code == 0, err.getvalue()
    lines = dict(line.split(" ", 1) for line in out.getvalue().splitlines())
    return float(lines["pred"]), int(lines["reads"])


def test_submitted_scripts_share_the_cache(tmp_path, job_dir):
    server_dir, cache_dir = tmp_path / "server", tmp_path / "cache"
    proc = start_server(server_dir, cache_dir)
    try:
        pred, reads = run(server_dir, "1.0")
        assert reads == 0
        again, reads = run(server_dir, "1.0")
        assert again == pred and reads > 0, "the second submission reads the first's work"
        # A different size: the bytecode cache checks size and whole-second mtime.
        write(job_dir / "helper.py", "OFFSET = 3  # edited\n")
        edited, _ = run(server_dir, "1.0")
        assert edited != pred, "an edited helper module is picked up"
        with StratumClient(server_dir=server_dir) as client:
            with pytest.raises(FileNotFoundError):
                client.start("missing.py")
            stats = client.stats()
        assert stats["jobs"] == 3 and stats["entries"] > 0
    finally:
        log = stop_server(server_dir, proc)
    assert "stratum server stopped" in log
    assert (cache_dir / "manifest.json").exists()

    # A restarted server finds what the previous one flushed.
    write(job_dir / "helper.py", "OFFSET = 2\n")
    proc = start_server(server_dir, cache_dir)
    try:
        restarted, reads = run(server_dir, "1.0")
        assert restarted == pred and reads > 0
    finally:
        stop_server(server_dir, proc)


def test_a_relative_cache_directory_is_the_servers(tmp_path, job_dir):
    # The jobs run from job_dir; the cache stays where the server started.
    server_dir = tmp_path / "server"
    proc = start_server(server_dir, Path("cache"), cwd=tmp_path)
    try:
        pred, _ = run(server_dir, "1.0")
        with StratumClient(server_dir=server_dir) as client:
            assert client.stats()["entries"] > 0, "the server reports the jobs' entries"
            client.clear()
            assert client.stats()["entries"] == 0
        again, reads = run(server_dir, "1.0")
        assert again == pred and reads == 0, "cleared where the jobs wrote"
    finally:
        stop_server(server_dir, proc)
    assert (tmp_path / "cache" / "manifest.json").exists()
    assert not (job_dir / "cache").exists()


def test_a_server_records_its_session(tmp_path, job_dir, sales_csv, pricing):
    server_dir, cache_dir, runs = tmp_path / "server", tmp_path / "cache", tmp_path / "runs"
    proc = start_server(server_dir, cache_dir, "--diagnostics-dir", str(runs))
    try:
        run(server_dir, "1.0")
        run(server_dir, "1.0")
        with StratumClient(server_dir=server_dir) as client:
            client.evaluate(sales_dag(sales_csv, pricing.net))
    finally:
        out = stop_server(server_dir, proc)
    [path] = (runs / "server").glob("*.json")
    session = json.loads(path.read_text())
    assert f"recording session {path}" in out
    assert session["status"] == "stopped" and session["schema_version"] == 1

    first, second, call = session["jobs"]
    assert first["exit_code"] == second["exit_code"] == call["exit_code"] == 0
    assert call["script"] == "<evaluate>"
    assert first["stats"]["puts"] > 0 and first["stats"]["planned_hits"] == 0
    assert second["stats"]["planned_hits"] > 0 and second["stats"]["puts"] == 0
    assert first["added_entries"] > 0 and second["added_entries"] == 0
    assert sum(job["added_entries"] for job in session["jobs"]) == len(session["entries"])
    assert {e["job_id"] for e in session["entries"]} == {first["job_id"], call["job_id"]}
    assert all(e["label"] for e in session["entries"])
    for job in (first, second, call):  # the server exports every job's runs
        [run_id] = job["run_ids"]
        artifact = json.loads((runs / f"{run_id}.json").read_text())
        assert artifact["job_id"] == job["job_id"]

    timeline = [json.loads(line) for line in
                (runs / "server" / session["timeline"]).read_text().splitlines()]
    events = [sample["event"] for sample in timeline]
    assert events[0] == "start" and events[-1] == "stop"
    assert events.count("job_start") == events.count("job_end") == 3
    assert max(sample["memory_usage"] for sample in timeline) > 0

    log = (runs / "server" / session["log"]).read_text()
    assert "job 1 started" in log and "job 2 finished: exit 0" in log
    assert "[1] pred" in log, "job output is logged"
    assert "admit " in log and "read " in log, "cache events are logged"


def test_the_server_answers_while_a_job_runs(tmp_path, job_dir, capsys):
    server_dir, cache_dir = tmp_path / "server", tmp_path / "cache"
    write(job_dir / "wait.py", """
        import pathlib, time
        print("waiting", flush=True)
        while not pathlib.Path("go").exists():
            time.sleep(0.05)
        print("done")
    """)
    proc = start_server(server_dir, cache_dir)
    try:
        client = StratumClient(server_dir=server_dir)
        job = client.start("wait.py")
        while client.job(job)["status"] != "running":
            time.sleep(0.05)
        assert client.health()["jobs"] == 1
        assert client.stats()["jobs"] == 1
        queued = client.start("wait.py")
        assert client.job(queued)["status"] == "queued"
        (job_dir / "go").touch()
        out = io.StringIO()
        assert client.follow(job, stdout=out) == 0
        assert out.getvalue() == "waiting\ndone\n"
        # Output is kept: a second follower (or a reconnecting one) sees all of it.
        out = io.StringIO()
        assert client.follow(job, stdout=out) == 0
        assert out.getvalue() == "waiting\ndone\n"
        assert client.follow(queued, stdout=io.StringIO()) == 0
        info = client.job(job)
        assert info["status"] == "finished" and info["exit_code"] == 0

        # A client that does not share the server's file system: the uploaded source
        # runs from a temporary directory.
        reply = client._request("POST", "/v1/jobs", json={
            "script": "/nowhere/remote.py", "cwd": "/nowhere",
            "source": "import os, sys; print(os.path.basename(sys.argv[0]), sys.argv[1:])",
            "argv": ["x"]})
        out = io.StringIO()
        assert client.follow(reply["id"], stdout=out) == 0
        assert out.getvalue() == "remote.py ['x']\n"

        with pytest.raises(ServerError, match="404"):
            client.job("nope")
        info = json.loads((server_dir / SERVER_INFO).read_text())
        assert oct((server_dir / SERVER_INFO).stat().st_mode & 0o777) == "0o600"
        with pytest.raises(ServerError, match="401"):
            StratumClient(info["url"], "wrong").stats()
        assert StratumClient(info["url"], "wrong").health()["ok"], "health needs no token"
        client.close()

        # The command line.
        assert main(["--server-dir", str(server_dir), "submit", "wait.py"]) == 0
        assert main(["--server-dir", str(server_dir), "submit", "missing.py"]) == 2
        assert main(["--server-dir", str(server_dir), "stats"]) == 0
        out, err = capsys.readouterr()
        assert out.startswith("waiting\ndone\n") and '"entries"' in out
        assert "can't open file" in err
    finally:
        stop_server(server_dir, proc)
    assert not (server_dir / SERVER_INFO).exists()


def test_calls_run_dags_built_by_the_client(tmp_path, job_dir, sales_csv, pricing):
    from tests.application.pds.test_q1_pipeline import build_q1, make_lineitem
    server_dir, cache_dir = tmp_path / "server", tmp_path / "cache"
    dag = sales_dag(sales_csv, pricing.net)
    local = grid_search(dag, cv=3, scoring="r2").results_.sort("id")
    q1 = build_q1(make_lineitem())
    proc = start_server(server_dir, cache_dir)
    try:
        with StratumClient(server_dir=server_dir) as client:
            results = client.grid_search(dag, cv=3, scoring="r2")
            polars.testing.assert_frame_equal(results.sort("id"), local)
            again, predictions = client.grid_search(dag, cv=3, scoring="r2",
                                                    return_predictions=True)
            polars.testing.assert_frame_equal(again.sort("id"), local)
            assert len(predictions) > 0
            assert client.stats()["stats"]["planned_hits"] > 0, "the second call reads the first's work"

            # Without the cache: nothing read, nothing added.
            before = client.stats()
            uncached = client.grid_search(dag, cv=3, scoring="r2", config={"reuse_cache": False})
            polars.testing.assert_frame_equal(uncached.sort("id"), local)
            after = client.stats()
            assert after.pop("jobs") == before.pop("jobs") + 1 and after == before

            # A query: no X and y, so evaluate returns its table.
            pd.testing.assert_frame_equal(client.evaluate(q1), evaluate(q1))

            # Refusals: another runtime, a setting the server owns, a failing DAG.
            payload = {"function": "evaluate", "payload": "", "config": {}, "cwd": "/",
                       "runtime": {"python": "2.7"}}
            with pytest.raises(ServerError, match="409.*python 2.7"):
                client._request("POST", "/v1/calls", json=payload)
            with pytest.raises(ServerError, match="400.*reuse_cache_dir"):
                client.evaluate(q1, config={"reuse_cache_dir": str(tmp_path)})
            with skrub.config_context(eager_data_ops=False):
                failing = st.var("t", pd.DataFrame({"v": [1]})).skb.apply_func(
                    lambda df: df["missing"])
            err = io.StringIO()
            with pytest.raises(JobFailed, match="evaluate failed on the server"):
                client.evaluate(failing, stderr=err)
            assert "KeyError: 'missing'" in err.getvalue()
            job = client.start("job.py", ["1.0"])
            with pytest.raises(ServerError, match="404.*script"):
                client.result(job)
            assert client.follow(job, stdout=io.StringIO()) == 0, "the server lives on"
    finally:
        stop_server(server_dir, proc)


def test_no_server(tmp_path):
    with pytest.raises(ServerNotRunning, match="No stratum server"):
        StratumClient(server_dir=tmp_path)
    assert main(["--server-dir", str(tmp_path), "stats"]) == 2
