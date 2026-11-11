"""Stratum as a server: scripts and DAGs run in one process and share its reuse cache."""
import base64
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
from stratum.frontend._api import evaluate, grid_search
from stratum._config import FLAGS
from stratum.server._client import (SERVER_INFO, JobFailed, ServerError, ServerNotRunning,
                                    StratumClient, dump_call, runtime)
from stratum.server._server import StratumServer, run_call, run_script

REPO = Path(__file__).resolve().parents[2]

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
    print("reads", active_reuse_cache().stats.hits)
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

    err = io.StringIO()
    assert run_script(write(tmp_path / "exit.py", "raise SystemExit('bad input')"),
                      stderr=err) == 1
    assert err.getvalue() == "bad input\n"


@pytest.mark.parametrize("ending, exit_code", [("", 0), ("raise RuntimeError('job failed')", 1)])
@pytest.mark.parametrize("inline", [False, True], ids=["file", "source"])
def test_run_script_restores_process_state(tmp_path, ending, exit_code, inline):
    write(tmp_path / "helper_mod.py", "VALUE = 1\n")
    script = write(tmp_path / "s.py", f"""
        import os, sys, stratum as st, skrub
        import helper_mod
        print(helper_mod.VALUE)
        st.set_config(stats=True, reuse_cache=False)
        skrub.set_config(eager_data_ops=False)
        os.environ["STRATUM_TEST_VAR"] = "1"
        sys.argv.append("x")
        {ending}
    """)
    import skrub
    before = (dict(vars(FLAGS)), skrub.get_config(), dict(os.environ), list(sys.argv),
              os.getcwd(), sys.modules["__main__"])
    out = io.StringIO()
    source = script.read_text() if inline else None
    assert run_script(script, cwd=str(tmp_path), stdout=out, source=source) == exit_code
    after = (dict(vars(FLAGS)), skrub.get_config(), dict(os.environ), list(sys.argv),
             os.getcwd(), sys.modules["__main__"])
    assert after == before
    assert "helper_mod" not in sys.modules, "user modules are dropped after a job"
    # A different size: the bytecode cache checks size and whole-second mtime.
    write(tmp_path / "helper_mod.py", "VALUE = 22\n")
    assert run_script(script, cwd=str(tmp_path), stdout=out, source=source) == exit_code
    assert out.getvalue().split() == ["1", "22"], "an edited helper is re-imported"


def test_inline_source_runs_in_its_own_main_module(tmp_path):
    source = textwrap.dedent("""
        import __main__, pickle
        from dataclasses import dataclass

        @dataclass
        class Record:
            value: int

        assert __main__.Record is Record
        assert Record.__annotations__['value'] is int
        assert __name__ == '__main__'
        assert __file__ == '<string>'
        print(pickle.loads(pickle.dumps(Record(7))).value)
    """)
    out, err = io.StringIO(), io.StringIO()
    assert run_script(None, cwd=str(tmp_path), stdout=out, stderr=err, source=source) == 0
    assert out.getvalue() == "7\n"
    assert err.getvalue() == ""
    assert run_script(None, cwd=str(tmp_path), stderr=err,
                      source="assert 'Record' not in globals()") == 0
    assert not list(tmp_path.iterdir()), "inline code needs no script file"


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

def start_server(server_dir: Path, *serve_args: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "stratum", "--server-dir", str(server_dir), "serve",
         *serve_args],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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


def run(server_dir, *argv, inline=False):
    out, err = io.StringIO(), io.StringIO()
    with StratumClient(server_dir=server_dir) as client:
        script = Path("job.py")
        code = client.submit(script.read_text() if inline else script, list(argv),
                             stdout=out, stderr=err)
    assert code == 0, err.getvalue()
    lines = dict(line.split(" ", 1) for line in out.getvalue().splitlines())
    return float(lines["pred"]), int(lines["reads"])


def test_submit_accepts_source_strings_and_bytes(tmp_path):
    server_dir = tmp_path / "server"
    # A string is always source, even when a file has the same name.
    write(tmp_path / "pass", "raise RuntimeError('a string was mistaken for a path')")
    (tmp_path / "data.txt").write_text("local data", encoding="utf-8")
    proc = start_server(server_dir)
    try:
        with StratumClient(server_dir=server_dir) as client:
            for source, expected in [
                ("", ""),
                (b"", ""),
                ("pass", ""),
                ("message = 'hello'\nprint(message)", "hello\n"),
                ("# coding: latin-1\nprint('café ☕')", "café ☕\n"),
                ("print('café ☕')".encode("utf-8"), "café ☕\n"),
                (b"\xef\xbb\xbfprint('BOM')", "BOM\n"),
                (b"# coding: latin-1\r\nprint('caf\xe9')\r\n", "café\n"),
            ]:
                out, err = io.StringIO(), io.StringIO()
                assert client.submit(source, cwd=tmp_path, stdout=out, stderr=err) == 0, err.getvalue()
                assert out.getvalue() == expected
                assert err.getvalue() == ""

            # start() accepts source too, with the same arguments, working
            # directory, output streams and exit-code handling as file jobs.
            source = ("import pathlib, sys\n"
                      "print(sys.argv)\n"
                      "print(pathlib.Path('data.txt').read_text())\n"
                      "print('stderr', file=sys.stderr)\n"
                      "raise SystemExit(7)")
            job = client.start(source, ["a b", "--alpha", "0.1"], cwd=tmp_path)
            out, err = io.StringIO(), io.StringIO()
            assert client.follow(job, stdout=out, stderr=err) == 7
            assert out.getvalue() == "['<string>', 'a b', '--alpha', '0.1']\nlocal data\n"
            assert err.getvalue() == "stderr\n"
            assert client.job(job)["script"] == "<string>"

            for source, error in [("1 / 0", "ZeroDivisionError"),
                                  (b"if:", "SyntaxError"),
                                  ("\x00", "SyntaxError")]:
                err = io.StringIO()
                assert client.submit(source, cwd=tmp_path, stderr=err) == 1
                assert error in err.getvalue()

            # A remote client's cwd need not exist on the server.
            out = io.StringIO()
            assert client.submit("import os; print(os.path.isdir(os.getcwd()))",
                                 cwd=tmp_path / "client-only", stdout=out) == 0
            assert out.getvalue() == "True\n", "the server survives failed source jobs"
    finally:
        stop_server(server_dir, proc)


@pytest.mark.parametrize("source, error", [
    (42, TypeError),
    (None, TypeError),
    (b"# coding: unknown-encoding\npass", SyntaxError),
    (b"print('\xff')", SyntaxError),
    (b"\n\nprint('\xff')", UnicodeDecodeError),
])
def test_invalid_source_is_rejected_before_submission(source, error):
    with StratumClient("http://unused", check=False) as client:
        with pytest.raises(error):
            client.start(source)


@pytest.mark.parametrize("inline", [False, True], ids=["file", "source"])
def test_submitted_scripts_share_the_cache(tmp_path, job_dir, capsys, inline):
    server_dir = tmp_path / "server"
    proc = start_server(server_dir)
    try:
        pred, reads = run(server_dir, "1.0", inline=inline)
        assert reads == 0
        again, reads = run(server_dir, "1.0", inline=inline)
        assert again == pred and reads > 0, "the second submission reads the first's work"
        # A different size: the bytecode cache checks size and whole-second mtime.
        write(job_dir / "helper.py", "OFFSET = 3  # edited\n")
        edited, _ = run(server_dir, "1.0", inline=inline)
        assert edited != pred, "an edited helper module is picked up"
        with StratumClient(server_dir=server_dir) as client:
            with pytest.raises(FileNotFoundError):
                client.start(Path("missing.py"))
            stats = client.stats()
        assert stats["jobs"] == 3 and stats["entries"] > 0
        assert main(["--server-dir", str(server_dir), "clear"]) == 0
        out, err = capsys.readouterr()
        assert err == ""
        cleared = json.loads(out)
        assert cleared["entries"] == cleared["memory_usage"] == 0
        after_clear, reads = run(server_dir, "1.0", inline=inline)
        assert after_clear == edited and reads == stats["stats"]["hits"]
    finally:
        log = stop_server(server_dir, proc)
    assert "stratum server stopped" in log

    # A new process starts cold: cache contents live only in memory.
    write(job_dir / "helper.py", "OFFSET = 2\n")
    proc = start_server(server_dir)
    try:
        restarted, reads = run(server_dir, "1.0", inline=inline)
        assert restarted == pred and reads == 0
    finally:
        stop_server(server_dir, proc)


def test_a_server_records_its_session(tmp_path, job_dir, sales_csv, pricing):
    server_dir, runs = tmp_path / "server", tmp_path / "runs"
    proc = start_server(server_dir, "--diagnostics-dir", str(runs))
    try:
        run(server_dir, "1.0", inline=True)
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
    assert first["script"] == "<string>"
    assert call["script"] == "<evaluate>"
    assert first["stats"]["puts"] > 0 and first["stats"]["hits"] == 0
    assert second["stats"]["hits"] > 0 and second["stats"]["puts"] == 0
    assert session["cache"]["stats_scope"] == "cache_lifetime"
    assert session["cache"]["entries"] > 0
    entries = session["entries"]
    assert len(entries) == session["cache"]["entries"]
    assert sum(entry["size"] for entry in entries) == session["cache"]["memory_usage"]
    assert all(entry["label"] and entry["job_id"] in (first["job_id"], call["job_id"])
               for entry in entries)
    assert any(entry["hits"] > 0 for entry in entries)
    assert first["added_entries"] > 0 and second["added_entries"] == 0
    for job in (first, second, call):  # the server exports every job's runs
        [run_id] = job["run_ids"]
        artifact = json.loads((runs / f"{run_id}.json").read_text())
        assert artifact["job_id"] == job["job_id"]
        assert artifact["reuse_cache"]["stats_scope"] == "run_interval"
        assert artifact["reuse_cache"]["stats"] == job["stats"]

    timeline = [json.loads(line) for line in
                (runs / "server" / session["timeline"]).read_text().splitlines()]
    events = [sample["event"] for sample in timeline]
    assert events[0] == "start" and events[-1] == "stop"
    assert events.count("job_start") == events.count("job_end") == 3
    assert max(sample["memory_usage"] for sample in timeline) > 0

    log = (runs / "server" / session["log"]).read_text()
    assert "job 1 started" in log and "job 2 finished: exit 0" in log
    assert "[1] pred" in log, "job output is logged"
    assert "inserted" in log and "hits" in log, "job cache statistics are logged"


def test_the_server_answers_while_a_job_runs(tmp_path, job_dir, capsys):
    server_dir = tmp_path / "server"
    write(job_dir / "wait.py", """
        import pathlib, time
        import stratum as st
        st.set_config(reuse_cache=False)
        print("waiting", flush=True)
        pathlib.Path("ready").touch()
        while not pathlib.Path("go").exists():
            time.sleep(0.05)
        print("done")
    """)
    proc = start_server(server_dir)
    try:
        client = StratumClient(server_dir=server_dir)
        job = client.start(Path("wait.py"))
        deadline = time.monotonic() + 20
        while not (job_dir / "ready").exists():
            assert time.monotonic() < deadline, "job did not become ready"
            time.sleep(0.05)
        assert client.health()["jobs"] == 1
        assert client.stats()["jobs"] == 1
        frame = pd.DataFrame({"v": [1, 2]})
        queued = client.start_call("evaluate", query(frame), cwd=tmp_path / "client-only")
        assert client.job(queued)["status"] == "queued"
        with pytest.raises(ServerError, match="409.*queued"):
            client.result(queued)
        (job_dir / "go").touch()
        out = io.StringIO()
        assert client.follow(job, stdout=out) == 0
        assert out.getvalue() == "waiting\ndone\n"
        # Output is kept: a second follower (or a reconnecting one) sees all of it.
        out = io.StringIO()
        assert client.follow(job, stdout=out) == 0
        assert out.getvalue() == "waiting\ndone\n"
        assert client.follow(queued, stdout=io.StringIO()) == 0
        pd.testing.assert_frame_equal(client.result(queued), frame.assign(w=[10, 20]))
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
        # Windows stat modes describe the read-only attribute, not POSIX
        # owner/group/other permissions. Keep the token/auth checks on every OS.
        if os.name == "posix":
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
        (job_dir / "go").touch()
        stop_server(server_dir, proc)
    assert not (server_dir / SERVER_INFO).exists()


def test_server_memory_budget_rejects_oversized_results(tmp_path, job_dir):
    server_dir = tmp_path / "server"
    proc = start_server(server_dir, "--memory-fraction", "1e-20")
    try:
        first, reads = run(server_dir, "1.0")
        again, more_reads = run(server_dir, "1.0")
        assert first == again and reads == more_reads == 0
        with StratumClient(server_dir=server_dir) as client:
            stats = client.stats()
        assert stats["memory_budget"] == 1
        assert stats["entries"] == stats["memory_usage"] == 0
        assert stats["stats"]["puts"] == 0 and stats["stats"]["rejected"] > 0
    finally:
        stop_server(server_dir, proc)


def test_calls_run_dags_built_by_the_client(tmp_path, job_dir, sales_csv, pricing):
    from tests.applications.pds.test_q1_pipeline import build_q1, make_lineitem
    server_dir = tmp_path / "server"
    dag = sales_dag(sales_csv, pricing.net)
    local = grid_search(dag, cv=3, scoring="r2").results_.sort("id")
    q1 = build_q1(make_lineitem())
    proc = start_server(server_dir)
    try:
        with StratumClient(server_dir=server_dir) as client:
            results = client.grid_search(dag, cv=3, scoring="r2")
            polars.testing.assert_frame_equal(results.sort("id"), local)
            again, predictions = client.grid_search(dag, cv=3, scoring="r2",
                                                    return_predictions=True)
            polars.testing.assert_frame_equal(again.sort("id"), local)
            assert len(predictions) > 0
            assert client.stats()["stats"]["hits"] > 0, "the second call reads the first's work"

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
            with pytest.raises(ServerError, match="400.*reuse_cache_memory_fraction"):
                client.evaluate(q1, config={"reuse_cache_memory_fraction": 0.5})
            with skrub.config_context(eager_data_ops=False):
                failing = st.var("t", pd.DataFrame({"v": [1]})).skb.apply_func(
                    lambda df: df["missing"])
            err = io.StringIO()
            with pytest.raises(JobFailed, match="evaluate failed on the server"):
                client.evaluate(failing, stderr=err)
            assert "KeyError: 'missing'" in err.getvalue()
            job = client.start(Path("job.py"), ["1.0"])
            with pytest.raises(ServerError, match="404.*script"):
                client.result(job)
            assert client.follow(job, stdout=io.StringIO()) == 0, "the server lives on"
    finally:
        stop_server(server_dir, proc)


@pytest.mark.parametrize("payload, error", [
    ("a", "base64"),
    (base64.b64encode(b"not a pickle").decode("ascii"), "UnpicklingError"),
])
def test_bad_call_payload_fails_the_job_and_the_server_keeps_running(tmp_path, payload, error):
    server_dir = tmp_path / "server"
    proc = start_server(server_dir)
    try:
        with StratumClient(server_dir=server_dir) as client:
            job = client._request("POST", "/v1/calls", json={
                "function": "evaluate", "payload": payload, "config": {},
                "runtime": runtime(), "cwd": str(tmp_path),
            })["id"]
            err = io.StringIO()
            assert client.follow(job, stderr=err) == 1
            assert error in err.getvalue()
            assert client.job(job)["status"] == "finished"
            with pytest.raises(ServerError, match="409.*failed with exit code 1"):
                client.result(job)

            frame = pd.DataFrame({"v": [1, 2]})
            pd.testing.assert_frame_equal(client.evaluate(query(frame)), frame.assign(w=[10, 20]))
    finally:
        stop_server(server_dir, proc)


def test_a_second_server_cannot_replace_a_running_server(tmp_path):
    server_dir = tmp_path / "server"
    proc = start_server(server_dir)
    try:
        info = (server_dir / SERVER_INFO).read_text()
        with pytest.raises(RuntimeError, match="already running"):
            StratumServer(server_dir).serve_forever()
        assert (server_dir / SERVER_INFO).read_text() == info
        with StratumClient(server_dir=server_dir) as client:
            assert client.health()["ok"]
    finally:
        stop_server(server_dir, proc)


def test_cli_shutdown_finishes_queued_jobs(tmp_path, capsys):
    server_dir = tmp_path / "server"
    waiting = write(tmp_path / "wait.py", """
        import pathlib, time
        pathlib.Path("ready").touch()
        while not pathlib.Path("go").exists():
            time.sleep(0.05)
        pathlib.Path("completed").write_text("first\\n")
    """)
    second = write(tmp_path / "second.py", """
        import pathlib
        with pathlib.Path("completed").open("a") as f:
            f.write("second\\n")
    """)
    proc = start_server(server_dir)
    try:
        with StratumClient(server_dir=server_dir) as client:
            client.start(waiting, cwd=tmp_path)
            deadline = time.monotonic() + 20
            while not (tmp_path / "ready").exists():
                assert time.monotonic() < deadline, "job did not start"
                time.sleep(0.05)
            queued = client.start(second, cwd=tmp_path)
            assert client.job(queued)["status"] == "queued"

        assert main(["--server-dir", str(server_dir), "shutdown"]) == 0
        assert capsys.readouterr() == ("", "")
        assert proc.poll() is None, "shutdown must wait for the running job"
        (tmp_path / "go").touch()
        out, _ = proc.communicate(timeout=30)
        assert proc.returncode == 0, out
        assert (tmp_path / "completed").read_text().splitlines() == ["first", "second"]
        assert not (server_dir / SERVER_INFO).exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=10)


def test_no_server(tmp_path, capsys):
    with pytest.raises(ServerNotRunning, match="No stratum server"):
        StratumClient(server_dir=tmp_path)
    assert main(["--server-dir", str(tmp_path), "stats"]) == 2
    out, err = capsys.readouterr()
    assert out == "" and "No stratum server" in err
