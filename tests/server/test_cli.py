"""CLI argument handling, exit codes, and interruption at the client boundary."""
from unittest.mock import MagicMock, Mock

import pytest

import stratum as st
from stratum._config import FLAGS
from stratum.server import _cli, _server
from stratum.server._client import ServerError, ServerNotRunning, StratumClient


@pytest.fixture
def client(monkeypatch):
    client = MagicMock(spec=StratumClient)
    client.__enter__.return_value = client
    monkeypatch.setattr(_cli, "StratumClient", lambda *args, **kwargs: client)
    return client


def test_submit_preserves_script_arguments_and_exit_code(client):
    client.start.return_value = "job-1"
    client.follow.return_value = 7

    assert _cli.main(["submit", "train.py", "--alpha", "0.1", "a b"]) == 7
    client.start.assert_called_once_with("train.py", ["--alpha", "0.1", "a b"])
    client.follow.assert_called_once_with("job-1")
    client.__exit__.assert_called_once()


def test_interrupting_submit_leaves_the_job_running(client, capsys):
    client.start.return_value = "job-1"
    client.follow.side_effect = KeyboardInterrupt

    assert _cli.main(["submit", "train.py"]) == 130
    out, err = capsys.readouterr()
    assert out == ""
    assert "job job-1 keeps running on the server" in err
    client.shutdown.assert_not_called()
    client.__exit__.assert_called_once()


@pytest.mark.parametrize("error, exit_code", [
    (ServerNotRunning("connection lost"), 2),
    (ServerError("401 Unauthorized: missing or wrong token"), 1),
])
def test_request_errors_go_to_stderr(client, capsys, error, exit_code):
    client.stats.side_effect = error

    assert _cli.main(["stats"]) == exit_code
    assert capsys.readouterr() == ("", f"{error}\n")
    client.__exit__.assert_called_once()


def test_serve_applies_options_and_exits_cleanly_on_interrupt(tmp_path, monkeypatch):
    server = Mock(spec=_server.StratumServer)
    server.serve_forever.side_effect = KeyboardInterrupt
    factory = Mock(return_value=server)
    monkeypatch.setattr(_server, "StratumServer", factory)
    server_dir, runs = str(tmp_path / "server"), str(tmp_path / "runs")

    with st.config(reuse_cache=False):
        assert _cli.main([
            "--server-dir", server_dir, "serve", "--host", "0.0.0.0", "--port", "8787",
            "--memory-fraction", "0.25", "--diagnostics-dir", runs, "--log-level", "DEBUG",
        ]) == 0
        assert FLAGS.reuse_cache is True
        assert FLAGS.reuse_cache_memory_fraction == 0.25

    factory.assert_called_once_with(server_dir, diagnostics_dir=runs,
                                    log_level="DEBUG", host="0.0.0.0", port=8787)
    server.serve_forever.assert_called_once()
