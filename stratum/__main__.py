"""Command line: ``python -m stratum {serve,submit,stats,clear,shutdown}``.

    python -m stratum serve [--host H] [--port P] [--cache-dir DIR] [--diagnostics-dir DIR] ...
    python -m stratum submit script.py [script args ...]
    python -m stratum stats

The client commands find the local server under ``--server-dir``, or a remote one
at ``--url`` (``$STRATUM_URL``) with ``$STRATUM_TOKEN``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from stratum._client import DEFAULT_SERVER_DIR, ServerError, StratumClient
from stratum._config import (REUSE_CACHE_ADMISSION_POLICIES, REUSE_CACHE_EVICTION_POLICIES,
                             set_config)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m stratum")
    parser.add_argument("--server-dir", default=str(DEFAULT_SERVER_DIR),
                        help="where the local server publishes its address and token "
                             "(default: %(default)s, or $STRATUM_SERVER_DIR)")
    parser.add_argument("--url", default=os.getenv("STRATUM_URL"),
                        help="a server's URL, instead of the local one (default: "
                             "$STRATUM_URL); its token is read from $STRATUM_TOKEN")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run the server in the foreground")
    serve.add_argument("--host", default="127.0.0.1",
                       help="interface to listen on (default: %(default)s; 0.0.0.0 for "
                            "every interface). Clients need the token, $STRATUM_TOKEN or "
                            "the one written under --server-dir")
    serve.add_argument("--port", type=int, default=0,
                       help="port to listen on (default: a free one)")
    serve.add_argument("--cache-dir", help="reuse cache directory")
    serve.add_argument("--memory-fraction", type=float,
                       help="share of host memory for the reuse cache's memory tier")
    serve.add_argument("--disk-budget", type=int, help="bytes the reuse cache may keep on disk")
    serve.add_argument("--eviction", choices=REUSE_CACHE_EVICTION_POLICIES)
    serve.add_argument("--admission", choices=REUSE_CACHE_ADMISSION_POLICIES)
    serve.add_argument("--diagnostics-dir",
                       help="record the session (jobs, cache entries and size over time, "
                            "logs) under DIR/server/ and the jobs' runs under DIR, "
                            "for the inspector")
    serve.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
                       help="level of the session log (default: %(default)s)")

    run = commands.add_parser("submit", help="run a script on the server")
    run.add_argument("script")
    run.add_argument("args", nargs=argparse.REMAINDER, help="arguments for the script")

    commands.add_parser("stats", help="print the server's reuse cache statistics")
    commands.add_parser("clear", help="delete every reuse cache entry")
    commands.add_parser("shutdown", help="flush the reuse cache and stop the server")

    args = parser.parse_args(argv)
    if args.command == "serve":
        from stratum._server import StratumServer

        set_config(reuse_cache=True, reuse_cache_dir=args.cache_dir,
                   reuse_cache_memory_fraction=args.memory_fraction,
                   reuse_cache_disk_budget=args.disk_budget,
                   reuse_cache_eviction=args.eviction,
                   reuse_cache_admission=args.admission)
        try:
            StratumServer(args.server_dir, diagnostics_dir=args.diagnostics_dir,
                          log_level=args.log_level, host=args.host, port=args.port
                          ).serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
    try:
        with StratumClient(args.url, server_dir=args.server_dir) as client:
            if args.command == "submit":
                return _submit(client, args.script, args.args)
            if args.command == "shutdown":
                client.shutdown()
                return 0
            reply = client.stats() if args.command == "stats" else client.clear()
    except ConnectionError as exc:
        print(exc, file=sys.stderr)
        return 2
    except ServerError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(json.dumps(reply, indent=2))
    return 0


def _submit(client: StratumClient, script: str, argv: list[str]) -> int:
    try:
        job = client.start(script, argv)
    except FileNotFoundError as exc:  # as `python missing.py` reports it
        print(f"can't open file: {exc}", file=sys.stderr)
        return 2
    try:
        return client.follow(job)
    except KeyboardInterrupt:
        print(f"\njob {job} keeps running on the server", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
