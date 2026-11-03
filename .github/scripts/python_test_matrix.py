"""Discover whole trajectory suites without importing their ML dependencies."""

import json
import os
from pathlib import Path
import re


TRAJECTORIES = Path("tests/application/trajectories")
ENVIRONMENTS = [
    ("ubuntu-latest", "3.11"),
    ("ubuntu-latest", "3.12"),
    ("ubuntu-latest", "3.13"),
    ("ubuntu-latest", "3.14"),
    ("macos-26", "3.12"),
    ("windows-latest", "3.12"),
]


def build_matrix(root: Path) -> dict:
    entries = [
        {"os": os_name, "python-version": version, "suite": "regular", "path": "tests"}
        for os_name, version in ENVIRONMENTS
    ]
    for path in sorted((root / TRAJECTORIES).iterdir()):
        if path.name.startswith((".", "__")):
            continue
        if path.is_file():
            if path.match("test_*.py") or path.match("*_test.py"):
                raise ValueError(f"Trajectory tests must be inside a suite directory: {path}")
            continue
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", path.name):
            raise ValueError(f"Invalid trajectory name: {path.name}")
        if not any(path.rglob("test_*.py")) and not any(path.rglob("*_test.py")):
            raise ValueError(f"Trajectory has no test modules: {path}")
        for selector in ("default", "greedy"):
            entries.append({
                "os": "ubuntu-latest",
                "python-version": "3.12",
                "suite": f"trajectory-{path.name}-{selector}",
                "path": path.relative_to(root).as_posix(),
                "selector": selector,
            })
    return {"include": entries}


def coverage_files(matrix: dict) -> list[str]:
    return [
        f".coverage.{entry['suite']}"
        for entry in matrix["include"]
        if entry["os"] == "ubuntu-latest" and entry["python-version"] == "3.12"
    ]


def split_matrix(matrix: dict) -> tuple[dict, dict]:
    """Only Linux jobs wait for the shared Linux extension."""
    linux = [entry for entry in matrix["include"] if entry["os"] == "ubuntu-latest"]
    platforms = [entry for entry in matrix["include"] if entry["os"] != "ubuntu-latest"]
    return {"include": linux}, {"include": platforms}


if __name__ == "__main__":
    matrix = build_matrix(Path(__file__).resolve().parents[2])
    linux, platforms = split_matrix(matrix)
    print(json.dumps(matrix, indent=2))
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(f"linux-matrix={json.dumps(linux)}\n")
            stream.write(f"platform-matrix={json.dumps(platforms)}\n")
            stream.write(f"coverage-files={json.dumps(coverage_files(matrix))}\n")
