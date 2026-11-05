"""Guard against silently dropping tests as new trajectory suites are added."""

from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]
CI = runpy.run_path(str(ROOT / ".github/scripts/python_test_matrix.py"))
TRAJECTORIES = CI["TRAJECTORIES"]


def test_new_trajectory_gets_a_job_and_coverage_input_per_selector(tmp_path):
    (tmp_path / TRAJECTORIES).mkdir(parents=True)
    before = CI["build_matrix"](tmp_path)
    suite = tmp_path / TRAJECTORIES / "new-task"
    suite.mkdir()
    (suite / "test_replay.py").write_text("def test_replay(): pass\n")
    # Python cache directories must not turn into CI jobs on local runs.
    (tmp_path / TRAJECTORIES / "__pycache__").mkdir()

    after = CI["build_matrix"](tmp_path)
    assert after["include"][:-2] == before["include"]
    assert after["include"][-2:] == [{
        "suite": f"trajectory-new-task-{selector}",
        "os": "ubuntu-latest",
        "python-version": "3.12",
        "path": (TRAJECTORIES / "new-task").as_posix(),
        "selector": selector,
    } for selector in ("default", "greedy")]
    assert CI["coverage_files"](after) == [
        ".coverage.regular", ".coverage.trajectory-new-task-default",
        ".coverage.trajectory-new-task-greedy",
    ]
    assert len(before["include"]) == 6


@pytest.mark.parametrize("filename", ["test_orphan.py", "orphan_test.py"])
def test_tests_outside_a_trajectory_directory_fail_discovery(tmp_path, filename):
    (tmp_path / TRAJECTORIES).mkdir(parents=True)
    (tmp_path / TRAJECTORIES / filename).touch()
    with pytest.raises(ValueError, match="inside a suite directory"):
        CI["build_matrix"](tmp_path)


def test_empty_trajectory_fails_discovery(tmp_path):
    (tmp_path / TRAJECTORIES / "unfinished").mkdir(parents=True)
    with pytest.raises(ValueError, match="no test modules"):
        CI["build_matrix"](tmp_path)


def test_platform_split_keeps_every_suite_and_coverage_input():
    matrix = CI["build_matrix"](ROOT)
    linux, windows, macos = CI["split_matrix"](matrix)
    assert all(entry["os"] == "ubuntu-latest" for entry in linux["include"])
    assert {entry["os"] for entry in windows["include"]} == {"windows-latest"}
    assert {entry["os"] for entry in macos["include"]} == {"macos-26"}
    assert all(entry["suite"] == "regular" for entry in windows["include"] + macos["include"])
    combined = linux["include"] + windows["include"] + macos["include"]
    assert len(combined) == len(matrix["include"])
    assert all(combined.count(entry) == 1 for entry in matrix["include"])
    assert CI["coverage_files"](linux) == CI["coverage_files"](matrix)


@pytest.mark.parametrize("selector", ["default", "greedy"])
def test_repository_test_files_belong_to_exactly_one_suite_per_selector(selector):
    def test_files(path):
        return set(path.rglob("test_*.py")) | set(path.rglob("*_test.py"))

    all_tests = test_files(ROOT / "tests")
    covered = all_tests - test_files(ROOT / TRAJECTORIES)
    for entry in CI["build_matrix"](ROOT)["include"]:
        if entry["suite"] == "regular":
            continue
        if entry["selector"] != selector:
            continue
        suite_tests = test_files(ROOT / entry["path"])
        assert suite_tests
        assert not covered.intersection(suite_tests)
        covered.update(suite_tests)
    assert covered == all_tests
