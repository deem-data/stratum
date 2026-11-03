"""Run trajectories with both selectors locally, or one per CI job.

Replay fixtures must depend on ``trajectory_selector`` so each selector gets a
fresh replay and its own cache.
"""

import pytest

from stratum import config


def pytest_addoption(parser):
    parser.addoption(
        "--trajectory-selector", choices=("default", "greedy"),
        help="Run only this implementation selector (default: run both).",
    )


def pytest_generate_tests(metafunc):
    if "trajectory_selector" in metafunc.fixturenames:
        selector = metafunc.config.getoption("--trajectory-selector")
        metafunc.parametrize(
            "trajectory_selector", [selector] if selector else ["default", "greedy"],
            indirect=True, scope="module",
        )


@pytest.fixture(scope="module", autouse=True)
def trajectory_selector(request):
    with config(implementation_selector=request.param):
        yield request.param
