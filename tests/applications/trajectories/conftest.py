"""Opt into trajectories with the default or greedy selector.

Replay fixtures must depend on ``trajectory_selector`` so each selector gets a
fresh replay and its own cache.
"""

import pytest

from stratum import config


@pytest.fixture(scope="session", autouse=True)
def require_trajectories(request):
    if not request.config.getoption("--trajectories"):
        pytest.skip("use --trajectories to run trajectory tests")


def pytest_generate_tests(metafunc):
    if "trajectory_selector" in metafunc.fixturenames:
        selector = metafunc.config.getoption("--selector")
        metafunc.parametrize(
            "trajectory_selector", [selector],
            indirect=True, scope="module",
        )


@pytest.fixture(scope="module", autouse=True)
def trajectory_selector(request):
    with config(implementation_selector=request.param):
        yield request.param
