"""Register test options for full-suite and targeted pytest runs."""


def pytest_addoption(parser):
    parser.addoption(
        "--trajectories", action="store_true", default=False,
        help="Run the slow NYC and forest-cover trajectory tests.",
    )
    parser.addoption(
        "--selector", choices=("default", "greedy"),
        default="default", help="Implementation selector for the replay (default: default).",
    )
