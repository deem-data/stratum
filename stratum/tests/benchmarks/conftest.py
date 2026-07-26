"""pytest configuration for the benchmark suite."""

# Stored by pytest_configure so test code can read back options.
_pytest_config = None


def pytest_addoption(parser):
    parser.addoption(
        "--bench-rows",
        type=int,
        default=100_000,
        help="Number of rows in benchmark DataFrames (default: 100_000).",
    )
    parser.addoption(
        "--bench-viz",
        action="store_true",
        default=False,
        help="Render before/after DAG PNGs via graphviz.",
    )
    parser.addoption(
        "--bench-json",
        type=str,
        default=None,
        help="Write the full benchmark report as JSON to this path.",
    )


def pytest_configure(config):
    global _pytest_config
    _pytest_config = config
