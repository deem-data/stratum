"""pytest configuration for algebraic rewrite benchmarks."""


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
        default=True,
        help="Render before/after DAG PNGs via graphviz.",
    )


def pytest_configure(config):
    import stratum.tests.logical_optimizer.algebraic_rewrites.benchmark_numeric as bn

    bn.BENCH_ROWS = config.getoption("--bench-rows", default=100_000)
    bn.BENCH_VIZ = config.getoption("--bench-viz", default=True)
