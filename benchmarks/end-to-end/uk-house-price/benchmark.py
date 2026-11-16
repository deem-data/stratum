"""Run the two UK housing searches and write their runtimes in seconds."""
import argparse
import csv
import multiprocessing
import os
from pathlib import Path
from time import perf_counter

MODES = ("skrub", "stratum-default", "stratum-greedy", "stratum-reuse")


def run_mode(mode, data_path, output, threads):
    import skrub
    from sklearn.model_selection import KFold
    from threadpoolctl import threadpool_limits
    from pipeline import FOLDS, SEED
    from two_level_gridsearch_exp import exploration
    from two_level_gridsearch_exp2 import refinement

    if mode != "skrub":
        import stratum as st
        st.set_config(scheduler=True, num_threads=threads, stats=False,
                      implementation_selector="default" if mode == "stratum-default" else "greedy",
                      reuse_cache=mode == "stratum-reuse")

    timings = []
    # Both searches share a process so refinement can reuse exploration results.
    with skrub.config_context(eager_data_ops=False), threadpool_limits(limits=threads):
        for stage, build in (("exploration", exploration), ("refinement", refinement)):
            start = perf_counter()
            dag = build(data_path, threads)
            search = dag.skb.make_grid_search(
                cv=KFold(FOLDS, shuffle=True, random_state=SEED), scoring="r2",
                n_jobs=1, fitted=True, refit=False, error_score="raise")
            timings.append(perf_counter() - start)
            print(f"{mode} {stage}: {timings[-1]:.3f}s", flush=True)
            del search, dag

    total = sum(timings)
    with output.open("a", newline="") as stream:
        csv.writer(stream).writerow([mode, threads, *(f"{seconds:.3f}" for seconds in (*timings, total))])
    print(f"{mode} total: {total:.3f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Input CSV")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES,
                        help="Configurations to run (default: all)")
    parser.add_argument("--threads", type=int, default=8, help="Threads per library (default: 8)")
    parser.add_argument("--output", type=Path, help="Timing CSV (default: beside the input)")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    data_path = args.data.resolve(strict=True)
    output = args.output or data_path.with_suffix(".timings.csv")

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS", "SKRUB_RUST_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ["PYTHONHASHSEED"] = "42"

    with output.open("w", newline="") as stream:
        csv.writer(stream).writerow(["mode", "threads", "exploration_seconds", "refinement_seconds", "total_seconds"])
    # A fresh process isolates each configuration and starts with an empty cache.
    for mode in args.modes:
        worker = multiprocessing.get_context("spawn").Process(
            target=run_mode, args=(mode, data_path, output, args.threads))
        worker.start()
        worker.join()
        if worker.exitcode:
            raise SystemExit(f"{mode} failed (exit code {worker.exitcode})")
    print(f"Runtimes: {output}")


if __name__ == "__main__":
    main()
