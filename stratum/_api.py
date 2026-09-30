import pandas as pd
from dataclasses import dataclass
from skrub import DataOp

from stratum._config import FLAGS
from stratum.optimizer._optimize import SearchConfig, optimize
from stratum.optimizer.logical._scoring import resolve_scoring
from stratum.runtime._scheduler import SequentialScheduler
from stratum.frontend._skrub_graph import get_data
from time import perf_counter


@dataclass(frozen=True)
class RunTimings:
    """Wall-clock time from optimizer entry through scheduler completion."""

    total: float
    optimization: float
    execution: float

#TODO: Rename this file
def grid_search(dag: DataOp, cv=None, scoring=None, return_predictions=False, env=None):
    """Perform grid search with cross-validation on a DataOp DAG. ``scoring`` is required."""
    if scoring is None:
        # A search ranks a set, so every candidate has to be measured the same way.
        # `.skb.make_grid_search()` still honours scoring=None, by handing the call to
        # skrub. See docs/adr/0004-a-search-always-names-its-metric.md.
        raise ValueError(
            "grid_search requires scoring=. Without it each candidate would be scored by"
            " its own estimator's `score`, so a batch mixing estimator kinds would rank"
            " an accuracy against an R². Pass a string naming an sklearn metric, a scorer"
            " built with `make_scorer`, or a callable `scorer(estimator, X, y)`."
        )
    t0 = perf_counter()
    #FIXME: Measure operator execution only if stats is enabled
    env_extra = env if env else {}
    env = get_data(dag)
    for k, v in env_extra.items():
        env[k] = v
    # The scorer is plan-time state: it decides what the plan's last operator computes,
    # and an unusable `scoring=` fails here rather than after a fold has been fitted.
    search = SearchConfig(metric=resolve_scoring(scoring),
                          return_predictions=return_predictions)
    # Resolve variables to constants at compile time, so the scheduler runs
    # without an environment.
    optimize_start = perf_counter()
    linearized_dag, split_pos, flagged_ops = optimize(dag, env=env, search=search)
    optimize_end = perf_counter()
    sched = SequentialScheduler(linearized_dag, split_pos, flagged_ops, FLAGS.stats, t0=t0)

    # Without an explicit `cv`, the folds come from the splitter the plan declares with
    # `mark_as_X(cv=...)`, which the plan computes itself (see BuildSplitterOp).
    preds = sched.grid_search(cv)
    execution_end = perf_counter()

    stats_printer(sched, RunTimings(
        total=execution_end - optimize_start,
        optimization=optimize_end - optimize_start,
        execution=execution_end - optimize_end,
    ))

    return (sched,preds) if return_predictions else sched


def evaluate(dag: DataOp, seed: int = 42, test_size = 0.2):
    """Evaluate a DataOp DAG with train/test split."""
    t0 = perf_counter()
    # Resolve variables to constants at compile time, so the scheduler runs
    # without an environment.
    env = get_data(dag)
    optimize_start = perf_counter()
    linearized_dag, split_pos, flagged_ops = optimize(dag, env=env)
    optimize_end = perf_counter()
    sched = SequentialScheduler(linearized_dag, split_pos, flagged_ops, FLAGS.stats, t0=t0)
    out = sched.evaluate(seed, test_size)
    execution_end = perf_counter()
    stats_printer(sched, RunTimings(
        total=execution_end - optimize_start,
        optimization=optimize_end - optimize_start,
        execution=execution_end - optimize_end,
    ))
    return out


def stats_printer(sched: SequentialScheduler, run: RunTimings):
    if FLAGS.stats:
        table = pd.DataFrame(sched.timings, columns=["Op", "time"])
        table = table.groupby("Op").aggregate(["sum", "count"])
        table.columns = ["Time", "Count"]
        table = table.reset_index().sort_values(by="Time", ascending=False)
        operator_time = table["Time"].sum()
        table["%"] = 100 * table["Time"] / operator_time if operator_time else 0.0
        table = table[["Op", "Count", "Time", "%"]]
        shown = table.head(FLAGS.stats_top_k)
        print("\n" + "=" * 80)
        print("Execution Statistics (seconds):")
        print(f"  Total:        {run.total:.4f}")
        print(f"  Optimization: {run.optimization:.4f}")
        print(f"  Execution:    {run.execution:.4f}")
        print("\nHeavy hitters (share of all operator processing time):\n")
        print(shown.to_string(
            index=False,
            formatters={"Time": "{:.4f}".format, "%": "{:.1f}%".format},
        ))
        print(f"Unshown operators: {operator_time - shown['Time'].sum():.4f}")
        print("=" * 80)
        print("BufferPool detail (serialize/deserialize times are included in Execution):")
        print(sched.pool.stats)
        print("=" * 80 + "\n")
