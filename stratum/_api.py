import pandas as pd
from skrub import DataOp

from stratum._config import FLAGS
from stratum.optimizer._optimize import SearchConfig, optimize
from stratum.optimizer.logical._scoring import resolve_scoring
from stratum.runtime._scheduler import SequentialScheduler
from stratum.frontend._skrub_graph import get_data
from time import perf_counter

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
    linearized_dag, split_pos, flagged_ops = optimize(dag, env=env, search=search)
    sched = SequentialScheduler(linearized_dag, split_pos, flagged_ops, FLAGS.stats, t0=t0)

    # Without an explicit `cv`, the folds come from the splitter the plan declares with
    # `mark_as_X(cv=...)`, which the plan computes itself (see BuildSplitterOp).
    preds = sched.grid_search(cv)

    stats_printer(sched)

    return (sched,preds) if return_predictions else sched


def evaluate(dag: DataOp, seed: int = 42, test_size = 0.2):
    """Evaluate a DataOp DAG with train/test split."""
    # Resolve variables to constants at compile time, so the scheduler runs
    # without an environment.
    linearized_dag, split_pos, flagged_ops = optimize(dag, env=get_data(dag))
    sched = SequentialScheduler(linearized_dag, split_pos, flagged_ops, FLAGS.stats)
    out = sched.evaluate(seed, test_size)
    stats_printer(sched)
    return out


def stats_printer(sched: SequentialScheduler):
    # FIXME: Measure operator execution only if stats is enabled
    # Heavy hitters
    if FLAGS.stats:
        table = pd.DataFrame(sched.timings, columns=["Op", "time"])
        table = table.groupby("Op").aggregate(["sum", "count"])
        table.columns = ["Time", "Count"]
        table = table.reset_index().sort_values(by="Time", ascending=False)
        # Share of total DataOp evaluation time, so heavy hitters stand out
        # relative to the whole run rather than only by absolute seconds.
        total_time = table["Time"].sum()
        table["%"] = 100 * table["Time"] / total_time if total_time else 0.0
        table = table[["Op", "Count", "Time", "%"]]
        print("\n" + "=" * 80)
        print(f"Heavy hitters (sorted by time spent in DataOp evaluation):\n")
        print(table.head(FLAGS.stats_top_k).to_string(
            index=False,
            formatters={"Time": "{:.4f}".format, "%": "{:.1f}%".format},
        ))
        print("=" * 80)
        print("Total BufferPool overhead during execution:", sched.buffer_pool_overhead)
        print("=" * 80 + "\n")
        print(sched.pool.stats)
        print("=" * 80 + "\n")
