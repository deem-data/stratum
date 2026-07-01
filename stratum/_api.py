import pandas as pd
from skrub import DataOp

from stratum._config import FLAGS
from stratum.optimizer._optimize import optimize
from stratum.runtime._scheduler import SequentialScheduler
from time import perf_counter

#TODO: Rename this file
def grid_search(dag: DataOp, cv=None, scoring=None, return_predictions=False, env=None):
    """Perform grid search with cross-validation on a DataOp DAG."""
    t0 = perf_counter()
    #FIXME: Measure operator execution only if stats is enabled
    env_extra = env if env else {}
    env = dag.skb.get_data()
    for k, v in env_extra.items():
        env[k] = v
    # Resolve variables to constants at compile time, so the scheduler runs
    # without an environment.
    linearized_dag, split_pos, flagged_ops = optimize(dag, env=env)
    sched = SequentialScheduler(linearized_dag, split_pos, flagged_ops, FLAGS.stats, t0=t0)

    preds = sched.grid_search(cv, scoring, return_predictions)

    stats_printer(sched)

    return (sched,preds) if return_predictions else sched


def evaluate(dag: DataOp, seed: int = 42, test_size = 0.2):
    """Evaluate a DataOp DAG with train/test split."""
    # Resolve variables to constants at compile time, so the scheduler runs
    # without an environment.
    linearized_dag, split_pos, flagged_ops = optimize(dag, env=dag.skb.get_data())
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
        print("\n" + "=" * 80)
        print(f"Heavy hitters (sorted by time spent in DataOp evaluation):\n")
        print(table.head(FLAGS.stats_top_k).to_string(index=False))
        print("=" * 80)
        print("Total BufferPool overhead during execution:", sched.buffer_pool_overhead)
        print("=" * 80 + "\n")
        print(sched.pool.stats)
        print("=" * 80 + "\n")
