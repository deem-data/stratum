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
    show_stats = FLAGS.stats
    stats_top_k = FLAGS.stats_top_k
    env_extra = env if env else {}
    env = dag.skb.get_data()
    for k, v in env_extra.items():
        env[k] = v
    linearized_dag, split_pos, flagged_ops = optimize(dag)
    sched = SequentialScheduler(linearized_dag, split_pos, flagged_ops, show_stats, env=env, t0=t0)

    preds = sched.grid_search(cv, scoring, return_predictions)

    # Heavy hitters
    if show_stats:
        table = pd.DataFrame(sched.timings, columns=["Op", "time"])
        table = table.groupby("Op").aggregate(["sum", "count"])
        table.columns = ["Time", "Count"]
        table = table.reset_index().sort_values(by="Time", ascending=False)
        print("\n" + "=" * 80)
        print(f"Heavy hitters (sorted by time spent in DataOp evaluation):\n")
        print(table.head(stats_top_k).to_string(index=False))
        print("=" * 80 + "\n")

    return (sched,preds) if return_predictions else sched

def evaluate(dag: DataOp, seed: int = 42, test_size = 0.2, cse: bool = False):
    """Evaluate a DataOp DAG with train/test split."""
    linearized_dag, split_pos, flagged_ops = optimize(dag)
    return SequentialScheduler(linearized_dag, split_pos, flagged_ops).evaluate(seed, test_size)