import pandas as pd
from skrub import DataOp

from stratum._config import FLAGS
from stratum.logical_optimizer._optimize import optimize
from stratum.runtime._scheduler import SequentialScheduler
from time import perf_counter

def grid_search(dag: DataOp, cv=None, scoring=None, return_predictions=False, env=None):
    """Perform grid search with cross-validation on a DataOp DAG."""
    t0 = perf_counter()
    show_stats = FLAGS.stats is not None
    env_extra = env if env else {}
    env = dag.skb.get_data()
    for k, v in env_extra.items():
        env[k] = v
    dag = optimize(dag)
    sched = SequentialScheduler(dag, show_stats, env=env, t0=t0)

    preds = sched.grid_search(cv, scoring, return_predictions)

    # Heavy hitters
    if show_stats:
        table = pd.DataFrame(sched.timings, columns=["Op", "time"])
        table = table.groupby("Op").aggregate(["sum", "count"])
        table.columns = ["Time", "Count"]
        table = table.reset_index().sort_values(by="Time", ascending=False)
        print("\n" + "=" * 80)
        print(f"Heavy hitters (sorted by time spent in DataOp evaluation):\n")
        print(table.head(FLAGS.stats).to_string(index=False))
        table.head(FLAGS.stats).to_csv("heavy_hitters.csv", index=False)
        print("=" * 80 + "\n")

    return (sched,preds) if return_predictions else sched

def evaluate(dag: DataOp, seed: int = 42, test_size = 0.2, cse: bool = False):
    """Evaluate a DataOp DAG with train/test split."""
    ops_ordered = optimize(dag)
    return SequentialScheduler(ops_ordered).evaluate(seed, test_size)