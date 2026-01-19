import pandas as pd
from skrub import DataOp

from stratum._config import FLAGS
from stratum.logical_optimizer._optimize import optimize
from stratum.runtime._scheduler import Scheduler


def grid_search(dag: DataOp, cv=None, scoring=None, return_predictions=False):
    """Perform grid search with cross-validation on a DataOp DAG."""

    show_stats = FLAGS.stats is not None
    ops_ordered = optimize(dag)
    sched = Scheduler(ops_ordered, show_stats)

    preds = sched.grid_search(cv, scoring, return_predictions)

    # Heavy hitters
    if show_stats:
        table = pd.DataFrame(sched.timings, columns=["Op", "time"])
        table = table.groupby("Op").aggregate(["sum", "count"])
        table.columns = ["Time", "Count"]
        table = table.reset_index().sort_values(by="Time", ascending=False)
        print("\n" + "=" * 80)
        print(f"Heavy hitters (sorted by time spent in DataOp evaluation):")
        print(table.head(FLAGS.stats).to_string(index=False))
        print("=" * 80 + "\n")
    return (sched,preds) if return_predictions else sched


def evaluate(dag: DataOp, seed: int = 42, test_size = 0.2, cse: bool = False):
    """Evaluate a DataOp DAG with train/test split."""
    ops_ordered = optimize(dag)
    return Scheduler(ops_ordered).evaluate(seed, test_size)