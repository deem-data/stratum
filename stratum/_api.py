import pandas as pd
from skrub import DataOp

from stratum._config import FLAGS
from stratum.logical_optimizer._optimize import optimize
from stratum.runtime._caching import Cache
from stratum.runtime._physical_planning import physical_planning
from stratum.runtime._scheduler import ParallelScheduler, SequentialScheduler


def grid_search(dag: DataOp, cv=None, scoring=None, return_predictions=False, env=None):
    """Perform grid search with cross-validation on a DataOp DAG."""

    show_stats = FLAGS.stats is not None
    env_extra = env if env else {}
    env = dag.skb.get_data()
    for k, v in env_extra.items():
        env[k] = v
    cache = None
    if FLAGS.caching:
        cache = Cache()
    dag = optimize(dag)
    if FLAGS.scheduler_parallelism is not None:
        dag = physical_planning(dag)
        sched = ParallelScheduler(dag, {}, show_stats, backend=FLAGS.scheduler_parallelism, cache=cache, env=env)
    else:
        sched = SequentialScheduler(dag, show_stats, cache=cache, env=env)

    preds = sched.grid_search(cv, scoring, return_predictions)

    if FLAGS.caching:
        # persist cache to disk
        cache.persist()

    # Heavy hitters
    if show_stats:
        table = pd.DataFrame(sched.timings, columns=["Op", "time"])
        table = table.groupby("Op").aggregate(["sum", "count"])
        table.columns = ["Time", "Count"]
        table = table.reset_index().sort_values(by="Time", ascending=False)
        print("\n" + "=" * 80)
        print(f"Heavy hitters (sorted by time spent in DataOp evaluation):\n")
        print(table.head(FLAGS.stats).to_string(index=False))
        print("=" * 80 + "\n")
        if FLAGS.caching and cache is not None:
            print("\n" + "=" * 80)
            print("Cache timing statistics:\n")
            cache_stats = []
            for op_name, duration in cache.timings:
                cache_stats.append({"Operation": op_name, "Time (s)": f"{duration:.4f}", "Count": "1"})
            if cache.hit_count > 0:
                cache_stats.append({"Operation": "cache_hits", "Time (s)": f"{cache.hit_time:.4f}", "Count": f"{cache.hit_count}"})
            if cache.miss_count > 0:
                cache_stats.append({"Operation": "cache_misses", "Time (s)": "-", "Count": f"{cache.miss_count}"})
            if cache.set_count > 0:
                cache_stats.append({"Operation": "cache_sets", "Time (s)": f"{cache.set_time:.4f}", "Count": f"{cache.set_count}"})
            if cache_stats:
                cache_table = pd.DataFrame(cache_stats)
                print(cache_table.to_string(index=False))
            print("=" * 80 + "\n")
            
    return (sched,preds) if return_predictions else sched

def evaluate(dag: DataOp, seed: int = 42, test_size = 0.2, cse: bool = False):
    """Evaluate a DataOp DAG with train/test split."""
    ops_ordered = optimize(dag)
    return SequentialScheduler(ops_ordered).evaluate(seed, test_size)