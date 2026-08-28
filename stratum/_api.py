import pandas as pd
from skrub import DataOp
from skrub._data_ops._data_ops import SplitX
from skrub._data_ops._estimator import _Splitter
# Aliased: this module defines its own `evaluate` for the train/test path.
from skrub._data_ops._evaluation import evaluate as skrub_evaluate
from skrub._data_ops._evaluation import needs_eval
from sklearn.model_selection import check_cv

from stratum._config import FLAGS
from stratum.optimizer._optimize import optimize
from stratum.runtime._scheduler import SequentialScheduler
from stratum.utils._skrub_graph import find_x_impl, get_data
from time import perf_counter

#TODO: Rename this file
def grid_search(dag: DataOp, cv=None, scoring=None, return_predictions=False, env=None):
    """Perform grid search with cross-validation on a DataOp DAG."""
    t0 = perf_counter()
    #FIXME: Measure operator execution only if stats is enabled
    env_extra = env if env else {}
    env = get_data(dag)
    for k, v in env_extra.items():
        env[k] = v
    cv = _resolve_cv(dag, cv, env)
    # Resolve variables to constants at compile time, so the scheduler runs
    # without an environment.
    linearized_dag, split_pos, flagged_ops = optimize(dag, env=env)
    sched = SequentialScheduler(linearized_dag, split_pos, flagged_ops, FLAGS.stats, t0=t0)

    preds = sched.grid_search(cv, scoring, return_predictions)

    stats_printer(sched)

    return (sched,preds) if return_predictions else sched


def _resolve_cv(dag: DataOp, cv, env: dict):
    """Resolve the splitter to cross-validate with.

    Mirrors skrub's ``_compute_cv_data``: an explicit ``cv`` is prioritized, otherwise the
    splitter declared on the plan via ``mark_as_X(cv=..., split_kwargs=...)``
    determines the folds.

    The declared splitter is wrapped in skrub's ``_Splitter`` so ``split_kwargs``
    (e.g. ``groups`` for ``GroupKFold``) reach it. ``split_kwargs`` defaults to
    None when only ``cv`` is passed, hence the normalization to an empty dict.
    """
    if cv is not None:
        return cv
    impl = find_x_impl(dag)
    if not isinstance(impl, SplitX) or impl.cv is None:
        return None
    declared_cv, split_kwargs = impl.cv, impl.split_kwargs
    if needs_eval((declared_cv, split_kwargs)):
        # Both may themselves be DataOps, which only the environment can resolve.
        resolved = skrub_evaluate(
            {"cv": declared_cv, "split_kwargs": split_kwargs},
            mode="fit_transform",
            environment=env,
            clear=True,
        )
        declared_cv, split_kwargs = resolved["cv"], resolved["split_kwargs"]
    return _Splitter(check_cv(declared_cv), split_kwargs or {})


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
