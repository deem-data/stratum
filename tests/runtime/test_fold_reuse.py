"""Reuse after the split: each fold takes what the cache holds of its own copy."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import stratum as st
from stratum._api import grid_search
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._optimize import SearchConfig, compile_plan
from stratum.optimizer.logical._scoring import resolve_scoring
from stratum.runtime._reuse_cache import active_reuse_cache
from stratum.runtime._scheduler import SequentialScheduler
from tests.runtime.test_reuse_planning import cache_dir, csv_path  # noqa: F401 (fixtures)


def search(csv_path, alphas=(0.1, 1.0)):
    data = st.as_data_op(str(csv_path)).skb.apply_func(pd.read_csv)
    X, y = data[["a", "b"]].skb.mark_as_X(), data["y"].skb.mark_as_y()
    alpha = st.choose_from(list(alphas), name="alpha")
    return X.skb.apply(StandardScaler()).skb.apply(Ridge(alpha=alpha), y=y)


@pytest.fixture
def run(monkeypatch):
    """Runs a search; returns the scheduler and, per fold, the ops it took from the cache."""
    def run(dag, cv=KFold(3), scoring="r2"):
        taken = []
        run_fold = SequentialScheduler.run_fold

        def recording(sched, indices, profile=None):
            out = run_fold(sched, indices, profile)
            taken.append(sorted(str(op) for op in sched.fold_ops if op.cache_hit is not None))
            return out

        with monkeypatch.context() as patch:
            patch.setattr(SequentialScheduler, "run_fold", recording)
            sched = grid_search(dag, cv=cv, scoring=scoring)
        return sched, taken
    return run


def test_a_repeated_search_takes_each_folds_predictions(csv_path, cache_dir, run):
    first, taken = run(search(csv_path))
    assert taken == [[]] * 3
    second, taken = run(search(csv_path))
    # The models' predictions on each fold's test rows: nothing before them runs.
    assert taken == [["SklearnRidge_predict", "SklearnRidge_predict"]] * 3
    assert len(second.fold_ops) < len(first.fold_ops)
    assert second.results_.equals(first.results_)
    assert not second.state_store and second.pool.active_count == 0


def test_a_changed_grid_reuses_what_the_candidates_share(csv_path, cache_dir, run):
    run(search(csv_path, alphas=(0.1, 1.0)))
    sched, taken = run(search(csv_path, alphas=(0.1, 10.0)))
    # The scaler, fitted and applied, and the model both grids hold; alpha=10 fits
    # on the scaler's cached output.
    assert taken == [["PassthroughTransformer_fit", "PassthroughTransformer_predict",
                      "SklearnRidge_predict"]] * 3
    theirs = search(csv_path, alphas=(0.1, 10.0)).skb.make_grid_search(
        fitted=True, cv=KFold(3), scoring="r2").results_
    np.testing.assert_allclose(sorted(sched.results_["scores"]),
                               sorted(theirs["mean_test_score"]))


def test_a_fold_only_takes_its_own_entries(csv_path, cache_dir, run):
    run(search(csv_path), cv=KFold(3))
    _, taken = run(search(csv_path), cv=KFold(2))
    assert taken == [[]] * 2


def test_a_fit_stage_is_taken_only_with_every_part_its_readers_read(csv_path, cache_dir, run):
    run(search(csv_path, alphas=(0.1, 1.0)))
    cache = active_reuse_cache()
    for entry in [e for e in cache.entries.values() if e.label == "PassthroughTransformer_fit state"]:
        cache._discard(entry)
    sched, taken = run(search(csv_path, alphas=(0.1, 10.0)))
    # The scaler's output is cached, but the test rows apply its state: it fits again.
    assert taken == [["PassthroughTransformer_predict", "SklearnRidge_predict"]] * 3
    first, _ = run(search(csv_path, alphas=(0.1, 10.0)))
    assert first.results_.equals(sched.results_)


def test_fitted_states_are_cached_with_their_size(csv_path, cache_dir, run):
    run(search(csv_path))
    states = [e for e in active_reuse_cache().entries.values() if e.label.endswith(" state")]
    # A scaler and two models per fold.
    assert len(states) == 9
    assert all(e.size > 1 for e in states)


def test_the_training_side_is_keyed_by_the_training_rows_only(csv_path, cache_dir):
    dag = search(csv_path)
    plan = compile_plan(dag, env=get_data(dag), search=SearchConfig(metric=resolve_scoring("r2")),
                        reuse_cache=active_reuse_cache())
    fold = plan.compile_folds()
    train = list(range(40))
    keys = []
    for test in (list(range(40, 60)), list(range(45, 60))):
        keys.append({op.instance_of: op.lineage for op in fold.instantiate((train, test))})
    fit_side = [op for op in fold.ops if op.stage == "fit" or getattr(op, "train", False)]
    assert fit_side and all(keys[0][op] == keys[1][op] is not None for op in fit_side)
    predicted = [op for op in fold.ops if op.stage == "predict"]
    assert all(keys[0][op] != keys[1][op] for op in predicted)


def test_the_test_rows_are_keyed_alike_whatever_the_scorer_calls(csv_path, cache_dir, run):
    from sklearn.linear_model import LogisticRegression
    data = st.as_data_op(str(csv_path)).skb.apply_func(pd.read_csv)

    def model():
        X = data[["a", "b"]].skb.mark_as_X()
        y = (data["y"] > .5).skb.mark_as_y()
        return X.skb.apply(StandardScaler()).skb.apply(LogisticRegression(), y=y)

    run(model(), scoring="accuracy")
    _, taken = run(model(), scoring="roc_auc")
    # The model answers with another method, but the scaler is fitted and applied
    # the same way.
    assert taken == [["PassthroughTransformer_fit", "PassthroughTransformer_predict"]] * 3


def test_a_hit_takes_only_the_parts_left_to_read(csv_path, cache_dir, run, monkeypatch):
    run(search(csv_path, alphas=(0.1, 1.0)))
    released = []
    monkeypatch.setattr(SequentialScheduler, "_release_states",
                        lambda sched, ops: released.extend(op for op in ops if op in sched.state_store))
    sched, _ = run(search(csv_path, alphas=(0.1, 10.0)))
    # Applied to the test rows, the scaler's state is read by no op left: alpha=10
    # fits on its cached output, and the predictions on the test rows are hits.
    scaler = next(op for op in sched.fold_ops if str(op) == "PassthroughTransformer_fit")
    assert scaler.cache_hit is not None
    assert scaler.cache_hit.has_output and not scaler.cache_hit.has_state
    assert not released and not sched.state_store and sched.pool.active_count == 0
