"""Reuse cached results for matching fold inputs."""
import gc
import hashlib
import weakref
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge
from skrub import ApplyToCols, StringEncoder, TableVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import stratum as st
from stratum.frontend._api import grid_search
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer._lineage import state_key
from stratum.optimizer._optimize import SearchConfig, optimize
from stratum.optimizer.logical._scoring import resolve_scoring
from stratum.optimizer.logical._split_ops import SplitOutput
from stratum.optimizer.physical._conversion import ConversionOp
from stratum.optimizer.physical._generic_execs import CallExec
from stratum.optimizer.physical._physical_ops import EstimatorExecution
from stratum.optimizer.physical._representation import FrameLib, frame_lib
from stratum.optimizer.physical._transform_execs import TableVectorizerOp
from stratum.runtime._reuse_cache import active_reuse_cache
from stratum.runtime._scheduler import SequentialScheduler
from stratum.runtime import _reuse_cache
from stratum.runtime._object_size import get_size
from tests.runtime.test_reuse_execution import csv_path  # noqa: F401 (fixture)


@pytest.fixture
def cache(monkeypatch):
    monkeypatch.setattr(_reuse_cache, "_active", None)
    with st.config(reuse_cache=True):
        yield active_reuse_cache()


@pytest.fixture
def stored(cache, monkeypatch):
    values = {}
    put = cache.put

    def record(key, value, **kwargs):
        accepted = put(key, value, **kwargs)
        if accepted:
            values[key] = value
        return accepted

    monkeypatch.setattr(cache, "put", record)
    return values


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

        def recording(sched, cv_id):
            out = run_fold(sched, cv_id)
            taken.append(sorted(str(op) for op in sched.fold_ops if op.cache_hit is not None))
            return out

        with monkeypatch.context() as patch:
            patch.setattr(SequentialScheduler, "run_fold", recording)
            sched = grid_search(dag, cv=cv, scoring=scoring)
        return sched, taken
    return run


def test_a_repeated_search_takes_each_folds_predictions(csv_path, cache, run):
    first, taken = run(search(csv_path))
    assert taken == [[]] * 3
    second, taken = run(search(csv_path))
    # The models' predictions on each fold's test rows: nothing before them runs.
    assert taken == [["SklearnRidge_predict_test", "SklearnRidge_predict_test"]] * 3
    assert len(second.fold_ops) < len(first.fold_ops)
    assert second.results_.equals(first.results_)
    assert not second.state_store and second.pool.active_count == 0


def test_every_fold_probes_entries_written_by_previous_folds(csv_path, cache, run):
    rows = np.arange(60)
    cv = [(rows[:40], rows[40:])] * 2
    sched, taken = run(search(csv_path), cv=cv)
    assert taken == [[], ["SklearnRidge_predict_test", "SklearnRidge_predict_test"]]
    assert sched.pool.active_count == 0 and not sched.state_store


def test_nested_concats_do_not_prevent_reuse_when_a_search_branch_is_removed(
        csv_path, cache, run, monkeypatch):
    def pipeline(explore):
        data = st.as_data_op(str(csv_path)).skb.apply_func(pd.read_csv)
        X, y = data[["a", "b"]].skb.mark_as_X(), data["y"].skb.mark_as_y()
        features = X.skb.apply(TableVectorizer(high_cardinality=StringEncoder(random_state=42)))
        features = features.skb.apply(StandardScaler())
        if explore:
            manual = X[["a"]].skb.concat([X[["b"]]], axis=1)
            manual = manual.skb.concat([X[["a"]].rename(columns={"a": "c"})], axis=1)
            features = st.choose_from({"manual": manual, "tv": features}, name="features").as_data_op()
        return features.skb.apply(Ridge(alpha=1.0 if explore else 2.0), y=y)

    executed = []
    process = SequentialScheduler.process_op

    def record(sched, op):
        if isinstance(op, TableVectorizerOp) and op.cache_hit is None:
            executed.append(op.mode)
        return process(sched, op)

    monkeypatch.setattr(SequentialScheduler, "process_op", record)
    with st.config(implementation_selector="greedy"), st.config_context(eager_data_ops=False):
        first, _ = run(pipeline(True))
        assert executed == ["fit_transform", "transform"] * 3
        assert frame_lib(first.linearized_dag[first.pos_split_op].inputs[0]) is FrameLib.POLARS
        assert not any(isinstance(op, ConversionOp) for op in first.linearized_dag[:first.pos_split_op])
        executed.clear()
        second, taken = run(pipeline(False))
        assert not executed
        assert all(any("StandardScaler" in name for name in hits) for hits in taken)
        with st.config(reuse_cache=False):
            fresh, _ = run(pipeline(False))
    np.testing.assert_allclose(second.results_["scores"], fresh.results_["scores"])
    assert second.pool.active_count == 0 and not second.state_store


def test_training_lineage_depends_only_on_training_rows(csv_path, cache):
    dag = search(csv_path)
    sched = SequentialScheduler(*optimize(
        dag, env=get_data(dag), search=SearchConfig(metric=resolve_scoring("r2")),
        reuse_cache=cache), reuse_cache=cache)
    try:
        sched.compute_xy(cv=3)
        sched.cv_id = 0
        keys = []
        for test in (np.arange(40, 60), np.arange(45, 60)):
            plan = sched.fold_plan.copy()
            rows = (np.arange(40, dtype=np.int64), test.astype(np.int64))
            for op in plan.ops:
                if isinstance(op, SplitOutput):
                    op.row_digest = hashlib.blake2b(
                        rows[0 if op.train else 1].tobytes(), digest_size=16).digest()
            plan = plan.recompile(0, available=sched._available, cache=cache)
            keys.append({op.instance_of: op.lineage for op in plan.ops})
        training = [op for op in sched.fold_plan.ops if op.train]
        assert training and all(keys[0][op] == keys[1][op] is not None for op in training)
        predicted = [op for op in sched.fold_plan.ops if op.state_ref is not None]
        assert predicted and all(keys[0][op] != keys[1][op] for op in predicted)
    finally:
        sched._finish()


@pytest.mark.parametrize("selector", ["default", "greedy"])
@pytest.mark.parametrize("also_reads_polars", [False, True])
def test_cached_conversions_keep_fold_inputs_bound(csv_path, cache, selector,
                                                  also_reads_polars):
    # An opaque call needs pandas, including an extra frame computed before CV.
    # Greedy produces that frame in polars and inserts a conversion. A warm
    # cache can supply the conversion and prune its original producer entirely.
    def add_feature(x, extra):
        return x.assign(c=len(extra))

    def pipeline(alpha):
        data = st.as_data_op(str(csv_path)).skb.apply_func(pd.read_csv)
        X, y = data[["a", "b"]].skb.mark_as_X(), data["y"].skb.mark_as_y()
        extra = data[["b"]].reset_index(drop=True)
        if also_reads_polars:
            # Keep both representations: the greedy join reads polars while the
            # opaque call reads pandas. The latter must reuse the cached
            # conversion rather than repeat it in each fold.
            X = X.merge(extra, on="b", how="left")
        return X.skb.apply_func(add_feature, extra).skb.apply(Ridge(alpha=alpha), y=y)

    with st.config(implementation_selector=selector):
        for i, alpha in enumerate((0.1, 0.1, 10.0)):
            cached = grid_search(pipeline(alpha), cv=KFold(3), scoring="r2")
            if i and selector == "greedy":
                hits = [op for op in cached.linearized_dag[:cached.pos_split_op]
                        if isinstance(op, ConversionOp) and op.cache_hit is not None]
                assert hits
                assert all(not op.inputs for op in hits)
                calls = [op for op in cached.fold_plan.ops if isinstance(op, CallExec)]
                assert len(calls) == 2  # training and validation
                assert all(op.inputs[1] in hits for op in calls)
            with st.config(reuse_cache=False):
                fresh = grid_search(pipeline(alpha), cv=KFold(3), scoring="r2")
            np.testing.assert_allclose(cached.results_["scores"], fresh.results_["scores"])
            assert cached.pool.active_count == 0 and not cached.state_store


def test_a_changed_grid_reuses_what_the_candidates_share(csv_path, cache, run):
    run(search(csv_path, alphas=(0.1, 1.0)))
    sched, taken = run(search(csv_path, alphas=(0.1, 10.0)))
    # Reuse the scaler's transformed outputs and cached predictions for the
    # candidate shared by both grids. Alpha=10 fits on the cached training output.
    assert taken == [["SklearnRidge_predict_test", "SklearnStandardScaler_fit_transform_train",
                      "SklearnStandardScaler_transform_test"]] * 3
    theirs = search(csv_path, alphas=(0.1, 10.0)).skb.make_grid_search(
        fitted=True, cv=KFold(3), scoring="r2").results_
    np.testing.assert_allclose(sorted(sched.results_["scores"]),
                               sorted(theirs["mean_test_score"]))


def test_changed_fold_rows_prevent_reuse(csv_path, cache, run):
    run(search(csv_path), cv=KFold(3))
    _, taken = run(search(csv_path), cv=KFold(2))
    assert taken == [[]] * 2


def test_fitting_reuse_requires_all_values_needed_by_consumers(
        csv_path, cache, run, stored, monkeypatch):
    run(search(csv_path, alphas=(0.1, 1.0)))
    cache = active_reuse_cache()
    missing = {key for key, value in stored.items() if isinstance(value, ApplyToCols)}
    assert len(missing) == 3
    get = cache.get
    monkeypatch.setattr(cache, "get", lambda key, default=None, **kwargs:
                        default if key in missing else get(key, default, **kwargs))
    sched, taken = run(search(csv_path, alphas=(0.1, 10.0)))
    # The plan still has a consumer of the scaler's fitted state, so its cached
    # output alone is insufficient.
    assert taken == [["SklearnRidge_predict_test", "SklearnStandardScaler_transform_test"]] * 3
    first, _ = run(search(csv_path, alphas=(0.1, 10.0)))
    assert first.results_.equals(sched.results_)


def test_fitted_states_are_cached_with_their_size(csv_path, cache, run, stored, monkeypatch):
    state_size = 123456
    monkeypatch.setattr(EstimatorExecution, "_measure_retained_state",
                        lambda self, estimator: state_size)
    sched, _ = run(search(csv_path))
    states = [value for value in stored.values() if isinstance(value, BaseEstimator)]
    # A scaler and two models per fold.
    assert len(states) == 9
    assert cache.memory_usage == sum(state_size if isinstance(value, BaseEstimator)
                                    else get_size(value) for value in stored.values())
    for op in sched.fold_ops:
        if op.produces_state and op.cache_key is not None:
            state, size = cache.get(state_key(op.cache_key), with_size=True)
            assert state is stored[state_key(op.cache_key)]
            assert size == op.retained_state_size() == state_size
            assert op.estimator is op.instance_of.estimator and op.estimator is not state


def test_reused_fitted_states_are_shared_with_read_only_execution(csv_path, cache, run,
                                                               stored, monkeypatch):
    first, _ = run(search(csv_path))
    models = {key: value for key, value in stored.items() if isinstance(value, Ridge)}
    coefficients = {key: model.coef_.copy() for key, model in models.items()}
    get = cache.get
    # Leave only fitted models available: prediction must run using their states.
    monkeypatch.setattr(cache, "get", lambda key, default=None, **kwargs:
                        get(key, default, **kwargs) if key in models else default)
    dag = search(csv_path)  # Build its eager preview before observing execution.
    predict = Ridge.predict
    seen = []

    def read_only_predict(model, X):
        seen.append(model)
        return predict(model, X)

    monkeypatch.setattr(Ridge, "predict", read_only_predict)
    second, taken = run(dag)
    assert taken == [["SklearnRidge_fit_train", "SklearnRidge_fit_train"]] * 3
    assert second.results_.equals(first.results_)
    assert {id(model) for model in seen} == {id(model) for model in models.values()}
    for key, model in models.items():
        np.testing.assert_array_equal(model.coef_, coefficients[key])
    assert not second.state_store and second.pool.active_count == 0


@pytest.mark.parametrize("failure", ["admission", "prediction"])
def test_cached_state_failure_releases_execution_and_pending_hits(
        csv_path, cache, run, stored, monkeypatch, failure):
    state_size = 123456
    monkeypatch.setattr(EstimatorExecution, "_measure_retained_state",
                        lambda self, estimator: state_size)
    run(search(csv_path))
    references = {key: weakref.ref(value) for key, value in stored.items()
                  if isinstance(value, Ridge)}
    stored.clear()
    # Restore several fitted states while recomputing their predictions.
    get = cache.get
    monkeypatch.setattr(cache, "get", lambda key, default=None, **kwargs:
                        get(key, default, **kwargs) if key in references else default)
    dag = search(csv_path)
    ops, split_pos = optimize(dag, env=get_data(dag), search=SearchConfig(metric=resolve_scoring("r2")),
                              reuse_cache=cache)
    sched = SequentialScheduler(ops, split_pos, reuse_cache=cache)
    if failure == "admission":
        def reject(*args, **kwargs):
            raise MemoryError("state admission failed")
        monkeypatch.setattr(sched.state_store, "put", reject)
        error, match = MemoryError, "state admission failed"
    else:
        def fail(model, X):
            raise ValueError("cached prediction failed")
        monkeypatch.setattr(Ridge, "predict", fail)
        error, match = RuntimeError, "cached prediction failed"
    with pytest.raises(error, match=match):
        sched.grid_search(KFold(3))
    assert sched.memory_usage == 0 and sched.pool.active_count == 0
    assert not sched.state_store
    cache.clear()
    stored.clear()
    gc.collect()
    assert references and all(ref() is None for ref in references.values())


def test_the_test_rows_are_keyed_alike_whatever_the_scorer_calls(csv_path, cache, run):
    from sklearn.linear_model import LogisticRegression
    data = st.as_data_op(str(csv_path)).skb.apply_func(pd.read_csv)

    def model():
        X = data[["a", "b"]].skb.mark_as_X()
        y = (data["y"] > .5).skb.mark_as_y()
        return X.skb.apply(StandardScaler()).skb.apply(LogisticRegression(), y=y)

    run(model(), scoring="accuracy")
    _, taken = run(model(), scoring="roc_auc")
    # The scorer requires a different estimator prediction method; the scaler's
    # fitting and transformation are unchanged.
    assert taken == [["SklearnStandardScaler_fit_transform_train", "SklearnStandardScaler_transform_test"]] * 3


def test_cache_hits_retain_only_values_needed_by_remaining_consumers(
        csv_path, cache, run, monkeypatch):
    run(search(csv_path, alphas=(0.1, 1.0)))
    released = []
    monkeypatch.setattr(SequentialScheduler, "_release_states",
                        lambda sched, ops: released.extend(op for op in ops if op in sched.state_store))
    sched, _ = run(search(csv_path, alphas=(0.1, 10.0)))
    # No remaining operator requires the scaler's fitted state: alpha=10 fits on
    # its cached output, and predictions for the test rows are also cached.
    scaler = next(op for op in sched.fold_ops if str(op) == "SklearnStandardScaler_fit_transform_train")
    assert scaler.cache_hit is not None
    assert scaler.cache_hit.has_output and not scaler.cache_hit.has_state
    assert not released and not sched.state_store and sched.pool.active_count == 0
