"""Operator state outlives its outputs and belongs to the plan, not the cache."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from stratum.optimizer.logical._base import All
from stratum.optimizer.logical._ops import ValueOp
from stratum.optimizer.physical._predictor_execs import SklearnKNeighbors
from stratum.runtime._reuse_cache import ReuseCache
from stratum.runtime._scheduler import SequentialScheduler


def test_model_memory_is_separate_from_outputs_and_survives_pool_cleanup():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = rng.integers(0, 2, len(X))
    source = ValueOp(X)
    op = SklearnKNeighbors(estimator=KNeighborsClassifier(), y=y, cols=All(), no_wrap=True)
    op.inputs = [source]
    op.remove_after = [source]
    op.cache_key = "predictions"
    cache = ReuseCache()
    scheduler = SequentialScheduler([source, op], None, [], reuse_cache=cache)
    assert scheduler.retained_state_memory_usage == 0

    scheduler.pool.put(source, X)
    scheduler.process_op(op)
    state_size = op.retained_state_size()
    assert state_size >= X.nbytes
    assert scheduler.retained_state_memory_usage == state_size
    assert scheduler.pool.memory_usage == cache.memory_usage == y.nbytes
    assert scheduler.memory_usage == state_size + y.nbytes

    # Evicting outputs cannot free the estimator held on the plan's operator.
    cache.clear()
    scheduler._finish()
    assert scheduler.pool.memory_usage == cache.memory_usage == 0
    assert scheduler.memory_usage == state_size

    op.y = y[:100]
    scheduler.pool.put(source, X[:100].copy())
    scheduler.process_op(op)
    assert 0 < scheduler.retained_state_memory_usage < state_size
    assert scheduler.memory_usage == op.retained_state_size() + y[:100].nbytes
    scheduler._finish()
