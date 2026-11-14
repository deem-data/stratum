import unittest
import weakref

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold, KFold, train_test_split

import stratum as st
from stratum.optimizer.logical._base import OperandRef
from stratum.optimizer.logical._split_ops import (
    BuildSplitterOp, SplitIndicesOp, SplitOp, SplitOutput)
from stratum.optimizer.physical._generic_execs import ValueExec
from stratum.optimizer.physical.rewrites._lifetime_planning import (
    compute_pinned_ops, plan_input_removals)
from stratum.optimizer.physical.rewrites._linearization import linearize_dag
from stratum.runtime._scheduler import SequentialScheduler
from tests.runtime.runtime_test_utils import simple_pipeline


def split_scheduler(tmp_path, declared=None):
    x = ValueExec(np.arange(24).reshape(12, 2))
    y = ValueExec(np.arange(12))
    indices = SplitIndicesOp(inputs=[x, y])
    if declared is not None:
        groups = ValueExec(np.repeat(np.arange(3), 4))
        splitter = BuildSplitterOp(declared, {"groups": OperandRef(0)}, inputs=[groups])
        groups.outputs = [splitter]
        splitter.outputs = [indices]
        indices.inputs.append(splitter)
    split = SplitOp(inputs=[x, y, indices])
    x.outputs = [indices, split]
    y.outputs = [indices, split]
    indices.outputs = [split]
    out = SplitOutput(inputs=[split])
    split.outputs = [out]
    ops, split_pos = linearize_dag(out)
    plan_input_removals(ops, compute_pinned_ops(ops, split_pos))
    sched = SequentialScheduler(ops, split_pos)
    sched.pool._spill_root = tmp_path
    sched.pool.memory_budget = 1
    return sched, split


class RecordingKFold(KFold):
    def __init__(self):
        super().__init__(n_splits=3)
        self.calls = 0
        self.references = []

    def split(self, X, y=None, groups=None):
        self.calls += 1
        self.references.extend((weakref.ref(X), weakref.ref(y)))
        for train, test in super().split(X, y, groups):
            self.references.extend((weakref.ref(train), weakref.ref(test)))
            yield train, test


def test_fold_indices_are_generated_once_and_spill_independently(tmp_path):
    sched, split = split_scheduler(tmp_path)
    cv = RecordingKFold()
    try:
        assert sched.compute_xy(cv=cv) is split
        assert cv.calls == 1
        assert split.indices.cv is None
        assert all(ref() is None for ref in cv.references)
        assert all(entry.pin_count == 0 for entry in sched.pool.live_variable_map.values())
        assert all(sched.pool.live_variable_map[(split.indices, i)].is_spilled
                   for i in range(3))

        for i, (_, test) in enumerate(KFold(3).split(np.arange(12))):
            result = sched.run_fold(i)
            np.testing.assert_array_equal(result, np.arange(24).reshape(12, 2)[test])
            assert (split.indices, i) not in sched.pool.live_variable_map
            assert all((split.indices, j) in sched.pool.live_variable_map
                       for j in range(i + 1, 3))
        assert cv.calls == 1
    finally:
        sched._finish()
    assert sched.memory_usage == 0
    assert not list(tmp_path.iterdir())


def test_declared_splitter_and_groups_are_released_before_folds(tmp_path):
    sched, split = split_scheduler(tmp_path, declared=GroupKFold(3))
    splitter = split.indices.inputs[2]
    groups = splitter.inputs[0]
    try:
        sched.compute_xy()
        assert splitter not in sched.pool.live_variable_map
        assert groups not in sched.pool.live_variable_map
        expected = GroupKFold(3).split(np.arange(12), groups=np.repeat(np.arange(3), 4))
        for i, (_, test) in enumerate(expected):
            np.testing.assert_array_equal(sched.run_fold(i),
                                          np.arange(24).reshape(12, 2)[test])
    finally:
        sched._finish()
    assert not list(tmp_path.iterdir())


def test_holdout_indices_preserve_seed_size_and_ignore_declared_cv(tmp_path):
    # This declared splitter would fail if holdout evaluation used it.
    sched, split = split_scheduler(tmp_path, declared=GroupKFold(20))
    try:
        sched.compute_xy(holdout=(17, .25))
        _, test = train_test_split(range(12), random_state=17, test_size=.25)
        np.testing.assert_array_equal(sched.run_fold(0),
                                      np.arange(24).reshape(12, 2)[test])
        assert split.indices.holdout is None
        assert (split.indices, 1) not in sched.pool.live_variable_map
    finally:
        sched._finish()


def test_index_generation_failure_releases_pins_and_runtime_configuration(tmp_path):
    class FailingCV:
        def split(self, X, y):
            yield np.arange(6), np.arange(6, 12)
            raise ValueError("split failed")

    sched, split = split_scheduler(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="SplitIndices.*split failed"):
            sched.compute_xy(cv=FailingCV())
        assert split.indices.cv is None
        assert all(entry.pin_count == 0 for entry in sched.pool.live_variable_map.values())
    finally:
        sched._finish()
    assert sched.memory_usage == 0
    assert not list(tmp_path.iterdir())


class SplitOpTest(unittest.TestCase):
    def test_train_test_split(self):
        pipeline = simple_pipeline()
        with st.config(scheduler=True):
            search = pipeline.skb.make_grid_search(scoring="neg_mean_squared_error")
            print(search.results_)


if __name__ == '__main__':
    unittest.main()
