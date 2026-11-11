"""Cache hits prune only unnecessary work and retain values across eviction."""
import os
import weakref

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from skrub._data_ops._data_ops import EvalMode

import stratum as st
from stratum.frontend._skrub_graph import get_data
from stratum.optimizer.physical.rewrites._input_removal_planning import compute_pinned_ops
from stratum.optimizer._lineage import compute_lineage_keys
from stratum.optimizer.physical.rewrites._linearization import is_recomputed, linearize_dag
from stratum.optimizer._op_utils import topological_iterator, validate_dag, validate_operands
from stratum.optimizer._optimize import OptConfig, optimize
from stratum.optimizer.physical.rewrites._reuse_planning import (
    CacheHit, apply_reuse_cache, cacheable_region, post_split_region)
from stratum.optimizer.logical._base import OperandRef
from stratum.optimizer.logical._ops import ImplOp, ValueOp
from stratum.optimizer.physical._impl_selection import FlagBasedSelector
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.runtime._reuse_cache import ReuseCache


class KeyedOp(PhysicalOp):
    fields = ["name", "refs"]
    cacheable = True

    def __init__(self, name, *inputs, cacheable=True, split=False):
        super().__init__(name=name, inputs=list(inputs))
        self.refs = [OperandRef(i) for i in range(len(inputs))]
        self.cacheable = cacheable
        self.is_split_op = split
        for producer in inputs:
            producer.outputs.append(self)


def graph_keys(root):
    return compute_lineage_keys(topological_iterator(root))


def hits(plan):
    return [op for op in plan if op.cache_hit is not None]


def assert_graph(root, expected):
    ops = list(topological_iterator(root))
    assert set(ops) == set(expected) and len(ops) == len(expected)
    for op in ops:
        for producer in op.inputs:
            assert producer in ops and op in producer.outputs
        for consumer in op.outputs:
            assert consumer in ops and op in consumer.inputs
    validate_dag(root)


def test_nearest_hit_prunes_a_whole_chain_and_keeps_root_identity():
    source = KeyedOp("source")
    middle = KeyedOp("middle", source)
    root = KeyedOp("root", middle)
    keys = graph_keys(root)
    cache = ReuseCache()
    for op in (source, middle, root):
        cache.put(keys[op], op.name)

    apply_reuse_cache(root, cache)

    assert_graph(root, [root])
    assert root.cache_hit.key == keys[root]
    assert root.cache_hit.take() == "root" and root.cache_key is None
    assert not source.inputs and not source.outputs
    assert not middle.inputs and not middle.outputs
    assert cache.stats.hits == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_ancestors_survive_for_an_uncached_branch(reverse):
    shared = KeyedOp("shared")
    private = KeyedOp("private", shared)
    cached = KeyedOp("cached", private)
    other = KeyedOp("other", shared)
    inputs = [cached, other]
    root = KeyedOp("root", *(reversed(inputs) if reverse else inputs), cacheable=False)
    keys = graph_keys(root)
    cache = ReuseCache()
    cache.put(keys[cached], "cached-result")

    apply_reuse_cache(root, cache)

    assert_graph(root, [shared, cached, other, root])
    assert cached.outputs == [root] and cached.inputs == []
    assert private.inputs == [] and private.outputs == []
    assert shared.outputs == [other]
    assert shared.cache_key == keys[shared] and other.cache_key == keys[other]
    assert root.cache_key is None and cache.stats.hits == 1


def test_shared_cached_ancestor_is_probed_once():
    source = KeyedOp("source")
    shared = KeyedOp("shared", source)
    left, right = KeyedOp("left", shared), KeyedOp("right", shared)
    root = KeyedOp("root", left, right, cacheable=False)
    cache = ReuseCache()
    cache.put(graph_keys(root)[shared], "shared-result")

    apply_reuse_cache(root, cache)

    assert_graph(root, [shared, left, right, root])
    assert shared.outputs == [left, right]
    assert hits([shared, left, right, root]) == [shared] and cache.stats.hits == 1


def test_repeated_input_edges_are_all_cut():
    shared = KeyedOp("shared")
    cached = KeyedOp("cached", shared, shared)
    other = KeyedOp("other", shared)
    root = KeyedOp("root", cached, cached, other, cacheable=False)
    cache = ReuseCache()
    cache.put(graph_keys(root)[cached], "cached-result")

    apply_reuse_cache(root, cache)

    assert_graph(root, [shared, cached, other, root])
    assert cached.outputs == [root, root] and shared.outputs == [other]
    assert set(linearize_dag(root)[0]) == {shared, cached, other, root}


def test_a_hit_can_prune_a_long_pipeline_without_recursion():
    source = KeyedOp("source")
    root = source
    for i in range(1500):
        root = KeyedOp(str(i), root)
    cache = ReuseCache()
    cache.put(graph_keys(root)[root], "result")
    apply_reuse_cache(root, cache)
    assert_graph(root, [root])
    assert not source.outputs


def test_every_split_and_its_descendants_are_excluded():
    source = KeyedOp("source")
    split_a, split_b = KeyedOp("split-a", source, split=True), KeyedOp("split-b", source, split=True)
    a, b = KeyedOp("a", split_a), KeyedOp("b", split_b)
    root = KeyedOp("root", a, b)
    ops = list(topological_iterator(root))
    keys = graph_keys(root)
    cache = ReuseCache()
    # Even entries supplied externally must never make fold-dependent ops hits.
    for op in ops:
        cache.put(keys[op], op.name)
    assert post_split_region(ops) == {split_a, split_b, a, b, root}
    assert cacheable_region(ops) == {source}

    apply_reuse_cache(root, cache)

    assert_graph(root, ops)
    assert hits(ops) == [source]
    assert all(op.cache_key is None for op in ops)


def test_recomputed_ops_and_their_dependents_are_never_reused():
    # Opt in explicitly to prove the region restriction is independent of config.
    class KeyedMode(ImplOp):
        fields = []
        cacheable = True

    mode = KeyedMode("mode", EvalMode())
    dependent = KeyedOp("dependent", mode)
    independent = KeyedOp("independent")
    root = KeyedOp("root", dependent, independent, cacheable=False)
    ops = list(topological_iterator(root))
    keys = graph_keys(root)
    cache = ReuseCache()
    for op in (mode, dependent, independent):
        cache.put(keys[op], op.name)

    apply_reuse_cache(root, cache)

    assert is_recomputed(mode) and mode not in cacheable_region(ops)
    assert mode.cache_hit is None and dependent.cache_hit is None
    assert mode.cache_key is None and dependent.cache_key is None
    assert hits(ops) == [independent]
    assert linearize_dag(root)[2] == [mode]


def test_constants_are_keyed_but_not_admitted():
    constant = ValueOp(3)
    root = KeyedOp("root", constant)
    keys = graph_keys(root)
    apply_reuse_cache(root, ReuseCache())
    assert constant.cache_key is None and constant.cache_hit is None
    assert keys[constant] is not None and root.cache_key == keys[root]


def test_unknown_ops_block_downstream_admission():
    source = KeyedOp("source")
    opaque = KeyedOp("opaque", source, cacheable=False)
    root = KeyedOp("root", opaque)
    apply_reuse_cache(root, ReuseCache())
    assert source.cache_key is not None
    assert opaque.cache_key is None and root.cache_key is None


def test_a_hit_owns_its_value_after_eviction_and_releases_it_on_handoff():
    root = KeyedOp("root")
    value = np.arange(10)
    ref = weakref.ref(value)
    cache = ReuseCache()
    cache.put(graph_keys(root)[root], value)
    apply_reuse_cache(root, cache)
    cache.clear()
    del value
    assert ref() is not None
    held = root.cache_hit.take()
    assert held is ref()
    del held
    assert ref() is None
    with pytest.raises(RuntimeError, match="optimize the plan again"):
        root.cache_hit.take()


def test_cached_none_is_a_hit_and_can_only_be_taken_once():
    root = KeyedOp("root")
    cache = ReuseCache()
    cache.put(graph_keys(root)[root], None)
    apply_reuse_cache(root, cache)
    assert root.cache_hit is not None and root.cache_hit.take() is None
    with pytest.raises(RuntimeError, match="already consumed"):
        root.cache_hit.take()


def test_eviction_during_lookup_leaves_the_computation_intact():
    class EvictOnGet(ReuseCache):
        def __contains__(self, key):
            raise AssertionError("planning must use one atomic lookup")

        def get(self, key, default=None):
            self.clear()
            return super().get(key, default)

    source = KeyedOp("source")
    root = KeyedOp("root", source)
    keys = graph_keys(root)
    cache = EvictOnGet()
    cache.put(keys[root], "result")
    apply_reuse_cache(root, cache)
    assert_graph(root, [source, root])
    assert not hits([source, root])
    assert root.cache_key == keys[root] and source.cache_key == keys[source]


def test_replanning_a_pruned_graph_is_rejected():
    root = KeyedOp("root")
    cache = ReuseCache()
    cache.put(graph_keys(root)[root], "result")
    apply_reuse_cache(root, cache)
    with pytest.raises(ValueError, match="unpruned graph"):
        apply_reuse_cache(root, cache)


def test_operand_validation_exempts_only_inputless_hits():
    root = KeyedOp("root")
    root.refs = [OperandRef(7)]
    with pytest.raises(ValueError, match="out of range"):
        validate_operands(root)
    root.cache_hit = CacheHit("key", None)
    validate_operands(root)
    root.inputs = [KeyedOp("source")]
    with pytest.raises(ValueError, match="must have no inputs"):
        validate_operands(root)


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": range(20), "b": range(20, 40), "y": range(40, 60)}).to_csv(
        path, index=False)
    return path


def features(path, factor=2):
    data = st.deferred(pd.read_csv)(str(path))
    return data.assign(c=data["a"] * factor + data["b"])


def model(path, alpha=1.0):
    data = features(path)
    X = data[["a", "b", "c"]].skb.mark_as_X()
    y = data["y"].skb.mark_as_y()
    return X.skb.apply(Ridge(alpha=alpha), y=y)


def plan_for(dag, cache=None, backend="pandas"):
    with st.config(validate_dag=True):
        return optimize(dag, OptConfig(selector=FlagBasedSelector(backend=backend)),
                        env=get_data(dag), reuse_cache=cache)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_cold_cache_flags_only_eligible_ops_before_the_split(csv_path, backend):
    cache = ReuseCache()
    plan, split_pos, flagged = plan_for(model(csv_path), cache, backend)
    assert split_pos is not None and not flagged and not hits(plan)
    assert cacheable_region(plan) == set(plan[:split_pos])
    keys = compute_lineage_keys(plan, cacheable_region(plan))
    expected = {op for op in plan if keys[op] is not None and not isinstance(op, ValueOp)}
    assert expected and expected == {op for op in plan if op.cache_key is not None}
    assert len(cache) == 0, "planning only flags future admissions"


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_optimizer_preserves_split_inputs_and_pins_hits(csv_path, backend):
    cache = ReuseCache()
    plan, split_pos, _ = plan_for(model(csv_path), cache, backend)
    x_op = plan[split_pos].inputs[0]
    cache.put(x_op.cache_key, "X-result")
    warm, split_pos, flagged = plan_for(model(csv_path), cache, backend)
    [hit] = hits(warm)
    split = warm[split_pos]
    assert type(hit) is type(x_op) and str(hit) == str(x_op)
    assert split.inputs[0] is hit and hit.outputs == [split] and hit.inputs == []
    assert split.inputs[1].cache_hit is None
    assert hit.cache_key is None and hit.cache_hit.key == x_op.cache_key
    assert hit in compute_pinned_ops(warm, split_pos, flagged)
    assert all(hit not in op.remove_after for op in warm)
    assert_graph(warm[-1], warm)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_a_whole_plan_hit_becomes_a_single_source(csv_path, backend):
    cache = ReuseCache()
    plan, _, _ = plan_for(features(csv_path), cache, backend)
    assert cacheable_region(plan) == set(plan)
    cache.put(plan[-1].cache_key, "features")
    warm, split_pos, flagged = plan_for(features(csv_path), cache, backend)
    assert len(warm) == 1 and split_pos is None and flagged == []
    assert warm[0].remove_after == [] and warm[0].cache_hit.take() == "features"


def test_changing_the_backend_or_configuration_invalidates_the_hit(csv_path):
    cache = ReuseCache()
    plan, _, _ = plan_for(features(csv_path), cache)
    cache.put(plan[-1].cache_key, "features")
    assert hits(plan_for(features(csv_path), cache)[0])
    assert not hits(plan_for(features(csv_path), cache, backend="polars")[0])
    assert not hits(plan_for(features(csv_path, factor=3), cache)[0])
    metadata = csv_path.stat()
    os.utime(csv_path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000_000))
    assert not hits(plan_for(features(csv_path), cache)[0])


def test_a_model_edit_keeps_reuse_before_the_split(csv_path):
    cache = ReuseCache()
    plan, split_pos, _ = plan_for(model(csv_path), cache)
    x_op = plan[split_pos].inputs[0]
    cache.put(x_op.cache_key, "X-result")
    edited, split_pos, _ = plan_for(model(csv_path, alpha=2.0), cache)
    assert edited[split_pos].inputs[0].cache_hit.key == x_op.cache_key
    assert all(op.cache_key is None and op.cache_hit is None for op in edited[split_pos:])


def test_without_an_explicit_cache_the_optimizer_does_not_plan_reuse(csv_path):
    plan, _, _ = plan_for(features(csv_path))
    assert all(op.cache_key is None and op.cache_hit is None for op in plan)
