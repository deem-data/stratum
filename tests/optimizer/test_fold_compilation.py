"""Staged compilation: the plan after the split, compiled at the split and copied per fold."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import stratum as st
from stratum._api import grid_search
from stratum._profiler._diagnostics import capture_plans
from stratum._profiler._observer import observe
from stratum.optimizer._optimize import OptConfig, SearchConfig, compile_plan
from stratum.optimizer.logical._scoring import resolve_scoring
from stratum.optimizer.logical._split_ops import BoundaryOp
from stratum.optimizer.physical._conversion import ConversionOp
from stratum.optimizer.physical._generic_execs import CallExec
from stratum.optimizer.physical._impl_selection import FlagBasedSelector
from stratum.runtime._scheduler import SequentialScheduler


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    df["y"] = 2 * df["a"] - df["b"] + rng.normal(scale=.1, size=60)
    return df


def search_pipeline(frame):
    data = st.as_data_op(frame)
    X, y = data[["a", "b"]].skb.mark_as_X(), data["y"].skb.mark_as_y()
    alpha = st.choose_from([0.1, 10.0], name="alpha")
    return X.skb.apply(StandardScaler()).skb.apply(Ridge(alpha=alpha), y=y)


def compiled(dag):
    return compile_plan(dag, search=SearchConfig(metric=resolve_scoring("r2")))


def shape(ops):
    return [(type(op).__name__, str(op), op.stage, op.mode) for op in ops]


def test_the_template_reads_the_plan_before_the_split_through_boundaries(frame):
    plan = compiled(search_pipeline(frame))
    before = set(plan.linearized_dag[:plan.split_pos])
    assert plan.template is not None and plan.template.boundaries
    assert all(isinstance(b, BoundaryOp) and not b.inputs for b in plan.template.boundaries)
    # Bound to the physical ops that ran: the split's X and y, at least.
    assert plan.template.sources() <= before
    split = plan.linearized_dag[plan.split_pos]
    assert set(split.inputs) <= plan.template.sources()


def test_compiling_at_the_split_decides_what_the_initial_compile_did(frame):
    plan = compiled(search_pipeline(frame))
    fold = plan.compile_folds()
    initial = plan.linearized_dag[plan.split_pos:]
    assert sorted(shape(fold.ops)) == sorted(shape(initial))
    assert not set(fold.ops) & set(initial), "new ops, compiled from the template"
    assert fold.ops[0].is_split_op
    before = set(plan.linearized_dag[:plan.split_pos])
    members = set(fold.ops)
    assert all(in_ in members or in_ in before for op in fold.ops for in_ in op.inputs)
    # The template stays as it was, to be compiled again.
    assert all(isinstance(b, BoundaryOp) for b in plan.template.boundaries)
    assert sorted(shape(plan.compile_folds().ops)) == sorted(shape(fold.ops))


def test_each_fold_runs_a_copy_of_the_fold_plan(frame):
    plan = compiled(search_pipeline(frame))
    sched = SequentialScheduler(plan.linearized_dag, plan.split_pos,
                                compile_folds=plan.compile_folds)
    timings = observe(sched).node_timings
    copies = []
    run_fold = sched.run_fold

    def recording(indices):
        out = run_fold(indices)
        copies.append(sched.fold_ops)
        return out

    sched.run_fold = recording
    sched.grid_search(KFold(3))

    fold_plan = sched.fold_plan.ops
    assert sched.linearized_dag[plan.split_pos:] == fold_plan
    assert len(copies) == 3
    for ops in copies:
        assert [op.instance_of for op in ops] == fold_plan
        assert ops[0].indices is None, "a fold's rows are not kept after it ran"
    assert not set(copies[0]) & set(copies[1])
    # Profiled as the op each copies: one entry per fold plan op, run in every fold.
    for op in fold_plan:
        assert {fold for _, fold in timings[op].phases} == {0, 1, 2}
    assert not sched.state_store
    assert sched.pool.active_count == 0


def test_a_search_scores_as_before(frame):
    ours = grid_search(search_pipeline(frame), cv=KFold(3), scoring="r2").results_
    theirs = search_pipeline(frame).skb.make_grid_search(
        fitted=True, cv=KFold(3), scoring="r2").results_
    np.testing.assert_allclose(sorted(ours["scores"]), sorted(theirs["mean_test_score"]))


def test_the_run_reports_the_plan_compiled_at_the_split(frame):
    with capture_plans() as plans:
        sched = grid_search(search_pipeline(frame), cv=KFold(2), scoring="r2")
    assert [p["name"] for p in plans][-2:] == ["physical_impl", "fold_impl"]
    fold_impl = plans[-1]
    assert fold_impl["split_pos"] == sched.pos_split_op
    assert [n["label"] for n in fold_impl["nodes"]] == [str(op) for op in sched.linearized_dag]


def test_a_plan_without_a_split_has_no_template(frame):
    plan = compile_plan(st.as_data_op(frame)[["a"]].skb.apply(StandardScaler()))
    assert plan.split_pos is None and plan.template is None


def test_a_conversion_made_before_the_split_is_not_repeated_per_fold(frame):
    data = st.as_data_op(frame)
    X, y = data[["a", "b"]].skb.mark_as_X(), data["y"].skb.mark_as_y()
    # An opaque call reads pandas, and `extra` is computed on polars before the split.
    extra = data[["b"]]
    dag = X.skb.apply_func(lambda x, e: x.assign(c=len(e)), extra).skb.apply(Ridge(), y=y)
    plan = compile_plan(dag, OptConfig(selector=FlagBasedSelector(backend="polars")))
    before = plan.linearized_dag[:plan.split_pos]
    fold = plan.compile_folds()
    assert not any(isinstance(op, ConversionOp) for op in fold.ops)
    calls = [op for op in fold.ops if isinstance(op, CallExec)]
    assert len(calls) == 2
    for call in calls:
        conversion = call.inputs[1]
        assert isinstance(conversion, ConversionOp) and conversion in before
