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
from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer._optimize import OptConfig, SearchConfig, compile_plan, pinned_before_split
from stratum.optimizer.logical._scoring import resolve_scoring
from stratum.optimizer.logical._split_ops import BoundaryOp
from stratum.optimizer.physical._conversion import ConversionOp
from stratum.optimizer.physical._generic_execs import CallExec
from stratum.optimizer.physical._impl_selection import (
    DefaultImplementationSelector, FlagBasedSelector, ImplementationSelector)
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

    def recording(indices, profile=None):
        out = run_fold(indices, profile)
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


class Recording(ImplementationSelector):
    """Chooses as ``chooser`` does, and notes what the profile holds of each op."""

    def __init__(self, chooser):
        self.chooser = chooser
        self.seen = {}

    def choose(self, op, candidates, ctx):
        if ctx.profile is not None:
            self.seen[str(op)] = ctx.profile.of(op)
        return self.chooser.choose(op, candidates, ctx)


def run_search(plan, cv=KFold(3), recompile_with=None):
    """Runs ``plan``'s search; returns the scheduler and each compile's profile.
    ``recompile_with`` is the selector the compile after the first fold uses."""
    profiles = []

    def compile_folds(profile=None, current=None):
        profiles.append(profile)
        if profile is not None and recompile_with is not None:
            plan.selector = recompile_with
        return plan.compile_folds(profile, current)

    sched = SequentialScheduler(plan.linearized_dag, plan.split_pos, compile_folds=compile_folds)
    sched.grid_search(cv)
    return sched, profiles


def test_the_plan_is_compiled_again_from_what_the_first_fold_measured(frame):
    selector = Recording(DefaultImplementationSelector())
    plan = compile_plan(search_pipeline(frame), OptConfig(selector=selector),
                        search=SearchConfig(metric=resolve_scoring("r2")))
    with capture_plans() as plans:
        sched, profiles = run_search(plan)
    # At the split, and after the first of three folds only.
    assert profiles[0] is None and len(profiles) == 2
    profile = profiles[1]
    template_ops = set(topological_iterator(plan.template.root))
    assert profile.ops and set(profile.ops) <= template_ops
    models = {stage: [m for op, m in profile.ops.items()
                      if type(op).__name__ == "PredictorOp" and op.stage == stage]
              for stage in ("fit", "predict")}
    assert len(models["fit"]) == len(models["predict"]) == 2
    # Fitted for its state alone, then applied.
    assert all(m.seconds > 0 and m.state_size > 0 and m.size == 0 for m in models["fit"])
    assert all(m.size > 0 and m.state_size == 0 for m in models["predict"])
    assert profile.peak > 0
    # The selector reads it by the op it is choosing for.
    assert selector.seen["RidgeOp_fit"] in models["fit"]
    # Deciding as before, the folds keep their plan: one fold plan is reported.
    assert [p["name"] for p in plans].count("fold_impl") == 1
    assert sched.fold_plan.ops == sched.linearized_dag[plan.split_pos:]


def test_a_single_fold_is_not_measured(frame):
    plan = compiled(search_pipeline(frame))
    _, profiles = run_search(plan, cv=[(np.arange(40), np.arange(40, 60))])
    assert profiles == [None]


def test_a_recompile_reads_what_the_plan_before_the_split_kept(frame):
    class PandasMerge(ImplementationSelector):
        """Polars, except the merge."""
        def choose(self, op, candidates, ctx):
            backend = "pandas" if "Join" in type(op).__name__ else "polars"
            return FlagBasedSelector(backend).choose(op, candidates, ctx)

    lookup = pd.DataFrame({"k": np.arange(6), "c": np.arange(6) * 1.5})

    def dag():
        data = st.as_data_op(frame.assign(k=np.arange(60) % 6))
        X, y = data[["k", "a"]].skb.mark_as_X(), data["y"].skb.mark_as_y()
        return X.merge(st.as_data_op(lookup)[["k", "c"]], on="k", how="left").skb.apply(Ridge(), y=y)

    def plan():
        return compile_plan(dag(), OptConfig(selector=PandasMerge()),
                            search=SearchConfig(metric=resolve_scoring("r2")))

    first = plan()
    before = first.linearized_dag[:first.split_pos]
    kept = pinned_before_split(before)
    # The merge reads the lookup as pandas: the polars one is freed before the split.
    lookup_op = next(op for op in before if str(op) == "PolarsColumnProjectionOp(cols[['k', 'c']])")
    assert lookup_op not in kept
    with capture_plans() as plans:
        sched, _ = run_search(first, recompile_with=FlagBasedSelector("polars"))
    fold = sched.fold_plan.ops
    assert [p["name"] for p in plans].count("fold_impl") == 2
    assert {type(op).__name__ for op in fold if "Join" in type(op).__name__} == {"PolarsJoinOp"}
    # Read through the conversion it kept, converted back.
    back = [op for op in fold if isinstance(op, ConversionOp) and op.inputs[0] in before]
    assert len(back) == 1 and back[0].inputs[0] in kept and back[0].inputs[0].inputs == [lookup_op]
    assert all(in_ in kept for op in fold for in_ in op.inputs if in_ not in fold)
    unchanged, _ = run_search(plan())
    assert sched.results_.equals(unchanged.results_)
    assert sched.pool.active_count == 0 and not sched.state_store
