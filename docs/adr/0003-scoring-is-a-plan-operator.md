# Scoring is a plan operator

A pipeline search ends in a `ScoreCandidates` operator: the plan fans in every
candidate's output, scores each against the test fold, and emits one row per candidate.
The scheduler loops folds and averages the rows. It decides neither what a candidate is
nor how a score is computed.

The reason is node identity. Choice unrolling already knows which sub-DAG produced which
candidate, and `install_candidate_set` carries that knowledge into the plan as input
edges, where the mapping cannot drift and the fold's labels have a declared lifetime.

## Considered options

**Keep scoring in the scheduler.** What Stratum did. Rejected: never given the mapping,
the scheduler rebuilt it by pattern-matching the last operator of the linearized plan and
zipping it positionally against the labels, which is correct only as long as two lists
that nothing ties together stay in the same order. It reached the fold's labels by
pinning a buffer removal planning could not see, so the pool under-counted live memory.

**Let the scheduler inject the metric before execution.** Rejected: plan mutation at run
time, which ADR 0001 and the `PlanContext` snapshot both exist to prevent, and an invalid
`scoring=` stays hidden until the first fold has been fitted.

**Swap the unrolled plan's `ChoiceOp` sink during lowering.** The smaller change, rejected
because the node is not a choice: after unrolling it selects nothing, it computes every
input and labels the results.

## Consequences

The metric resolves at plan time, which makes a pass's **mode** the response method that
metric reads (`predict`, `predict_proba`, `decision_function`), as it already is in
skrub. So those keyword-argument groups on `.skb.apply()` become executable, `score`
stays rejected for the reason in ADR 0004, a metric no single response serves for every
candidate is refused when the plan is built, and `return_predictions=True` is refused
alongside a non-`predict` response, whose values are responses rather than predictions.

`evaluate` and `grid_search` no longer share a plan. Both end in the same fan-in family,
`CollectCandidates` for `evaluate` and `ScoreCandidates` for a search, and a plan with no
split operator gets neither, because it is not evaluated fold by fold.

What the operator is handed in place of an estimator is ADR 0005.
