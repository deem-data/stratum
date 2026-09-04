# A pipeline search always names its metric

`stratum._api.grid_search`, Stratum's own entry point, refuses `scoring=None` and raises.
The skrub-compatibility patch on `SkrubNamespace.make_grid_search` does not: with no
metric named it hands the call to unpatched skrub, so `import stratum as skrub` keeps
behaving exactly as skrub does.

`scoring=None` means "score each estimator with its own `score` method". For a single
estimator that is merely a default. For a pipeline search it is a correctness bug,
because the candidates in one batch need not be the same kind of estimator: a batch
choosing between a classifier and a regressor reports accuracy for one and R² for the
other, then ranks the two by raw value, and nothing in the results table records that
they do not measure the same thing. Ranking candidates is the entire output of a search.

It also removes the only reason scoring ever had to ask a candidate for its own `score`,
which is what made a candidate have to look like an estimator at all. What it is asked
for instead is ADR 0005.

## Considered options

**Support `scoring=None` as sklearn and skrub do.** Rejected: a per-estimator default is
coherent where one estimator is scored, but Stratum ranks a set, and a ranking over two
different metrics is not a ranking.

**Fall back to a fixed default metric.** Rejected: it is the bug this work removed.
Substituting a metric the caller did not ask for produces a plausible-looking number that
answers a different question.

**Refuse in the compatibility patch too.** Rejected because it breaks ADR 0002. A bare
`pipeline.skb.make_grid_search()` works under real skrub whenever the final estimator has
a `score` method, and agent-generated pipelines are exactly the code that would hit it.

## Consequences

The two entry points disagree on purpose: the same plan raises through
`stratum._api.grid_search` and runs, on skrub and therefore slowly, through
`.skb.make_grid_search()`. That fallback is silent, which is its price: a search naming
no metric loses the acceleration with nothing said about it, and a test meaning to
exercise Stratum has to name a metric to stay on that path.

A plan whose final estimator has no `score` method now needs an explicit metric, which
plain skrub needs too, raising `TypeError` for the same case. `score_kwargs` on
`.skb.apply()` stays rejected: it is the one method group Stratum will not gain, because
nothing here calls an estimator's `score`.
