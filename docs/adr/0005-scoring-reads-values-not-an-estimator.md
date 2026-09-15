# Scoring reads the values a plan produced, not an estimator

`scoring=` resolves to a `Metric`: a function, the response method it reads, its keyword
arguments, and its sign. Scoring one candidate on one fold is then
`sign * fn(y_true, values)`. Stratum never calls a scikit-learn scorer.

A scorer needs an estimator object to interrogate before it will use anything that
estimator produced: `__sklearn_tags__` for whether this is a classifier at all,
`classes_` for which probability column is the positive class. A candidate is a sub-DAG
of a merged plan, so there is nothing to hand over. The plan already produces the values
a metric reads (ADR 0003), so taking scikit-learn's scoring API as a vocabulary rather
than an implementation removes the problem instead of managing it. How a name resolves is
the module docstring of `stratum/optimizer/ir/_scoring.py`.

## Considered options

**Call the scorer, with a stand-in in the estimator slot.** What Stratum did. Rejected:
the stand-in answers from the estimator fitted by the candidate's last step, so it
fabricates claims about what a candidate *is*, keeps that estimator and its predict-time
features alive past the point the plan called them dead, and has no honest answer when
the last step is not an estimator.

**Reimplement every metric.** Rejected as a blanket policy. A metric is 0.05 to 0.2
percent of a fold on realistic data, so there is no performance argument, and a
hand-written `roc_auc` differing from scikit-learn's in the fourth decimal is the worst
bug class in this codebase. Metrics move into `_NATIVE` one at a time, where there is a
reason, each pinned against scikit-learn's by a test.

**Read the metric off the scorer at every call.** Rejected: those fields are scikit-learn
internals. Read once, when the plan is built, in one function, a version bump fails there
rather than at every fold.

## Consequences

`_decompose` reads `_score_func`, `_response_method`, `_kwargs` and `_sign`. It is the
only place a scikit-learn version bump can invalidate scoring.

A callable `scoring=` is a metric, `f(y_true, y_pred)`, not a scorer. `make_scorer(...)`
is accepted and decomposed, which is how a caller asks for keyword arguments or for a
response other than `predict`, and a `scorer(estimator, X, y)` is refused with a reason,
because there is no estimator to pass it. A scorer reading the fold's raw `X` cannot be
expressed at all: the operator takes the candidates' values and the fold's labels, and no
`X` edge.
