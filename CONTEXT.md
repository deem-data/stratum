# Stratum

Stratum executes large-scale agentic pipeline search: it takes the ML pipelines an MLE
agent produces, represents them as one lazily evaluated plan, optimizes that plan, and
executes it across heterogeneous backends.

This file is the project's glossary and nothing else. It has two tiers: the research
framing that the paper and README speak in, and the system internals that the code
speaks in. The seam between them is stated explicitly, because the two vocabularies are
otherwise disjoint.

## Tier 1: research framing

**MLE agent**:
An external LLM agent that authors ML pipelines. Not part of Stratum; Stratum is what
evaluates its output.
_Avoid_: AutoML system (AutoML is the class of method Stratum's approach contrasts with)

**Pipeline**:
One end-to-end ML program authored by an MLE agent, written in skrub DataOps syntax.
_Avoid_: script, workflow, model, experiment

**Trajectory**:
The search path one MLE agent produces: the ordered sequence of pipelines it explored
and evaluated over its iterative search.
_Avoid_: run, history, episode

**Batch**:
A set of pipeline candidates submitted for evaluation together, drawn from one or more
trajectories.

**Pipeline search**:
Evaluating a batch or trajectory of pipelines to identify the best one.
_Avoid_: hyperparameter search (that is one choice inside a single pipeline, not the
search over pipelines)

**Merging** (also **consolidation**):
Combining a batch of pipeline candidates into a single plan containing choices, so
shared work is executed once. This is multi-query optimization applied to pipelines.
_Avoid_: fusing, fusion. Pipeline merging has historically been called fusion in this
project and in conversation; the word is now reserved for operator fusion (tier 2).

## Tier 2: system internals

### The seam

A **pipeline** (tier 1) is a **candidate** (tier 2). Merging a batch of candidates
produces one plan whose branch points are **choices**; **choice unrolling** expands
those choices back into one sub-DAG per candidate, and **CSE** collapses the work they
share. A candidate is therefore a sub-DAG of a merged plan, never an estimator object.

### Plan and structure

**Plan**:
The optimizer's output artifact, the thing that gets executed. Exists at three levels:
`logical` (after logical rewrites), `physical` (after lowering), and `physical_impl`
(after implementation selection, i.e. the executable plan).
_Avoid_: DAG or graph when the artifact, not its shape, is meant

**DAG**:
The graph structure of a plan. Use this word when edges, topological order, reachability
or cycles are the point.
_Avoid_: graph, tree

**Graph**:
Reserved for visualization (rendering a plan for inspection) and for generic
graph algorithms. Never a synonym for plan or DAG.

**Linearization**:
Flattening a physical plan's DAG into an execution order. Its result is a list, so it is
a linearized plan, not a linearized DAG.

### Operators

**Logical operator** (**op**):
A node saying *what* to compute, independent of backend. Named `<Family>Op`, subclass of
`Op`, which subclasses `IRNode`. The `Op` suffix means logical and nothing else.

**Physical operator**:
A node in the plan after lowering, saying *how* to compute. Abstract physical operators
are named `Physical<Family>` and carry configuration but cannot run; concrete ones are
named `<Backend><Family>` and run on exactly one backend.
_Avoid_: exec, executor. An `Exec` suffix previously marked abstract physical operators,
which are precisely the ones that cannot execute.

**Lowering**:
Translating logical operators into physical ones. One logical operator may lower to
several physical operators.

**Implementation** (**impl**):
A concrete physical operator registered as able to run a given abstract one.
`PhysicalImpl` is its registry record. The short form `impl` refers only to this;
`skrub_impl` is upstream skrub's own term for a `DataOpImpl` and is unrelated.

**Implementation selection**:
The pass that replaces each abstract physical operator with a concrete implementation,
before execution.

**Opaque operator**:
A skrub `DataOpImpl` that Stratum has no native logical operator for. The optimizer
cannot see inside it, so no rewrite and no CSE applies; it executes by delegating back
to skrub.

**Adapter**:
A scikit-learn or skrub compatible estimator subclass under `stratum/adapters/`. An
adapter is an estimator; a physical operator is a plan node. They are different kinds of
thing and must not share a name.

**Backend**:
The substrate a concrete implementation runs on: pandas, polars, rust, numpy, or
sklearn/skrub.

**Operand**:
A positional reference (`OperandRef`) into an operator's `inputs` list, identifying
which input fills a given slot.

### Transformations

**Rewrite**:
A semantics-preserving transformation of the logical plan. A rewrite that changes results
is the worst class of bug in this codebase.

**Fusion**:
Collapsing multiple operations into a single kernel or pass. Covers the vertical case (a
sequence of logical operators becomes one kernel) and the horizontal case (a per-column
apply loop becomes one pass).
_Avoid_: merging, consolidation (both reserved for pipeline batches, tier 1)

**CSE** (common subexpression elimination):
Deduplicating structurally identical operators so shared work is computed once. The
mechanism that makes merging a batch cheaper than evaluating its candidates separately.

### Execution

**Choice**:
A branch point in a plan (`ChoiceOp`) with named alternatives. Present only before choice
unrolling; an unrolled plan contains none.

**Outcome**:
One alternative of a choice.

**Candidate**:
One pipeline as a sub-DAG of a merged plan. See the seam above.

**Candidate set**:
The fan-in over every candidate's output that terminates an unrolled plan, carrying one
label per candidate. It stands where the choice that unrolling consumed used to be.
_Avoid_: final choice (an unrolled plan has no choices left to be final)

**Fold**:
One train/test split of a cross-validation.

**Mode**:
Which method one pass over a plan calls on its estimators: a fitting mode, or the response
method a scorer asked for. An operator that cannot serve the mode falls back to
`transform`.

**Fit-dead**:
An operator whose output nothing in the fitting pass needs: it fits nothing itself, and
every consumer is fit-dead too. Skipped entirely while fitting.

**Response**:
What an estimator emits for a scorer to consume: the output of `predict`, `predict_proba`
or `decision_function`. The **response method** is which of those the scorer asked for.
_Avoid_: prediction, when a response other than `predict` is meant

**Scorer**:
A callable obeying scikit-learn's `scorer(estimator, X, y)` protocol. The scorer, not its
caller, decides the metric, the arguments it is given and the sign of the result.
_Avoid_: metric. A metric is `f(y_true, y_pred)`; a scorer is what wraps one.

**Score**:
The number a scorer returns for one candidate on one fold. Always a utility: greater is
better, whatever the metric.
_Avoid_: loss, error

**Buffer pool**:
The runtime's store for plan intermediates, with size accounting and eviction, keyed on
operator node identity.
