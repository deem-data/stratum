"""Construct the candidate collection or scoring operator for a logical plan."""
from __future__ import annotations

from typing import TYPE_CHECKING

from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer.logical._candidate_ops import (
    CollectCandidatesOp, DeclaredScoringOp, ScoreCandidatesOp,
)
from stratum.optimizer.logical._ops import (
    BaseEstimatorOp, ChoiceOp, Op, OperandRef, remap_operand_refs,
)
from stratum.optimizer.logical._split_ops import SplitOutput
from stratum.utils._utils import start_time, log_time

if TYPE_CHECKING:
    from stratum.optimizer._optimize import SearchConfig


def install_candidate_set(root: Op, search: SearchConfig | None) -> Op:
    """Terminate the plan in a candidate-set operator.

    Replace the sink `ChoiceOp` left by unrolling with explicit candidate inputs
    and labels. For a plan without choices, append a candidate-set operator with
    one input so the runtime uses the same result representation.
    """
    start = start_time()
    fold = _split_outputs(root)
    if fold is None:
        # Without a split, evaluate returns the final operator result;
        # grid_search requires X/y markers.
        return root
    replacing = isinstance(root, ChoiceOp)
    names = root.make_outcome_names() if replacing else ["default"]
    candidates = list(root.inputs) if replacing else [root]
    # Postprocessing may follow the declaration of the plan's shared metric.
    # Unrolling can clone that declaration, so retain its kwargs bindings for
    # each candidate while still scoring the candidate's final output.
    declarations = _scoring_declarations(root)
    declared = [declarations[c] for c in candidates]

    if search is None or search.metric is None:
        node = CollectCandidatesOp(names)
    else:
        method = _prediction_method(search.metric, names, candidates)
        if search.return_predictions and method != "predict":
            raise ValueError(
                "return_predictions=True requires `predict` outputs, but scoring="
                f"{search.metric.name!r} requires `{method}`. Request scores without"
                " predictions, or use a metric that reads `predict` outputs."
            )
        node = ScoreCandidatesOp(names, search.metric, prediction_method=method,
                                 emit_predictions=search.return_predictions)
    node.inputs = list(candidates)
    for candidate in candidates:
        if replacing:
            # The choice is the root, so it is the only consumer being redirected. A
            # rebuild rather than `replace_output` because one operator may fill two
            # candidate slots and therefore be redirected twice.
            candidate.outputs = [node if out is root else out
                                 for out in candidate.outputs]
        else:
            candidate.add_output(node)

    if isinstance(node, ScoreCandidatesOp):
        _, y_op = fold
        node.y_ref = OperandRef(node.add_input(y_op))
        y_op.add_output(node)
        if search.declared_kwargs:
            if any(d is None for d in declared):
                raise ValueError(
                    "not every candidate contains `.skb.with_scoring()`, so the declared"
                    " scorer cannot score them all. Pass `scoring=` instead.")
            node.kwargs_refs = [_rebind(d.scorers[0]["kwargs"], d.inputs, node)
                                for d in declared]
    for d in dict.fromkeys(declared):
        if d is not None:
            # Bypass the metadata wherever it occurs, keeping postprocessing and
            # the scorer's kwargs alive before pruning the declaration.
            pred = d.inputs[0]
            for out in d.outputs:
                while any(in_ is d for in_ in out.inputs):
                    out.replace_input(d, pred)
                # Positional candidate slots may now share one producer. Keep
                # one reverse edge per slot for the topological indegree count.
                pred.outputs = [o for o in pred.outputs if o is not out]
                pred.outputs.extend(out for in_ in out.inputs if in_ is pred)
            d.detach_and_prune()
    log_time("installing the candidate set took", start)
    return node


def _scoring_declarations(root: Op) -> dict[Op, DeclaredScoringOp | None]:
    """Propagate the shared scoring declaration through downstream operators.

    Walk the graph once, including when several candidates share its upstream
    nodes. The metric itself is already resolved in SearchConfig.
    """
    declarations = {}
    for op in topological_iterator(root):
        declarations[op] = op if isinstance(op, DeclaredScoringOp) else next(
            (declarations[in_] for in_ in op.inputs if declarations[in_] is not None), None)
    return declarations


def _rebind(value, operands: list[Op], consumer: Op):
    """``value`` with its OperandRefs into ``operands`` re-pointed at inputs of ``consumer``."""
    mapping = {}
    for k in _referenced(value):
        if k not in mapping:
            mapping[k] = consumer.add_input(operands[k])
            operands[k].add_output(consumer)
    return remap_operand_refs(value, mapping)


def _referenced(value):
    """The operand indices ``value`` refers to, through nested tuples, lists and dicts."""
    if isinstance(value, OperandRef):
        yield value.k
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _referenced(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _referenced(v)


def _prediction_method(metric, names: list[str], candidates: list[Op]) -> str:
    """Select a prediction method supported by every candidate in the scoring pass.

    - Try the metric's prediction methods in preference order. For example,
      `roc_auc` accepts decision scores or probabilities.
    - Accept `predict` for any candidate, including a precomputed prediction DAG.
      Other methods require a compatible estimator in each candidate.
    - Raise when no common prediction method is available, before executing folds.
    """
    estimators = [_last_estimator(c) for c in candidates]
    for method in metric.prediction_methods:
        if method == "predict" or all(
                e is not None and method in e.supported_prediction_methods for e in estimators):
            return method
    wanted = " or ".join(metric.prediction_methods)
    cannot = [name for name, e in zip(names, estimators)
              if e is None or not set(metric.prediction_methods) & e.supported_prediction_methods]
    raise ValueError(
        f"scoring={metric.name!r} reads {wanted}, but no single prediction method is"
        f" supported by every candidate in this plan: {', '.join(cannot)} cannot provide it."
        " Use a metric that reads predictions, or a model that provides what this one needs."
    )


def _last_estimator(op: Op) -> BaseEstimatorOp | None:
    """The estimator nearest the end of a candidate's path, or None if it has none.

    Search input edges breadth-first, allowing for post-processing after an estimator.
    """
    seen, queue = set(), [op]
    while queue:
        node = queue.pop(0)
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, BaseEstimatorOp):
            return node
        queue.extend(node.inputs)
    return None


def _split_outputs(root: Op) -> tuple[Op, Op] | None:
    """The plan's X and y split outputs, or None when the plan has no split op."""
    for op in topological_iterator(root):
        if not op.is_split_op:
            continue
        x = next((o for o in op.outputs if isinstance(o, SplitOutput) and o.is_x), None)
        y = next((o for o in op.outputs if isinstance(o, SplitOutput) and not o.is_x), None)
        assert x is not None and y is not None, f"split op {op} without both outputs"
        return x, y
    return None
