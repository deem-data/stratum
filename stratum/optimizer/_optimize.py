from skrub._data_ops._evaluation import _Graph
from skrub._data_ops import DataOp
from skrub._data_ops._subsampling import SubsamplePreviews
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Any
from .logical._op_cse import apply_op_cse
from .logical._dataframe_ops import extract_dataframe_op, add_splitting_op
from .logical._numeric_ops import extract_numeric_op
from .logical._candidate_ops import CollectCandidatesOp, ScoreCandidatesOp
from .logical._ops import BaseEstimatorOp, ChoiceOp, Op, OperandRef, as_op
from .logical._split_ops import SplitOutput
from ._op_utils import clone_sub_dag, find_choice_naive, replace_op_in_outputs, show_graph, topological_iterator, validate_dag
from ._explain import explain_linear_plan
from .logical._algebraic_rewrites import algebraic_rewrites, AlgebraicRewritesConfig
from ._linearization import linearize_dag
from ._fit_pass_planning import mark_fit_dead_ops
from ._input_removal_planning import compute_pinned_ops, plan_input_removals
from .physical._plan_context import PlanContext
from .physical._lowering import lower_to_physical
from .physical._impl_selection import (ImplementationSelector, get_implementation_selector, select_implementations)
# Importing the physical exec modules and their lowering rules.
from .physical import _source_execs  # noqa: F401
from .physical import _transform_execs  # noqa: F401
from stratum.frontend._skrub_graph import build_graph
import logging
from stratum._config import FLAGS
from stratum.utils._utils import start_time, log_time

logger = logging.getLogger(__name__)


def topological_traverse(nodes, parents, children):
    """ Compute a topological order of the DAG in skrub IR. """
    # Compute in-degree (number of children for each node)
    indegree = {n: len(children.get(n, [])) for n in nodes}

    # Initialize queue with nodes having no children
    queue = deque([n for n, deg in indegree.items() if deg == 0])
    topo_order = []

    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for parent in parents.get(node, []):
            indegree[parent] -= 1
            if indegree[parent] == 0:
                queue.append(parent)

    return topo_order


@dataclass(frozen=True)
class SearchConfig:
    """What a plan needs at build time to score the candidates it evaluates.

    Present only when the plan is being built for a search; ``evaluate`` passes None and
    gets a plan that collects candidates without scoring them.
    """

    metric: Any
    return_predictions: bool = False


class OptConfig():
    # TODO we should move this class to the _config.py file
    def __init__(
        self,
        cse: bool = True,
        unroll_choices: bool = True,
        dataframe_ops: bool = True,
        numeric_ops: bool = True,
        algebraic_rewrites: bool = True,
        algebraic_rewrite_config: AlgebraicRewritesConfig | None = None,
    ):
        self.cse = cse
        self.dataframe_ops = dataframe_ops
        self.unroll_choices = unroll_choices
        self.numeric_ops = numeric_ops
        self.algebraic_rewrites = algebraic_rewrites
        if algebraic_rewrite_config is None:
            algebraic_rewrite_config = AlgebraicRewritesConfig()
        self.algebraic_rewrite_config = algebraic_rewrite_config

def _debug_show_graph(root: Op, name: str):
    if FLAGS.debug_graph:
        show_graph(root, name)

def _debug_explain_linear_plan(name: str, linearized_dag: list, split_pos: int | None):
    """Print the final executable plan (post implementation selection)."""
    if "physical_impl" in FLAGS.explain:
        explain_linear_plan(name, linearized_dag, split_pos)


def _debug_explain_dag(level: str, root: Op):
    """Print a topological plan for a non-linearized IR level (logical/physical).

    These levels have no split/CV structure yet, so the plan is a flat
    topological listing (``split_pos=None``)."""
    if level in FLAGS.explain:
        explain_linear_plan(level, list(topological_iterator(root)), split_pos=None)


def _debug_validate_dag(root: Op):
    """Assert every OperandRef indexes a valid input edge across the whole DAG.

    Gated by FLAGS.validate_dag so a rewrite that drops/reorders inputs without
    renumbering its operand refs fails loudly instead of silently miswiring."""
    if FLAGS.validate_dag:
        validate_dag(root)

def optimize(dag_root: DataOp, config: OptConfig = None, env: dict = None,
             search: SearchConfig | None = None):
    """Entry point for the optimizer. Runs the three planning phases and returns
    the linearized physical plan ``(linearized_dag, split_pos, flagged_ops)``.

    The steps are:

    1. :func:`logical_optimize` -- compile the Skrub DataOp DAG to the logical
       IR and run all backend-agnostic rewrites (extraction, CSE, choice
       unrolling, algebraic rewrites).
    2. :func:`~stratum.optimizer.physical._lowering.lower_to_physical` -- lower
       logical ops to physical ops (one logical op may become several).
    3. :func:`physical_optimize` -- select a concrete implementation per
       physical op, then linearize and plan intermediate last use as the final step.

    ``env`` (variable name -> value), when supplied, lets the converter resolve
    variables to compile-time constants (ValueOps) instead of VariableOps. ``search``,
    when supplied, makes this a search plan: it ends in a scoring operator rather than
    one that only labels its candidates."""
    start = start_time()
    if config is None:
        config = OptConfig()

    # Step 1: Convert the Skrub DataOp DAG to the logical IR and apply rewrites.
    root = logical_optimize(dag_root, config, env, search)
    _debug_explain_dag("logical", root)

    # Steps 2 & 3 read the config that drives operator selection once, here, so
    # execution carries no operator-selection control flow.
    ctx = PlanContext.from_flags()

    # Step 2: lower logical ops to physical ops.
    root = lower_to_physical(root, ctx)
    _debug_validate_dag(root)  # operand refs after lowering
    _debug_show_graph(root, "lowered")
    _debug_explain_dag("physical", root)

    # Step 3: physical optimization (implementation selection + linearization).
    # TODO: May need physical-level rewrites before or after operator selection
    result = physical_optimize(root, ctx)

    log_time("Optimization took in total", start)
    return result


def logical_optimize(dag_root: DataOp, config: OptConfig, env: dict = None,
                     search: SearchConfig | None = None) -> Op:
    """Step 1: build the logical IR and run all backend-agnostic rewrites.
    Returns the logical DAG root, ready to be lowered to physical ops."""

    # Convert to logical operator DAG
    root = convert_to_ops(dag_root, env)

    # Add splitting op
    root = add_splitting_op(root)
    _debug_validate_dag(root)  # operand refs as wired by as_op

    # Extract specialized operators from generic MethodCallOp / CallOp.
    if config.dataframe_ops:
        root = extract_frame_operators(root)
    if config.numeric_ops:
        root = extract_numeric_operators(root)

    # Apply CSE on the Op IR *after* extraction, so it can dedup whole specialized
    # ops (e.g. two identical mask SelectionOps). Running it earlier would merge a
    # mask's shared sub-expressions into a node with multiple consumers, which then
    # blocks selection folding.
    if FLAGS.cse:
        root = run_op_cse_pass(root)

    # Unrolling of choices to a dag with only a single ChoiceOp at the end
    if config.unroll_choices:
        root = choice_unrolling(root)

    # Final logical DAG
    if config.algebraic_rewrites:
        root = algebraic_rewrites(root, config.algebraic_rewrite_config)
        _debug_show_graph(root, "algebraic_rewrite")

    # Last, so every rewrite above sees the plan shape it was written against. A plan
    # whose choices were not unrolled is not an executable candidate set, so it gets no
    # terminating operator.
    if config.unroll_choices:
        root = install_candidate_set(root, search)

    _debug_validate_dag(root)  # operand refs after all logical rewrites, before lowering
    return root


def physical_optimize(root: Op, ctx: PlanContext, registry=None,
                      selector: ImplementationSelector | None = None):
    """Step 3: select concrete implementations, then linearize and plan removals.
    Implementation selection resolves each op with registered candidates to a
    concrete implementation (consulting the default PhysicalRegistry unless one
    is injected). Linearization and buffer-removal planning run last, as the
    final step of the whole pipeline."""

    # TODO: add physical rewrites here (or afterwards)
    if selector is None:
        selector = get_implementation_selector(ctx.implementation_selector)
    root = select_implementations(root, ctx, registry=registry, selector=selector)
    _debug_validate_dag(root)
    _debug_show_graph(root, "physical")
    # similarly we can do the things like parallelization planning here

    # Linearize and mark last used intermediates for removal
    linearized_dag, split_pos, flagged_ops = linearize_dag(root)
    pinned_ops = compute_pinned_ops(linearized_dag, split_pos, flagged_ops)
    plan_input_removals(linearized_dag, pinned_ops)
    mark_fit_dead_ops(linearized_dag, split_pos, flagged_ops)

    _debug_explain_linear_plan("physical_impl", linearized_dag, split_pos)
    return linearized_dag, split_pos, flagged_ops


def run_op_cse_pass(root: Op) -> Op:
    """Apply CSE on the Op IR (post-conversion) and return the deduplicated root."""
    start = start_time()
    root = apply_op_cse(root)
    log_time("Op CSE took", start)
    _debug_validate_dag(root)
    _debug_show_graph(root, "op_cse")
    return root


def extract_frame_operators(root):
    """ Rewrite the dataframe ops in the dag to the new dataframe ops."""
    start = start_time()
    for op in topological_iterator(root):
        root, _ = extract_dataframe_op(op, root, FLAGS.make_selection_op,
                                       FLAGS.make_map_op, FLAGS.make_column_projection)
    log_time("dataframe_rewrite took", start)
    _debug_show_graph(root, "frame_rewrite")
    return root


def extract_numeric_operators(root):
    """ Rewrite the dataframe ops in the dag to the new dataframe ops."""
    start = start_time()
    for op in topological_iterator(root):
        root, _ = extract_numeric_op(op, root)
    log_time("to_numeric took", start)
    _debug_show_graph(root, "numeric_rewrite")
    return root


def convert_to_ops(dag: DataOp, env: dict = None) -> Op:
    """Convert a Skrub DataOp DAG to stratum's logical IR (Op DAG).

    Single fused topological pass: ``as_op`` builds each op together with its
    de-duplicated ``inputs`` list, operand references, and output edges. Inputs
    are resolved through ``ids_to_ops`` (keyed by ``id(DataOp)``), which is
    guaranteed populated because we walk in topological, inputs-first order.
    """
    start = start_time()
    children, nodes, parents = get_dataops_graph(dag)
    order = topological_traverse(nodes, parents, children)
    root_id = order[-1]

    # id(DataOp) -> Op. Keyed by DataOp identity (not the graph's node keys) so
    # as_op's operand binder can resolve inputs found in the impl fields directly.
    ids_to_ops = {}
    for node_key in order:
        skrub_op = nodes[node_key]
        impl = skrub_op._skrub_impl
        if isinstance(impl, SubsamplePreviews):
            # Drop the preview node: route its single input straight to consumers.
            # Consumers reference this node's DataOp in their fields, so mapping it
            # to the input op makes them wire to the input op directly.
            input_key = children.get(node_key, [])[0]
            ids_to_ops[id(skrub_op)] = ids_to_ops[id(nodes[input_key])]
            continue
        ids_to_ops[id(skrub_op)] = as_op(skrub_op, ids_to_ops, env)

    root = ids_to_ops[id(nodes[root_id])]
    log_time("conversion took", start)
    _debug_show_graph(root, "conversion")
    return root


def install_candidate_set(root: Op, search: SearchConfig | None) -> Op:
    """Terminate the plan in a candidate-set operator.

    Unrolling leaves a `ChoiceOp` at the sink that no longer chooses anything, so it is
    replaced. A plan with no choice has one candidate and gets the same node appended: a
    set of one is still a set, and the runtime then has one shape to read rather than
    two. See `docs/adr/0003-scoring-is-a-plan-operator.md`.
    """
    start = start_time()
    fold = _split_outputs(root)
    if fold is None:
        # No split op, so the plan is not evaluated against folds and has no candidates
        # to set against each other. `Scheduler.evaluate` reports that itself.
        return root
    replacing = isinstance(root, ChoiceOp)
    names = root.make_outcome_names() if replacing else ["default"]
    candidates = list(root.inputs) if replacing else [root]

    if search is None or search.metric is None:
        node = CollectCandidatesOp(names)
    else:
        mode = _response_mode(search.metric, names, candidates)
        if search.return_predictions and mode != "predict":
            raise ValueError(
                f"return_predictions=True cannot be honoured with scoring="
                f"{search.metric.name!r}: that metric reads `{mode}`, so the values this"
                " plan produces are responses, not predictions. Ask for one or the other."
            )
        node = ScoreCandidatesOp(names, search.metric, response_mode=mode,
                                 emit_predictions=search.return_predictions)
    node.inputs = list(candidates)
    for candidate in candidates:
        if replacing:
            # The choice is the root, so it is the only consumer being redirected. A
            # rebuild rather than `replace_output` because one operator may fill two
            # candidate slots and therefore be redirected twice.
            candidate.outputs = [node if out is root else out for out in candidate.outputs]
        else:
            candidate.add_output(node)

    if isinstance(node, ScoreCandidatesOp):
        _, y_op = fold
        node.y_ref = OperandRef(node.add_input(y_op))
        y_op.add_output(node)
    log_time("installing the candidate set took", start)
    return node


def _response_mode(metric, names: list[str], candidates: list[Op]) -> str:
    """The response method the scoring pass runs, chosen once here.

    A metric declares what it can read, best first: `roc_auc` takes a decision function
    or probabilities. One pass has one mode, so the choice has to serve every candidate.
    `predict` always does, because a predict pass produces predictions whether or not a
    candidate ends in an estimator; anything else has to come from one.

    A metric no candidate can feed is refused here, before any data is touched, rather
    than at the first fold.
    """
    estimators = [_last_estimator(c) for c in candidates]
    for response in metric.response_method:
        if response == "predict" or all(e is not None and response in e.supported_modes
                                        for e in estimators):
            return response
    wanted = " or ".join(metric.response_method)
    cannot = [name for name, e in zip(names, estimators)
              if e is None or not set(metric.response_method) & e.supported_modes]
    raise ValueError(
        f"scoring={metric.name!r} reads {wanted}, which no single response provides for"
        f" every candidate in this plan: {', '.join(cannot)} cannot produce it. Use a"
        " metric that reads predictions, or a model that provides what this one needs."
    )


def _last_estimator(op: Op) -> BaseEstimatorOp | None:
    """The estimator nearest the end of a candidate's path, or None if it has none.

    The candidate's own tail may be post-processing; what decides which responses the
    plan can produce is the estimator feeding it, as it is the final `Apply` in skrub.
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


def get_dataops_graph(dag: DataOp) -> tuple[dict, dict, dict]:
    start = start_time()
    g = build_graph(dag)
    nodes = g["nodes"]
    parents = g["parents"]
    children = g["children"]
    log_time("Conversion dag took", start)
    return children, nodes, parents


def choice_unrolling(root: Op):
    """ Rewrite for unrolling the dag after choice op into separate dags for each outcome."""
    start = start_time()
    contains_choice = True
    while contains_choice:
        dag_iter = topological_iterator(root)
        contains_choice = False
        for op in dag_iter:
            if op.is_choice():
                outcomes = op.inputs

                # check if we find any choice in the sub-dag of the current choice
                last_op, is_choice = find_choice_naive(op)
                if last_op is op:
                    # the choice has no consumers left: unrolling is finished
                    contains_choice = False
                    break
                if is_choice:
                    unroll_nested_choice(last_op, op, outcomes)
                    contains_choice = True
                else:
                    assert root is last_op, "Root should be the last op in the dag"
                    # we reached the end of the dag
                    logger.debug(f"Unrolling simple choice: {op}")
                    root = unroll_simple_choice(root, op, outcomes)
                    logger.debug(f"New root after unrolling: {root}")

                del op
                break
    log_time("unrolled took", start)
    _debug_show_graph(root, "unrolled")
    return root



def unroll_simple_choice(root: Op, op: ChoiceOp, outcomes: list) -> Op:
    """ Unroll a simple choice op, which has no choice in the sub-dag."""
    dag_root = ChoiceOp(outcome_names=op.outcome_names, append_choice_name=False)
    dag_root.inputs = [root]

    # clones sub-dag after choice op for all outcomes[1:]
    for outcome in outcomes[1:]:
        outcome.outputs = []
        leafs = clone_sub_dag(op, new_root_op=outcome)
        assert len(leafs) == 1
        dag_root.add_input(leafs[0])
        leafs[0].add_output(dag_root)

    # reuse sub-dag for the first outcome
    outcomes[0].outputs = []
    replace_op_in_outputs(op, replacement=outcomes[0])
    root.add_output(dag_root)
    return dag_root


def unroll_nested_choice(last_op: ChoiceOp, op: ChoiceOp, outcomes):
    """ Unroll a nested choice op, which has choice in the sub-dag."""
    n_outcomes = len(last_op.outcome_names)

    # clone the sub-dag for each outcome of the current choice
    for outcome, outcome_name in zip(outcomes[1:], op.outcome_names[1:]):
        outcome.outputs = []
        clone_sub_dag(op, new_root_op=outcome, stop_at_op=last_op)
        for i in range(n_outcomes):
            last_op.outcome_names.append(last_op.outcome_names[i] + outcome_name)

    # reuse sub-dag for the first outcome
    outcomes[0].outputs = [op.outputs[0]]
    for i in range(n_outcomes):
        last_op.outcome_names[i] += op.outcome_names[0]
    outcomes[0].outputs = []
    replace_op_in_outputs(op, replacement=outcomes[0])
