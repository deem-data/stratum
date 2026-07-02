from skrub._data_ops._evaluation import _Graph
from skrub._data_ops import DataOp
from skrub._data_ops._subsampling import SubsamplePreviews
from collections import deque, defaultdict
from ._op_cse import apply_op_cse
from .ir._dataframe_ops import extract_dataframe_op, add_splitting_op
from .ir._numeric_ops import extract_numeric_op
from .ir._ops import ChoiceOp, Op, SearchEvalOp, as_op
from ._op_utils import clone_sub_dag, find_choice_naive, replace_op_in_outputs, show_graph, topological_iterator, validate_dag
from ._explain import explain_linear_plan
from ._algebraic_rewrites import algebraic_rewrites, AlgebraicRewritesConfig
from ._linearization import linearize_dag
from ._input_removal_planning import compute_pinned_ops, plan_input_removals
from stratum.utils._skrub_graph import build_graph
import logging
from stratum._config import FLAGS
from stratum.utils._utils import start_time, log_time

logger = logging.getLogger(__name__)
EVAL_OP_ENABLED = False


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
    if FLAGS.explain_linear_plan:
        explain_linear_plan(name, linearized_dag, split_pos)


def _debug_validate_dag(root: Op):
    """Assert every OperandRef indexes a valid input edge across the whole DAG.

    Gated by FLAGS.validate_dag so a rewrite that drops/reorders inputs without
    renumbering its operand refs fails loudly instead of silently miswiring."""
    if FLAGS.validate_dag:
        validate_dag(root)

def optimize(dag_root: DataOp, config: OptConfig = None, env: dict = None):
    """ Entry point for the logical optimizer. Takes a Skrub DataOp DAG, applies logical optimizations,
    and returns an Op root node.

    ``env`` (variable name -> value), when supplied, lets the converter resolve
    variables to compile-time constants (ValueOps) instead of VariableOps."""
    start = start_time()
    if config is None:
        config = OptConfig()


    # Convert to Op DAG
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

    # Final optimized DAG
    if config.algebraic_rewrites:
        root = algebraic_rewrites(root, config.algebraic_rewrite_config)
        _debug_show_graph(root, "algebraic_rewrite")

    # Final passes: linearization and buffer removal planning
    _debug_validate_dag(root)  # operand refs after all rewrites, before linearization
    linearized_dag, split_pos, flagged_ops = linearize_dag(root)
    pinned_ops = compute_pinned_ops(linearized_dag, split_pos, flagged_ops)
    plan_input_removals(linearized_dag, pinned_ops)

    _debug_explain_linear_plan("explain_linear_plan", linearized_dag, split_pos)

    log_time("Optimization took in total", start)
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
        root, _ = extract_dataframe_op(op, root, FLAGS.make_selection_op)
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


def get_dataops_graph(dag: DataOp) -> tuple[dict, dict, dict]:
    start = start_time()
    if FLAGS.fast_dataops_convert:
        g = build_graph(dag)
    else:
        g = _Graph().run(dag)
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
                no_children = last_op is op
                if no_children:
                    if EVAL_OP_ENABLED:
                        # TODO add handle for no_children --> replace choice with eval op
                        raise NotImplementedError("Fix me")
                    else:
                        # unrolling finished
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
    dag_root = (SearchEvalOp(outcome_names=op.outcome_names, parent=[root]) if EVAL_OP_ENABLED
                          else ChoiceOp(outcome_names=op.outcome_names, append_choice_name=False))
    if not EVAL_OP_ENABLED:
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
