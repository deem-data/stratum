from skrub._data_ops._evaluation import _Graph
from skrub._data_ops import DataOp
from skrub._data_ops._subsampling import SubsamplePreviews
from collections import deque
from ._cse import apply_cse
from .ir._dataframe_ops import extract_dataframe_op, add_splitting_op
from .ir._numeric_ops import extract_numeric_op
from .ir._ops import ChoiceOp, ImplOp, Op, SearchEvalOp, as_op
from ._op_utils import clone_sub_dag, find_choice_naive, replace_op_in_outputs, show_graph, topological_iterator
from ._algebraic_rewrites import algebraic_rewrites, AlgebraicRewritesConfig
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


def apply_cse_on_skrub_ir(dag: DataOp):
    """ Apply CSE on a Skrub DataOp DAG and return the deduplicated DAG. (Deprecated versio of optimize function)"""
    children, nodes, parents = get_dataops_graph(dag)
    order = topological_traverse(nodes, parents, children)
    apply_cse(dag, nodes, order, parents)
    return dag

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
    if FLAGS.DEBUG:
        show_graph(root, name)

def optimize(dag_root: DataOp, config: OptConfig = None):
    """ Entry point for the logical optimizer. Takes a Skrub DataOp DAG, applies logical optimizations,
    and returns an Op root node."""
    start = start_time()
    if config is None:
        config = OptConfig()

    children, nodes, parents = get_dataops_graph(dag_root)
    order = topological_traverse(nodes, parents, children)

    # Apply CSE on skrub IR
    if FLAGS.cse:
        run_cse_pass(dag_root, nodes, order, parents)

    # Convert to Op DAG and add splitting op
    root = convert_to_ops(dag_root, debug_graph=False)
    root = add_splitting_op(root)

    # Extract specialized operators from generic MethodCallOp / CallOp
    if config.dataframe_ops:
        if config.numeric_ops:
            # Fused extraction of frame and numeric ops
            root = extract_frame_and_numeric_operators(root)
        else:
            # Extract only dataframe ops
            root = extract_frame_operators(root)
    elif config.numeric_ops:
        # Extract only numeric ops
        root = extract_numeric_operators(root)

    # Unrolling of choices to a dag with only a single ChoiceOp at the end
    if config.unroll_choices:
        root = choice_unrolling(root)

    # Final optimized DAG
    if config.algebraic_rewrites:
        root = algebraic_rewrites(root, config.algebraic_rewrite_config)

    log_time("Optimization took in total", start)
    return root


def run_cse_pass(dag_root: DataOp, nodes: dict, order: list, parents: dict):
    """ Apply CSE on a Skrub DataOp DAG and return the deduplicated DAG."""
    start = start_time()
    apply_cse(dag_root, nodes, order, parents)
    # TODO cse should directly return the new list of ops ordered so we dont have to iterate again
    log_time("CSE took", start)


def extract_frame_operators(root):
    """ Rewrite the dataframe ops in the dag to the new dataframe ops."""
    start = start_time()
    for op in topological_iterator(root):
        root, _ = extract_dataframe_op(op, root)
    log_time("dataframe_rewrite took", start)
    return root


def extract_numeric_operators(root):
    """ Rewrite the dataframe ops in the dag to the new dataframe ops."""
    start = start_time()
    for op in topological_iterator(root):
        root, _ = extract_numeric_op(op, root)
    log_time("to_numeric took", start)
    return root


def extract_frame_and_numeric_operators(root):
    """ Rewrite the dataframe ops in the dag to the new dataframe ops."""
    start = start_time()
    for op in topological_iterator(root):
        root, matched = extract_dataframe_op(op, root)
        if not matched:
            root, _ = extract_numeric_op(op, root)
    log_time("frame_and_numeric_rewrite took", start)
    return root


def convert_to_ops(dag: DataOp, debug_graph: bool = True) -> Op:
    """ Convert a Skrub DataOp DAG to a stratum's logical IR (Op DAG)"""
    start = start_time()
    children, nodes, parents = get_dataops_graph(dag)
    order = topological_traverse(nodes, parents, children)
    root_id = order[-1]

    # make logical IR:
    # we start by making unconnected ops
    ids_to_ops = {node: as_op(nodes[node]) for node in order}
    # we then connect the ops to a graph
    for node in order:
        op = ids_to_ops[node]
        if isinstance(op, ImplOp) and isinstance(op.skrub_impl, SubsamplePreviews):
            output_ids = parents.get(node, [])
            output_ops = [ids_to_ops[output] for output in output_ids]
            input_id = children.get(node, [])[0]
            input_op = ids_to_ops[input_id]
            input_op.outputs.remove(op)
            input_op.outputs.extend(output_ops)
            for output_id in output_ids:
                children[output_id].remove(node)
                children[output_id].append(input_id)
            del ids_to_ops[node]
        else:
            op.outputs = [ids_to_ops[output] for output in parents.get(node, [])]

            if op.is_choice():
                convert_handle_choice(node, op, ids_to_ops, children)
            else:
                op.inputs = [ids_to_ops[input] for input in children.get(node, [])]
    root = ids_to_ops[root_id]
    log_time("conversion took", start)
    if debug_graph:
        _debug_show_graph(dag, "conversion")
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


def convert_handle_choice(node, op, ids_to_ops, children):
    input_iter = iter(ids_to_ops[input] for input in children.get(node, []))
    for j, p in enumerate(op.inputs):
        if p == 0:
            op.inputs[j] = next(input_iter)
        else:
            p.outputs = [op]


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
