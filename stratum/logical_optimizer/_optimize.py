from skrub._data_ops._data_ops import CallMethod, Value
from skrub._data_ops._evaluation import _Graph
from skrub._data_ops import DataOp
from collections import deque
from ._cse import apply_cse
from ._ops import ChoiceOp, Op, SearchEvalOp, as_op
from ._op_utils import clone_sub_dag, find_choice_naive, replace_op_in_children, show_graph, topological_sort_ir
from time import perf_counter
import logging


logger = logging.getLogger(__name__)
EVAL_OP_ENABLED = False
DEBUG_FLAG = False


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
    graph = _Graph().run(dag)
    nodes = graph["nodes"]
    parents = graph["parents"]
    children = graph["children"]

    order = topological_traverse(nodes, parents, children)
    apply_cse(dag, nodes, order, parents)
    return dag

class OptConfig():
    def __init__(self, cse: bool = False, unroll_choices: bool = True):
        self.cse = cse
        self.unroll_choices = unroll_choices


def optimize(dag: DataOp, config: OptConfig = None):
    """ Entry point for the logical optimizer. Takes a Skrub DataOp DAG, applies logical optimizations 
    and returns a topologically sorted list of Op nodes."""
    t0 = perf_counter()
    if config is None:
        config = OptConfig()
        
    graph = _Graph().run(dag)
    nodes = graph["nodes"]
    parents = graph["parents"]
    children = graph["children"]

    order = topological_traverse(nodes, parents, children)
    if config.cse:
        apply_cse(dag, nodes, order, parents)
        # TODO cse should direcly return the new list of ops ordered so we dont have to iterate again

    op_order = convert_to_ops(dag)

    # TODO add rewrite handling here

    if config.unroll_choices:
        op_order = choice_unrolling(op_order)
    if DEBUG_FLAG:
        show_graph(op_order, "optimized")
    t1 = perf_counter()
    logger.info(f"Optimization took {t1 - t0:.2f} seconds")
    return op_order


def convert_to_ops(dag: DataOp) -> list[Op]:
    """ Convert a Skrub DataOp DAG to a stratum's logical IR (Op DAG)"""
    graph = _Graph().run(dag)
    nodes = graph["nodes"]
    parents = graph["parents"]
    children = graph["children"]

    order = topological_traverse(nodes, parents, children)

    # make logical IR:
    ids_to_ops = {node: as_op(nodes[node]) for node in order}
    op_order = []
    for node in order:
        op = ids_to_ops[node]
        op.children = [ids_to_ops[child] for child in parents.get(node, [])]
        if op.is_choice():
            parent_ops_iter = iter([ids_to_ops[parent] for parent in children.get(node, [])])
            for i, p in enumerate(op.parents):
                if p == 0:
                    op.parents[i] = next(parent_ops_iter)
                else:
                    p.children = [op]
                    op_order.append(p)
        else:
            op.parents = [ids_to_ops[parent] for parent in children.get(node, [])]
        op_order.append(op)
    return op_order


def choice_unrolling(ops_ordered: list[Op]):
    """ Rewrite for unrolling the dag after choice op into separate dags for each outcome."""
    i = 0
    while len(ops_ordered) > i:
        op = ops_ordered[i]
        if op.is_choice():
            outcomes = op.parents

            # check if we find any choice in the sub-dag of the current choice
            last_op, is_choice = find_choice_naive(op)
            no_children = last_op is op
            if no_children:
                if EVAL_OP_ENABLED:
                    # TODO add handle for no_children --> replace choice with eval op
                    raise NotImplementedError("Fix me")
                else:
                    # nothing to do
                    i += 1
                    continue
            if is_choice:
                new_ops = unroll_nested_choice(last_op, op, outcomes)
            else:
                new_ops = unroll_simple_choice(last_op, op, outcomes)

            ops_ordered.remove(op)
            ops_ordered.extend(new_ops)
            ops_ordered = topological_sort_ir(ops_ordered)
            if DEBUG_FLAG:
                show_graph(ops_ordered, f"choice-unrolled={i}")
            del op
        else:
            i += 1
    return ops_ordered


def unroll_simple_choice(last_op: Op, op: ChoiceOp, outcomes: list) -> list[SearchEvalOp | ChoiceOp]:
    """ Unroll a simple choice op, which has no choice in the sub-dag."""
    dag_sink = (SearchEvalOp(outcome_names=op.outcome_names, parent=last_op) if EVAL_OP_ENABLED
                          else ChoiceOp(outcome_names=op.outcome_names, append_choice_name=False))
    if not EVAL_OP_ENABLED:
        dag_sink.parents = [last_op]
        dag_sink.children = []
    new_ops = [dag_sink]

    # clones sub-dag after choice op for all outcomes[1:]
    for outcome in outcomes[1:]:
        outcome.children = []
        tmp_new_ops, leafs = clone_sub_dag(op, new_root_op=outcome)
        assert len(leafs) == 1
        dag_sink.add_parent(leafs[0])
        leafs[0].add_child(dag_sink)
        new_ops.extend(tmp_new_ops)

    # reuse sub-dag for the first outcome
    outcomes[0].children = []
    replace_op_in_children(op, replacement=outcomes[0])
    last_op.add_child(dag_sink)
    return new_ops


def unroll_nested_choice(last_op: ChoiceOp, op: ChoiceOp, outcomes) -> list[Op]:
    """ Unroll a nested choice op, which has choice in the sub-dag."""
    new_ops, n_outcomes = [], len(last_op.outcome_names)

    # clone the sub-dag for each outcome of the current choice
    for outcome, outcome_name in zip(outcomes[1:], op.outcome_names[1:]):
        outcome.children = []
        tmp_new_ops, _ = clone_sub_dag(op, new_root_op=outcome, stop_at_op=last_op)
        for i in range(n_outcomes):
            last_op.outcome_names.append(last_op.outcome_names[i] + outcome_name)
        new_ops.extend(tmp_new_ops)

    # reuse sub-dag for the first outcome
    outcomes[0].children = [op.children[0]]
    for i in range(n_outcomes):
        last_op.outcome_names[i] += op.outcome_names[0]
    outcomes[0].children = []
    replace_op_in_children(op, replacement=outcomes[0])
    return new_ops