from skrub._data_ops import DataOp
from skrub._data_ops._subsampling import SubsamplePreviews
from collections import deque
from ._cse import apply_cse
from ._dataframe_ops import add_splitting_op
from ._dataframe_ops import rewrite_dataframe_ops, group_dataframe_ops
from ._ops import ChoiceOp, ImplOp, Op, SearchEvalOp, as_op
from ._op_utils import clone_sub_dag, find_choice_naive, replace_op_in_outputs, show_graph, topological_iterator
from ._skrub_graph import build_graph
from time import perf_counter
import logging
from stratum._config import FLAGS

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
    graph = build_graph(dag)
    nodes = graph["nodes"]
    parents = graph["parents"]
    children = graph["children"]

    order = topological_traverse(nodes, parents, children)
    apply_cse(dag, nodes, order, parents)
    return dag

class OptConfig():
    # TODO we should move this class to the _config.py file
    def __init__(self, cse: bool = True, unroll_choices: bool = True, dataframe_ops: bool = True):
        self.cse = cse
        self.dataframe_ops = dataframe_ops
        self.unroll_choices = unroll_choices

def _debug_show_graph(sink: Op, name: str):
    if FLAGS.DEBUG:
        show_graph(sink, name)

def optimize(dag_sink: DataOp, config: OptConfig = None):
    """ Entry point for the logical optimizer. Takes a Skrub DataOp DAG, applies logical optimizations 
    and returns an Op sink node."""
    t0 = perf_counter()
    if config is None:
        config = OptConfig()
        
    t0_graph = perf_counter()
    g = build_graph(dag_sink)
    nodes = g["nodes"]
    parents = g["parents"]
    children = g["children"]
    t1_graph = perf_counter()
    logger.info(f"Graph construction took {t1_graph - t0_graph:.2f} seconds")
    order = topological_traverse(nodes, parents, children)
    if FLAGS.cse:
        t0_cse = perf_counter()
        apply_cse(dag_sink, nodes, order, parents)
        # TODO cse should direcly return the new list of ops ordered so we dont have to iterate again
        t1_cse = perf_counter()
        logger.info(f"CSE took {t1_cse - t0_cse:.2f} seconds")

    t0_convert = perf_counter()
    sink = convert_to_ops(dag_sink)
    t1_convert = perf_counter()
    logger.info(f"Conversion took {t1_convert - t0_convert:.2f} seconds")

    t0_splitting = perf_counter()
    sink = add_splitting_op(sink)
    t1_splitting = perf_counter()
    logger.info(f"Splitting took {t1_splitting - t0_splitting:.2f} seconds")


    _debug_show_graph(sink, "convertion")
    t1_splitting = perf_counter()
    logger.info(f"Splitting took {t1_splitting - t0_splitting:.2f} seconds")
    # Rewrites:

    # Parsing of dataframe ops
    if config.dataframe_ops:
        t0_dataframe = perf_counter()
        sink = rewrite_dataframe_ops(sink)
        sink = group_dataframe_ops(sink)
        _debug_show_graph(sink, "dataframe_rewrite")
        t1_dataframe = perf_counter()
        logger.info(f"Dataframe rewrite took {t1_dataframe - t0_dataframe:.2f} seconds")
    # Unrolling of choices to a dag wit only a single choice op at the end
    if config.unroll_choices:
        t0_choices = perf_counter()
        sink = choice_unrolling(sink)
        _debug_show_graph(sink, "unrolled")
        t1_choices = perf_counter()
        logger.info(f"Choices unrolling took {t1_choices - t0_choices:.2f} seconds")
    
    # Final optimized DAG
    
    t1 = perf_counter()
    logger.info(f"Optimization took in total {t1 - t0:.2f} seconds")
    return sink


def convert_to_ops(dag: DataOp) -> Op:
    """ Convert a Skrub DataOp DAG to a stratum's logical IR (Op DAG)"""
    t0_convert = perf_counter()
    g = build_graph(dag)
    nodes = g["nodes"]
    parents = g["parents"]
    children = g["children"]
    t1_convert = perf_counter()
    logger.info(f"Conversion dag took {t1_convert - t0_convert:.2f} seconds")
    order = topological_traverse(nodes, parents, children)
    sink_id = order[-1]

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
    return ids_to_ops[sink_id]


def convert_handle_choice(node, op, ids_to_ops, children):
    input_iter = iter(ids_to_ops[input] for input in children.get(node, []))
    for j, p in enumerate(op.inputs):
        if p == 0:
            op.inputs[j] = next(input_iter)
        else:
            p.outputs = [op]


def choice_unrolling(sink: Op):
    """ Rewrite for unrolling the dag after choice op into separate dags for each outcome."""
    contains_choice = True
    i = 0
    while contains_choice:
        dag_iter = topological_iterator(sink)
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
                    assert sink is last_op, "Sink should be the last op in the dag"
                    # we reached the end of the dag
                    logger.debug(f"Unrolling simple choice: {op}")
                    sink = unroll_simple_choice(sink, op, outcomes)
                    logger.debug(f"New sink after unrolling: {sink}")

                # if FLAGS.DEBUG:
                #     show_graph(sink, f"choice-unrolled={i}")
                del op
                break
    return sink



def unroll_simple_choice(sink: Op, op: ChoiceOp, outcomes: list) -> Op:
    """ Unroll a simple choice op, which has no choice in the sub-dag."""
    dag_sink = (SearchEvalOp(outcome_names=op.outcome_names, parent=[sink]) if EVAL_OP_ENABLED
                          else ChoiceOp(outcome_names=op.outcome_names, append_choice_name=False))
    if not EVAL_OP_ENABLED:
        dag_sink.inputs = [sink]

    # clones sub-dag after choice op for all outcomes[1:]
    for outcome in outcomes[1:]:
        outcome.outputs = []
        leafs = clone_sub_dag(op, new_root_op=outcome)
        assert len(leafs) == 1
        dag_sink.add_input(leafs[0])
        leafs[0].add_output(dag_sink)

    # reuse sub-dag for the first outcome
    outcomes[0].outputs = []
    replace_op_in_outputs(op, replacement=outcomes[0])
    sink.add_output(dag_sink)
    return dag_sink


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