from skrub._data_ops._data_ops import CallMethod, Value
from skrub._data_ops._evaluation import _Graph
from skrub._data_ops import DataOp
from collections import deque
from ._cse import apply_cse
from ._op_utils import choice_unrolling
from ._ops import Op, as_op

import logging
logger = logging.getLogger(__name__)


def topological_traverse(nodes, parents, children):
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