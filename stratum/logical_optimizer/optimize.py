from skrub._data_ops._data_ops import CallMethod
from skrub._data_ops._evaluation import _Graph
from skrub._data_ops import DataOp
from collections import deque
from stratum.logical_optimizer.cse import apply_cse
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


def optimize(dag: DataOp):
    graph = _Graph().run(dag)
    nodes = graph["nodes"]
    parents = graph["parents"]
    children = graph["children"]

    order = topological_traverse(nodes, parents, children)
    apply_cse(dag, nodes, order, parents)
    return dag