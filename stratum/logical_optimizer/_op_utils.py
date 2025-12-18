from collections import deque
from typing import Iterator
from graphviz import Digraph
from stratum.logical_optimizer._ops import Op, ChoiceOp
from stratum._config import get_config
import os


def replace_op_in_children(op: Op, replacement: Op):
    """Replace op in children of op with replacement op."""
    for c in op.outputs:
        c.inputs = [replacement if p is op else p for p in c.inputs]
        replacement.add_child(c)


def find_choice_naive(op: Op) -> ChoiceOp:
    """
    Find the choice operation in the sub-dag using a naive approach. Might return incorrect results if there are multiple choices in the sub-dag.
    """
    # TODO check and improve find_choice(op: Op)
    last_op = op
    contains_choice = False
    while len(last_op.outputs) > 0 and not contains_choice:
        last_op = last_op.outputs[0]
        contains_choice = last_op.is_choice()
    return last_op, contains_choice


def get_all_children(op: Op, stop_at_op: Op = None):
    """Returns a list of all children. If stop_at_op is given, the children of the stop_at_op are not included."""
    queue = [op]
    visited = set()
    parents_internal = {}
    while queue:
        node = queue.pop(0)
        if node.has_children():
            for child in node.outputs:
                if child in visited:
                    parents_internal[child].append(node)
                elif child is not stop_at_op:
                    visited.add(child)
                    queue.append(child)
                    parents_internal[child] = [node]
                    
    return list(visited), parents_internal
    

def clone_sub_dag(root_op: Op, stop_at_op: Op = None, new_root_op: Op = None):
    """Clones a sub-dag of the given Op. Excluding the given Op, but including all its internal children. 
    Returns a list of all ops in the sub-dag, a list of the root ops of the sub-dag and a list of the leaf nodes of the sub-dag.
    """
    if new_root_op is None:
        new_root_op = root_op

    # Topological search inside sub-dag --> parents_internal contains only the parents inside the dag
    children, parents_internal = get_all_children(root_op, stop_at_op)
    indegree = {c: len(parents_internal[c]) for c in children}
    queue = []

    # clone_look_up: Look-up table for setting parents correctly
    sub_dag_leaves, clone_look_up = [], {root_op: new_root_op}
    for c in root_op.outputs:
        queue.append(c)

    while queue:
        op = queue.pop(0)
        op_clone = op.clone()
        clone_look_up[op] = op_clone

        # update op_clones's parents and the parents's chidlren
        for p in op.inputs:
            p = clone_look_up.get(p, p)
            op_clone.add_parent(p)
            p.add_child(op_clone)

        if op.outputs is not None and len(op.outputs) > 0:
            for child in op.outputs:
                if child is stop_at_op:
                    op_clone.add_child(stop_at_op)
                    stop_at_op.add_parent(op_clone)
                    # we dont add the child to the sub-dag and dont add it to op_clone's children
                    assert len(op.outputs) == 1, "Op before stop Op should have only one child"
                    sub_dag_leaves.append(op_clone)
                    continue
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        else:
            sub_dag_leaves.append(op_clone)
    return sub_dag_leaves


def topological_iterator(sink: Op) -> Iterator[Op]:
    """
    Iterate over the Op DAG in topological order.
    """

    # first we need to bfs for finding all sources in the dag
    queue1 = deque([sink])
    indegree = {sink: 0 if not sink.inputs else len(sink.inputs)}
    queue2 = deque()
    while queue1:
        op = queue1.popleft()
        if not op.inputs or len(op.inputs) == 0:
            queue2.append(op)
        else:
            for in_op in op.inputs:
                if in_op not in indegree:
                    indegree[in_op] = 0 if not in_op.inputs else len(in_op.inputs)
                    queue1.append(in_op)

    # now we can do topological traversal
    while queue2:
        op = queue2.popleft()
        yield op
        for out_op in op.outputs:
            indegree[out_op] -= 1
            if indegree[out_op] == 0:
                queue2.append(out_op)


def show_graph(sink: Op, filename: str = 'plan'):  
    """Show the runtime plan of the DataOp DAG."""

    dot = Digraph(comment=filename)
    for current_op in topological_iterator(sink):
        current_op.update_name()
        name = current_op.name
        name = name.replace("<","'").replace(">","'") if name is not None else "None"
        dot.node(str(id(current_op)), name)
        for child in current_op.outputs:
            dot.edge(str(id(current_op)), str(id(child)))
    filename = "graphs/" + filename
    # make sure folder exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    dot.render(filename, view=get_config()["open_graph"],cleanup=True)