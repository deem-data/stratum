from __future__ import annotations
from collections import deque
from typing import Iterator
from graphviz import Digraph
from stratum.optimizer.ir._ops import DATA_OP_PLACEHOLDER, Op, ChoiceOp
from stratum._config import get_config
import os
from dataclasses import dataclass

@dataclass
class IteratorFlags:
    bfs = False

FLAGS = IteratorFlags()

def replace_op_in_outputs(op: Op, replacement: Op):
    """Replace op in all its outputs with a replacement op."""
    for out_ in op.outputs:
        for i,in_ in enumerate(out_.inputs):
            if in_ is op:
                out_.inputs[i] = replacement
                break
        replacement.add_output(out_)


def find_choice_naive(op: Op) -> tuple[ChoiceOp, bool]:
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


def get_all_outputs(op: Op, stop_at_op: Op = None):
    """Returns a list of all output ops. If stop_at_op is given, the outputs of the stop_at_op are not included."""
    queue = [op]
    visited = set()
    inputs_internal = {}
    while queue:
        node = queue.pop(0)
        if node.has_outputs():
            for out_ in node.outputs:
                if out_ in visited:
                    inputs_internal[out_].append(node)
                elif out_ is not stop_at_op:
                    visited.add(out_)
                    queue.append(out_)
                    inputs_internal[out_] = [node]

    return list(visited), inputs_internal


def clone_sub_dag(root_op: Op, stop_at_op: Op = None, new_root_op: Op = None):
    """Clones a sub-dag of the given Op. Excluding the given Op, but including all its internal outputs.
    Returns a list of all ops in the sub-dag, a list of the root ops of the sub-dag and a list of the leaf nodes of the sub-dag.
    """
    if new_root_op is None:
        new_root_op = root_op

    # Topological search inside sub-dag --> inputs_internal contains only the input ops, which are inside the sub-dag
    outputs, inputs_internal = get_all_outputs(root_op, stop_at_op)
    indegree = {c: len(inputs_internal[c]) for c in outputs}
    queue = []

    # clone_look_up: Look-up table for setting parents correctly
    sub_dag_leaves, clone_look_up = [], {root_op: new_root_op}
    for out_ in root_op.outputs:
        queue.append(out_)

    while queue:
        op = queue.pop(0)
        op_clone = op.clone()
        clone_look_up[op] = op_clone

        # update op_clones's parents and the parents's chidlren
        for in_ in op.inputs:
            in_ = clone_look_up.get(in_, in_)
            op_clone.add_input(in_)
            in_.add_output(op_clone)

        if op.outputs is not None and len(op.outputs) > 0:
            for out_ in op.outputs:
                if out_ is stop_at_op:
                    op_clone.add_output(stop_at_op)
                    stop_at_op.add_input(op_clone)
                    # we dont add the output to the sub-dag and dont add it to op_clone's outputs
                    assert len(op.outputs) == 1, "Op before stop Op should have only one output"
                    sub_dag_leaves.append(op_clone)
                    continue
                indegree[out_] -= 1
                if indegree[out_] == 0:
                    queue.append(out_)
        else:
            sub_dag_leaves.append(op_clone)
    return sub_dag_leaves


def topological_iterator(root: Op) -> Iterator[Op]:
    """
    Iterate over the Op DAG in topological order.
    """

    # first we need to bfs for finding all sources in the dag
    queue1 = deque([root])
    indegree = {root: 0 if not root.inputs else len(root.inputs)}
    queue2 = deque()
    while queue1:
        op = queue1.popleft()
        if not op.inputs or len(op.inputs) == 0:
            queue2.append(op)
        else:
            for in_op in op.inputs:
                if in_op not in indegree:
                    if in_op is DATA_OP_PLACEHOLDER:
                        raise RuntimeError(f"Encountered DATA_OP_PLACEHOLDER as input of op {op}, which should not happen.")
                    curr_indegree = len(in_op.inputs)
                    indegree[in_op] = curr_indegree
                    queue1.append(in_op)

    # now we can do topological traversal
    if FLAGS.bfs:
        return topological_iterator_bfs(queue2, indegree)
    else:
        return topological_iterator_dfs(queue2, indegree)

def topological_iterator_bfs(queue, indegree) -> Iterator[Op]:
    while queue:
        op = queue.popleft()
        yield op
        op_outputs = op.outputs
        for out_op in op_outputs:
            if out_op not in indegree:
                raise RuntimeError(f"Encountered op {out_op} which should not exist in the DAG. Probably due to a buggy rewrite, which did not updated the its inputs / outputs correctly.")
            indegree[out_op] -= 1
            if indegree[out_op] == 0:
                queue.append(out_op)

def topological_iterator_dfs(queue, indegree) -> Iterator[Op]:
    stack = list(queue)
    while stack:
        op = stack.pop()
        yield op
        op_outputs = op.outputs
        for out_op in op_outputs:
            if out_op not in indegree:
                raise RuntimeError(f"Encountered op {out_op} which should not exist in the DAG. Probably due to a buggy rewrite, which did not updated the its inputs / outputs correctly.")
            indegree[out_op] -= 1
            if indegree[out_op] == 0:
                stack.append(out_op)

def show_graph(root: Op, filename: str = 'plan'):
    """Show the runtime plan of the DataOp DAG."""
    if get_config()["open_graph"]:
        dot = Digraph(comment=filename)
        for current_op in topological_iterator(root):
            current_op.update_name()
            name = str(current_op) if not isinstance(current_op, ChoiceOp) else current_op.name
            name = name.replace("<","'").replace(">","'") if name is not None else "None"
            dot.node(str(id(current_op)), name)
            for outputs in current_op.outputs:
                dot.edge(str(id(current_op)), str(id(outputs)))
        filename = "graphs/" + filename
        # make sure folder exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        dot.render(filename, view=True,cleanup=True)


def rewrite_pass(match_fn, action_fn):
    """Create a rewrite that does one full DAG pass.

    Parameters
    ----------
    match_fn : (Op) -> tuple | None
        Called on every op in topological order. Returns a tuple of
        matched ops (passed to action_fn) or None if the pattern
        doesn't match.
    action_fn : (*matched_ops, root: Op) -> Op
        Called when match_fn returns a match. Receives the matched ops
        and the current root. Must return the (possibly updated) root.

    Returns
    -------
    Callable[[Op], Op]
        A rewrite function: root -> updated_root
    """
    def run(root):
        for op in topological_iterator(root):
            matched = match_fn(op)
            if matched is not None:
                root = action_fn(*matched, root)
        return root
    return run
