from __future__ import annotations
from collections import deque
from typing import Iterator
from graphviz import Digraph
from stratum.optimizer.logical._ops import OperandRef, Op, ChoiceOp
from stratum._config import get_config
import os
from dataclasses import dataclass
import webbrowser

@dataclass
class IteratorFlags:
    bfs = False

FLAGS = IteratorFlags()

def replace_op_in_outputs(op: Op, replacement: Op):
    """Replace op in all its outputs with a replacement op."""
    for out_ in op.outputs:
        out_.replace_input(op, replacement)
        replacement.add_output(out_)


def get_all_outputs(op: Op, stop_at_op: Op = None):
    """Return reachable consumers and their inputs within the traversed subgraph.

    Do not visit ``stop_at_op`` or traverse its output edges.
    """
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
    """Clone consumers downstream of ``root_op`` and return the cloned leaves.

    Rebind references to ``root_op`` to ``new_root_op`` when supplied. Keep
    external inputs shared and connect clones to ``stop_at_op`` where the
    original edges reach it.
    """
    if new_root_op is None:
        new_root_op = root_op

    # Count only input edges within the subgraph when ordering its clones.
    outputs, inputs_internal = get_all_outputs(root_op, stop_at_op)
    indegree = {c: len(inputs_internal[c]) for c in outputs}
    queue = []

    # Map original operators to clones when reconnecting their inputs.
    sub_dag_leaves, clone_look_up = [], {root_op: new_root_op}
    for out_ in root_op.outputs:
        queue.append(out_)

    while queue:
        op = queue.pop(0)
        op_clone = op.clone()
        clone_look_up[op] = op_clone

        # Connect the clone's inputs and their reverse output edges.
        for in_ in op.inputs:
            in_ = clone_look_up.get(in_, in_)
            op_clone.add_input(in_)
            in_.add_output(op_clone)

        if op.outputs is not None and len(op.outputs) > 0:
            for out_ in op.outputs:
                if out_ is stop_at_op:
                    op_clone.add_output(stop_at_op)
                    stop_at_op.add_input(op_clone)
                    # Connect to the boundary operator without cloning or visiting it.
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
    indegree, queue2 = compute_graph_node_indegree(root)

    # now we can do topological traversal
    if FLAGS.bfs:
        return topological_iterator_bfs(queue2, indegree)
    else:
        return topological_iterator_dfs(queue2, indegree)


# TODO: Correct the return annotation to tuple[dict[Op, int], deque[Op]].
def compute_graph_node_indegree(root: Op) -> tuple[deque[Op], dict[Op, int]]:
    # Traverse input edges breadth-first to find the graph's source operators.
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
                    if isinstance(in_op, OperandRef):
                        raise RuntimeError(
                            f"Encountered OperandRef as input of op {op}, which should not happen.")
                    curr_indegree = len(in_op.inputs)
                    indegree[in_op] = curr_indegree
                    queue1.append(in_op)
    return indegree, queue2


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

def _iter_operand_refs(value):
    """Yield every OperandRef nested in value (recurses lists/tuples/dicts/slices,
    and column-expression trees that expose ``iter_operand_refs``)."""
    if isinstance(value, OperandRef):
        yield value
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_operand_refs(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_operand_refs(v)
    elif isinstance(value, slice):
        for bound in (value.start, value.stop, value.step):
            yield from _iter_operand_refs(bound)
    elif hasattr(value, "iter_operand_refs"):
        yield from value.iter_operand_refs()


def validate_operands(op: Op) -> None:
    """Assert every OperandRef stored on `op` indexes a valid entry of op.inputs.

    Catches rewrites that drop/reorder inputs without renumbering operand refs.
    A cache hit retains its original configuration for inspection, but those
    fields no longer execute and its former OperandRefs are unused.
    """
    if op.cache_hit is not None:
        if op.inputs:
            raise ValueError(f"A cache hit must have no inputs: {op!r}")
        return
    n = len(op.inputs)
    for attr, value in op.__dict__.items():
        if attr in ("inputs", "outputs", "remove_after"):
            continue
        for ref in _iter_operand_refs(value):
            if not (0 <= ref.k < n):
                raise ValueError(
                    f"Operand {ref} on {op!r} (attr '{attr}') is out of range for "
                    f"{n} input(s); a rewrite likely changed inputs without renumbering.")


def validate_dag(root: Op) -> None:
    """Run `validate_operands` over every op reachable from root."""
    for op in topological_iterator(root):
        validate_operands(op)


def show_graph(root: Op, filename: str = 'plan'):
    """Show the runtime plan of the DataOp DAG.

    Render as SVG by default (see ``graph_format``). Large plans can exceed
    Cairo's bitmap dimensions, causing Graphviz to scale down the drawing and
    make labels difficult to read. SVG avoids that bitmap size limit.

    The render call removes the temporary DOT source after creating the output.
    """
    dot = Digraph(
        comment=filename,
        format=get_config()["graph_format"],
        graph_attr={'rankdir': 'BT', 'ranksep': '0.3', 'nodesep': '0.2'},
        node_attr={'fontname': 'Helvetica', 'fontsize': '11'},
    )
    for current_op in topological_iterator(root):
        validate_operands(current_op)
        current_op.update_name()
        name = str(current_op) if not isinstance(current_op, ChoiceOp) else current_op.name
        name = name.replace("<","'").replace(">","'") if name is not None else "None"
        dot.node(str(id(current_op)), name)
        for outputs in current_op.outputs:
            dot.edge(str(id(current_op)), str(id(outputs)))
    base = os.path.join("graphs", filename)
    os.makedirs(os.path.dirname(base), exist_ok=True)
    out_path = os.path.abspath(
        dot.render(filename=base + ".dot", outfile=f"{base}.{dot.format}",
                   view=False, cleanup=True))
    if get_config()["open_graph"]:
        webbrowser.open(f"file://{out_path}")
        

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
        # A rewrite may detach the op being visited. A live iterator reads that
        # op's outputs after yielding it, so the traversal would stop before its
        # downstream consumers. Freeze the visit order first, and skip ops that
        # an earlier rewrite in this pass detached.
        for op in list(topological_iterator(root)):
            if op is not root and not op.outputs:
                continue
            matched = match_fn(op)
            if matched is not None:
                root = action_fn(*matched, root)
        return root
    return run
