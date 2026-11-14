"""Remove unused operators after train/test unrolling of the physical plan."""
from __future__ import annotations

from stratum.optimizer.logical._base import IRNode
from stratum.utils._utils import start_time, log_time


def eliminate_dead_code(root: IRNode) -> IRNode:
    """Keep the root and its transitive data and fitted-state dependencies.

    Applies to logical and physical DAGs. Operators are retained only through
    input edges; statefulness alone does not make an operator live. Side effects
    outside the explicit dependencies are not part of this contract.

    Dead branches may remain in producers' outputs after a rewrite, so discover
    nodes directly rather than using the topological iterator. Detach those
    branches, including their private producers, before normal traversal resumes.
    """
    start = start_time()
    live = set()
    stack = [root]
    while stack:
        op = stack.pop()
        if op in live:
            continue
        live.add(op)
        stack.extend(op.inputs)

    dead = set()
    stack = [out for op in live for out in op.outputs if out not in live]
    while stack:
        op = stack.pop()
        if op in live or op in dead:
            continue
        dead.add(op)
        stack.extend(op.inputs)
        stack.extend(op.outputs)
    for op in dead:
        op.detach()

    log_time("Dead code elimination took", start)
    return root
