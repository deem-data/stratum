from __future__ import annotations

from stratum.optimizer._op_utils import compute_graph_node_indegree
from stratum.optimizer.logical._ops import Op
from stratum.utils._utils import start_time, log_time

import logging
logger = logging.getLogger(__name__)


def linearize_dag(dag_sink: Op) -> tuple[list[Op], int | None]:
    """Topologically sort a DAG and enforce the split invariant.

    Use Kahn's algorithm with a stack of ready operators. When the split
    operator and another operator are ready, defer the split operator.

    For a DAG with one split, this places all non-descendants before it.
    ``ops[:split_pos]`` are pre-split operators; ``ops[split_pos:]`` contains
    the split operator and its descendants.

    Returns:
        linearized_dag: Linearized list of ops with the split invariant.
        split_pos: Index of the SplitOp in linearized_dag, or None if absent.
    """
    start = start_time()
    indegree, sources = compute_graph_node_indegree(dag_sink)

    linearized_ops = []
    stack = list(sources)
    split_pos = None
    i = 0
    while stack:
        if stack[-1].is_split_op and len(stack) > 1:
            # Defer split op: pop something else instead
            op = stack.pop(-2)
        else:
            op = stack.pop()

        if op.is_split_op:
            split_pos = i

        linearized_ops.append(op)
        for out_op in op.outputs:
            if out_op not in indegree:
                raise RuntimeError(
                    f"Encountered op {out_op} which should not exist in the DAG. "
                    f"Probably due to a buggy rewrite, which did not update its inputs / outputs correctly."
                )
            indegree[out_op] -= 1
            if indegree[out_op] == 0:
                stack.append(out_op)
        i += 1

    log_time("linearization took", start)
    return linearized_ops, split_pos
