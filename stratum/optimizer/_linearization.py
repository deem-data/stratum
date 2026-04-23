from __future__ import annotations

from skrub._data_ops._data_ops import EvalMode
from stratum.optimizer.ir._dataframe_ops import SplitOp
from stratum.optimizer._op_utils import compute_graph_node_indegree
from stratum.optimizer.ir._ops import ImplOp, Op
from stratum.utils._utils import start_time, log_time

import logging
logger = logging.getLogger(__name__)


def linearize_dag(dag_sink: Op) -> tuple[list[Op], int | None, list[Op]]:
    """Topologically sort a DAG and enforce the split invariant.

    Single-pass DFS that defers the split op on the stack: whenever the
    split op is ready but other ops are too, prefer the non-split ops.
    This naturally ensures all non-descendants are processed first.

    Guarantees that every op at index >= split_pos is a descendant of the
    split op (i.e. in its subtree). This makes downstream scheduling
    trivial: ops[:split_pos] are pre-split, ops[split_pos:] are post-split.

    Returns:
        linearized_dag: Linearized list of ops with the split invariant.
        split_pos: Index of the SplitOp in linearized_dag, or None if absent.
        flagged_ops: Ops flagged for recomputation (ImplOps with EvalMode).
    """
    start = start_time()
    indegree, sources = compute_graph_node_indegree(dag_sink)

    linearized_ops = []
    flagged_ops = []
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
        if isinstance(op, ImplOp) and isinstance(op.skrub_impl, EvalMode):
            flagged_ops.append(op)

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
    return linearized_ops, split_pos, flagged_ops
