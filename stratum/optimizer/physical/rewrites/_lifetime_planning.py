"""Plan intermediate retention and buffer-pool releases for the linearized DAG.

Use final-consumer information to schedule releases, retaining intermediates
needed across passes and CV folds.

- Last-use analysis counts remaining uses in the fixed execution order and
  records releases in each consumer's ``op.remove_after``.
- Pre-split outputs are retained if they are not recomputed and feed post-split
  or re-executed ops. They may still spill to disk; ``pinned_ops`` does not
  imply ``BufferPool.pin`` residency.
- Operators skipped during fitting still perform their scheduled releases,
  so a release need not follow an actual read in that pass.
- The scheduler calls ``pool.remove()`` for scheduled releases. The reuse
  cache may retain the same objects after their buffer-pool entries are removed.
"""
from __future__ import annotations

from stratum.optimizer.logical._ops import Op
from stratum.utils._utils import start_time, log_time

import logging
logger = logging.getLogger(__name__)


def compute_pinned_ops(
    linearized_dag: list[Op],
    split_pos: int | None,
    recompute_ops: list[Op],
) -> set[Op]:
    """Return pre-split ops whose outputs must remain available across CV folds."""
    start = start_time()
    if split_pos is None:
        return set()

    pinned: set[Op] = set()
    re_executed = set(linearized_dag[split_pos:]) | set(recompute_ops)
    for op in linearized_dag[:split_pos]:
        if op not in re_executed:
            for out_op in op.outputs:
                if out_op in re_executed:
                    pinned.add(op)
                    break
    log_time("compute_pinned_ops took", start)
    return pinned


def plan_input_removals(
    linearized_dag: list[Op],
    pinned_ops: set[Op],
) -> None:
    """Set ``op.remove_after`` for every op in the linearized DAG."""
    start = start_time()
    remaining = {op: max(len(op.outputs), 1) for op in linearized_dag}

    for op in linearized_dag:
        remove = []
        for in_op in op.inputs:
            remaining[in_op] -= 1
            if remaining[in_op] <= 0 and in_op not in pinned_ops:
                remove.append(in_op)
        op.remove_after = remove

    logger.debug(
        f"Removal planning done: pinned={len(pinned_ops)} ops, "
        f"total={len(linearized_dag)} ops"
    )
    log_time("plan_input_removals took", start)
