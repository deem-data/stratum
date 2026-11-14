"""Plan data and fitted-state lifetimes for the linearized DAG.

Use final-consumer information to schedule releases, retaining intermediates
needed across CV folds.

- Last-use analysis counts data and fitted-state inputs separately, recording
  releases in ``op.remove_after`` and ``op.release_state_after``.
- Pre-split outputs that feed post-split operators remain available across folds.
  They may still spill to disk; ``pinned_ops`` does not imply ``BufferPool.pin``
  residency.
- The scheduler removes data from the buffer pool and state from the state store
  after their last consumers. The reuse cache may retain the same objects.
"""
from __future__ import annotations

from stratum.optimizer.logical._ops import Op
from stratum.utils._utils import start_time, log_time

import logging
logger = logging.getLogger(__name__)


def compute_pinned_ops(
    linearized_dag: list[Op],
    split_pos: int | None,
) -> set[Op]:
    """Return pre-split ops whose outputs must remain available across CV folds."""
    start = start_time()
    if split_pos is None:
        return set()

    pinned: set[Op] = set()
    per_fold = set(linearized_dag[split_pos:])
    for op in linearized_dag[:split_pos]:
        if any(out_op in per_fold for out_op in op.outputs):
            pinned.add(op)
    log_time("compute_pinned_ops took", start)
    return pinned


def plan_input_removals(
    linearized_dag: list[Op],
    pinned_ops: set[Op],
) -> None:
    """Set ``op.remove_after`` and ``op.release_state_after`` for every op in the
    linearized DAG.

    Count data and fitted-state consumers separately. Schedule each value's
    release after its last use, except for values retained across folds.
    """
    start = start_time()
    data_left: dict[Op, int] = {}
    state_left: dict[Op, int] = {}
    for op in linearized_dag:
        for k, in_op in enumerate(op.inputs):
            left = state_left if op.reads_state(k) else data_left
            left[in_op] = left.get(in_op, 0) + 1

    for op in linearized_dag:
        remove, release = [], []
        for k, in_op in enumerate(op.inputs):
            state = op.reads_state(k)
            left = state_left if state else data_left
            left[in_op] -= 1
            if left[in_op] == 0 and in_op not in pinned_ops:
                (release if state else remove).append(in_op)
        op.remove_after = remove
        op.release_state_after = release

    logger.debug(
        f"Removal planning done: pinned={len(pinned_ops)} ops, "
        f"total={len(linearized_dag)} ops"
    )
    log_time("plan_input_removals took", start)
