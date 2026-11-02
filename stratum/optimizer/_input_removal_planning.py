"""Input removal planning for the linearized Op DAG.

Removal planning marks the last use of each intermediate in the DAG. The list of
intermediates to be removed is materialized in each op's ``op.remove_after`` attribute.
Skrub/stratum's execution plan does not include control flow, so we can mark the last
use of each intermediate in the compile time.

How it works:
- Each op starts with a count of the input edges that read it.
- Walking the linearized order, every time an op appears as an input of
  another op, its remaining count is decremented. Edges reading a fitted state
  are counted apart from the ones reading data.
- When the count hits zero, the intermediate is scheduled for removal right after
  the current op finishes (set via ``op.remove_after``).
- Pinned ops (pre-split ops that feed post-split ops) are never removed by
  the schedule; they persist across CV folds.
- At execution time the scheduler calls ``buffers.remove()`` for every
  entry in ``op.remove_after``, and drops the state of every entry in
  ``op.release_state_after``.
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
    """Return pre-split ops that must persist across CV folds."""
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

    An op that fits state is consumed twice over: as data by the ops reading its
    output, and as state by its "predict" stages. The two are counted apart, so
    each is released after its own last reader.
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
