"""Input release planning for the linearized Op DAG.

After linearization, each op produces an intermediate buffer stored in the
BufferPool.  Release planning decides *when* each buffer can be freed so
that peak memory stays low while correctness is preserved.

How it works:
- Each op starts with a consumer count equal to ``len(op.outputs)`` (min 1).
- Walking the linearized order, every time an op appears as an input of
  another op, its remaining count is decremented.
- When the count hits zero the buffer is scheduled for release right after
  the current op finishes (set via ``op.release_after``).
- Pinned ops (pre-split ops that feed post-split / re-executed ops) are
  never released by the schedule; they persist across CV folds.
- At execution time ``Op.release_inputs()`` calls ``buffers.release()`` for
  every entry in its ``release_after`` list.
"""
from __future__ import annotations

from stratum.optimizer.ir._ops import Op

import logging
logger = logging.getLogger(__name__)


def compute_pinned_ops(
    linearized_dag: list[Op],
    split_pos: int | None,
    recompute_ops: list[Op],
) -> set[Op]:
    """Return pre-split ops whose buffers must persist across CV folds."""
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
    return pinned


def plan_input_releases(
    linearized_dag: list[Op],
    pinned_ops: set[Op],
) -> None:
    """Set ``op.release_after`` for every op in the linearized DAG."""
    remaining = {op: max(len(op.outputs), 1) for op in linearized_dag}

    for op in linearized_dag:
        release = []
        for in_op in op.inputs:
            remaining[in_op] -= 1
            if remaining[in_op] <= 0 and in_op not in pinned_ops:
                release.append(in_op)
        op.release_after = release

    logger.debug(
        f"Release planning done: pinned={len(pinned_ops)} ops, "
        f"total={len(linearized_dag)} ops"
    )
