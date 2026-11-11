"""Dead code elimination for the fitting pass.

- Traverse post-split operators in reverse topological order.
- Keep estimator operators that update fitted state, and operators scheduled
  for recomputation.
- Mark any other operator whose consumers are all marked as dead during fitting.
- Skip marked operators during fitting; retain them for prediction and scoring.
  Skip input pinning and output computation/storage, but perform input removals.
- Mark each operator individually: dead code may be interleaved with estimators
  in the linearized plan, rather than forming a contiguous suffix.
  Both orders are valid:
  ``[A_model, A_postprocess, B_model, B_postprocess, sink]`` and
  ``[A_model, B_model, A_postprocess, B_postprocess, sink]``.
"""
from __future__ import annotations

from stratum.optimizer.logical._ops import BaseEstimatorOp, Op
from stratum.utils._utils import start_time, log_time

import logging
logger = logging.getLogger(__name__)


def mark_dead_ops_during_fit(linearized_dag: list[Op], split_pos: int | None,
                             flagged_ops: list[Op]) -> set[Op]:
    """Mark operators not executed during fitting and return the marked set.

    Set ``op.dead_in_fit`` on each marked operator for the scheduler.
    """
    start = start_time()
    dead: set[Op] = set()
    if split_pos is None:
        # Plans without a split do not have a separate fitting pass.
        return dead

    protected = set(flagged_ops)
    for op in reversed(linearized_dag[split_pos:]):
        if isinstance(op, BaseEstimatorOp):
            # Preserve fitted-state updates even when the output is unused.
            # This includes transformers applied after a predictor.
            continue
        if op in protected:
            # Recomputed operators also execute outside the linearized order;
            # consumer analysis alone cannot determine whether they are needed.
            continue
        if all(out in dead for out in op.outputs):
            op.dead_in_fit = True
            dead.add(op)

    logger.debug(f"Dead code elimination during fitting: {len(dead)} of "
                 f"{len(linearized_dag) - split_pos} "
                 f"post-split ops are dead while fitting")
    log_time("dead code elimination during fitting took", start)
    return dead
