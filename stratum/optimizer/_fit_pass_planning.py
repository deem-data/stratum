"""Marking the work a fitting pass does not need to do.

Cross-validation runs each fold twice: once to fit, once to get the responses the fold is
scored on. The fitting pass computes the whole plan, including everything downstream of
the last thing that needs fitting -- a predictor's own predictions over the training
fold, the post-processing applied to them, and the candidate set that labels and scores
them. All of it is computed, stored, and then dropped unread.

An op is *fit-dead* when nothing that runs in the fitting pass needs its output: it fits
nothing itself, and every consumer is fit-dead too. That is a backward reachability
question, answered once here rather than by a mode check inside each operator, so a
fit-dead op is never pinned, run, or stored at all.

The dead set is not a contiguous suffix of the linearized plan. Two candidates linearize
as ``[A_model, A_tail, B_model, B_tail, sink]`` just as readily as
``[A_model, B_model, A_tail, B_tail, sink]``, so a single cut-off index would be wrong
and the mark has to be per operator.
"""
from __future__ import annotations

from stratum.optimizer.logical._ops import BaseEstimatorOp, Op
from stratum.utils._utils import start_time, log_time

import logging
logger = logging.getLogger(__name__)


def mark_fit_dead_ops(linearized_dag: list[Op], split_pos: int | None,
                      flagged_ops: list[Op]) -> set[Op]:
    """Set ``op.dead_in_fit`` for every op the fitting pass can skip. Returns them."""
    start = start_time()
    dead: set[Op] = set()
    if split_pos is None:
        # Nothing is executed twice, so there is no pass to spare work in.
        return dead

    protected = set(flagged_ops)
    for op in reversed(linearized_dag[split_pos:]):
        if isinstance(op, BaseEstimatorOp):
            # Fitting is the whole point of the pass for these, transformers included:
            # a step after the predictor still has to be fitted on its output.
            continue
        if op in protected:
            # Re-executed ops are scheduled outside the plan's own order, so their
            # consumers here do not describe who needs them.
            continue
        if all(out in dead for out in op.outputs):
            op.dead_in_fit = True
            dead.add(op)

    logger.debug(f"Fit-pass planning: {len(dead)} of {len(linearized_dag) - split_pos} "
                 f"post-split ops are dead while fitting")
    log_time("fit-pass planning took", start)
    return dead
