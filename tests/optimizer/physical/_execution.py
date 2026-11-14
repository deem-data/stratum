"""Explicit fitted-state inputs for direct estimator execution tests."""
import copy

from stratum.optimizer.logical._base import OperandRef


def inference_op(fit, input_count=1):
    """Copy a bound estimator for its default inference method with state appended."""
    inference = copy.copy(fit)
    inference.mode = fit.fallback_mode
    inference.state_ref = OperandRef(input_count)
    return inference
