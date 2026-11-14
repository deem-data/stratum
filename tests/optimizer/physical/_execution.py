"""Explicit fitted-state inputs for direct estimator execution tests."""
import copy

from stratum.optimizer.logical._base import OperandRef


def response_op(fit, input_count=1):
    """Copy a bound estimator for its default response with state appended."""
    response = copy.copy(fit)
    response.mode = fit.fallback_mode
    response.state_ref = OperandRef(input_count)
    return response
