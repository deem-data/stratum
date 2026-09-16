"""The operator a plan ends in when it evaluates candidates.

Choice unrolling replaces a plan's branch points with one sub-DAG per candidate, so by
the time the plan is executable there is nothing left to choose: every candidate is
computed. What terminates such a plan is a fan-in that labels each candidate's output
and, for a search, scores it.

Leaving that fan-in as a ``ChoiceOp`` cost the runtime the one thing the planner already
knew. The scheduler had to recover which operator produced which candidate by
pattern-matching the last node of the linearized plan and zipping it positionally against
the labels, and it reached the fold's labels by pinning a buffer the plan's own removal
planning could not see. Both are input edges here, so the mapping cannot drift and the
lifetime is declared.

Scoring itself is a call against those values and the fold's labels; see
``_scoring.py`` for why no estimator is involved.
"""
from __future__ import annotations

import polars as pl

from stratum.optimizer.logical._base import OutputType
from stratum.optimizer.logical._ops import FITTING_MODE, Op, OperandRef
from stratum.optimizer.logical._scoring import Metric, read_response

import logging
logger = logging.getLogger(__name__)

_MAX_LABEL = 50


class CandidateSetOp(Op):
    """Fan-in over every candidate's output, carrying one label per candidate.

    The first ``len(candidate_names)`` inputs are the candidate tails, in label order.
    Subclasses may append further operands and address them by ``OperandRef``.
    """

    logical_family = "CandidateSet"
    fields = ["candidate_names"]

    def __init__(self, candidate_names: list[str], inputs: list | None = None):
        super().__init__(inputs=list(inputs) if inputs else [])
        self.candidate_names = list(candidate_names)
        self.update_name()

    def update_name(self):
        label = " | ".join(self.candidate_names)
        self.name = label if len(label) <= _MAX_LABEL else label[:_MAX_LABEL] + "..."

    @property
    def candidates(self) -> list[Op]:
        """The operator producing each candidate's output, in label order."""
        return self.inputs[:len(self.candidate_names)]

    def consumes_inputs_positionally(self) -> bool:
        # Each candidate is read by position, so two candidates that happen to be the
        # same operator must still occupy distinct slots.
        return True

    def structure_key(self):
        # A plan has one candidate set; merging it with anything is never valid.
        return None

    def clone(self):
        new_op = type(self)(self.candidate_names)
        new_op.was_cloned = True
        return new_op


class CollectCandidatesOp(CandidateSetOp):
    """Label each candidate's output. Terminates a plan that is evaluated, not searched."""

    logical_family = "CollectCandidates"

    def process(self, mode: str, inputs: list):
        return [{"id": name, "vals": inputs[i]}
                for i, name in enumerate(self.candidate_names)]


class ScoreCandidatesOp(CandidateSetOp):
    """Score each candidate against the test fold, one row per candidate.

    Inputs are the candidate tails followed by the fold's ``y``, as ``mark_as_y``
    produced it, addressed by ``y_ref``. ``response_mode`` is the method the pass ran, so
    it names what the candidate values are -- predictions, probabilities, or a decision
    function -- and the metric reads them directly.
    """

    logical_family = "ScoreCandidates"
    fields = ["candidate_names", "metric", "response_mode", "emit_predictions"]

    def __init__(self, candidate_names: list[str], metric: Metric,
                 inputs: list | None = None, response_mode: str = "predict",
                 emit_predictions: bool = False):
        super().__init__(candidate_names, inputs)
        self.metric = metric
        self.response_mode = response_mode
        self.emit_predictions = emit_predictions
        self.output_type = OutputType.FRAME
        self.y_ref: OperandRef | None = None

    def clone(self):
        new_op = type(self)(self.candidate_names, self.metric,
                            response_mode=self.response_mode,
                            emit_predictions=self.emit_predictions)
        new_op.was_cloned = True
        return new_op

    def process(self, mode: str, inputs: list):
        if mode == FITTING_MODE:
            # A fold is scored on the fold it was not fitted on, so there is nothing to
            # score during the fitting pass and nothing downstream to consume it.
            return None
        y_true = inputs[self.y_ref.k]
        ids, scores, values = [], [], []
        for i, name in enumerate(self.candidate_names):
            produced = inputs[i]
            try:
                scores.append(self.metric(y_true, read_response(produced, self.response_mode)))
            except Exception as e:
                raise RuntimeError(
                    f"[scoring] {self.metric!r} failed on candidate {name!r}"
                    f" reading its {self.response_mode!r} values: {e}"
                ) from e
            ids.append(name)
            values.append(produced)
        table = {"id": ids, "scores": scores}
        if self.emit_predictions:
            table["vals"] = values
        return pl.DataFrame(table)
