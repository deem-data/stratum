"""Operators for collecting and scoring candidate results.

- Choice unrolling creates one sub-DAG per candidate. The final operator
  combines their outputs and associates each with its candidate name.
- Search plans also score those outputs against the test labels.
- Explicit input edges preserve candidate ordering and expose test-label
  dependencies to buffer-lifetime planning.

Scoring uses the computed responses and test labels directly; see ``_scoring.py``.
"""
from __future__ import annotations

import polars as pl

from stratum.optimizer.logical import _schema
from stratum.optimizer.logical._base import OutputType, _resolve_operand, clone_value
from stratum.optimizer.logical._ops import Op, OperandRef
from stratum.optimizer.logical._scoring import Metric, read_response

import logging
logger = logging.getLogger(__name__)

_MAX_LABEL = 50


class DeclaredScoringOp(Op):
    """``pred.skb.with_scoring(...)``: ``pred`` unchanged, plus the scorers it declares.

    The scorers keep their skrub shape (``{"scoring", "kwargs", "name"}``), with every
    DataOp in them bound to an operand of this op, so a ``sample_weight`` computed from
    the data is part of the plan. Like skrub's ``Scoring`` node it passes ``pred``
    through; the candidate set strips it and, when the search scores with the declared
    metric, reads the kwargs operands per fold (see ``install_candidate_set``).
    """

    logical_family = "DeclaredScoring"
    fields = ["scorers"]

    def __init__(self, scorers: list, inputs: list | None = None):
        super().__init__(inputs=list(inputs) if inputs else [])
        self.scorers = scorers

    def process(self, mode: str, inputs: list):
        return inputs[0]


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
        # Keep the terminal candidate set out of common-subexpression elimination.
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

    Inputs include candidate outputs and the test labels addressed by ``y_ref``.
    ``response_mode`` identifies the response method selected for test rows during
    train/test unrolling. The metric consumes the resulting predictions,
    probabilities or decision-function values directly.
    """

    logical_family = "ScoreCandidates"
    fields = ["candidate_names", "metric", "response_mode", "emit_predictions", "y_ref",
              "kwargs_refs"]

    def __init__(self, candidate_names: list[str], metric: Metric,
                 inputs: list | None = None, response_mode: str = "predict",
                 emit_predictions: bool = False, y_ref: OperandRef | None = None,
                 kwargs_refs: list | None = None):
        super().__init__(candidate_names, inputs)
        self.metric = metric
        self.response_mode = response_mode
        self.emit_predictions = emit_predictions
        self.output_type = OutputType.FRAME
        self.y_ref = y_ref
        # One entry per candidate: the kwargs its declared scorer reads, with
        # OperandRefs for the ones the plan computes on the fold; None for none.
        self.kwargs_refs = kwargs_refs

    def propagate_output_schema(self):
        """Describe scoring results: candidate name, score, and optional predictions.

        There is one row per candidate. Scoring runs on test rows only;
        the training copy has no candidate set.
        """
        # candidate_names are strings and Metric returns a float; the values are
        # whatever the candidate produced (array / frame / series), so untyped.
        schema = {"id": pl.String, "scores": pl.Float64}
        if self.emit_predictions:
            schema["vals"] = _schema.UNKNOWN_DTYPE
        self.output_schema = pl.Schema(schema)

    def clone(self):
        new_op = type(self)(self.candidate_names, self.metric,
                            response_mode=self.response_mode,
                            emit_predictions=self.emit_predictions, y_ref=self.y_ref,
                            kwargs_refs=clone_value(self.kwargs_refs))
        new_op.was_cloned = True
        return new_op

    def process(self, mode: str, inputs: list):
        y_true = inputs[self.y_ref.k]
        ids, scores, values = [], [], []
        for i, name in enumerate(self.candidate_names):
            produced = inputs[i]
            kwargs = _resolve_operand(self.kwargs_refs[i], inputs) if self.kwargs_refs else None
            try:
                scores.append(self.metric(y_true, read_response(produced, self.response_mode),
                                          **(kwargs or {})))
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
