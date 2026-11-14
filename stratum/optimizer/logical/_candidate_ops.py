"""Operators for collecting and scoring candidate results.

- Choice unrolling creates one sub-DAG per candidate. The final operator
  combines their outputs and associates each with its candidate name.
- Search plans also score those outputs against the test labels.
- Explicit input edges preserve candidate ordering and expose test-label
  dependencies to buffer-lifetime planning.

Scoring uses the computed outputs and test labels directly; see ``_scoring.py``.
"""
from __future__ import annotations

import polars as pl

from stratum.optimizer.logical import _schema
from stratum.optimizer.logical._base import OutputType, _resolve_operand, clone_value
from stratum.optimizer.logical._ops import Op, OperandRef
from stratum.optimizer.logical._scoring import Metric, read_prediction

import logging
logger = logging.getLogger(__name__)

_MAX_LABEL = 50


class DeclaredScoringOp(Op):
    """Annotate ``pred`` with the scorers declared by ``pred.skb.with_scoring(...)``.

    Scorers retain skrub's ``{"scoring", "kwargs", "name"}`` representation.
    DataOps in scorer arguments, such as ``sample_weight``, bind to input operands.
    This operator passes ``pred`` through unchanged. Candidate construction removes
    the annotation and binds scorer kwargs for per-fold evaluation when using the
    declared metric (see ``install_candidate_set``).
    """

    logical_family = "DeclaredScoring"
    fields = ["scorers"]

    def __init__(self, scorers: list, inputs: list | None = None):
        super().__init__(inputs=list(inputs) if inputs else [])
        self.scorers = scorers

    def process(self, inputs: list):
        return inputs[0]


class CandidateSetOp(Op):
    """Fan-in over every candidate's output, carrying one label per candidate.

    The first ``len(candidate_names)`` inputs are candidate output operators,
    in label order.
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
        # Positional candidate access requires distinct slots for repeated operators.
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

    def process(self, inputs: list):
        return [{"id": name, "vals": inputs[i]}
                for i, name in enumerate(self.candidate_names)]


class ScoreCandidatesOp(CandidateSetOp):
    """Score each candidate against the test fold, one row per candidate.

    Inputs include candidate outputs and the test labels addressed by ``y_ref``.
    ``prediction_method`` identifies the method selected for test rows during
    train/test unrolling. The metric consumes the resulting predictions,
    probabilities or decision-function values directly.
    """

    logical_family = "ScoreCandidates"
    fields = ["candidate_names", "metric", "prediction_method", "emit_predictions", "y_ref",
              "kwargs_refs"]

    def __init__(self, candidate_names: list[str], metric: Metric,
                 inputs: list | None = None, prediction_method: str = "predict",
                 emit_predictions: bool = False, y_ref: OperandRef | None = None,
                 kwargs_refs: list | None = None):
        super().__init__(candidate_names, inputs)
        self.metric = metric
        self.prediction_method = prediction_method
        self.emit_predictions = emit_predictions
        self.output_type = OutputType.FRAME
        self.y_ref = y_ref
        # One entry per candidate's scorer kwargs, with OperandRefs for graph-fed
        # values evaluated per fold; None denotes absent kwargs.
        self.kwargs_refs = kwargs_refs

    def propagate_output_schema(self):
        """Describe scoring results: candidate name, score, and optional predictions.

        There is one row per candidate. Scoring runs on test rows only;
        the training copy has no candidate set.
        """
        # Candidate names are strings and scores are floats. Prediction values
        # may be arrays, frames or series, so their dtype remains unknown.
        schema = {"id": pl.String, "scores": pl.Float64}
        if self.emit_predictions:
            schema["vals"] = _schema.UNKNOWN_DTYPE
        self.output_schema = pl.Schema(schema)

    def clone(self):
        new_op = type(self)(self.candidate_names, self.metric,
                            prediction_method=self.prediction_method,
                            emit_predictions=self.emit_predictions, y_ref=self.y_ref,
                            kwargs_refs=clone_value(self.kwargs_refs))
        new_op.was_cloned = True
        return new_op

    def process(self, inputs: list):
        y_true = inputs[self.y_ref.k]
        ids, scores, values = [], [], []
        for i, name in enumerate(self.candidate_names):
            produced = inputs[i]
            kwargs = _resolve_operand(self.kwargs_refs[i], inputs) if self.kwargs_refs else None
            try:
                scores.append(self.metric(
                    y_true, read_prediction(produced, self.prediction_method), **(kwargs or {})))
            except Exception as e:
                raise RuntimeError(
                    f"[scoring] {self.metric!r} failed on candidate {name!r}"
                    f" reading its {self.prediction_method!r} values: {e}"
                ) from e
            ids.append(name)
            values.append(produced)
        table = {"id": ids, "scores": scores}
        if self.emit_predictions:
            table["vals"] = values
        return pl.DataFrame(table)
