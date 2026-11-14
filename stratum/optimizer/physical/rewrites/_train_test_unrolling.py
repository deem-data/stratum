"""Bind execution modes and unroll train/test copies of the selected plan.

- Copy bound implementations and their configuration after selection, so train
  and test copies use the same backend without repeating initialization.
- Duplicate the plan downstream of the split into train and test copies: fit on
  training rows and evaluate on test rows within one execution per fold.
- Stateful fitting returns data and fitted state. Response operators read state
  through ``state_ref``; transformers fit with ``fit_transform``.
- Predictors use separate ``fit`` and ``predict`` operators, including before
  the split and in plans without a split.
- Bind each op's ``mode``. The test copy uses the requested response method,
  resolved per estimator, with separate ``eval_mode()`` sources.
- Retain only each estimator method's required inputs. Responses omit ``y``,
  fit kwargs and graph-fed hyperparameters. A subsequent dead-code elimination
  pass removes unused branches and training predictions.
"""
from __future__ import annotations

import copy

from stratum.optimizer._op_utils import _iter_operand_refs, topological_iterator
from stratum.optimizer.logical._base import IRNode, OperandRef, remap_operand_refs
from stratum.optimizer.logical._candidate_ops import ScoreCandidatesOp
from stratum.optimizer.logical._ops import BaseEstimatorOp, ImplOp, PredictorOp
from stratum.optimizer.logical._split_ops import SplitOutput
from skrub._data_ops._data_ops import EvalMode
from stratum.utils._utils import start_time, log_time

import logging
logger = logging.getLogger(__name__)


def unroll_train_test(root: IRNode) -> IRNode:
    """Bind modes and construct explicit fitted-state dependencies.

    With a split, return the root of the test copy. Otherwise return the root
    producing the full-data result. Train/test metadata is only set on copies.

    Run once after implementation selection, before reuse and lifetime planning.
    Run dead-code elimination before traversing the resulting DAG: unused train
    copies remain connected through producers' output edges.
    """
    start = start_time()
    ops = list(topological_iterator(root))
    split = next((op for op in ops if op.is_split_op), None)
    response = root.response_mode if isinstance(root, ScoreCandidatesOp) else "predict"

    region = _downstream(split, ops) if split is not None else []
    train = {op: _clone(op) for op in region}
    for op in region:
        _wire_clone(op, train)
    for op in region:
        train[op].train = True
        op.train = False
        op.mode = response

    bound = [node for op in region if op.has_state
             for node in _split_stateful(op, train[op])]
    # Full-data predictors also need explicit state and a separate data output.
    for op in ops:
        if op in train:
            continue
        if isinstance(op, PredictorOp):
            predict = _add_prediction(op)
            if root is op:
                root = predict
            bound.extend((op, predict))
        elif isinstance(op, BaseEstimatorOp):
            bound.append(op)
    for op in bound:
        if isinstance(op, BaseEstimatorOp):
            _keep_mode_inputs(op)

    _split_eval_mode_sources(ops, set(region), response)
    log_time("train/test unrolling took", start)
    return root


def _downstream(split: IRNode, ops: list[IRNode]) -> list[IRNode]:
    """The ops downstream of ``split``, in topological order."""
    reached = {split}
    region = []
    for op in ops:
        if op is not split and any(in_ in reached for in_ in op.inputs):
            reached.add(op)
            region.append(op)
    return region


def _clone(op: IRNode) -> IRNode:
    """Copy bound configuration without rerunning constructors or selection hooks.

    Configuration is shared: this rewrite replaces fields, and execution fits
    fresh estimators. Graph edges and lifetime bookkeeping belong to each copy.
    """
    clone = copy.copy(op)
    clone.inputs, clone.outputs = [], []
    clone.remove_after, clone.release_state_after = [], []
    clone.was_cloned = True
    if isinstance(op, SplitOutput):
        clone.train = True
        clone.name = clone.name.replace("_test", "_train")
    return clone


def _wire_clone(op: IRNode, train: dict) -> None:
    """Bind the train copy's inputs to cloned producers where available.

    Preserve the multiplicity of each producer-consumer edge.
    """
    clone = train[op]
    clone.inputs = [train.get(in_, in_) for in_ in op.inputs]
    for in_ in dict.fromkeys(op.inputs):
        producer = train.get(in_, in_)
        producer.outputs.extend([clone] * sum(out is op for out in in_.outputs))


def _split_stateful(test: IRNode, fit: IRNode) -> list[IRNode]:
    """Bind fitting and response modes and connect their fitted state.

    The test copy consumes state produced by the train copy. For predictors with
    training-output consumers, insert a separate prediction operator on training rows.
    """
    fit.mode = fit.fitting_mode
    bound = [fit, test]
    if isinstance(fit, PredictorOp) and fit.outputs:
        bound.append(_add_prediction(fit))
    _add_state_edge(test, fit)
    return bound


def _add_prediction(fit: PredictorOp) -> PredictorOp:
    """Move a predictor's data consumers to a separate prediction operator."""
    fit.mode = "fit"
    predict = _clone(fit)
    predict.inputs = list(fit.inputs)
    for in_ in dict.fromkeys(fit.inputs):
        in_.outputs.extend([predict] * sum(out is fit for out in in_.outputs))
    for out in fit.outputs:
        out.inputs = [predict if in_ is fit else in_ for in_ in out.inputs]
    predict.outputs, fit.outputs = fit.outputs, []
    predict.mode = "predict"
    _add_state_edge(predict, fit)
    return predict


def _add_state_edge(op: IRNode, fit: IRNode) -> None:
    op.state_ref = OperandRef(len(op.inputs))
    op.inputs.append(fit)
    fit.add_output(op)
    if isinstance(op, BaseEstimatorOp):
        # Resolve the requested response to a supported method or its fallback.
        op.mode = op.effective_mode(op.mode)
    else:
        op.mode = "transform"


def _keep_mode_inputs(op: BaseEstimatorOp) -> None:
    """Retain the bound method's arguments and prune unreferenced inputs.

    Fitting arguments include ``X``, ``y``, column selectors, graph-fed
    hyperparameters and fitting kwargs. Response arguments are limited to
    ``X``, fitted state and kwargs for the resolved response method.
    """
    if op.produces_state:
        keep = {op.fit_kwargs_key}
    else:
        keep = {op.mode}
        op.y = None
        op.cols = op.exclude_cols = None
        op.no_wrap = False
        op.param_refs = {}
    op.kwargs = type(op.kwargs)((k, v) for k, v in op.kwargs.items() if k in keep)
    _drop_unreferenced_inputs(op)


def _drop_unreferenced_inputs(op: IRNode) -> None:
    """Prune unreferenced inputs and renumber operand references.

    Retain ``X`` at input 0, the state edge and inputs referenced by operator
    fields. Dead-code elimination removes unused producers.
    """
    used = {0}
    for name in type(op).fields:
        used.update(ref.k for ref in _iter_operand_refs(getattr(op, name)))
    if op.state_ref is not None:
        used.add(op.state_ref.k)
    dropped = [in_ for k, in_ in enumerate(op.inputs) if k not in used]
    if not dropped:
        return
    mapping = {old: new for new, old in enumerate(sorted(used))}
    op.inputs = [op.inputs[k] for k in sorted(used)]
    for name in type(op).fields:
        setattr(op, name, remap_operand_refs(getattr(op, name), mapping))
    if op.state_ref is not None:
        op.state_ref = remap_operand_refs(op.state_ref, mapping)
    for in_ in dict.fromkeys(dropped):
        if any(kept is in_ for kept in op.inputs):
            continue
        in_.outputs = [out for out in in_.outputs if out is not op]


def _split_eval_mode_sources(ops: list[IRNode], test_copy: set[IRNode], response: str) -> None:
    """Clone ``eval_mode()`` sources for the test copy's response mode.

    The train copy and operators before the split retain the original sources
    in fitting mode.
    """
    for source in ops:
        if not (isinstance(source, ImplOp) and isinstance(source.skrub_impl, EvalMode)):
            continue
        readers = [out for out in source.outputs if out in test_copy]
        if not readers:
            continue
        in_test = _clone(source)
        in_test.mode = response
        in_test.train = False
        for out in readers:
            out.inputs = [in_test if in_ is source else in_ for in_ in out.inputs]
            in_test.outputs.append(out)
        source.outputs = [out for out in source.outputs if out not in test_copy]
