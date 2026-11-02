"""Unrolling fit and predict within a fold.

Every fold fits on its training rows and answers on its test rows. Rather than
running the plan after the split twice, once per mode, the plan holds both: a
*train copy* of everything downstream of the split, which runs in the fitting
mode on the training rows, and a *test copy*, which runs in the response mode on
the test rows. One run of the plan after the split is then one fold.

An op with state (:attr:`IRNode.has_state`: an estimator, a column selector, a
stateful skrub impl) is split in two. Its train copy becomes the ``"fit"`` stage,
which fits the state on the training rows; its test copy the ``"predict"``
stage, which applies that state, read through a *state edge* from the fit stage
(``state_ref``). A transformer's fit stage is ``fit_transform``, so it outputs
the transformed training rows as well; a predictor's only fits, and its
predictions on the training rows, where the train copy reads them, are a
``"predict"`` stage of their own.

Each op's mode is fixed here (``pass_mode``): the test copy's is the response
the plan is scored on, narrowed per estimator to the method it serves. An
``eval_mode()`` source the test copy reads gets a copy of its own, in that mode.

What a copy does not need goes: the train copy keeps only what feeds a fit
stage, and a stage keeps only the inputs its own call reads (a predict stage
never reads ``y``, the fit kwargs or graph-fed hyper-parameters), so whatever
only fed those in the test copy is pruned.
"""
from __future__ import annotations

from stratum.optimizer._op_utils import _iter_operand_refs, topological_iterator
from stratum.optimizer.logical._base import OperandRef, remap_operand_refs
from stratum.optimizer.logical._candidate_ops import ScoreCandidatesOp
from stratum.optimizer.logical._ops import BaseEstimatorOp, ImplOp, Op, PredictorOp
from stratum.optimizer.logical._split_ops import SplitOutput
from skrub._data_ops._data_ops import EvalMode
from stratum.utils._utils import start_time, log_time

import logging
logger = logging.getLogger(__name__)


def unroll_fit_predict(root: Op) -> Op:
    """Unroll the plan after the split into a train and a test copy. Returns the root,
    which is the test copy's (the plan answers with the test rows)."""
    start = start_time()
    ops = list(topological_iterator(root))
    split = next((op for op in ops if op.is_split_op), None)
    if split is None:
        # No fold, so nothing is fitted on one set of rows and applied to another.
        return root
    response = root.response_mode if isinstance(root, ScoreCandidatesOp) else "predict"

    region = _downstream(split, ops)
    train = {op: _clone(op) for op in region}
    for op in region:
        _wire_clone(op, train)
    for op in region:
        op.pass_mode = response

    # The training rows are only for fitting: drop the train copy's tail that no
    # fit reads, before it makes a predictor look read on the training rows.
    for op in reversed(list(train.values())):
        if not op.outputs and not op.has_state:
            op.detach()
    stages = [stage for op in region if op.has_state
              for stage in _split_stages(op, train[op])]
    for stage in stages:
        if isinstance(stage, BaseEstimatorOp):
            _keep_stage_inputs(stage)
    for op in reversed([*train.values(), *stages]):
        if not op.outputs and op.inputs and op is not root:
            # What no stage reads any more, now each reads only what it calls with.
            op.detach_and_prune()

    _split_eval_mode_sources(ops, set(region), response)
    log_time("fit/predict unrolling took", start)
    return root


def _downstream(split: Op, ops: list[Op]) -> list[Op]:
    """The ops downstream of ``split``, in topological order."""
    reached = {split}
    region = []
    for op in ops:
        if op is not split and any(in_ in reached for in_ in op.inputs):
            reached.add(op)
            region.append(op)
    return region


def _clone(op: Op) -> Op:
    """A copy of ``op`` for the train copy, without edges yet."""
    clone = op.clone()
    if isinstance(op, SplitOutput):
        clone.train = True
        clone.name = clone.name.replace("_test", "_train")
    # Constructors set these from what they are given, which a rewrite may since
    # have refined; schema propagation has run, so the schema is current.
    clone.output_type = op.output_type
    clone.output_schema = op.output_schema
    return clone


def _wire_clone(op: Op, train: dict) -> None:
    """Give ``op``'s train copy ``op``'s inputs, as their train copies where they have
    one. A producer keeps one output entry per entry it has for ``op``."""
    clone = train[op]
    clone.inputs = [train.get(in_, in_) for in_ in op.inputs]
    for in_ in dict.fromkeys(op.inputs):
        producer = train.get(in_, in_)
        producer.outputs.extend([clone] * sum(out is op for out in in_.outputs))


def _split_stages(test: Op, fit: Op) -> list[Op]:
    """Make ``fit`` (the train copy) the fit stage and ``test`` a predict stage
    applying its state. A predictor's predictions on the training rows, if the
    train copy reads them, become a predict stage of their own. Returns the stages."""
    fit.stage = "fit"
    stages = [fit, test]
    if isinstance(fit, PredictorOp) and fit.outputs:
        on_train = _clone(fit)
        on_train.inputs = list(fit.inputs)
        for in_ in dict.fromkeys(fit.inputs):
            in_.outputs.extend([on_train] * sum(out is fit for out in in_.outputs))
        for out in fit.outputs:
            out.inputs = [on_train if in_ is fit else in_ for in_ in out.inputs]
        on_train.outputs, fit.outputs = fit.outputs, []
        # What fitting followed by `predict` returns today.
        on_train.pass_mode = "predict"
        _add_state_edge(on_train, fit)
        stages.append(on_train)
    _add_state_edge(test, fit)
    return stages


def _add_state_edge(op: Op, fit: Op) -> None:
    op.stage = "predict"
    op.state_ref = OperandRef(len(op.inputs))
    op.inputs.append(fit)
    fit.add_output(op)
    if isinstance(op, BaseEstimatorOp):
        # The method it calls, as decided at run time before: a response the
        # estimator serves, or what it does instead.
        op.pass_mode = op.effective_mode(op.mode)


def _keep_stage_inputs(op: BaseEstimatorOp) -> None:
    """Narrow an estimator stage to what its own call reads, and drop the inputs
    nothing else it holds refers to.

    A fit stage reads ``y``, the column selection, graph-fed hyper-parameters and
    the kwargs of its fitting call; a predict stage only the kwargs of the method
    it calls, besides X and the state.
    """
    if op.stage == "fit":
        keep = {op.fit_kwargs_key}
    else:
        keep = {op.mode}
        op.y = None
        op.cols = op.exclude_cols = None
        op.no_wrap = False
        op.param_refs = {}
    op.kwargs = type(op.kwargs)((k, v) for k, v in op.kwargs.items() if k in keep)
    _drop_unreferenced_inputs(op)


def _drop_unreferenced_inputs(op: Op) -> None:
    """Remove the inputs of ``op`` that no field (nor X, input 0, nor the state edge)
    refers to, renumbering the refs, and prune what only they fed."""
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
        if not in_.outputs:
            in_.detach_and_prune()


def _split_eval_mode_sources(ops: list[Op], test_copy: set[Op], response: str) -> None:
    """Give the test copy its own ``eval_mode()`` sources, in its mode.

    The train copy and the ops before the split keep the original, in the fitting
    mode, as they read it before.
    """
    for source in ops:
        if not (isinstance(source, ImplOp) and isinstance(source.skrub_impl, EvalMode)):
            continue
        readers = [out for out in source.outputs if out in test_copy]
        if not readers:
            continue
        in_test = source.clone()
        in_test.pass_mode = response
        for out in readers:
            out.inputs = [in_test if in_ is source else in_ for in_ in out.inputs]
            in_test.outputs.append(out)
        source.outputs = [out for out in source.outputs if out not in test_copy]
