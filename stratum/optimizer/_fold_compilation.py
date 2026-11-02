"""Staged compilation: the plan after the split is compiled at the split, and copied per fold.

The optimizer compiles the whole plan up front, so that the ops before the split
are chosen knowing what reads them after it. Those run once. What follows the
split runs once per fold, and is compiled again when the scheduler reaches the
split, from a :class:`FoldTemplate` the optimizer keeps: the logical plan from
the split on, as fit/predict unrolling left it. By then the ops before the split
have run and are fixed; the template reads each through a
:class:`~stratum.optimizer.logical._split_ops.BoundaryOp`, which implementation
selection takes as given.

Compiling the template gives a :class:`FoldPlan`, which each fold runs a fresh
copy of (:meth:`FoldPlan.instantiate`), with the fold's rows bound to its split.

Today the compile at the split decides what the initial compile did; it is where
decisions made from what the run has measured will go.
"""
from __future__ import annotations

import copy

from stratum.optimizer._input_removal_planning import plan_input_removals
from stratum.optimizer._linearization import linearize_dag
from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer.logical._base import IRNode
from stratum.optimizer.logical._split_ops import BoundaryOp
from stratum.optimizer.physical._conversion import ConversionOp
from stratum.optimizer.physical._impl_selection import select_implementations
from stratum.optimizer.physical._lowering import lower_to_physical
from stratum.utils._utils import start_time, log_time

import logging
logger = logging.getLogger(__name__)


def post_split_region(ops: list[IRNode]) -> set[IRNode]:
    """The split op and everything downstream of it: what runs once per fold."""
    split = next((op for op in ops if op.is_split_op), None)
    post_split = set()
    stack = [split] if split is not None else []
    while stack:
        op = stack.pop()
        if op not in post_split:
            post_split.add(op)
            stack.extend(op.outputs)
    return post_split


def copy_op(op: IRNode) -> IRNode:
    """A copy of ``op`` with the same class and attributes, without edges or buffer
    lifetimes.

    Attributes are shared, not copied: compiling and running an op reassign them.
    A stateful skrub impl is the exception, as it fits itself in place; the copies
    share it as the folds of one op did, which is sound while folds run one at a
    time.
    """
    new = copy.copy(op)
    new.inputs, new.outputs = [], []
    new.remove_after, new.release_state_after = [], []
    return new


def _copy_region(ops: list[IRNode], outside=None) -> dict[IRNode, IRNode]:
    """Copies of ``ops`` (in topological order), wired among themselves. An input
    from outside is ``outside(op)``, or the op itself when ``outside`` is None."""
    copies = {op: copy_op(op) for op in ops}
    for op in ops:
        new = copies[op]
        new.inputs = [copies[in_] if in_ in copies
                      else in_ if outside is None else outside(in_) for in_ in op.inputs]
        # An op downstream of the split feeds only ops downstream of it.
        new.outputs = [copies[out] for out in op.outputs]
    return copies


class FoldTemplate:
    """The logical plan from the split on, retained to compile the folds from.

    ``root`` is the plan's root. ``boundaries`` read the ops before the split, one
    BoundaryOp per op.
    """

    def __init__(self, root: IRNode, boundaries: list[BoundaryOp]):
        self.root = root
        self.boundaries = boundaries

    @classmethod
    def retain(cls, ops: list[IRNode], root: IRNode) -> FoldTemplate | None:
        """Copy the plan ``ops`` (topological, ending in ``root``) from the split on.
        None if there is no split, or the root does not follow it."""
        region = post_split_region(ops)
        if root not in region:
            return None
        boundaries: dict[IRNode, BoundaryOp] = {}

        def boundary(op):
            if op not in boundaries:
                boundaries[op] = BoundaryOp(op)
            return boundaries[op]

        copies = _copy_region([op for op in ops if op in region], boundary)
        for op, b in boundaries.items():
            # One entry per edge, as the op has.
            b.outputs = [copies[out] for out in op.outputs if out in copies]
        return cls(copies[root], list(boundaries.values()))

    def bind(self, replaced: dict[IRNode, IRNode]) -> None:
        """Point the boundaries at the physical ops lowering put in place of the
        logical ones they stand for (``replaced``)."""
        for b in self.boundaries:
            b.stands_for = replaced.get(b.stands_for, b.stands_for)

    def sources(self) -> set[IRNode]:
        """The ops before the split the plan after it reads."""
        return {b.stands_for for b in self.boundaries}

    def compile(self, ctx, selector, live: set[IRNode], registry=None) -> FoldPlan:
        """Lower a copy of the template and select its implementations.

        ``live`` are the ops before the split whose values the pool keeps for the
        folds. A conversion of one of them that the plan before the split already
        made is read from there rather than repeated in every fold.
        """
        start = start_time()
        # The boundaries are the template's sources, so they are copied too.
        copies = _copy_region(list(topological_iterator(self.root)))
        root = lower_to_physical(copies[self.root], ctx)
        root = select_implementations(root, ctx, registry=registry, selector=selector)

        order, _ = linearize_dag(root)
        # Inputs from before the split: the ops the boundaries stand for, or the
        # conversions of them made there.
        before: dict[IRNode, IRNode] = {}
        fold = []
        for op in order:
            if isinstance(op, BoundaryOp):
                continue
            if isinstance(op, ConversionOp) and isinstance(op.inputs[0], BoundaryOp):
                done = _done_conversion(op, op.inputs[0].stands_for, live)
                if done is not None:
                    before[op] = done
                    continue
            fold.append(op)
        for op in fold:
            op.inputs = [before[in_] if in_ in before
                         else _live(in_.stands_for, live) if isinstance(in_, BoundaryOp)
                         else in_ for in_ in op.inputs]
        log_time("compiling the plan after the split took", start)
        return FoldPlan(fold)


def _live(op: IRNode, live: set[IRNode]) -> IRNode:
    if op not in live:
        raise RuntimeError(
            f"The plan after the split reads {op}, which the plan before it did not"
            " keep for the folds")
    return op


def _done_conversion(conversion: ConversionOp, source: IRNode,
                     live: set[IRNode]) -> IRNode | None:
    """The same conversion of ``source``, if the plan before the split made it and
    kept it for the folds."""
    return next((out for out in source.outputs
                 if isinstance(out, ConversionOp) and out.target is conversion.target
                 and out in live), None)


class FoldPlan:
    """The plan of one fold, compiled: ops in execution order, from the split op to
    the root. Inputs from before the split are those ops themselves."""

    def __init__(self, ops: list[IRNode]):
        self.ops = ops

    def instantiate(self, indices) -> list[IRNode]:
        """A copy of the plan to run one fold, whose rows are ``indices``, the
        ``(train, test)`` row positions. Each copy's ``instance_of`` is the op it
        copies, and its buffers are released after their last reader in the fold.
        """
        copies = _copy_region(self.ops)
        fold = []
        for op in self.ops:
            new = copies[op]
            new.instance_of = op
            if new.is_split_op:
                new.indices = indices
            fold.append(new)
        # Read by every fold: what comes before the split stays in the pool.
        members = set(fold)
        plan_input_removals(fold, {in_ for op in fold for in_ in op.inputs
                                   if in_ not in members})
        return fold
