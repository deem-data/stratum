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
With a reuse cache, the copy is where reuse is planned for the fold: its keys
follow from its rows, and what the cache holds of it is taken and pruned.

The template is compiled again after the first fold, with what that fold
measured (a :class:`FoldProfile`, which implementation selection reads from the
plan context). The remaining folds run the plan that compile gives, or keep the
one they had if it decides the same. No selector reads the profile yet, so today
they always keep it.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from stratum.optimizer._input_removal_planning import plan_input_removals
from stratum.optimizer._linearization import linearize_dag
from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer._reuse_planning import plan_fold_reuse, post_split_region
from stratum.optimizer.logical._base import IRNode
from stratum.optimizer.logical._split_ops import BoundaryOp
from stratum.optimizer.physical._conversion import ConversionOp
from stratum.optimizer.physical._impl_selection import select_implementations
from stratum.optimizer.physical._lowering import lower_to_physical
from stratum.utils._utils import start_time, log_time

import logging
logger = logging.getLogger(__name__)


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

    def compile(self, ctx, selector, live: set[IRNode], registry=None,
                known: dict | None = None) -> FoldPlan:
        """Lower a copy of the template and select its implementations.

        ``live`` are the ops before the split whose values the pool keeps for the
        folds. ``known`` maps some of them to their value, a plain scalar, which
        the plan is compiled knowing (``BoundaryOp.known_value``). A conversion of one of them that the plan before the split already
        made is read from there rather than repeated in every fold. An op before
        the split the pool did not keep is read through a conversion of it that it
        kept, converted back if the plan needs it so.
        """
        start = start_time()
        # The boundaries are the template's sources, so they are copied too.
        copies = _copy_region(list(topological_iterator(self.root)))
        for op, new in copies.items():
            if isinstance(new, BoundaryOp):
                new.stands_for = _kept(op.stands_for, live)
                if known and new.stands_for in known:
                    new.known_value = known[new.stands_for]
        replaced = {}
        root = lower_to_physical(copies[self.root], ctx, replaced)
        for op, new in copies.items():
            replaced.get(new, new).origin = op
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
                         else in_.stands_for if isinstance(in_, BoundaryOp)
                         else in_ for in_ in op.inputs]
        log_time("compiling the plan after the split took", start)
        return FoldPlan(fold)


#: The values a plan after the split is compiled knowing: plain scalars (numpy's
#: among them), cheap to read from the pool and small enough to hold in the plan.
KNOWN_VALUE_TYPES = (str, int, float, bool)


def _kept(op: IRNode, live: set[IRNode]) -> IRNode:
    """``op``, or a conversion of it, whichever the pool keeps for the folds.

    The plan before the split frees an op read only through a conversion of it,
    which a compile choosing other libraries may no longer read.
    """
    if op in live:
        return op
    for out in op.outputs:
        if isinstance(out, ConversionOp) and out in live:
            return out
    raise RuntimeError(
        f"The plan after the split reads {op}, which the plan before it did not"
        " keep for the folds")


def _done_conversion(conversion: ConversionOp, source: IRNode,
                     live: set[IRNode]) -> IRNode | None:
    """The same conversion of ``source``, if the plan before the split made it and
    kept it for the folds."""
    return next((out for out in source.outputs
                 if isinstance(out, ConversionOp) and out.target is conversion.target
                 and (out._tracks_index or not conversion._tracks_index)
                 and out in live), None)


class FoldPlan:
    """The plan of one fold, compiled: ops in execution order, from the split op to
    the root. Inputs from before the split are those ops themselves.

    ``reuse``, when given, is the ``(cache, admission policy)`` to plan each fold's
    reuse with.
    """

    def __init__(self, ops: list[IRNode], reuse=None):
        self.ops = ops
        self.reuse = reuse

    def decides_as(self, other: FoldPlan) -> bool:
        """Whether the two plans run the same implementations in the same order on
        the same inputs: compiled from the same template ops, with the same
        choices made for them."""
        return _decisions(self.ops) == _decisions(other.ops)

    def instantiate(self, indices) -> list[IRNode]:
        """A copy of the plan to run one fold, whose rows are ``indices``, the
        ``(train, test)`` row positions. Each copy's ``instance_of`` is the op it
        copies, and its buffers are released after their last reader in the fold.
        With ``reuse``, a copy the cache holds is a hit, and what only hits read is
        left out.
        """
        copies = _copy_region(self.ops)
        fold = []
        for op in self.ops:
            new = copies[op]
            new.instance_of = op
            if new.is_split_op:
                new.indices = indices
            fold.append(new)
        if self.reuse is not None:
            fold = plan_fold_reuse(fold, *self.reuse)
        # Read by every fold: what comes before the split stays in the pool.
        members = set(fold)
        plan_input_removals(fold, {in_ for op in fold for in_ in op.inputs
                                   if in_ not in members})
        return fold


def _decisions(ops: list[IRNode]) -> list[tuple]:
    at = {op: i for i, op in enumerate(ops)}
    return [(type(op), getattr(op, "_frame_lib", None), op._tracks_index, op.origin,
             op.stage, op.mode,
             tuple(("fold", at[in_]) if in_ in at else ("before", id(in_))
                   for in_ in op.inputs))
            for op in ops]


@dataclass
class Measured:
    """What running an op in a fold measured: ``seconds`` it took, and the bytes
    of its output (``size``) and of the state it fitted (``state_size``), each 0
    if it kept none."""
    seconds: float
    size: int = 0
    state_size: int = 0


class FoldProfile:
    """What running one fold measured, for the compile after it to decide from.

    ``ops`` maps each template op (see ``IRNode.origin``) to what its compiled op
    measured. An op the fold took from the cache did not run and is not measured,
    nor is an op compiling added that no template op stands for, such as a
    conversion. ``peak`` is the most bytes the fold's outputs and states held at
    once, over what the pool held when the fold started.
    """

    def __init__(self):
        self.ops: dict[IRNode, Measured] = {}
        self.peak = 0

    def of(self, op: IRNode) -> Measured | None:
        """What was measured of the template op ``op`` was compiled from."""
        return self.ops.get(op.origin) if op.origin is not None else None
