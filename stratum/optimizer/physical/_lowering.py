"""Logical -> physical lowering pass.

Walks the logical DAG in topological order and replaces each logical op for
which a lowering rule is registered with its physical equivalent, splicing the
new node into the graph in place (the same mutate-in-place style as the frame
extraction pass). A logical op with no registered rule is left untouched -- it
passes through and still executes via its own ``process``. This is what lets op
families migrate to the physical layer one at a time without a flag day: an
un-lowered family simply keeps running as it does today.

A rule maps ``logical op -> physical op`` and may be 1:1 or (later) 1:N. It
returns the *sink* physical node; :func:`install_lowered` copies the logical
op's edges and X/y/type metadata onto it and rewires the neighbours. The
returned node keeps the logical op's place in the DAG.
"""
from __future__ import annotations

from typing import Callable

from stratum.optimizer.ir._base import IRNode, OutputType
from stratum.optimizer.ir._ops import Op
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer._op_utils import topological_iterator
from stratum.utils._utils import start_time, log_time

import logging
logger = logging.getLogger(__name__)


# logical op type -> rule(logical_op, ctx) -> PhysicalOp (the sink node).
_LOWERING_RULES: dict[type, Callable] = {}


def lowering_rule(*op_types):
    """Register ``fn`` as the lowering rule for each logical op type given."""
    def deco(fn):
        for t in op_types:
            _LOWERING_RULES[t] = fn
        return fn
    return deco


def install_lowered(old: Op, new: PhysicalOp, root: IRNode) -> IRNode:
    """Splice ``new`` into the DAG in place of the logical op ``old``.

    Copies ``old``'s edges and X/y/type metadata onto ``new`` (unless the rule
    already set a specific ``output_type``) and rewires every neighbour, so the
    physical node occupies exactly the logical node's position. Returns the new
    root when ``old`` was the root, else ``root`` unchanged.
    """
    new.inputs = old.inputs
    new.outputs = old.outputs
    new.is_X = old.is_X
    new.is_y = old.is_y
    new.output_schema = old.output_schema
    if new.output_type is OutputType.UNKNOWN:
        new.output_type = old.output_type
    for in_ in new.inputs:
        in_.replace_output(old, new)
    for out_ in new.outputs:
        out_.replace_input(old, new)
    return new if root is old else root


def lower_to_physical(root: IRNode, ctx) -> IRNode:
    """Rewrite the logical DAG into a (partly) physical DAG under ``ctx``.

    Ops with a registered rule are replaced by physical nodes; ops without one
    pass through unchanged. Returns the (possibly new) root.
    """
    start = start_time()
    for op in topological_iterator(root):
        rule = _LOWERING_RULES.get(type(op))
        if rule is None:
            continue
        new_op = rule(op, ctx)
        if new_op is None or new_op is op:
            continue
        root = install_lowered(op, new_op, root)
    log_time("lowering took", start)
    return root
