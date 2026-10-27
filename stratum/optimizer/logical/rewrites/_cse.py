"""Common Subexpression Elimination (CSE) on stratum's Op IR.

Runs on the Op DAG produced by ``convert_to_ops`` (an earlier skrub-DataOp-IR CSE
that ran before conversion has been removed in favour of this pass).

The Op DAG is an explicit, bidirectional graph (``Op.inputs`` / ``Op.outputs``)
visited in topological order (inputs before outputs), which makes CSE a classic
*value-numbering* pass: each op is identified by ``Op.structure_key()`` (equal
iff two ops are the same computation), the first op with a given key is canonical,
and later duplicates are merged into it.

Equality/hashing of ops themselves stays identity-based (the graph layer keys ops
by identity in many sets/dicts), so the structural key lives in ``structure_key``
rather than in ``__eq__``/``__hash__``.

Merging redirects a duplicate's consumers to the canonical op. Most ops address
their operands by index via :class:`OperandRef`, so when input edges are
de-duplicated their refs are renumbered. Positional consumers that do *not* use
operand refs (``ChoiceOp`` outcomes, ``ImplOp``'s ``operand_index``) cannot have
two input slots collapsed into one, so such a merge is skipped (see ``_can_merge``).
"""
from __future__ import annotations

from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer.logical._ops import Op
import logging

logger = logging.getLogger(__name__)

def apply_op_cse(root: Op) -> Op:
    """Run CSE on an Op DAG in place and return the (possibly new) root.

    A single topological pass assigns each op a structure key and merges any op
    whose key was already seen into the first (canonical) op carrying that key.
    """
    table: dict = {}
    new_root = root
    # Materialize the order *before* mutating: merging rewires inputs/outputs,
    # which would otherwise corrupt the lazy iterator's indegree bookkeeping.
    for op in list(topological_iterator(root)):
        key = op.structure_key()
        if key is None:
            continue
        canonical = table.get(key)
        if canonical is None:
            table[key] = op
        elif canonical is not op and _can_merge(op, canonical):
            logger.debug("CSE: eliminating %r in favor of %r", op, canonical)
            _merge_op(op, canonical)
    return new_root


def _can_merge(op: Op, canonical: Op) -> bool:
    """Whether ``op`` can be merged into ``canonical`` without corrupting a consumer.

    Merging redirects each consumer's edge from ``op`` to ``canonical``. If a
    consumer already references ``canonical``, that collapses two of its input
    slots into one. For OperandRef-based ops this is fine (refs are renumbered),
    but a positional consumer (ChoiceOp/ImplOp) needs its slots kept distinct, so
    we refuse the merge and leave both ops in place.
    """
    for out in op.outputs:
        if out.consumes_inputs_positionally() and any(i is canonical for i in out.inputs):
            return False
    return True


def _merge_op(op: Op, canonical: Op) -> None:
    """Redirect ``op``'s consumers to ``canonical`` and detach ``op``.

    ``op`` and ``canonical`` share the same input objects (that is why they are
    equal), so we only move the output edges: every consumer is rebound from
    ``op`` to ``canonical``, then ``op`` is dropped from its inputs' output lists.
    ``IRNode.replace_input`` collapses the consumer's slot onto ``canonical``'s
    when it already reads it, renumbering its refs, or swaps it in place for a
    positional consumer (``_can_merge`` refuses the case where that would be
    needed).
    """
    for out in list(op.outputs):
        while any(i is op for i in out.inputs):
            out.replace_input(op, canonical)
        canonical.add_output(out)
    op.detach()
