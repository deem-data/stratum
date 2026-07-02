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
from stratum.optimizer.ir._ops import ChoiceOp, ImplOp, Op, OperandRef
import logging

logger = logging.getLogger(__name__)

# Attributes that are graph structure rather than configuration; never rewritten
# when renumbering operand refs. Mirrors the exclusion set in `validate_operands`.
_STRUCTURAL_ATTRS = frozenset({"inputs", "outputs", "remove_after"})

# Consumers whose inputs are positional and *not* addressed by OperandRef:
# ChoiceOp consumes its outcomes by position, ImplOp resolves inputs via a cached
# id(DataOp)->index map. Their input slots cannot be collapsed by edge dedup.
_POSITIONAL_CONSUMER_TYPES = (ChoiceOp, ImplOp)


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
        if isinstance(out, _POSITIONAL_CONSUMER_TYPES) and any(i is canonical for i in out.inputs):
            return False
    return True


def _merge_op(op: Op, canonical: Op) -> None:
    """Redirect ``op``'s consumers to ``canonical`` and detach ``op``.

    ``op`` and ``canonical`` share the same input objects (that is why they are
    equal), so we only move the output edges: every consumer is rebound from
    ``op`` to ``canonical``, then ``op`` is dropped from its inputs' output lists.
    """
    for out in list(op.outputs):
        _rebind_consumer(out, op, canonical)
        canonical.add_output(out)
    for inp in op.inputs:
        inp.outputs = [o for o in inp.outputs if o is not op]
    op.inputs = []
    op.outputs = []


def _rebind_consumer(out: Op, old_op: Op, new_op: Op) -> None:
    """Replace ``old_op`` with ``new_op`` in ``out``'s inputs and renumber refs.

    Input edges are de-duplicated, so if ``out`` already references ``new_op`` the
    substitution collapses two edges into one and the consumer's :class:`OperandRef`s
    are remapped to the new, compacted index space. ``_can_merge`` guarantees such a
    collapse only happens for OperandRef-based consumers.
    """
    substituted = [new_op if x is old_op else x for x in out.inputs]

    new_inputs: list = []
    id_to_new: dict = {}
    old_to_new: dict = {}
    for old_k, x in enumerate(substituted):
        new_k = id_to_new.get(id(x))
        if new_k is None:
            new_k = len(new_inputs)
            new_inputs.append(x)
            id_to_new[id(x)] = new_k
        old_to_new[old_k] = new_k

    # Operand refs live in the declared config `fields`; fall back to scanning all
    # config attributes for the rare op that carries refs without declaring fields.
    fields = getattr(type(out), "fields", None)
    attrs = fields if fields is not None else [a for a in out.__dict__ if a not in _STRUCTURAL_ATTRS]
    for attr in attrs:
        setattr(out, attr, _remap_refs(getattr(out, attr), old_to_new))
    out.inputs = new_inputs


def _remap_refs(value, mapping: dict):
    """Return ``value`` with every nested OperandRef remapped via ``mapping``."""
    if isinstance(value, OperandRef):
        return OperandRef(mapping[value.k])
    if isinstance(value, tuple):
        return tuple(_remap_refs(v, mapping) for v in value)
    if isinstance(value, list):
        return [_remap_refs(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: _remap_refs(v, mapping) for k, v in value.items()}
    if hasattr(value, "remap_operand_refs"):
        # Column-expression tree (immutable) -> rebuild with remapped refs.
        return value.remap_operand_refs(mapping)
    return value
