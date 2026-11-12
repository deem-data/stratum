"""Inline scalar ``ValueOp``s into consumers that can hold the literal directly."""
from stratum.optimizer._op_utils import _iter_operand_refs, topological_iterator
from stratum.optimizer.logical._base import _resolve_operand
from stratum.optimizer.logical._dataframe_ops import ConcatOp
from stratum.optimizer.logical._ops import (BaseEstimatorOp, CallOp, GetItemOp,
                                             MethodCallOp, Op, OperandRef, ValueOp,
                                             _getitem_key_name)


def _is_literal(op: Op) -> bool:
    return (isinstance(op, ValueOp) and not (op.is_X or op.is_y)
            and type(op.value) in (type(None), bool, int, float, str, bytes))


def _inlinable(op: Op, k: int) -> bool:
    """Whether input slot ``k`` of ``op`` may be replaced by its literal."""
    if isinstance(op, ConcatOp):
        return op.axis == OperandRef(k) and type(op.inputs[k].value) in (int, str)
    if isinstance(op, (MethodCallOp, GetItemOp)):
        return k != 0  # the receiver / container stays a graph input
    if isinstance(op, CallOp):
        # Numeric extraction reads a numpy call's first argument as its operand.
        return k != 0 or getattr(op.func, "__module__", None) != "numpy"
    return False


def literal_inlining(root: Op) -> Op:
    """Substitute eligible literals and remove input edges with no remaining use."""
    # Snapshot the traversal before changing edges.
    for op in list(topological_iterator(root)):
        if isinstance(op, BaseEstimatorOp):
            _inline_estimator_params(op)
        elif slots := {k for k, in_ in enumerate(op.inputs)
                       if _is_literal(in_) and _inlinable(op, k)}:
            _inline_slots(op, slots)
    return root


def _inline_estimator_params(op: BaseEstimatorOp) -> None:
    """Move scalar parameter inputs into an owned, unfitted estimator prototype.

    ``param_refs`` retains only runtime bindings. An input also used by X, y or
    another Apply field stays an input even after its parameter use is inlined.
    """
    literals = {name: ref for name, ref in op.param_refs.items()
                if _is_literal(op.inputs[ref.k])}
    if not literals:
        return

    # Each converted op owns both prototypes. Planning and fitting must see the same
    # parameters, so update both before removing their runtime bindings.
    params = {name: op.inputs[ref.k].value for name, ref in literals.items()}
    op.estimator.set_params(**params)
    op.original_estimator.set_params(**params)
    op.supported_modes = frozenset(m for m in op.response_modes if hasattr(op.estimator, m))
    op.param_refs = {name: ref for name, ref in op.param_refs.items() if name not in literals}

    used = {0}  # X is implicit, so it does not appear in the fields below.
    used.update(ref.k for field in op.fields
                for ref in _iter_operand_refs(getattr(op, field)))
    if unused := {ref.k for ref in literals.values()} - used:
        _inline_slots(op, unused)


def _inline_slots(op: Op, inlined: set[int]) -> None:
    """One substitution both inlines and renumbers: an inlined slot resolves to
    its value, a kept slot to a ref to its new position."""
    substitute, kept = [], []
    for k, in_ in enumerate(op.inputs):
        if k in inlined:
            substitute.append(in_.value)
            in_.outputs = [out for out in in_.outputs if out is not op]
        else:
            substitute.append(OperandRef(len(kept)))
            kept.append(in_)
    for field in op.fields:
        setattr(op, field, _resolve_operand(getattr(op, field), substitute))
    op.inputs = kept
    if isinstance(op, GetItemOp) and next(_iter_operand_refs(op.key), None) is None:
        op.name = _getitem_key_name(op.key)
