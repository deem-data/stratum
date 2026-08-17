import numbers

from stratum.optimizer.ir._numeric_ops import NumericOp, NumericOpType
from stratum.optimizer._op_utils import rewrite_pass, replace_op_in_outputs
from stratum.optimizer.ir._ops import Op, ValueOp, OperandRef
import numpy as np


def _is_scalar_const(value) -> bool:
    """True if ``value`` is a scalar numeric constant safe to compare with ``==``.

    Guards against ndarray constants (e.g. ``df * np.array([...])``), whose
    ``== const`` yields an array and raises "truth value of an array is
    ambiguous" when used in a boolean context.

    ``numbers.Real`` rather than ``(int, float)`` so numpy scalars match too:
    ``np.int64`` is not an ``int`` subclass and would otherwise skip every
    identity rewrite. ``Real`` still excludes ndarray and ``complex``.
    """
    return isinstance(value, numbers.Real)


def _matches_scalar_const(op, const, reversed=None):
    """Return whether ``op`` has the requested scalar constant operand."""
    return (
        op.opt_operand is None
        and _is_scalar_const(op.constant)
        and op.constant == const
        and (reversed is None or op.reversed == reversed)
    )


def match_two_op_chain(op_cls, type1, type2, *, match1=None, match2=None):
    """Match a typed two-op chain with optional per-operation predicates."""
    def match(op1):
        if (
            isinstance(op1, op_cls)
            and op1.type is type1
            and len(op1.outputs) == 1
            and (match1 is None or match1(op1))
        ):
            op2 = op1.outputs[0]
            if (
                isinstance(op2, op_cls)
                and op2.type is type2
                and (match2 is None or match2(op2))
            ):
                return (op1, op2)
        return None
    return match


def match_identity_operation(op_cls, type1, const, reversed=None):
    """Match a var-const NumericOp that performs an identity transformation.

    Parameters
    ----------
    reversed : bool or None
        If None, the ``reversed`` flag is not checked (e.g. multiplication is
        commutative so ``x*1`` and ``1*x`` are both identities).  Set to
        ``True`` / ``False`` for non-commutative operations like subtraction.
    """
    def match(op1):
        if isinstance(op1, op_cls) and op1.type == type1:
            if _matches_scalar_const(op1, const, reversed):
                return (op1,)
        return None
    return match


def match_self_operation(op_cls, op_type):
    """Match a var-var operation whose two operands are the same node (``x <op> x``).

    Counterpart to :func:`match_identity_operation` for the var-var case: instead
    of a constant operand it requires the optional operand to be an
    :class:`OperandRef` pointing back at input 0. ``x <op> x`` binds a single
    de-duplicated input edge (see :class:`~stratum.optimizer.ir._ops.OperandBinder`),
    so both operands resolving to ``inputs[0]`` is exactly the self-operation case.
    ``reversed`` is irrelevant here -- both sides are the same node.
    """
    def match(op):
        if (
            isinstance(op, op_cls)
            and op.type is op_type
            and isinstance(op.opt_operand, OperandRef)
            and op.inputs[op.opt_operand.k] is op.inputs[0]
        ):
            return (op,)
        return None
    return match


def eliminate_single_op_chain_root_safe(op, root):
    eliminate_single_op_chain(op)
    if op is root:
        root = op.inputs[0]
    return root


def eliminate_single_op_chain(op):
    primary = op.inputs[0]
    op.replace_input_of_outputs(primary)
    primary.outputs.remove(op)
    for out_ in op.outputs:
        primary.add_output(out_)


def eliminate_two_op_chain(op1, op2):
    """Remove a redundant pair of inverse ops: y = f(op2(op1(x))) -> y = f(x).

    Rewires the DAG in-place so that op1's input connects directly to op2's output.
    """
    x = op1.inputs[0]
    x.outputs = [out for out in x.outputs if out is not op1]
    replace_op_in_outputs(op2, x)


def eliminate_two_op_chain_root_safe(op1: Op, op2: Op, root: Op) -> Op:
    """Wrapper around eliminate_two_op_chain that handles the case where
    op2 is the root (last node) of the DAG -- returns the updated root."""
    eliminate_two_op_chain(op1, op2)
    if op2 is root:
        root = op1.inputs[0]
    return root


def replace_two_op_chain(op1: Op, op2: Op, replacement: Op):
    """Replace op1 -> op2 with replacement: x -> replacement -> downstream."""
    x = op1.inputs[0]
    x.replace_output(op1, replacement)
    replacement.add_input(x)
    for downstream in op2.outputs:
        replacement.add_output(downstream)
        downstream.replace_input(op2, replacement)


def make_replace_two_op_chain_root_safe(make_replacement):
    """Action factory: replace a two-op chain with a new op from make_replacement()."""
    def action(op1: Op, op2: Op, root: Op) -> Op:
        replacement = make_replacement()
        replace_two_op_chain(op1, op2, replacement)
        if op2 is root:
            root = replacement
        return root
    return action

def fold_to_zero(op: Op, root: Op) -> Op:
    """Constant-fold ``x * 0`` (or ``0 * x``) to ``0``.

    Unlike the identity rewrites, the result is not the input but a constant, so
    we drop the multiply and its now-dead operand edges and rewire downstream
    consumers to a :class:`ValueOp` holding ``0.0``. A ValueOp is a source node
    (no inputs) whose ``process`` returns the constant directly, so the whole
    ``x`` subgraph is never computed.
    """
    zero_op = ValueOp(0.0)
    for operand in op.inputs:
        operand.outputs = [out for out in operand.outputs if out is not op]
    replace_op_in_outputs(op, zero_op)
    return zero_op if op is root else root


def fold_to_one(op: Op, root: Op) -> Op:
    """Constant-fold ``x ** 0`` to ``1``.

    Parallels :func:`fold_to_zero`: drops the pow op and its dead operand edges
    and rewires downstream consumers to a :class:`ValueOp` holding ``1``.
    """
    one_op = ValueOp(1)
    for operand in op.inputs:
        operand.outputs = [out for out in operand.outputs if out is not op]
    replace_op_in_outputs(op, one_op)
    return one_op if op is root else root


def fold_to_square(op: Op, root: Op) -> Op:
    """Rewrite a self-multiply into a square op, keeping its operand (x * x -> square(x)).

    The operand keeps its single input edge, so the only rewiring needed is to
    swap the multiply for the square in ``x``'s outputs and in the inputs of the
    multiply's consumers.
    """
    x = op.inputs[0]
    square_op = NumericOp(inputs=[], outputs=[], type=NumericOpType.SQUARE)
    x.replace_output(op, square_op)
    square_op.add_input(x)
    replace_op_in_outputs(op, square_op)
    return square_op if op is root else root


match_exp_minus_one = match_two_op_chain(NumericOp, NumericOpType.EXP, NumericOpType.SUBTRACT,
    match2=lambda op: _matches_scalar_const(op, 1, reversed=False),
)

match_add_one_then_log = match_two_op_chain(NumericOp, NumericOpType.ADD, NumericOpType.LOG,
    match1=lambda op: _matches_scalar_const(op, 1),
)

_replace_with_abs = make_replace_two_op_chain_root_safe(
    lambda: NumericOp(inputs=[], outputs=[], type=NumericOpType.ABS)
)

_replace_with_expm1 = make_replace_two_op_chain_root_safe(
    lambda: NumericOp(inputs=[], outputs=[], type=NumericOpType.EXPM1)
)

_replace_with_log1p = make_replace_two_op_chain_root_safe(
    lambda: NumericOp(inputs=[], outputs=[], type=NumericOpType.LOG1P)
)

eliminate_log_exp = rewrite_pass(
    match_two_op_chain(NumericOp, NumericOpType.LOG, NumericOpType.EXP),
    eliminate_two_op_chain_root_safe,
)

eliminate_exp_log = rewrite_pass(
    match_two_op_chain(NumericOp, NumericOpType.EXP, NumericOpType.LOG),
    eliminate_two_op_chain_root_safe,
)

eliminate_expm1_log1p = rewrite_pass(
    match_two_op_chain(NumericOp, NumericOpType.EXPM1, NumericOpType.LOG1P),
    eliminate_two_op_chain_root_safe,
)

eliminate_log1p_expm1 = rewrite_pass(
    match_two_op_chain(NumericOp, NumericOpType.LOG1P, NumericOpType.EXPM1),
    eliminate_two_op_chain_root_safe,
)

eliminate_sqrt_square = rewrite_pass(
    match_two_op_chain(NumericOp, NumericOpType.SQUARE, NumericOpType.SQRT),
    _replace_with_abs,
)

eliminate_identity_operation = rewrite_pass(
    match_identity_operation(NumericOp, NumericOpType.MULTIPLY, 1),
    eliminate_single_op_chain_root_safe,
)

eliminate_abs_abs = rewrite_pass(
    match_two_op_chain(NumericOp, NumericOpType.ABS, NumericOpType.ABS),
    _replace_with_abs,
)

eliminate_add_zero = rewrite_pass(
    match_identity_operation(NumericOp, NumericOpType.ADD, 0),
    eliminate_single_op_chain_root_safe,
)

fold_exp_minus_one = rewrite_pass(match_exp_minus_one, _replace_with_expm1)

eliminate_identity_subtract = rewrite_pass(
    match_identity_operation(NumericOp, NumericOpType.SUBTRACT, 0, reversed=False),
    eliminate_single_op_chain_root_safe,
)


eliminate_any_mul_zero = rewrite_pass(
    match_identity_operation(NumericOp, NumericOpType.MULTIPLY, 0),
    fold_to_zero,
)

eliminate_pow_zero = rewrite_pass(
    match_identity_operation(NumericOp, NumericOpType.POW, 0, reversed=False),
    fold_to_one,
)

# `x ** 1 -> x`. Matched as a first-class identity on `NumericOpType.POW`, reusing the
# same helper as `x * 1` / `x / 1`. `reversed=False` keeps `1 ** x` (which is 1, not x)
# untouched.
#
# TODO(dtype): like `x / 1` below, not always dtype-preserving. `int_array ** 1.0`
# yields float64, and under NEP 50 `int8_array ** np.int64(1)` yields int64, while
# the eliminated result keeps the original dtype. Matching only a weak `int` 1 would
# avoid this.
eliminate_pow_by_one = rewrite_pass(
    match_identity_operation(NumericOp, NumericOpType.POW, 1, reversed=False),
    eliminate_single_op_chain_root_safe,
)

# TODO(dtype): unlike the other identity rewrites (`x*1`, `x+0`, `x-0`), dropping
# `x / 1` is not dtype-preserving. `np.divide` always performs true division, so
# `int_array / 1` yields float64 while the eliminated result keeps the original
# integer dtype. The values are equal but the dtype changes.
eliminate_div_by_one = rewrite_pass(
    match_identity_operation(NumericOp, NumericOpType.DIVIDE, 1, reversed=False),
    eliminate_single_op_chain_root_safe,
)

fold_log_plus_one = rewrite_pass(match_add_one_then_log, _replace_with_log1p)

fold_self_multiply = rewrite_pass(
    match_self_operation(NumericOp, NumericOpType.MULTIPLY),
    fold_to_square,
)


def replace_three_op_chain(op1: Op, op2: Op, op3: Op, replacement: Op):
    """Replace op1 -> op2 -> op3 with replacement: x -> replacement -> downstream."""
    x = op1.inputs[0]
    x.replace_output(op1, replacement)
    replacement.add_input(x)
    for downstream in op3.outputs:
        replacement.add_output(downstream)
        downstream.replace_input(op3, replacement)


def make_replace_three_op_chain_root_safe(make_replacement):
    """Action factory: replace a three-op chain with a new op from make_replacement()."""
    def action(op1: Op, op2: Op, op3: Op, root: Op) -> Op:
        replacement = make_replacement()
        replace_three_op_chain(op1, op2, op3, replacement)
        if op3 is root:
            root = replacement
        return root
    return action


def _logsumexp(x):
    """Numerically stable log(sum(exp(x))): ``c + log(sum(exp(x - c)))``, ``c = max(x)``.

    Note: this is a *stability* rewrite, not a speed one. There is no fused
    kernel here -- the replacement still runs separate NumPy passes, and the
    extra ``max`` / ``subtract`` make it marginally more work than the original
    three-op chain. What it buys is that inputs large enough to overflow ``exp``
    (e.g. ``x = 1000``) return a finite result instead of ``inf``. Collapsing
    three IR nodes into one is a side effect, not the motivation.
    """
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))


def match_log_sum_exp(op):
    """Match ``EXP -> SUM -> LOG`` chain."""
    if not isinstance(op, NumericOp):
        return None
    if op.type is not NumericOpType.EXP:
        return None
    if len(op.outputs) != 1:
        return None

    op2 = op.outputs[0]
    if not isinstance(op2, NumericOp):
        return None
    if op2.type is not NumericOpType.SUM:
        return None
    if len(op2.outputs) != 1:
        return None
    # `_logsumexp` always reduces over the whole array, so any reduction
    # modifier (axis, keepdims, dtype, where, ...) would be silently dropped.
    if op2.args or op2.kwargs:
        return None

    op3 = op2.outputs[0]
    if not isinstance(op3, NumericOp):
        return None
    if op3.type is not NumericOpType.LOG:
        return None

    return (op, op2, op3)


_replace_with_logsumexp = make_replace_three_op_chain_root_safe(
    lambda: NumericOp(inputs=[], outputs=[], func=_logsumexp)
)

eliminate_log_sum_exp = rewrite_pass(
    match_log_sum_exp,
    _replace_with_logsumexp,
)
