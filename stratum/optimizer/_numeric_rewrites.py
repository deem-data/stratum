from stratum.optimizer.ir._numeric_ops import NumericOp, NumericOpType
from stratum.optimizer._op_utils import rewrite_pass, replace_op_in_outputs
from stratum.optimizer.ir._ops import Op, ValueOp


def _is_scalar_const(value) -> bool:
    """True if ``value`` is a scalar numeric constant safe to compare with ``==``.

    Guards against ndarray constants (e.g. ``df * np.array([...])``), whose
    ``== const`` yields an array and raises "truth value of an array is
    ambiguous" when used in a boolean context.
    """
    return isinstance(value, (int, float))


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
    """Replace ``op`` with a ``ValueOp(0.0)`` and drop its operand edges.

    Mechanism only: the caller must ensure folding to 0 is valid for its op
    (``x * 0`` always is; ``0 / x`` only when x is non-zero and non-NaN).
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

# NOT semantics-preserving in general: 0/x == 0 only when x != 0 (0/0 = NaN),
# and folding skips evaluating the divisor. Opt-in via the zero_div flag
# (default off); enable only when the divisor is known non-zero.
eliminate_zero_div = rewrite_pass(
    match_identity_operation(NumericOp, NumericOpType.DIVIDE, 0, reversed=True),
    fold_to_zero,
)

eliminate_pow_zero = rewrite_pass(
    match_identity_operation(NumericOp, NumericOpType.POW, 0, reversed=False),
    fold_to_one,
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
