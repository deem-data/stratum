from stratum.optimizer.ir._numeric_ops import NumericOp, NumericOpType
from stratum.optimizer._op_utils import rewrite_pass, replace_op_in_outputs
from stratum.optimizer.ir._ops import Op, ValueOp, BinOp, OperandRef
import operator


def match_two_op_chain(op_cls, type1, type2):
    """Match predicate for two consecutive ops of the same class with given types."""
    def match(op):
        if isinstance(op, op_cls) and op.type is type1 and len(op.outputs) == 1:
            op2 = op.outputs[0]
            if isinstance(op2, op_cls) and op2.type is type2:
                return (op, op2)
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
            if op1.opt_operand is None and op1.constant == const:
                if reversed is None or op1.reversed == reversed:
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


def match_exp_minus_one(op):
    """Match exp(x) - 1"""
    if isinstance(op, NumericOp) and op.type is NumericOpType.EXP and len(op.outputs) == 1:
        op2 = op.outputs[0]
        if (isinstance(op2, NumericOp) and op2.type is NumericOpType.SUBTRACT
                and op2.opt_operand is None and op2.constant == 1 and not op2.reversed):
            return (op, op2)
    return None

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

_replace_with_abs = make_replace_two_op_chain_root_safe(
    lambda: NumericOp(inputs=[], outputs=[], type=NumericOpType.ABS)
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

_replace_with_expm1 = make_replace_two_op_chain_root_safe(
    lambda: NumericOp(inputs=[], outputs=[], type=NumericOpType.EXPM1)
)

eliminate_exp_minus_one = rewrite_pass(match_exp_minus_one, _replace_with_expm1)

eliminate_identity_subtract = rewrite_pass(
    match_identity_operation(NumericOp, NumericOpType.SUBTRACT, 0, reversed=False),
    eliminate_single_op_chain_root_safe,
)


eliminate_any_mul_zero = rewrite_pass(
    match_identity_operation(NumericOp, NumericOpType.MULTIPLY, 0),
    fold_to_zero,
)


def match_pow_zero(op):
    """Match ``x ** 0`` (a variable raised to the power zero).

    Only matches when the base is a variable (OperandRef), not a constant,
    so that ``0 ** 0`` is *not* rewritten — the ``x != 0`` semantics are
    the caller's responsibility.
    """
    if isinstance(op, BinOp) and op.op is operator.pow:
        if op.right == 0 and isinstance(op.left, OperandRef):
            return (op,)
    return None


def annihilate_pow_zero_root_safe(op, root):
    """Replace ``x ** 0`` with ``ValueOp(1)``."""
    new_op = ValueOp(1)
    replace_op_in_outputs(op, new_op)
    if op is root:
        root = new_op
    return root


eliminate_pow_zero = rewrite_pass(
    match_pow_zero,
    annihilate_pow_zero_root_safe,
)
