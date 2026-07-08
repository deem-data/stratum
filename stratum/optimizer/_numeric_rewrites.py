from stratum.optimizer.ir._numeric_ops import NumericOp, NumericOpType
from stratum.optimizer._op_utils import rewrite_pass, replace_op_in_outputs
from stratum.optimizer.ir._ops import Op, ValueOp


def match_two_op_chain(op_cls, type1, type2):
    """Match predicate for two consecutive ops of the same class with given types."""
    def match(op):
        if isinstance(op, op_cls) and op.type is type1 and len(op.outputs) == 1:
            op2 = op.outputs[0]
            if isinstance(op2, op_cls) and op2.type is type2:
                return (op, op2)
        return None
    return match


def match_identity_operation(op_cls, type1, const):
    def match(op1):
        if isinstance(op1, op_cls) and op1.type == type1:
            if op1.opt_operand is None and op1.constant == const:
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


def match_constant_foldable(op):
    """Match a NumericOp whose variable inputs are all ValueOps (constants).

    For binary ops with a constant operand (``opt_operand is None``) only the
    primary input must be a ValueOp; the constant side is already a Python scalar.
    For var-var binary ops both inputs must be ValueOps.
    """
    if not isinstance(op, NumericOp):
        return None
    if not op.inputs:
        return None
    if op.opt_operand is not None:
        # var-var binary: every input must be a ValueOp
        if not all(isinstance(inp, ValueOp) for inp in op.inputs):
            return None
    else:
        # unary or var-const binary: only the primary input must be a ValueOp
        if not isinstance(op.inputs[0], ValueOp):
            return None
    return (op,)


def fold_constant_op_root_safe(op, root):
    """Evaluate *op* with its constant inputs and replace it with a ValueOp."""
    input_values = [inp.value for inp in op.inputs]
    result = op.process("fit", input_values)
    new_op = ValueOp(result)
    replace_op_in_outputs(op, new_op)
    if op is root:
        root = new_op
    return root



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

eliminate_constant_folding = rewrite_pass(
    match_constant_foldable,
    fold_constant_op_root_safe,
)
