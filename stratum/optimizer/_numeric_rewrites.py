from stratum.optimizer.ir._numeric_ops import NumericOp, NumericOpType
from stratum.optimizer._op_utils import rewrite_pass
from stratum.optimizer.ir._ops import Op

def match_two_op_chain(op_cls, type1, type2):
    """Match predicate for two consecutive ops of the same class with given types."""
    def match(op):
        if isinstance(op, op_cls) and op.type is type1 and len(op.outputs) == 1:
            op2 = op.outputs[0]
            if isinstance(op2, op_cls) and op2.type is type2:
                return (op, op2)
        return None
    return match

def eliminate_two_op_chain(op1, op2):
    """Remove a redundant pair of inverse ops: y = f(op2(op1(x))) -> y = f(x).

    Rewires the DAG in-place so that op1's input connects directly to op2's output.
    """
    x = op1.inputs[0]
    if len(op2.outputs) == 1:
        y = op2.outputs[0]
        y.replace_input(op2, x)
        x.replace_output(op1, y)
    else:
        x.outputs = []

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



eliminate_log_exp = rewrite_pass(
    match_two_op_chain(NumericOp, NumericOpType.LOG, NumericOpType.EXP),
    eliminate_two_op_chain_root_safe,
)

eliminate_exp_log = rewrite_pass(
    match_two_op_chain(NumericOp, NumericOpType.EXP, NumericOpType.LOG),
    eliminate_two_op_chain_root_safe,
)

_replace_with_abs = make_replace_two_op_chain_root_safe(lambda : NumericOp(inputs=[], outputs=[], type=NumericOpType.ABS))

eliminate_sqrt_square = rewrite_pass(
    match_two_op_chain(NumericOp, NumericOpType.SQUARE, NumericOpType.SQRT),
    _replace_with_abs,
)

