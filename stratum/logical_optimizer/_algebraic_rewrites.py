from stratum.logical_optimizer._numeric_ops import NumericOp
from stratum.logical_optimizer._op_utils import topological_iterator
from stratum.logical_optimizer._numeric_ops import NumericOpType

def eliminate_two_op_chain(op1, op2):
    # y = f(op2(op1(x))) -> y = f(x)
    x = op1.inputs[0]
    if len(op2.outputs) == 1:
        y = op2.outputs[0]
        y.replace_input(op2, x)
        x.replace_output(op1, y)
    else:
        x.outputs = []

def algebraic_rewrites(sink):
    for op1 in topological_iterator(sink):
        if isinstance(op1, NumericOp):
            if len(op1.outputs) == 1 and isinstance(op1.outputs[0], NumericOp):
                op2 = op1.outputs[0]
                type1 = op1.type
                type2 = op2.type
                if type1 == NumericOpType.LOG and type2 == NumericOpType.EXP or type1 == NumericOpType.EXP and type2 == NumericOpType.LOG:
                    # y = f(log(exp(x)))  OR  y = f(exp(log(x))) -> y = f(x)
                    eliminate_two_op_chain(op1, op2)
                    if op2 is sink:
                        sink = op1.inputs[0]
    return sink