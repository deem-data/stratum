from stratum.optimizer.ir._ops import CallOp, Op
import numpy as np
from enum import Enum

class NumericOpType(Enum):
    GENERIC = "generic"
    LOG = "log"
    EXP = "exp"
class NumericOp(Op):
    fields = ["func", "args", "kwargs", "type"]
    func = None

    def __init__(self, func, args, kwargs, inputs, outputs):
        if func is np.log:
            self.type = NumericOpType.LOG
            name = "log"
        elif func is np.exp:
            self.type = NumericOpType.EXP
            name = "exp"
        else:
            self.type = NumericOpType.GENERIC
            self.func = func
            name = func.__name__
        super().__init__(name=name, inputs=inputs, outputs=outputs)

        self.args = args
        self.kwargs = kwargs

    def process(self, mode: str, environment: dict, inputs: list):
        if self.type == NumericOpType.GENERIC:
            return self.func(inputs[0], *self.args, **self.kwargs)
        elif self.type == NumericOpType.LOG:
            return np.log(inputs[0])
        elif self.type == NumericOpType.EXP:
            return np.exp(inputs[0])
        else:
            raise ValueError(f"Unsupported numeric operation type: {self.type}")


def make_numeric_op(op: CallOp) -> NumericOp:
    op.args = op.args[1:]
    new_op = NumericOp(func=op.func, args=op.args, kwargs=op.kwargs, inputs=op.inputs, outputs=op.outputs)
    return new_op

def extract_numeric_op(op: Op, root: Op) -> tuple[Op, bool]:
    new_op = None
    if isinstance(op, CallOp):
        if op.func is np.log:
            new_op = make_numeric_op(op)
        elif op.func is np.exp:
            new_op = make_numeric_op(op)
        # if op is some other function from np package, make a generic numeric op
        elif op.func.__module__ == "numpy":
            new_op = make_numeric_op(op)

    if new_op is None:
        return root, False
    else:
        op.replace_input_of_outputs(new_op)
        op.replace_output_of_inputs(new_op)
        if op is root:
            root = new_op
        return root, True