from stratum.optimizer.ir._ops import BinOp, CallOp, Op
import operator
import numpy as np
from enum import Enum

class NumericOpType(Enum):
    GENERIC = "generic"
    LOG = "log"
    EXP = "exp"
    SQRT = "sqrt"
    ABS = "abs"
    SQUARE = "square"

class NumericOp(Op):
    fields = ["func", "args", "kwargs", "type"]
    func = None

    def __init__(self, inputs, outputs, func=None, args=(), kwargs=None, type: NumericOpType = None):
        if func is not None:
            if func is np.log:
                self.type = NumericOpType.LOG
                name = "log"
            elif func is np.exp:
                self.type = NumericOpType.EXP
                name = "exp"
            elif func is np.sqrt:
                self.type = NumericOpType.SQRT
                name = "sqrt"
            elif func is np.abs:
                self.type = NumericOpType.ABS
                name = "abs"
            elif func is np.square:
                self.type = NumericOpType.SQUARE
                name = "square"
            else:
                self.type = NumericOpType.GENERIC
                self.func = func
                name = func.__name__
        elif type is not None:
            if type == NumericOpType.GENERIC:
                raise ValueError("GENERIC type requires a func")
            self.type = type
            name = type.value
        else:
            raise ValueError("Either func or type must be provided")

        super().__init__(name=name, inputs=inputs, outputs=outputs)
        self.args = args
        self.kwargs = kwargs or {}

    def process(self, mode: str, environment: dict, inputs: list):
        if self.type == NumericOpType.GENERIC:
            return self.func(inputs[0], *self.args, **self.kwargs)
        elif self.type == NumericOpType.LOG:
            return np.log(inputs[0])
        elif self.type == NumericOpType.EXP:
            return np.exp(inputs[0])
        elif self.type == NumericOpType.SQRT:
            return np.sqrt(inputs[0])
        elif self.type == NumericOpType.ABS:
            return np.abs(inputs[0])
        elif self.type == NumericOpType.SQUARE:
            return np.square(inputs[0])
        else:
            raise ValueError(f"Unsupported numeric operation type: {self.type}")


def make_numeric_op(op: CallOp) -> NumericOp:
    op.args = op.args[1:]
    new_op = NumericOp(func=op.func, args=op.args, kwargs=op.kwargs, inputs=op.inputs, outputs=op.outputs)
    return new_op

def extract_numeric_op(op: Op, root: Op) -> tuple[Op, bool]:
    new_op = None
    if isinstance(op, BinOp) and op.op is operator.pow and op.right == 2:
        new_op = NumericOp(func=np.square, args=(), kwargs={}, inputs=op.inputs, outputs=op.outputs)
    elif isinstance(op, CallOp):
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