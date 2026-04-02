from stratum.optimizer.ir._ops import CallOp, Op
from stratum.optimizer._op_utils import topological_iterator
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

    def process(self, mode: str, environment: dict):
        if self.type == NumericOpType.GENERIC:
            self.intermediate = self.func(self.inputs[0].intermediate, *self.args, **self.kwargs)
        elif self.type == NumericOpType.LOG:
            self.intermediate = np.log(self.inputs[0].intermediate)
        elif self.type == NumericOpType.EXP:
            self.intermediate = np.exp(self.inputs[0].intermediate)
        else:
            raise ValueError(f"Unsupported numeric operation type: {self.type}")


def make_numeric_op(op: CallOp) -> NumericOp:
    op.args = op.args[1:]
    new_op = NumericOp(func=op.func, args=op.args, kwargs=op.kwargs, inputs=op.inputs, outputs=op.outputs)
    return new_op

def to_numeric_op(sink: Op) -> Op:
    """ Detect and convert the numeric ops in the dag to the stratum's NumericOps."""
    for op in topological_iterator(sink):
        new_op = None
        if isinstance(op, CallOp):
            if op.func is np.log:
                new_op = make_numeric_op(op)
            elif op.func is np.exp:
                new_op = make_numeric_op(op)
            # if op is some other function from np package, make a generic numeric op
            elif op.func.__module__ == "numpy":
                new_op = make_numeric_op(op)



        if new_op is not None:
            op.replace_input_of_outputs(new_op)
            op.replace_output_of_inputs(new_op)
            if op is sink:
                sink = new_op
    return sink