from stratum.optimizer.ir._ops import BinOp, CallOp, Op, DATA_OP_PLACEHOLDER
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
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"

_ARITH_OP_MAP = {
    operator.add: NumericOpType.ADD,
    operator.sub: NumericOpType.SUBTRACT,
    operator.mul: NumericOpType.MULTIPLY,
    operator.truediv: NumericOpType.DIVIDE,
}

_NUMPY_BINARY_MAP = {
    np.add: NumericOpType.ADD,
    np.subtract: NumericOpType.SUBTRACT,
    np.multiply: NumericOpType.MULTIPLY,
    np.divide: NumericOpType.DIVIDE,
}

_BINARY_TYPES = frozenset(_ARITH_OP_MAP.values())
_BINARY_NUMPY_FUNCS = frozenset(_NUMPY_BINARY_MAP.keys())

class NumericOp(Op):
    fields = ["func", "args", "kwargs", "type", "constant", "reversed"]
    func = None

    def __init__(self, inputs=None, outputs=None, func=None, args=(), kwargs=None, type: NumericOpType = None, constant=None, reversed=False):
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
            elif func is np.add:
                self.type = NumericOpType.ADD
                name = "add"
            elif func is np.subtract:
                self.type = NumericOpType.SUBTRACT
                name = "subtract"
            elif func is np.multiply:
                self.type = NumericOpType.MULTIPLY
                name = "multiply"
            elif func is np.divide:
                self.type = NumericOpType.DIVIDE
                name = "divide"
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
        self.constant = constant
        self.reversed = reversed

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
        elif self.type in _BINARY_TYPES:
            left, right = (self.constant, inputs[0]) if self.reversed else (inputs[0], self.constant)
            if self.type == NumericOpType.ADD:
                return np.add(left, right)
            elif self.type == NumericOpType.SUBTRACT:
                return np.subtract(left, right)
            elif self.type == NumericOpType.MULTIPLY:
                return np.multiply(left, right)
            elif self.type == NumericOpType.DIVIDE:
                return np.divide(left, right)
            else:
                raise ValueError(f"Unsupported binary numeric operation type: {self.type}")
        else:
            raise ValueError(f"Unsupported numeric operation type: {self.type}")


def make_numeric_op(op: CallOp) -> NumericOp:
    remaining_args = op.args[1:]
    return NumericOp(func=op.func, args=remaining_args, kwargs=op.kwargs, inputs=op.inputs, outputs=op.outputs)

def make_binary_numeric_op(op: CallOp, type: NumericOpType) -> NumericOp:
    args = op.args or ()
    if len(args) == 2 and args[0] is DATA_OP_PLACEHOLDER:
        constant, reversed = args[1], False
    elif len(args) == 2 and args[1] is DATA_OP_PLACEHOLDER:
        constant, reversed = args[0], True
    else:
        raise ValueError(                                                                                                                                        
            f"make_binary_numeric_op called with args that are not a single-placeholder pair: {args}"                                                          
        )      
    return NumericOp(type=type, constant=constant, reversed=reversed, inputs=op.inputs, outputs=op.outputs)


def _is_binary_extractable(op: CallOp) -> bool:
    args = op.args or ()
    if len(args) != 2:
        return False
    l_ph = args[0] is DATA_OP_PLACEHOLDER
    r_ph = args[1] is DATA_OP_PLACEHOLDER
    return l_ph != r_ph

def extract_numeric_op(op: Op, root: Op) -> tuple[Op, bool]:
    new_op = None
    if isinstance(op, BinOp) and op.op is operator.pow and op.right == 2:
        new_op = NumericOp(func=np.square, args=(), kwargs={}, inputs=op.inputs, outputs=op.outputs)
    elif isinstance(op, BinOp) and op.op in _ARITH_OP_MAP:
        l_ph = op.left is DATA_OP_PLACEHOLDER
        r_ph = op.right is DATA_OP_PLACEHOLDER
        if l_ph != r_ph:  # var op const or const op var, not var op var
            constant = op.right if l_ph else op.left
            new_op = NumericOp(
                type=_ARITH_OP_MAP[op.op],
                constant=constant,
                reversed=not l_ph,  # True when const is on the left
                inputs=op.inputs,
                outputs=op.outputs,
            )
    elif isinstance(op, CallOp):
        if op.func is np.log:
            new_op = make_numeric_op(op)
        elif op.func is np.exp:
            new_op = make_numeric_op(op)
        elif op.func is np.sqrt:
            new_op = make_numeric_op(op)
        elif op.func is np.abs:
            new_op = make_numeric_op(op)
        elif op.func is np.square:
            new_op = make_numeric_op(op)
        elif op.func in _NUMPY_BINARY_MAP and _is_binary_extractable(op):
            new_op = make_binary_numeric_op(op, _NUMPY_BINARY_MAP[op.func])
        # if op is some other function from np package, make a generic numeric op
        elif op.func.__module__ == "numpy" and op.func not in _BINARY_NUMPY_FUNCS:
            new_op = make_numeric_op(op)

    if new_op is None:
        return root, False
    else:
        op.replace_input_of_outputs(new_op)
        op.replace_output_of_inputs(new_op)
        if op is root:
            root = new_op
        return root, True