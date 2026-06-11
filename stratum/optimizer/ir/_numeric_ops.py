from stratum.optimizer.ir._ops import BinOp, CallOp, Op, OperandRef
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
    LOG1P = "log1p"
    EXPM1 = "expm1"
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

_NUMPY_UNARY_MAP = {
    np.log: NumericOpType.LOG,
    np.exp: NumericOpType.EXP,
    np.sqrt: NumericOpType.SQRT,
    np.abs: NumericOpType.ABS,
    np.square: NumericOpType.SQUARE,
    np.log1p: NumericOpType.LOG1P,
    np.expm1: NumericOpType.EXPM1,
}

_UNARY_NUMPY_FUNCS = frozenset(_NUMPY_UNARY_MAP.keys())
_BINARY_TYPES = frozenset(_ARITH_OP_MAP.values())
_BINARY_NUMPY_FUNCS = frozenset(_NUMPY_BINARY_MAP.keys())

class NumericOp(Op):
    fields = ["func", "args", "kwargs", "type", "constant", "opt_operand", "reversed"]
    func = None

    def __init__(self, inputs=None, outputs=None, func=None, args=(), kwargs=None, type: NumericOpType = None, constant=None, opt_operand=None, reversed=False):
        if func is not None:
            self.type = _NUMPY_UNARY_MAP.get(func)
            self.type = self.type if self.type else _NUMPY_BINARY_MAP.get(func)
            if self.type is None:
                self.type = NumericOpType.GENERIC
                self.func = func
                name = func.__name__
            else:
                name = self.type.value
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
        self.opt_operand = opt_operand
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
        elif self.type == NumericOpType.LOG1P:
            return np.log1p(inputs[0])
        elif self.type == NumericOpType.EXPM1:
            return np.expm1(inputs[0])
        elif self.type in _BINARY_TYPES:
            # The primary operand is always input 0 (bound first); the optional
            # second operand is referenced explicitly so x op x (single edge) works.
            primary = inputs[0]
            operand = inputs[self.opt_operand.k] if isinstance(self.opt_operand, OperandRef) else self.constant
            left, right = (operand, primary) if self.reversed else (primary, operand)
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


def make_unary_numeric_op(op: CallOp) -> NumericOp:
    remaining_args = op.args[1:]
    return NumericOp(func=op.func, args=remaining_args, kwargs=op.kwargs, inputs=op.inputs, outputs=op.outputs)

def make_binary_numeric_op(op: CallOp, type: NumericOpType) -> NumericOp:
    args = op.args or ()
    if len(args) != 2:
        raise ValueError(
            f"make_binary_numeric_op called with args that are not a pair: {args}"
        )
    l_ph = isinstance(args[0], OperandRef)
    r_ph = isinstance(args[1], OperandRef)
    if l_ph and r_ph:
        extra = dict(opt_operand=args[1], reversed=False)
    elif l_ph:
        extra = dict(constant=args[1], reversed=False)
    elif r_ph:
        extra = dict(constant=args[0], reversed=True)
    else:
        raise ValueError(
            f"make_binary_numeric_op called with args that have no placeholder: {args}"
        )
    return NumericOp(type=type, inputs=op.inputs, outputs=op.outputs, **extra)


def extract_numeric_op(op: Op, root: Op) -> tuple[Op, bool]:
    new_op = None
    if isinstance(op, BinOp) and op.op is operator.pow and op.right == 2:
        new_op = NumericOp(func=np.square, args=(), kwargs={}, inputs=op.inputs, outputs=op.outputs)
    elif isinstance(op, BinOp) and op.op in _ARITH_OP_MAP:
        l_ph = isinstance(op.left, OperandRef)
        r_ph = isinstance(op.right, OperandRef)
        extra = None
        if l_ph and r_ph:
            extra = dict(opt_operand=op.right, reversed=False)
        elif l_ph:
            extra = dict(constant=op.right, reversed=False)
        elif r_ph:
            extra = dict(constant=op.left, reversed=True)
        if extra is not None:
            new_op = NumericOp(
                type=_ARITH_OP_MAP[op.op],
                inputs=op.inputs,
                outputs=op.outputs,
                **extra,
            )
    elif isinstance(op, CallOp):
        if op.func in _UNARY_NUMPY_FUNCS:
            new_op = make_unary_numeric_op(op)
        elif op.func in _NUMPY_BINARY_MAP:
            new_op = make_binary_numeric_op(op, _NUMPY_BINARY_MAP[op.func])
        # if op is some other function from np package, make a generic numeric op
        elif op.func.__module__ == "numpy" and op.func not in _BINARY_NUMPY_FUNCS:
            new_op = make_unary_numeric_op(op)

    if new_op is None:
        return root, False
    else:
        op.replace_input_of_outputs(new_op)
        op.replace_output_of_inputs(new_op)
        if op is root:
            root = new_op
        return root, True
