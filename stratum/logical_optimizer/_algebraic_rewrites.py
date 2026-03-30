from dataclasses import dataclass
from typing import Any, Callable

from stratum.logical_optimizer._numeric_ops import NumericOp
from stratum.logical_optimizer._op_utils import topological_iterator
from stratum.logical_optimizer._numeric_ops import NumericOpType

RewriteFn = Callable[[NumericOp, Any], Any]

@dataclass(frozen=True, slots=True)
class AlgebraicRewritesConfig:
    log_exp: bool = True

def eliminate_two_op_chain(op1, op2):
    # y = f(op2(op1(x))) -> y = f(x)
    x = op1.inputs[0]
    if len(op2.outputs) == 1:
        y = op2.outputs[0]
        y.replace_input(op2, x)
        x.replace_output(op1, y)
    else:
        x.outputs = []


def eliminate_two_op_chain_sink_safe(op1: NumericOp, op2: NumericOp, sink: Any) -> Any:
    eliminate_two_op_chain(op1, op2)
    if op2 is sink:
        sink = op1.inputs[0]
    return sink


def _build_numeric_rewrite_dispatch(config: AlgebraicRewritesConfig) -> dict[tuple[NumericOpType, NumericOpType], RewriteFn]:
    dispatch: dict[tuple[NumericOpType, NumericOpType], RewriteFn] = {}
    if config.log_exp :
        dispatch[(NumericOpType.LOG, NumericOpType.EXP)] = eliminate_two_op_chain_sink_safe
        dispatch[(NumericOpType.EXP, NumericOpType.LOG)] = eliminate_two_op_chain_sink_safe
    return dispatch


def algebraic_rewrites(sink, config: AlgebraicRewritesConfig | None = None):
    numeric_rewrite_dispatch = _build_numeric_rewrite_dispatch(config)
    if not numeric_rewrite_dispatch:
        return sink

    for op1 in topological_iterator(sink):
        if not isinstance(op1, NumericOp) or len(op1.outputs) != 1:
            continue

        op2 = op1.outputs[0]
        if not isinstance(op2, NumericOp):
            continue

        rewrite_fn = numeric_rewrite_dispatch.get((op1.type, op2.type))
        if rewrite_fn is not None:
            sink = rewrite_fn(op1, op2,sink)
    return sink