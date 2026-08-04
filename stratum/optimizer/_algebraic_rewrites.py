from dataclasses import dataclass
from stratum.optimizer._numeric_rewrites import (
    eliminate_exp_log,
    eliminate_expm1_log1p,
    eliminate_log_exp,
    eliminate_log1p_expm1,
    eliminate_sqrt_square,
    eliminate_identity_operation,
    eliminate_abs_abs,
    eliminate_add_zero,
    fold_exp_minus_one,
    eliminate_pow_zero,
    eliminate_zero_div,
    eliminate_identity_subtract,
    eliminate_any_mul_zero,
    eliminate_div_by_one,
    fold_log_plus_one,
)
from stratum.optimizer.ir._ops import Op
from stratum.utils._utils import start_time, log_time
import logging
from time import perf_counter

logger = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class AlgebraicRewritesConfig:
    log_exp: bool = True
    exp_log: bool = True
    sqrt_square: bool = True
    log1p_expm1: bool = True
    expm1_log1p: bool = True
    identity_op: bool = True
    abs_abs: bool = True
    add_zero: bool = True
    exp_minus_one: bool = True
    pow_zero: bool = True
    zero_div: bool = False
    identity_subtract: bool = True
    any_mul_zero: bool = True
    div_by_one: bool = True
    log_plus_one: bool = True


def algebraic_rewrites(root: Op, config: AlgebraicRewritesConfig) -> Op:
    """Run all enabled algebraic rewrites, one pass per rewrite."""
    start = start_time()
    if config.identity_op:
        root = eliminate_identity_operation(root)
    if config.add_zero:
        root = eliminate_add_zero(root)
    if config.div_by_one:
        root = eliminate_div_by_one(root)
    if config.zero_div:
        root = eliminate_zero_div(root)
    if config.log_exp:
        root = eliminate_log_exp(root)
    if config.exp_log:
        root = eliminate_exp_log(root)
    if config.abs_abs:
        root = eliminate_abs_abs(root)
    if config.sqrt_square:
        root = eliminate_sqrt_square(root)
    if config.exp_minus_one:
        root = fold_exp_minus_one(root)
    if config.log_plus_one:
        root = fold_log_plus_one(root)
    if config.log1p_expm1:
        root = eliminate_log1p_expm1(root)
    if config.expm1_log1p:
        root = eliminate_expm1_log1p(root)
    if config.identity_subtract:
        root = eliminate_identity_subtract(root)
    if config.pow_zero:
        root = eliminate_pow_zero(root)
    if config.any_mul_zero:
        root = eliminate_any_mul_zero(root)
    log_time("algebraic_rewrite", start)
    return root
