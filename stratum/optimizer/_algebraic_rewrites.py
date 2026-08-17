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
    eliminate_pow_by_one,
    eliminate_identity_subtract,
    eliminate_any_mul_zero,
    eliminate_div_by_one,
    fold_log_plus_one,
    eliminate_log_sum_exp,
    fold_self_multiply,
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
    pow_by_one: bool = True
    identity_subtract: bool = True
    any_mul_zero: bool = True
    div_by_one: bool = True
    log_plus_one: bool = True
    log_sum_exp: bool = True
    self_multiply: bool = True


def algebraic_rewrites(root: Op, config: AlgebraicRewritesConfig) -> Op:
    """Run all enabled algebraic rewrites, one pass per rewrite.

    Each rewrite runs exactly once, so a rewrite that *produces* a pattern another
    rewrite *consumes* only fires if it is ordered first. `self_multiply` is placed
    before `sqrt_square` for that reason: it turns `sqrt(x * x)` into
    `sqrt(square(x))`, which `sqrt_square` then folds to `abs(x)`.

    A single order cannot satisfy every such dependency (the enables-relation over
    the rewrites has cycles), so some reducible graphs survive -- e.g. `(x - 0) * x`
    stays a multiply because `identity_subtract` runs after `self_multiply`. See
    #188 for running the whole set in a bounded loop, which removes the dependency
    on ordering altogether.
    """
    start = start_time()
    if config.identity_op:
        root = eliminate_identity_operation(root)
    if config.add_zero:
        root = eliminate_add_zero(root)
    if config.div_by_one:
        root = eliminate_div_by_one(root)
    if config.log_exp:
        root = eliminate_log_exp(root)
    if config.exp_log:
        root = eliminate_exp_log(root)
    if config.abs_abs:
        root = eliminate_abs_abs(root)
    if config.self_multiply:
        root = fold_self_multiply(root)
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
    if config.pow_by_one:
        root = eliminate_pow_by_one(root)
    if config.any_mul_zero:
        root = eliminate_any_mul_zero(root)
    if config.log_sum_exp:
        # Stability, not speed -- see _logsumexp. Enabled by default because the
        # unstable form silently overflows to inf.
        root = eliminate_log_sum_exp(root)
    log_time("algebraic_rewrite", start)
    return root
