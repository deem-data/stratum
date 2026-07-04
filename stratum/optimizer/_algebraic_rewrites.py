from dataclasses import dataclass
from stratum.optimizer._numeric_rewrites import (
    eliminate_exp_log,
    eliminate_expm1_log1p,
    eliminate_log_exp,
    eliminate_log1p_expm1,
    eliminate_sqrt_square,
    eliminate_one_mul_x,
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
    one_mul_x: bool = True    


def algebraic_rewrites(root: Op, config: AlgebraicRewritesConfig) -> Op:
    """Run all enabled algebraic rewrites, one pass per rewrite."""
    start = start_time()
    if config.log_exp:
        root = eliminate_log_exp(root)
    if config.exp_log:
        root = eliminate_exp_log(root)
    if config.sqrt_square:
        root = eliminate_sqrt_square(root)
    if config.log1p_expm1:
        root = eliminate_log1p_expm1(root)
    if config.expm1_log1p:
        root = eliminate_expm1_log1p(root)
    if config.x_mul_one:
        root = eliminate_one_mul_x(root)

        
    log_time("algebraic_rewrite", start)
    return root
