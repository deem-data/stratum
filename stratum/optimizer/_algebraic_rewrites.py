from dataclasses import dataclass
from stratum.optimizer._numeric_rewrites import eliminate_log_exp, eliminate_exp_log, eliminate_sqrt_square
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


def algebraic_rewrites(root: Op, config: AlgebraicRewritesConfig) -> Op:
    """Run all enabled algebraic rewrites, one pass per rewrite."""
    start = start_time()
    if config.log_exp:
        root = eliminate_log_exp(root)
    if config.exp_log:
        root = eliminate_exp_log(root)
    if config.sqrt_square:
        root = eliminate_sqrt_square(root)
    log_time("algebraic_rewrite", start)
    return root
