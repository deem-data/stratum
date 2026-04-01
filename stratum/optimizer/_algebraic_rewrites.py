from dataclasses import dataclass
from stratum.optimizer._numeric_rewrites import eliminate_log_exp, eliminate_exp_log
from stratum.optimizer.ir._ops import Op

@dataclass(frozen=True, slots=True)
class AlgebraicRewritesConfig:
    log_exp: bool = True
    exp_log: bool = True


def algebraic_rewrites(root: Op, config: AlgebraicRewritesConfig) -> Op:
    """Run all enabled algebraic rewrites, one pass per rewrite."""
    if config.log_exp:
        root = eliminate_log_exp(root)
    if config.exp_log:
        root = eliminate_exp_log(root)
    return root
