"""Dataframe plan rewrites (topology / column-binding composition).

Parallel to :mod:`stratum.optimizer._algebraic_rewrites`, which handles numeric
expression identities. This pass family rewrites dataframe plan structure.
"""
from __future__ import annotations

from dataclasses import dataclass

from stratum.optimizer._map_rewrites import fuse_assign_maps
from stratum.optimizer.ir._ops import Op
from stratum.utils._utils import start_time, log_time


@dataclass(frozen=True, slots=True)
class DataframeRewritesConfig:
    fuse_assign_maps: bool = True
    # Room for later: selection fusion, projection pushdown, ...


def dataframe_rewrites(root: Op, config: DataframeRewritesConfig) -> Op:
    """Run enabled dataframe rewrites, one pass per rewrite."""
    start = start_time()
    if config.fuse_assign_maps:
        root = fuse_assign_maps(root)
    log_time("dataframe_rewrites", start)
    return root
