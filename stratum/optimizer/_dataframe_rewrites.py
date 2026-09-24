from __future__ import annotations

from dataclasses import dataclass

from stratum.optimizer._map_rewrites import fuse_assign_maps
from stratum.optimizer.logical._ops import Op
from stratum.utils._utils import start_time, log_time


@dataclass(frozen=True, slots=True)
class DataframeRewritesConfig:
    fuse_assign_maps: bool = True


def dataframe_rewrites(root: Op, config: DataframeRewritesConfig) -> Op:
    if config.fuse_assign_maps:
        start = start_time()
        root = fuse_assign_maps(root)
        log_time("map_fusion took", start)
    return root
